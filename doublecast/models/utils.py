"""
Copyright 2026 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import json
import torch
import re
import pandas as pd
import numpy as np
from functools import partial

from doublecast.models.dual_t5 import DualT5DecoderBlock
from doublecast.data.prompts import PROMPTS

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from textwrap import wrap

from transformers.trainer_callback import TrainerCallback

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import sys
import json
from pathlib import Path
from functools import partial
from typing import Dict, List, Tuple, Union, Any, Optional

# Add the parent directory to the Python path
script_dir = Path(__file__).absolute().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch import nn
from transformers import (
    Trainer,
    set_seed
)
set_seed(42)

from transformers.trainer import nested_detach, is_sagemaker_mp_enabled
from transformers.trainer_callback import TrainerCallback

# Create stubs for SageMaker MP functions
def smp_forward_only(model, inputs):
    """Stub function that just calls the model directly."""
    return model(**inputs)

def smp_nested_concat(tensor):
    """Stub function that returns the tensor unchanged."""
    return tensor


class StagedCustomTrainer(Trainer):
    def __init__(self, *args, ffn_lr_multiplier: float = 0.1, text_lr_multiplier: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffn_lr_multiplier = ffn_lr_multiplier
        self.text_lr_multiplier = text_lr_multiplier
        print(f"Initialized CustomTrainer. FFN LR = Base LR * {self.ffn_lr_multiplier}, "
              f"Text LR = Base LR * {self.text_lr_multiplier}")

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Save the ground truth labels from the batch for CRPS
        future_targets = inputs.get("future_targets")
        scale = inputs.get("scale")

        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        # labels may be popped; grab them first
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        # Also produce stochastic predictions for CRPS downstream
        inputs_for_sampling = dict(inputs)
        inputs_for_sampling["labels"] = None
        with torch.no_grad():
            preds = model(**inputs_for_sampling)

        preds_with_scale = (preds, future_targets, scale)
        logits = (logits, preds_with_scale)
        return (loss, logits, labels)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        args = self.args
        base_lr = args.learning_rate

        # Buckets
        custom_attn_params = []   # decoder.layer[2].EncDecAttention + layer_norm
        ffn_params = []           # decoder.layer[3]
        text_params = []          # any param with "text_encoder" in its name
        unexpected_trainable_params = []

        model_base = getattr(model, "module", model)  # DDP unwrap
        processed_params = set()

        # Identify decoder groups by structure
        decoder = getattr(getattr(model_base, "model", None), "decoder", None)
        if decoder and hasattr(decoder, "block"):
            print(f"Scanning {len(decoder.block)} decoder blocks for parameter grouping...")
            for i, block in enumerate(decoder.block):
                if isinstance(block, DualT5DecoderBlock):
                    # FFN (layer[3])
                    if hasattr(block, "layer") and len(block.layer) > 3 and block.layer[3] is not None:
                        for _, p in block.layer[3].named_parameters():
                            if p.requires_grad:
                                ffn_params.append(p)
                                processed_params.add(p)
                    # Cross-attn + LN (layer[2])
                    if hasattr(block, "layer") and len(block.layer) > 2 and block.layer[2] is not None:
                        custom_layer = block.layer[2]
                        if hasattr(custom_layer, "EncDecAttention"):
                            for _, p in custom_layer.EncDecAttention.named_parameters():
                                if p.requires_grad:
                                    custom_attn_params.append(p)
                                    processed_params.add(p)
                        if hasattr(custom_layer, "layer_norm"):
                            for _, p in custom_layer.layer_norm.named_parameters():
                                if p.requires_grad:
                                    custom_attn_params.append(p)
                                    processed_params.add(p)
        else:
            print("Warning: decoder blocks not found at model.model.decoder.block; grouping may be incomplete.")

        # Collect text encoder params (name-based)
        for n, p in model.named_parameters():
            if p.requires_grad and "text_encoder" in n:
                text_params.append(p)
                processed_params.add(p)

        # Safety: any other trainables not already bucketed
        for n, p in model.named_parameters():
            if p.requires_grad and p not in processed_params:
                unexpected_trainable_params.append(p)

        ffn_lr = base_lr * self.ffn_lr_multiplier
        text_lr = base_lr * self.text_lr_multiplier

        print("-" * 50)
        print("Optimizer parameter groups:")
        print(f"  Base LR: {base_lr}")
        print(f"  FFN LR (x{self.ffn_lr_multiplier}): {ffn_lr} | count={len(ffn_params)}")
        print(f"  XAttn+LN LR (base): {base_lr} | count={len(custom_attn_params)}")
        print(f"  Text encoder LR (x{self.text_lr_multiplier}): {text_lr} | count={len(text_params)}")
        if unexpected_trainable_params:
            print(f"  Unexpected trainables (base LR): count={len(unexpected_trainable_params)}")
        print("-" * 50)

        optimizer_grouped_parameters = []
        if custom_attn_params:
            optimizer_grouped_parameters.append({"params": custom_attn_params, "lr": base_lr})
        if ffn_params:
            optimizer_grouped_parameters.append({"params": ffn_params, "lr": ffn_lr})
        if text_params:
            optimizer_grouped_parameters.append({"params": text_params, "lr": text_lr})
        if unexpected_trainable_params:
            optimizer_grouped_parameters.append({"params": unexpected_trainable_params, "lr": base_lr})

        if not optimizer_grouped_parameters:
            print("WARNING: No trainable parameters found. Ensure unfreezing ran before Trainer construction.")
            raise ValueError("No trainable parameters found.")

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
        if "weight_decay" not in optimizer_kwargs:
            optimizer_kwargs["weight_decay"] = args.weight_decay
        optimizer_kwargs.pop("lr", None)  # per-group LR in use

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        print(f"Created optimizer: {type(self.optimizer).__name__} with {len(optimizer_grouped_parameters)} groups.")
        return self.optimizer



class CustomTrainer(Trainer):
    def __init__(self, *args, ffn_lr_multiplier=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffn_lr_multiplier = ffn_lr_multiplier
        print(f"Initialized DifferentialLrTrainer. FFN LR = Base LR * {self.ffn_lr_multiplier}")

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        # Save the ground truth labels from the batch
        future_targets = inputs.get("future_targets")
        scale = inputs.get("scale")

        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        # Step 2: Run model without labels to get samples (for CRPS)
        inputs_for_sampling = dict(inputs)
        inputs_for_sampling["labels"] = None  # Force sampling mode

        with torch.no_grad():
            preds = model(**inputs_for_sampling)

        # Create tuple with both predictions and scale
        preds_with_scale = (preds, future_targets, scale)

        logits = (logits, preds_with_scale)

        return (loss, logits, labels)

    def create_optimizer(self):
        model = self.model
        args = self.args

        # --- 1. Prepare Parameter Groups based on user's specified trainable layers ---
        ffn_params = []  # layer[3]
        custom_attn_params = []  # layer[2].EncDecAttention + layer[2].layer_norm
        unexpected_trainable_params = []  # Should remain empty

        # Adapt this logic to your model structure as needed.
        model_base = getattr(model, 'module', model)  # Handle DDP wrapping
        processed_params = set()

        decoder = getattr(getattr(model_base, 'model', None), 'decoder', None)
        if decoder and hasattr(decoder, 'block'):
            print(f"Scanning {len(decoder.block)} decoder blocks for specified trainable parameters...")
            for i, block in enumerate(decoder.block):
                # Check if it is the custom block type containing the layers to be trained
                if isinstance(block, DualT5DecoderBlock):  # Use the actual class name
                    # FFN Layer (layer[3])
                    if hasattr(block, 'layer') and len(block.layer) > 3 and block.layer[3] is not None:
                        print(f"  Block {i}: Checking FFN (layer[3]) parameters...")
                        for name, param in block.layer[3].named_parameters():
                            if param.requires_grad:
                                ffn_params.append(param)
                                processed_params.add(param)
                    else:
                        print(f"  Block {i}: FFN layer (layer[3]) not found or block structure mismatch.")

                    # Custom Attention Layer Components (layer[2])
                    if hasattr(block, 'layer') and len(block.layer) > 2 and block.layer[2] is not None:
                        custom_layer = block.layer[2]
                        # layer[2].EncDecAttention parameters
                        if hasattr(custom_layer, 'EncDecAttention'):
                            print(f"  Block {i}: Checking Custom Attn (layer[2].EncDecAttention) parameters...")
                            for name, param in custom_layer.EncDecAttention.named_parameters():
                                if param.requires_grad:
                                    custom_attn_params.append(param)
                                    processed_params.add(param)
                        # layer[2].layer_norm parameters
                        if hasattr(custom_layer, 'layer_norm'):
                            print(f"  Block {i}: Checking Custom Attn (layer[2].layer_norm) parameters...")
                            for name, param in custom_layer.layer_norm.named_parameters():
                                if param.requires_grad:
                                    custom_attn_params.append(param)
                                    processed_params.add(param)
                    else:
                        print(f"  Block {i}: Custom Attn layer (layer[2]) not found or block structure mismatch.")
        else:
            print("Warning: Could not find decoder blocks at expected location (model.decoder.block). Adapt parameter identification logic if needed.")

        # --- Safety Check: Find any other trainable parameters ---
        print("Running safety check for unexpected trainable parameters...")
        for name, param in model.named_parameters():
            if param.requires_grad and param not in processed_params:
                print(f"  WARNING: Found unexpected trainable parameter: {name}")
                unexpected_trainable_params.append(param)

        # --- 2. Define Learning Rates ---
        base_lr = args.learning_rate
        ffn_lr = base_lr * self.ffn_lr_multiplier

        print("-" * 30)
        print("Parameter Group Summary:")
        print(f"  Base LR: {base_lr}")
        print(f"  FFN LR (Multiplier: {self.ffn_lr_multiplier}): {ffn_lr}")
        print(f"  Number of FFN params (low LR): {len(ffn_params)}")
        print(f"  Number of Custom Attn params (base LR): {len(custom_attn_params)}")
        if unexpected_trainable_params:
            print(f"  WARNING: Number of unexpected trainable params (base LR): {len(unexpected_trainable_params)}")
        print("-" * 30)

        # --- 3. Create Optimizer Grouped Parameters ---
        optimizer_grouped_parameters = []
        if custom_attn_params:
            optimizer_grouped_parameters.append({"params": custom_attn_params, "lr": base_lr})
        if ffn_params:
            optimizer_grouped_parameters.append({"params": ffn_params, "lr": ffn_lr})
        if unexpected_trainable_params:
            optimizer_grouped_parameters.append({"params": unexpected_trainable_params, "lr": base_lr})

        if not optimizer_grouped_parameters:
            if not ffn_params and not custom_attn_params:
                print("WARNING: No parameters designated for training were found with requires_grad=True. Did you unfreeze them correctly before creating the Trainer?")
            else:
                raise ValueError("Parameter grouping failed unexpectedly.")

        # --- 4. Instantiate Optimizer using args.optim ---
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
        if "weight_decay" not in optimizer_kwargs:
            optimizer_kwargs["weight_decay"] = args.weight_decay
        optimizer_kwargs.pop("lr", None)  # LR is now specified per group

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        print(f"Created optimizer: {type(self.optimizer).__name__} with {len(optimizer_grouped_parameters)} parameter group(s).")
        return self.optimizer


class AttentionRatioCallback(TrainerCallback):
    """Callback to log cross-attention ratios during training and evaluation."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log attention ratios after each logging step."""
        if model is None or logs is None:
            return

        # Determine if this is an eval logging event
        is_eval = any(key.startswith("eval_") for key in logs.keys())
        prefix = "eval_" if is_eval else ""

        model_module = model.module if hasattr(model, "module") else model

        # Collect attention ratios from all dual blocks
        attention_ratios = []
        if hasattr(model_module, "model") and hasattr(model_module.model, "decoder"):
            for i, block in enumerate(model_module.model.decoder.block):
                if isinstance(block, DualT5DecoderBlock) and hasattr(block.layer[2], "log_mean_attention_ratio"):
                    ratio = block.layer[2].log_mean_attention_ratio.item()
                    attention_ratios.append(ratio)
                    # Use the prefix for standard logs
                    logs[f"{prefix}attention_ratio_block_{i}"] = ratio

                    # For debugging (optional - can be commented out in production)
                    print(f"{'Eval' if is_eval else 'Train'} - Block {i} attention ratio: {ratio:.4f}")

        if attention_ratios:
            # Use the prefix for the mean attention ratio
            logs[f"{prefix}mean_attention_ratio"] = sum(attention_ratios) / len(attention_ratios)
            print(f"{'Eval' if is_eval else 'Train'} - Mean attention ratio: {logs[f'{prefix}mean_attention_ratio']:.4f}")

            # Direct WandB logging with phase distinction
            try:
                import wandb
                if wandb.run is not None:
                    # Use different namespaces for train vs eval in WandB
                    phase = "eval/attention" if is_eval else "train/attention"
                    wandb_log = {f"{phase}/block_{i}": ratio for i, ratio in enumerate(attention_ratios)}
                    wandb_log[f"{phase}/mean"] = logs[f"{prefix}mean_attention_ratio"]

                    # Include global step for proper timeline alignment
                    wandb.log(wandb_log, step=state.global_step)
                    print(f"Directly logged {phase} metrics to WandB")
            except Exception as e:
                print(f"Error directly logging to WandB: {str(e)}")


def format_params(count):
    if count >= 1e9:
        return f"{count/1e9:.2f}B"  # Billions
    return f"{count/1e6:.2f}M"      # Millions


def visualize_checkpoint_results(
    dual_pipeline,
    chronos_pipeline,
    dataset_test,
    sample_indices,
    results_dir,
    checkpoint_type,
    prompt_type
):
    """
    Visualize model predictions from a specific checkpoint against baseline Chronos model.
    Updated to handle both instruction/response format and minimal dataset format.

    Args:
        dual_pipeline: Loaded DoubleCastPipeline with checkpoint
        chronos_pipeline: Baseline ChronosPipeline
        dataset_test: Test dataset to sample from
        sample_indices: List of indices to visualize
        results_dir: Directory to save visualization results
        checkpoint_type: String identifier ("best" or "final")
        prompt_type: Type of prompt to use
    """
    SELECTED_PROMPT = PROMPTS[prompt_type]

    for i in sample_indices:
        data = dataset_test[i]

        # Extract data from example, handling both formats
        if "instruction" in data and "response" in data:
            past_timestamps, past_values, text_context = extract_from_instruction(data["instruction"])
            future_timestamps, future_values = extract_future_from_response(data["response"])
        else:
            # Minimal format with direct access to values
            past_values = data.get("past_target", [])
            future_values = data.get("future_target", [])
            text_context = data.get("context_abs", "")
            past_timestamps = data.get("past_timestamp", None)
            future_timestamps = data.get("future_timestamp", None)

            # Convert string timestamps to datetime objects if needed
            if past_timestamps is not None and isinstance(past_timestamps[0], str):
                past_timestamps = pd.to_datetime(past_timestamps)
            if future_timestamps is not None and isinstance(future_timestamps[0], str):
                future_timestamps = pd.to_datetime(future_timestamps)

        # Get frequency from data
        freq = data.get("freq", None)

        # Convert to tensor
        context = torch.tensor(
            past_values,
            dtype=torch.bfloat16,
        ).flatten()

        prediction_length = len(future_values)

        # Run baseline Chronos model
        chronos_preds = chronos_pipeline.predict(
            context=context,
            prediction_length=prediction_length,
            num_samples=20,
            limit_prediction_length=False,
        )

        # Create prompt function
        prompt_func = partial(
            SELECTED_PROMPT,
            past_timestamps=past_timestamps,
            future_timestamps=future_timestamps,
            freq=freq
        )

        # Run dual model
        dual_preds = dual_pipeline.predict(
            context=context,
            text_context=text_context,
            prediction_length=prediction_length,
            num_samples=20,
            limit_prediction_length=False,
            format_prompt=prompt_func,
        )

        # Plot comparison
        plot_comparison(
            result=dual_preds,
            chronos_result=chronos_preds,
            past_timestamps=past_timestamps,
            future_timestamps=future_timestamps,
            past_target=past_values,
            future_target=future_values,
            context=text_context,
            save_dir=results_dir,
            checkpoint_type=checkpoint_type,
            index=i,
        )


def plot_comparison(
    result,  # (B, n_sample, pred_len)
    chronos_result,  # (B, n_sample, pred_len)
    past_timestamps,
    future_timestamps,
    past_target,
    future_target,
    context,
    save_dir,
    filename_prefix="plot",
    checkpoint_type="best",  # or "final"
    index=0
):
    # Prepare output path
    folder = os.path.join(save_dir, f"{checkpoint_type}-checkpoint")
    os.makedirs(folder, exist_ok=True)
    filename = f"{filename_prefix}_{index:03d}.png"
    full_path = os.path.join(folder, filename)

    # Initialize plot
    plt.figure(figsize=(10, 6), dpi=150)

    # Plot history and ground truth
    plt.plot(past_timestamps, past_target, label="History", color="black", linewidth=3)
    plt.plot(future_timestamps, future_target, label="Ground Truth", color="forestgreen", linewidth=3)

    # Chronos forecast quantiles
    chronos_sample = chronos_result[0]  # shape: (n_sample, pred_len)
    c_q10 = torch.quantile(chronos_sample, 0.1, dim=0).cpu().numpy()
    c_q25 = torch.quantile(chronos_sample, 0.25, dim=0).cpu().numpy()
    c_q50 = torch.quantile(chronos_sample, 0.5, dim=0).cpu().numpy()
    c_q75 = torch.quantile(chronos_sample, 0.75, dim=0).cpu().numpy()
    c_q90 = torch.quantile(chronos_sample, 0.9, dim=0).cpu().numpy()

    plt.fill_between(future_timestamps, c_q10, c_q90, alpha=0.2, color="#FFB347", label="Chronos (10–90%)")
    plt.fill_between(future_timestamps, c_q25, c_q75, alpha=0.35, color="#FF7F50", label="Chronos (25–75%)")
    plt.plot(future_timestamps, c_q50, label="Chronos Median", color="#E65100", linewidth=1.5)

    # Forecast quantiles for your model
    forecast_sample = result[0]  # shape: (n_sample, pred_len)
    f_q10 = torch.quantile(forecast_sample, 0.1, dim=0).cpu().numpy()
    f_q25 = torch.quantile(forecast_sample, 0.25, dim=0).cpu().numpy()
    f_q50 = torch.quantile(forecast_sample, 0.5, dim=0).cpu().numpy()
    f_q75 = torch.quantile(forecast_sample, 0.75, dim=0).cpu().numpy()
    f_q90 = torch.quantile(forecast_sample, 0.9, dim=0).cpu().numpy()

    plt.fill_between(future_timestamps, f_q10, f_q90, alpha=0.2, color="#7EC8E3", label="DoubleCast (10–90%)")
    plt.fill_between(future_timestamps, f_q25, f_q75, alpha=0.35, color="#1E90FF", label="DoubleCast (25–75%)")
    plt.plot(future_timestamps, f_q50, label="DoubleCast Median", color="#003f5c", linewidth=1.5)

    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=30, ha="right")

    # Styling
    plt.title("\n".join(wrap(context, 80)), fontsize=12)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    # Save
    plt.savefig(full_path)
    plt.close()

    print(f"Saved plot to: {full_path}")


def is_valid(example, min_past=6, min_future=1):
    """
    Check if an example is valid for training/evaluation.
    Compatible with both instruction/response format and minimal format.

    Args:
        example: Dict with time series data in either format
        min_past: Minimum number of historical points required
        min_future: Minimum number of future points required

    Returns:
        bool: True if example is valid, False otherwise
    """
    try:
        if "instruction" in example and "response" in example:
            instruction = example["instruction"]
            response = example["response"]
            past_timestamps, past_values, context = extract_from_instruction(instruction)
            future_timestamps, future_values = extract_future_from_response(response)
        else:
            past_values = example.get("past_target", [])
            future_values = example.get("future_target", [])
            start_timestamp = example.get("start_timestamp", None)
            freq = example.get("freq", None)
            past_len = example.get("past_len", len(past_values))
            future_len = example.get("future_len", len(future_values))

            past_timestamps, future_timestamps = infer_timestamps_from_metadata(
                start_timestamp, freq, past_len, future_len
            )

        # Check that history contains enough values
        if past_values is None or len(past_values) < min_past:
            return False

        # Check that forecast values exist and meet minimum requirement
        if future_values is None or len(future_values) < min_future:
            return False

        # Check for NaN values
        if np.isnan(past_values).any() or np.isnan(future_values).any():
            return False

        # Extra checks to avoid bfloat16 conversion errors
        try:
            past_target = torch.tensor(past_values, dtype=torch.bfloat16)
            future_target = torch.tensor(future_values, dtype=torch.bfloat16)
            if past_target.ndim != 1 or future_target.ndim != 1:
                return False
        except Exception:
            return False

        # If timestamps are available, validate them
        if past_timestamps is not None and future_timestamps is not None:
            # Check that timestamps are valid (not NaT)
            if len(past_timestamps) == 0 or pd.isna(past_timestamps).any():
                return False

            if len(future_timestamps) == 0 or pd.isna(future_timestamps).any():
                return False

            # Optional: Check that future timestamps follow past timestamps
            if len(future_timestamps) > 0 and len(past_timestamps) > 0:
                if future_timestamps[0] <= past_timestamps[-1]:
                    return False

        # Example is valid if we got this far
        return True

    except Exception as e:
        # Any unexpected error makes the example invalid
        # print(f"Validation error: {str(e)}")  # Uncomment for debugging
        return False


def extract_from_instruction(instruction):
    """
    Extracts the historical values and context from the instruction.
    Returns:
      - past_time: a list of numerical values from the <history> section.
      - context: a string extracted from the <context> section.
    """
    # Extract context between <context> and </context>
    context_match = re.search(r"<context>(.*?)</context>", instruction, re.DOTALL)
    context = context_match.group(1).strip() if context_match else ""

    # Extract history between <history> and </history>
    history_match = re.search(r"<history>(.*?)</history>", instruction, re.DOTALL)
    history_str = history_match.group(1).strip() if history_match else ""

    data = {}
    for line in history_str.splitlines():
        line = line.strip()
        if line.startswith("(") and line.endswith(")"):
            # Remove the parentheses and split by comma.
            content = line[1:-1]
            parts = content.split(",")
            if len(parts) >= 2:
                try:
                    # Extract and clean the timestamp.
                    timestamp = parts[0].strip().strip("'").strip('"')
                    value = float(parts[1].strip())
                    data[timestamp] = value
                except ValueError:
                    continue
    s = pd.Series(data)
    # Convert the index (timestamps) to datetime objects.
    s.index = pd.to_datetime(s.index, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    return s.index, s.values, context


def extract_future_from_response(response):
    """
    Extracts forecast values from the response.
    The response should contain a forecast block between <forecast> and </forecast>.
    Returns:
      - future_time: a list of numerical values from the forecast lines.
    """
    forecast_match = re.search(r"<forecast>(.*?)</forecast>", response, re.DOTALL)
    forecast_str = forecast_match.group(1).strip() if forecast_match else ""

    data = {}
    for line in forecast_str.splitlines():
        line = line.strip()
        if line.startswith("(") and line.endswith(")"):
            # Remove the parentheses and split by comma.
            content = line[1:-1]
            parts = content.split(",")
            if len(parts) >= 2:
                try:
                    # Extract and clean the timestamp.
                    timestamp = parts[0].strip().strip("'").strip('"')
                    value = float(parts[1].strip())
                    data[timestamp] = value
                except ValueError:
                    continue
    s = pd.Series(data)
    # Convert the index (timestamps) to datetime objects.
    s.index = pd.to_datetime(s.index, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    return s.index, s.values


def infer_timestamps_from_metadata(start_timestamp, freq, past_len, future_len):
    """
    Infer past and future timestamps based on frequency and lengths.

    Args:
        start_timestamp: Timestamp of the first observation
        freq: Pandas frequency string (e.g., 'D', 'H', '5T')
        past_len: Number of historical observations
        future_len: Number of future observations

    Returns:
        tuple: (past_timestamps, future_timestamps)
    """
    if start_timestamp is None or freq is None:
        return None, None

    # Convert to pandas Timestamp if not already
    if not isinstance(start_timestamp, pd.Timestamp):
        try:
            start_timestamp = pd.Timestamp(start_timestamp)
        except Exception:
            return None, None

    # Generate timestamp sequence
    total_len = past_len + future_len
    try:
        all_timestamps = pd.date_range(start=start_timestamp, periods=total_len, freq=freq)
        past_timestamps = all_timestamps[:past_len]
        future_timestamps = all_timestamps[past_len:past_len+future_len]
        return past_timestamps, future_timestamps
    except Exception as e:
        print(f"Error inferring timestamps: {e}")
        return None, None


def dual_text_timeseries_collator(features, num_tokenizer, text_tokenizer, prompt):
    """
    Collator for dual Chronos training.
    Updated to handle timestamps stored as strings.
    """
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    list_past_targets = []
    list_future_targets = []
    list_past_timestamps = []
    list_future_timestamps = []
    list_context = []
    list_freq = []
    prompted_contexts = []

    # Process each feature without padding.
    for f in features:
        if "instruction" in f and "response" in f:
            instruction = f["instruction"]
            response = f["response"]
            freq = f.get("freq", None)
            past_timestamps, past_values, context = extract_from_instruction(instruction)
            future_timestamps, future_values = extract_future_from_response(response)
        else:
            past_values = f.get("past_target", [])
            future_values = f.get("future_target", [])
            context = f.get("context_abs", "")
            freq = f.get("freq", None)

            # Get timestamps and convert from strings to datetime objects if needed
            past_timestamps = f.get("past_timestamp", None)
            future_timestamps = f.get("future_timestamp", None)

            # Convert string timestamps to datetime objects
            if past_timestamps is not None and isinstance(past_timestamps[0], str):
                past_timestamps = pd.to_datetime(past_timestamps, format="%Y-%m-%d %H:%M:%S", errors="coerce")
            if future_timestamps is not None and isinstance(future_timestamps[0], str):
                future_timestamps = pd.to_datetime(future_timestamps, format="%Y-%m-%d %H:%M:%S", errors="coerce")

        past_target = torch.tensor(past_values, dtype=torch.bfloat16)
        future_target = torch.tensor(future_values, dtype=torch.bfloat16)

        list_past_targets.append(past_target)
        list_future_targets.append(future_target)
        list_past_timestamps.append(past_timestamps)
        list_future_timestamps.append(future_timestamps)
        list_context.append(context)
        list_freq.append(freq)

    # If there are no valid examples, return an empty dict.
    if len(list_past_targets) == 0:
        return {}

    # Determine maximum lengths for past and future sequences in the batch.
    max_past_len = max(t.size(0) for t in list_past_targets)
    max_future_len = num_tokenizer.config.prediction_length

    padded_pasts = []
    padded_futures = []

    # Pad past targets with torch.nan on the left.
    for past in list_past_targets:
        pad_len = max_past_len - past.size(0)
        if pad_len > 0:
            padded = torch.cat(
                [torch.full((pad_len,), float("nan"), dtype=past.dtype, device=past.device), past],
                dim=0,
            )
        else:
            padded = past
        padded_pasts.append(padded.unsqueeze(0))  # shape: (1, max_past_len)

    # Pad future targets with torch.nan on the right.
    for future in list_future_targets:
        pad_len = max_future_len - future.size(0)
        if pad_len > 0:
            padded = torch.cat(
                [future, torch.full((pad_len,), float("nan"), dtype=future.dtype, device=future.device)],
                dim=0,
            )
        else:
            padded = future[:max_future_len]
        padded_futures.append(padded.unsqueeze(0))  # shape: (1, max_future_len)

    # Stack into batch tensors.
    batch_past = torch.cat(padded_pasts, dim=0)    # shape: (batch_size, max_past_len)
    batch_future = torch.cat(padded_futures, dim=0)  # shape: (batch_size, max_future_len)

    # Use num_tokenizer transforms on batched data.
    input_ids, attention_mask, scale = num_tokenizer.context_input_transform(batch_past)
    labels, labels_mask = num_tokenizer.label_input_transform(batch_future, scale)
    labels[labels_mask == 0] = -100

    # Format prompted contexts per sample.
    for i in range(len(list_context)):
        prompted_context = PROMPTS[prompt](
            list_past_timestamps[i], list_future_timestamps[i], list_context[i], batch_past[i], scale[i], list_freq[i]
        )
        prompted_contexts.append(prompted_context)

    text_outputs = text_tokenizer(
        prompted_contexts,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "text_input_ids": text_outputs["input_ids"],
        "text_attention_mask": text_outputs["attention_mask"],
        "future_targets": batch_future,
        "scale": scale,
    }
    return batch


def dual_bolt_text_timeseries_collator(features, text_tokenizer, prompt):
    """
    Collator for Dual Chronos Bolt training that:
    1. Passes raw time series data directly to the model
    2. Tokenizes the text inputs for cross-attention

    Updated to ensure mask tensors are boolean type.
    """
    if text_tokenizer.pad_token is None and hasattr(text_tokenizer, 'eos_token'):
        text_tokenizer.pad_token = text_tokenizer.eos_token

    list_past_targets = []
    list_future_targets = []
    list_past_timestamps = []
    list_future_timestamps = []
    list_context = []

    # Extract data from features
    for f in features:
        instruction = f["instruction"]
        response = f["response"]
        past_timestamps, past_values, context = extract_from_instruction(instruction)
        future_timestamps, future_values = extract_future_from_response(response)

        past_target = torch.tensor(past_values, dtype=torch.float32)
        future_target = torch.tensor(future_values, dtype=torch.float32)

        if 'thinking_trace' in f.keys():
            context += f["thinking_trace"]

        list_past_targets.append(past_target)
        list_future_targets.append(future_target)
        list_past_timestamps.append(past_timestamps)
        list_future_timestamps.append(future_timestamps)
        list_context.append(context)

    # If there are no valid examples, return an empty dict
    if len(list_past_targets) == 0:
        return {}

    # Determine maximum lengths for past and future sequences
    max_past_len = max(t.size(0) for t in list_past_targets)
    max_future_len = 64  # Hard-coded to 64 as requested

    # Batch preparation - LEFT padding with NaN for Chronos Bolt
    padded_pasts = []
    for past in list_past_targets:
        pad_len = max_past_len - past.size(0)
        if pad_len > 0:
            padded = torch.cat(
                [torch.full((pad_len,), float("nan"), dtype=past.dtype), past],  # LEFT padding
                dim=0,
            )
        else:
            padded = past
        padded_pasts.append(padded.unsqueeze(0))

    # Future targets - RIGHT padding with NaN
    padded_futures = []
    future_targets_mask = []
    for future in list_future_targets:
        pad_len = max_future_len - future.size(0)
        # Create boolean mask directly using torch.ones(..., dtype=torch.bool)
        mask = torch.ones(max_future_len, dtype=torch.bool)
        if pad_len > 0:
            padded = torch.cat(
                [future, torch.full((pad_len,), float("nan"), dtype=future.dtype)],  # RIGHT padding
                dim=0,
            )
            mask[future.size(0):] = False  # Use False instead of 0.0
        else:
            padded = future[:max_future_len]
        padded_futures.append(padded.unsqueeze(0))
        future_targets_mask.append(mask.unsqueeze(0))

    # Stack into batch tensors
    batch_past = torch.cat(padded_pasts, dim=0)
    batch_future = torch.cat(padded_futures, dim=0)
    # Ensure batch_future_mask is boolean type
    batch_future_mask = torch.cat(future_targets_mask, dim=0)

    # Create context mask from NaN values (already boolean)
    context_mask = ~torch.isnan(batch_past)

    # Format prompted contexts and tokenize
    prompted_contexts = []
    for i, context_text in enumerate(list_context):
        prompted_context = PROMPTS[prompt](
            list_past_timestamps[i],
            list_future_timestamps[i],
            context_text,
            list_past_targets[i],
            None
        )
        prompted_contexts.append(prompted_context)

    text_outputs = text_tokenizer(
        prompted_contexts,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )

    # Return with keys matching model's forward parameters
    return {
        "context": batch_past,
        "text_input_ids": text_outputs["input_ids"],
        "text_attention_mask": text_outputs["attention_mask"],
        "mask": context_mask,
        "target": batch_future,
        "target_mask": batch_future_mask,  # Now guaranteed to be boolean
    }

