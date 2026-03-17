# Code partially taken from chronos.chronos

import logging
import os
import json
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, GenerationConfig, PreTrainedModel,
    PretrainedConfig
)
from transformers.models.t5.configuration_t5 import T5Config
from safetensors.torch import load_file

import chronos
from chronos import ChronosConfig, ChronosTokenizer, ChronosPipeline
from chronos.base import ForecastType

from doublecast.models.dual_t5 import DualT5ForConditionalGeneration

logger = logging.getLogger(__file__)


class DoubleCastConfig(T5Config):
    """
    Configuration class for DoubleCastModel.

    This configuration class stores configuration for a dual model that combines
    a T5-based time series model with a text encoder for context-aware forecasting.
    """
    model_type = "double_cast"

    def __init__(
        self,
        text_encoder_path: str = "google/t5-efficient-large",
        cross_attention_model_size: int = 1024,
        dual_block_placement: Union[str, List[int]] = "all",
        chronos_path: str = "amazon/chronos-t5-large",
        chronos_config: Optional[Dict[str, Any]] = None,
        text_encoder_layer_index: int = -2,
        **kwargs
    ):
        # Remove problematic keys that T5Config doesn't understand
        kwargs.pop('loss_type', None)

        super().__init__(**kwargs)

        # Set dual-specific attributes
        self.text_encoder_path = text_encoder_path
        self.cross_attention_model_size = cross_attention_model_size
        self.dual_block_placement = dual_block_placement
        self.chronos_path = chronos_path
        self.chronos_config = chronos_config or {}
        self.text_encoder_layer_index = text_encoder_layer_index

        # Validation
        if self.cross_attention_model_size <= 0:
            raise ValueError("cross_attention_model_size must be positive")

    @property
    def prediction_length(self):
        """Get prediction_length from chronos_config"""
        return self.chronos_config.get('prediction_length', 64)

    @prediction_length.setter
    def prediction_length(self, value):
        """Set prediction_length in chronos_config"""
        self.chronos_config['prediction_length'] = value

    def to_dict(self):
        """Override to include custom attributes in serialization"""
        output = super().to_dict()
        output.update({
            'text_encoder_path': self.text_encoder_path,
            'cross_attention_model_size': self.cross_attention_model_size,
            'dual_block_placement': self.dual_block_placement,
            'chronos_path': self.chronos_path,
            'chronos_config': self.chronos_config,
            'text_encoder_layer_index': self.text_encoder_layer_index,
        })
        return output


class DoubleCastModel(PreTrainedModel):
    """
    DoubleCastModel combines a T5-based time series forecasting model with a text encoder
    for context-aware time series forecasting.

    The model consists of:
    - A DualT5ForConditionalGeneration model (self.model) for time series processing
    - A text encoder (self.text_encoder) for processing textual context
    """

    config_class = DoubleCastConfig
    base_model_prefix = "double_cast"

    # Tied weights - these should match the actual model structure
    _tied_weights_keys = [
        "model.encoder.embed_tokens.weight",
        "model.decoder.embed_tokens.weight",
        "model.lm_head.weight",
    ]

    # Keys to ignore when they're unexpected during loading
    _keys_to_ignore_on_load_unexpected = [
        "model.decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: DoubleCastConfig):
        super().__init__(config)

        self.model = DualT5ForConditionalGeneration(config=config)

        self.text_encoder = AutoModel.from_pretrained(config.text_encoder_path)

        if hasattr(self.text_encoder, 'encoder'):
            self.text_encoder = self.text_encoder.encoder

        # Copy weights from base Chronos model
        if config.chronos_config.get('model_type') == "seq2seq":
            base_model = AutoModelForSeq2SeqLM.from_pretrained(config.chronos_path)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(config.chronos_path)

        self.model.copy_weights_from_original(base_model)
        del base_model

        self.post_init()

    def _tie_weights(self):
        """Tie weights for the inner model"""
        if hasattr(self.model, 'tie_weights'):
            self.model.tie_weights()
        elif hasattr(self.model, '_tie_weights'):
            self.model._tie_weights()

    def get_input_embeddings(self):
        """Required by PreTrainedModel for tied weights"""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        """Required by PreTrainedModel for tied weights"""
        self.model.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        """Required by PreTrainedModel for tied weights"""
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """Required by PreTrainedModel for tied weights"""
        self.model.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load DoubleCastModel from a local checkpoint or HuggingFace Hub repo.

        - Local path: loads all weights directly from the checkpoint (full load).
        - HF Hub repo ID: the HF repo only contains the trained delta weights
          (decoder layer.2 and layer.3). The base model is first initialised from
          the Chronos and text-encoder weights stored in config.json, then the
          delta weights are applied on top.
        """

        device = kwargs.pop('device_map', 'cpu')
        torch_dtype = kwargs.pop('torch_dtype', None)

        # Consume build-from-scratch kwargs — not used when loading a checkpoint
        kwargs.pop('text_encoder_path', None)
        kwargs.pop('dual_block_placement', None)
        kwargs.pop('text_encoder_layer_index', None)
        kwargs.pop('chronos_path', None)

        hf_kwargs = {k: kwargs.pop(k) for k in ('token', 'cache_dir', 'revision', 'local_files_only') if k in kwargs}

        checkpoint_path = Path(pretrained_model_name_or_path)
        is_hf_hub = not checkpoint_path.is_dir()

        if is_hf_hub:
            from huggingface_hub import snapshot_download
            checkpoint_path = Path(snapshot_download(str(pretrained_model_name_or_path), **hf_kwargs))

        config = DoubleCastConfig.from_pretrained(checkpoint_path, **kwargs)
        model = cls(config)
        model = model.to(device=device, dtype=torch_dtype)

        state_dict = cls._load_state_dict_from_checkpoint(checkpoint_path, device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if hasattr(model, '_tie_weights'):
            model._tie_weights()

        if is_hf_hub:
            # Delta load — missing frozen weights are expected, only warn on unexpected keys
            if unexpected_keys:
                logger.warning(f"Unexpected keys in delta checkpoint: {unexpected_keys[:5]}")
            logger.info(f"Loaded delta checkpoint from HF Hub '{pretrained_model_name_or_path}' "
                        f"({len(state_dict)} trained tensors applied on top of base models)")
        else:
            cls._log_loading_info(missing_keys, unexpected_keys, model._tied_weights_keys)
            logger.info(f"Loaded full checkpoint from: {pretrained_model_name_or_path}")

        return model

    @staticmethod
    def _load_state_dict_from_checkpoint(checkpoint_path: Path, device):
        """Load state dict from various checkpoint formats"""
        state_dict = {}

        # First, check for safetensors (newer format)
        safetensors_file = checkpoint_path / "model.safetensors"
        if safetensors_file.exists():
            logger.info("Loading from safetensors format")
            state_dict = load_file(safetensors_file, device=str(device))

        # Check for sharded safetensors
        elif (checkpoint_path / "model.safetensors.index.json").exists():
            logger.info("Loading from sharded safetensors format")
            index_file = checkpoint_path / "model.safetensors.index.json"
            with open(index_file, 'r') as f:
                index = json.load(f)

            for shard_file in set(index["weight_map"].values()):
                shard_path = checkpoint_path / shard_file
                shard_state_dict = load_file(shard_path, device=str(device))
                state_dict.update(shard_state_dict)

        # Check for sharded pytorch tensors
        elif (checkpoint_path / "pytorch_model.bin.index.json").exists():
            logger.info("Loading from sharded pytorch tensors format")
            index_file = checkpoint_path / "pytorch_model.bin.index.json"
            with open(index_file, 'r') as f:
                index = json.load(f)

            for shard_file in set(index["weight_map"].values()):
                shard_path = checkpoint_path / shard_file
                shard_state_dict = torch.load(shard_path, map_location=device, weights_only=True)
                state_dict.update(shard_state_dict)

        # Try single pytorch_model.bin file
        elif (checkpoint_path / "pytorch_model.bin").exists():
            logger.info("Loading from single pytorch tensor file")
            model_file = checkpoint_path / "pytorch_model.bin"
            state_dict = torch.load(model_file, map_location=device, weights_only=True)

        else:
            raise FileNotFoundError("No model files found in checkpoint")

        return state_dict

    @staticmethod
    def _log_loading_info(missing_keys, unexpected_keys, tied_weights_keys):
        """Log information about missing and unexpected keys"""
        if missing_keys:
            # Filter out expected missing keys (tied weights)
            critical_missing = [k for k in missing_keys if k not in tied_weights_keys]
            if critical_missing:
                logger.error(f"Critical missing keys: {critical_missing}")
                raise RuntimeError(f"Missing critical weights: {critical_missing}")
            else:
                logger.info(f"Missing {len(missing_keys)} tied weight keys (expected): {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected {len(unexpected_keys)} keys: {unexpected_keys[:5]}...")

    @classmethod
    def _build_from_base_models(cls, pretrained_model_name_or_path, text_encoder_path,
                               dual_block_placement, chronos_path, device, torch_dtype, **kwargs):
        """Build model from base models when checkpoint loading fails"""

        # Extract text_encoder_layer_index from kwargs
        text_encoder_layer_index = kwargs.pop("text_encoder_layer_index", -2)

        # Use provided chronos_path or fallback to pretrained_model_name_or_path
        if chronos_path is None:
            chronos_path = pretrained_model_name_or_path

        # Load base chronos config
        base_config = AutoConfig.from_pretrained(chronos_path, **kwargs)

        # Load text encoder config
        text_encoder_config = AutoConfig.from_pretrained(text_encoder_path, **kwargs)
        cross_attention_model_size = (
            text_encoder_config.d_model if hasattr(text_encoder_config, 'd_model')
            else text_encoder_config.hidden_size
        )

        # Create config dict
        config_dict = base_config.to_dict()
        config_dict.update({
            'text_encoder_path': text_encoder_path,
            'cross_attention_model_size': cross_attention_model_size,
            'dual_block_placement': dual_block_placement,
            'text_encoder_layer_index': text_encoder_layer_index,  # Add this line
            'chronos_path': chronos_path,
            'chronos_config': config_dict.get('chronos_config', {}),
        })

        # Create DoubleCastConfig
        dual_config = DoubleCastConfig(**config_dict)

        # Create model
        model = cls(dual_config)

        # Transfer to device and dtype
        model = model.to(device=device, dtype=torch_dtype)

        logger.info(f"Built DoubleCastModel from base models")
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass of the dual model"""

        # Obtain text encoder embeddings from text_input_ids
        text_encoder_output = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            output_hidden_states=True,
        )
        # Use the configurable layer index instead of hardcoded -2
        text_encoder_embeds = text_encoder_output.hidden_states[self.config.text_encoder_layer_index]

        if labels is not None:
            # Training mode
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                text_encoder_embeds=text_encoder_embeds,
                text_encoder_attention_mask=text_attention_mask,
                labels=labels,
            )
        else:
            # Inference mode
            if prediction_length is None:
                prediction_length = self.config.chronos_config.get('prediction_length')
            if num_samples is None:
                num_samples = self.config.chronos_config.get('num_samples')
            if temperature is None:
                temperature = self.config.chronos_config.get('temperature')
            if top_k is None:
                top_k = self.config.chronos_config.get('top_k')
            if top_p is None:
                top_p = self.config.chronos_config.get('top_p')

            preds = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                text_encoder_embeds=text_encoder_embeds,
                text_encoder_attention_mask=text_attention_mask,
                generation_config=GenerationConfig(
                    min_new_tokens=prediction_length,
                    max_new_tokens=prediction_length,
                    do_sample=True,
                    num_return_sequences=num_samples,
                    eos_token_id=self.config.eos_token_id,
                    pad_token_id=self.config.pad_token_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                ),
            )

            if self.config.chronos_config.get('model_type') == "seq2seq":
                preds = preds[..., 1:]  # remove the decoder start token
            else:
                assert self.config.chronos_config.get('model_type') == "causal"
                assert preds.size(-1) == input_ids.size(-1) + prediction_length
                preds = preds[..., -prediction_length:]

            return preds.reshape(input_ids.size(0), num_samples, -1)


class DoubleCastPipeline(ChronosPipeline):
    """Pipeline for DoubleCastModel that handles both numerical and textual inputs"""

    tokenizer: ChronosTokenizer
    text_tokenizer: AutoTokenizer
    model: DoubleCastModel
    forecast_type: ForecastType = ForecastType.SAMPLES

    def __init__(self, tokenizer, text_tokenizer, model):
        super().__init__(tokenizer=tokenizer, model=model)
        self.tokenizer = tokenizer
        self.text_tokenizer = text_tokenizer
        self.model = model

    def predict(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        text_context: Union[str, List[str]],
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        limit_prediction_length: bool = False,
        format_prompt: Optional[Union[Callable, List[Callable]]] = None,
    ) -> torch.Tensor:
        """Predict future values given numerical and textual context"""

        # Process numerical inputs
        context_tensor = self._prepare_and_validate_context(context=context)

        if prediction_length is None:
            prediction_length = self.model.config.chronos_config.get('prediction_length')

        if prediction_length > self.model.config.chronos_config.get('prediction_length'):
            msg = (
                f"We recommend keeping prediction length <= {self.model.config.chronos_config.get('prediction_length')}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            logger.warning(msg)

        predictions = []
        remaining = prediction_length

        while remaining > 0:
            token_ids, attention_mask, scale = self.tokenizer.context_input_transform(
                context_tensor
            )

            # Handle format_prompt - either single callable or list of callables
            if format_prompt is not None:
                if isinstance(format_prompt, list):
                    # List of functions for batch
                    text_context = [
                        fmt_fn(context=txt, past_target=context_tensor[i], scale=scale[i] if scale.dim() > 0 else scale)
                        for i, (txt, fmt_fn) in enumerate(zip(text_context, format_prompt))
                    ]
                else:
                    # Single function
                    text_context = format_prompt(
                        context=text_context, past_target=context_tensor.squeeze(0), scale=scale
                    )

            # Process text inputs
            if isinstance(text_context, str):
                text_inputs = self.text_tokenizer([text_context])
            else:
                text_inputs = self.text_tokenizer(text_context)
            text_input_ids = torch.LongTensor(text_inputs["input_ids"])
            text_attention_mask = torch.BoolTensor(text_inputs["attention_mask"])

            samples = self.model(
                token_ids.to(self.model.device),
                attention_mask.to(self.model.device),
                text_input_ids.to(self.model.device),
                text_attention_mask.to(self.model.device),
                None,
                min(remaining, self.model.config.chronos_config.get('prediction_length')),
                num_samples,
                temperature,
                top_k,
                top_p,
            )
            prediction = self.tokenizer.output_transform(
                samples.to(scale.device), scale
            )

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            context_tensor = torch.cat(
                [context_tensor, prediction.median(dim=1).values], dim=-1
            )

        return torch.cat(predictions, dim=-1).to(dtype=torch.float32, device="cpu")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "DoubleCastPipeline":
        """Create a DoubleCastPipeline from a checkpoint or HuggingFace Hub repo.

        There are two supported calling conventions:

        **New style** — single argument is a DoubleCast checkpoint or HF Hub repo ID.
        All configuration (text_encoder_path, chronos_path, etc.) is read from config.json:

            pipeline = DoubleCastPipeline.from_pretrained("your-org/doublecast")

        **Legacy style** — first argument is the base Chronos model; a separate
        ``checkpoint_path`` kwarg points to the DoubleCast checkpoint (used by
        training scripts that build or resume from a run directory):

            pipeline = DoubleCastPipeline.from_pretrained(
                "amazon/chronos-t5-large",
                text_encoder_path="Qwen/Qwen3-14B",
                checkpoint_path="/path/to/checkpoint",
            )
        """
        checkpoint_path = kwargs.pop("checkpoint_path", None)

        if checkpoint_path is not None:
            # Legacy style: chronos base model + separate checkpoint path
            dual_model = DoubleCastModel.from_pretrained(checkpoint_path, **kwargs)
        else:
            # New style: single arg is the DoubleCast checkpoint / HF Hub repo
            text_encoder_path = kwargs.pop("text_encoder_path", None)
            dual_block_placement = kwargs.pop("dual_block_placement", None)
            text_encoder_layer_index = kwargs.pop("text_encoder_layer_index", None)

            if text_encoder_path or dual_block_placement or text_encoder_layer_index:
                # Build a fresh model from base components (training from scratch)
                chronos_path = pretrained_model_name_or_path
                text_encoder_path = text_encoder_path or "google/t5-efficient-large"
                dual_block_placement = dual_block_placement or "all"
                text_encoder_layer_index = text_encoder_layer_index if text_encoder_layer_index is not None else -2

                device = kwargs.pop('device_map', 'cpu')
                torch_dtype = kwargs.pop('torch_dtype', None)

                text_encoder_config = AutoConfig.from_pretrained(text_encoder_path)
                cross_attention_model_size = (
                    text_encoder_config.d_model if hasattr(text_encoder_config, 'd_model')
                    else text_encoder_config.hidden_size
                )
                base_config = AutoConfig.from_pretrained(chronos_path)
                config_dict = base_config.to_dict()
                config_dict.update({
                    'text_encoder_path': text_encoder_path,
                    'cross_attention_model_size': cross_attention_model_size,
                    'dual_block_placement': dual_block_placement,
                    'text_encoder_layer_index': text_encoder_layer_index,
                    'chronos_path': chronos_path,
                    'chronos_config': config_dict.get('chronos_config', {}),
                })
                dual_config = DoubleCastConfig(**config_dict)
                dual_model = DoubleCastModel(config=dual_config)
                dual_model = dual_model.to(device=device, dtype=torch_dtype)
            else:
                # Load from DoubleCast checkpoint or HF Hub repo
                dual_model = DoubleCastModel.from_pretrained(pretrained_model_name_or_path, **kwargs)

        dual_config = dual_model.config
        chronos_config = ChronosConfig(**dual_config.chronos_config)
        tokenizer = chronos_config.create_tokenizer()
        text_tokenizer = AutoTokenizer.from_pretrained(dual_config.text_encoder_path)

        return cls(tokenizer=tokenizer, text_tokenizer=text_tokenizer, model=dual_model)
