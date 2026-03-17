import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "Context-aided forecasting"

import torch
from torch import nn
import wandb
from accelerate import Accelerator
from datasets import (
    interleave_datasets,
    load_dataset,
    load_from_disk,
)
from transformers import (
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from doublecast.evaluation.metrics import compute_metrics
from doublecast.models.dual_t5 import DualT5DecoderBlock
from doublecast.models.doublecast import DoubleCastPipeline
from doublecast.models.utils import (
    is_valid,
    AttentionRatioCallback,
    StagedCustomTrainer,
    dual_text_timeseries_collator,
    format_params,
)

set_seed(42)
accelerator = Accelerator()
wandb.login()

# Enable TF32 on Hopper for faster matmuls (does not affect bf16)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


STAGE_CFG = {
    "1": {
        "desc": "Adapters only",
        "unfreeze_last_n_blocks": 0,
        "unfreeze_text_last_n": 0,
        "learning_rate": 1e-4,
        "ffn_lr_multiplier": 0.01,
        "text_lr_multiplier": 0.0,
        "warmup_ratio": 0.005,
    },
    "2": {
        "desc": "Progressive unfreeze",
        "unfreeze_last_n_blocks": 2,
        "unfreeze_text_last_n": 0,
        "learning_rate": 8e-5,
        "ffn_lr_multiplier": 1.0,
        "text_lr_multiplier": 0.0,
        "warmup_ratio": 0.005,
    },
    "3": {
        "desc": "Fine polish",
        "unfreeze_last_n_blocks": 2,
        "unfreeze_text_last_n": 4,
        "learning_rate": 5e-5,
        "ffn_lr_multiplier": 1.0,
        "text_lr_multiplier": 0.5,
        "warmup_ratio": 0.005,
    },
}

def freeze_and_unfreeze(model: torch.nn.Module, last_n_backbone: int, last_n_text: int):
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Always unfreeze dual cross-attn + LN + FFN in DualT5DecoderBlock
    for block in model.model.decoder.block:
        if isinstance(block, DualT5DecoderBlock):
            for p in block.layer[2].EncDecAttention.parameters():
                p.requires_grad = True
            for p in block.layer[2].layer_norm.parameters():
                p.requires_grad = True
            for p in block.layer[3].parameters():
                p.requires_grad = True

    # Progressive unfreeze backbone: interpret -1 as "all decoder blocks"
    dec_blocks = list(model.model.decoder.block)
    if last_n_backbone is not None:
        if last_n_backbone == -1:
            n_backbone = len(dec_blocks)
        else:
            n_backbone = max(0, min(last_n_backbone, len(dec_blocks)))
        if n_backbone > 0:
            for layer in dec_blocks[-n_backbone:]:
                for p in layer.parameters():
                    p.requires_grad = True

    # Progressive unfreeze: last N layers of text encoder (by name pattern)
    def _unfreeze_text_last_n_by_name(m: nn.Module, last_n: int):
        if last_n is None or last_n == 0:
            return []
        layer_regex = re.compile(r"text_encoder.*layers\.(\d+)\.")
        layer_ids = set()
        for n, _ in m.named_parameters():
            match = layer_regex.search(n)
            if match:
                layer_ids.add(int(match.group(1)))

        # If -1: unfreeze all text encoder params
        if last_n == -1:
            for n, p in m.named_parameters():
                if "text_encoder" in n:
                    p.requires_grad = True
            return ["all"]

        unfrozen_layers = []
        if layer_ids:
            sorted_ids = sorted(layer_ids)
            target = set(sorted_ids[-last_n:])
            for n, p in m.named_parameters():
                match = layer_regex.search(n)
                if match and int(match.group(1)) in target:
                    p.requires_grad = True
            unfrozen_layers = sorted(list(target))
            # also unfreeze common final norms if present
            for n, p in m.named_parameters():
                if "text_encoder" in n and any(k in n for k in [".ln_f", ".final_layernorm"]) and "layers" not in n:
                    p.requires_grad = True
        else:
            # Fallback: unfreeze all text encoder params if we can't find layer indices
            for n, p in m.named_parameters():
                if "text_encoder" in n:
                    p.requires_grad = True
            unfrozen_layers = ["all"]
        return unfrozen_layers

    unfrozen_te = _unfreeze_text_last_n_by_name(model, last_n_text)
    return unfrozen_te

def apply_stage_defaults(args, stage_cfg):
    if args.learning_rate is None:
        args.learning_rate = stage_cfg["learning_rate"]
    if args.warmup_ratio is None:
        args.warmup_ratio = stage_cfg["warmup_ratio"]
    if args.ffn_lr_multiplier is None:
        args.ffn_lr_multiplier = stage_cfg["ffn_lr_multiplier"]
    if args.text_lr_multiplier is None:
        args.text_lr_multiplier = stage_cfg["text_lr_multiplier"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chronos_path", type=str, default="amazon/chronos-t5-large")
    parser.add_argument(
        "--text_encoder_path", type=str, default="Qwen/Qwen2.5-1.5B"
    )
    parser.add_argument(
        "--dual_block_placement",
        nargs="+",
        default="all",
        help="List of block indices for DualT5DecoderBlock, or 'all' to enable all blocks.",
    )
    parser.add_argument(
        "--text_encoder_layer_index",
        type=int,
        default=-2,
        help="Which layer of text encoder to use for embeddings (default: -2 for second-to-last layer)"
    )
    parser.add_argument(
        "--train_dataset",
        nargs="+",
        type=str,
        help="List of training dataset paths to interleave"
    )
    parser.add_argument(
        "--train_dataset_names",
        nargs="+",
        type=str,
        help="Names for training datasets (must match number of train_dataset paths)",
        default=None
    )
    parser.add_argument(
        "--interleave_probabilities",
        nargs="+",
        type=float,
        help="Probabilities for dataset interleaving (must sum to 1)",
        default=None
    )
    parser.add_argument(
        "--val_dataset",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--val_dataset_names",
        nargs="+",
        type=str,
        help="Names for validation datasets (must match number of val_dataset paths)",
        default=None
    )
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--prompt_type", type=str, default="naive")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Max gradient norm for gradient clipping")
    parser.add_argument("--warmup_ratio", type=float, default=0.005,
               help="Ratio of training steps for learning rate warmup")
    parser.add_argument("--weight_decay", type=float, default=0.01,
            help="Weight decay for AdamW optimizer")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                help="Number of subprocesses to use for data loading")
    parser.add_argument("--ffn_lr_multiplier", type=float, default=None,
                help="Learning rate multiplier for dual-block FFN params (None => stage default)")
    parser.add_argument("--stage", choices=["1", "2", "3"], default="1",
                help="Training stage: 1=Adapters only, 2=Progressive unfreeze, 3=Polish")
    parser.add_argument("--unfreeze_last_n_blocks", type=int, default=None,
                help="Override stage default for number of decoder blocks to unfreeze (from the end)")
    parser.add_argument("--unfreeze_text_last_n", type=int, default=None,
                help="Override stage default for number of text-encoder layers to unfreeze (from the end)")
    parser.add_argument("--text_lr_multiplier", type=float, default=None,
                help="LR multiplier applied to text encoder params (None => stage default)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                help="Path to a specific checkpoint folder to resume training from")
    parser.add_argument("--disable_wandb", action="store_true",
                help="Disable Weights & Biases logging")
    parser.add_argument("--use_fsdp", action="store_true",
                help="Enable FSDP full_shard auto_wrap.")
    parser.add_argument("--fsdp_min_params", type=int, default=5_000_000,
                help="Min params per layer for auto wrap.")
    parser.add_argument("--deepspeed", type=str, default=None,
                help="Path to DeepSpeed ZeRO config JSON")

    args = parser.parse_args()

    chronos_size = Path(args.chronos_path).name.split("-")[-1]
    text_encoder_short = Path(args.text_encoder_path).name.lower()
    if args.dual_block_placement == ["all"] or args.dual_block_placement == "all":
        dual_block_str = "all"
    else:
        dual_block_str = "blocks_" + "_".join(map(str, args.dual_block_placement))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{chronos_size}_{text_encoder_short}_{dual_block_str}_{args.prompt_type}_s{args.stage}_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)

    dual_pipeline = DoubleCastPipeline.from_pretrained(
        args.chronos_path,
        text_encoder_path=args.text_encoder_path,
        dual_block_placement=args.dual_block_placement,
        text_encoder_layer_index=args.text_encoder_layer_index,
        checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path is not None else None,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model = dual_pipeline.model

    # ---- apply stage config (LRs, warmup, unfreezing) ----
    stage_cfg = STAGE_CFG[args.stage]
    apply_stage_defaults(args, stage_cfg)
    last_n_backbone = args.unfreeze_last_n_blocks if args.unfreeze_last_n_blocks is not None else stage_cfg["unfreeze_last_n_blocks"]
    last_n_text = args.unfreeze_text_last_n if args.unfreeze_text_last_n is not None else stage_cfg["unfreeze_text_last_n"]
    unfrozen_te = freeze_and_unfreeze(model, last_n_backbone, last_n_text)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_formatted = format_params(total_params)
    trainable_params_formatted = format_params(trainable_params)
    trainable_layer_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            parts = name.split('.')
            if len(parts) >= 4:
                layer_name = '.'.join(parts[:4])
                if layer_name not in trainable_layer_names:
                    trainable_layer_names.append(layer_name)

    args.total_parameters = total_params
    args.trainable_parameters = trainable_params
    args.trainable_layers = trainable_layer_names
    args.total_parameters_formatted = total_params_formatted
    args.trainable_parameters_formatted = trainable_params_formatted

    with open(Path(output_dir) / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_datasets = []
    train_sizes_before = []
    train_sizes_after = []

    train_dataset_names = args.train_dataset_names if args.train_dataset_names else [
        os.path.basename(path).split('.')[0] for path in args.train_dataset
    ]

    if len(train_dataset_names) != len(args.train_dataset):
        train_dataset_names = [f"train_{i}" for i in range(len(args.train_dataset))]

    logger.info("Loading training datasets:")
    for i, (train_path, name) in enumerate(zip(args.train_dataset, train_dataset_names)):
        logger.info(f"  Dataset {i+1}: {name} from {train_path}")
        if train_path.endswith('.json'):
            dataset = load_dataset("json", data_files=train_path)["train"]
        else:
            dataset = load_from_disk(train_path)
        train_size_before = len(dataset)
        train_sizes_before.append(train_size_before)

        dataset = dataset.filter(partial(is_valid))

        train_size_after = len(dataset)
        train_sizes_after.append(train_size_after)

        filtered = train_size_before - train_size_after
        logger.info(f"    {filtered}/{train_size_before} examples filtered out ({filtered/train_size_before*100:.2f}%)")
        logger.info(f"    {train_size_after}/{train_size_before} examples kept ({train_size_after/train_size_before*100:.2f}%)")

        train_datasets.append(dataset)

    # Check if probabilities are provided and valid
    if args.interleave_probabilities:
        if len(args.interleave_probabilities) != len(train_datasets):
            raise ValueError(f"Number of interleave probabilities ({len(args.interleave_probabilities)}) must match number of datasets ({len(train_datasets)})")

        # Check if probabilities sum to approximately 1
        prob_sum = sum(args.interleave_probabilities)
        if not 0.99 <= prob_sum <= 1.01:
            logger.warning(f"Interleave probabilities sum to {prob_sum}, not 1.0. Normalizing.")
            probs = [p/prob_sum for p in args.interleave_probabilities]
        else:
            probs = args.interleave_probabilities
    else:
        # Default to equal probabilities
        probs = [1/len(train_datasets)] * len(train_datasets)

    # If there's only one dataset, skip interleaving completely
    if len(train_datasets) == 1:
        logger.info("Only one dataset detected, skipping interleaving")
        dataset_train = train_datasets[0]
        logger.info(f"Training dataset size: {len(dataset_train)}")
    else:
        logger.info(f"Interleaving {len(train_datasets)} datasets with probabilities: {probs}")
        dataset_train = interleave_datasets(train_datasets, probabilities=probs)
        logger.info(f"Final interleaved dataset size: {len(dataset_train)}")

    # Debugging option for small runs
    if "debug" in args.output_dir:
        dataset_train = dataset_train.select(range(min(16 * len(train_datasets), len(dataset_train))))
        logger.info(f"Debug mode: reduced training dataset to {len(dataset_train)} samples")

    # Load multiple validation datasets
    dataset_names = args.val_dataset_names if args.val_dataset_names else [
        os.path.basename(path).split('.')[0] for path in args.val_dataset
    ]

    if len(dataset_names) != len(args.val_dataset):
        dataset_names = [f"dataset_{i}" for i in range(len(args.val_dataset))]

    eval_datasets = {}
    for i, (val_path, name) in enumerate(zip(args.val_dataset, dataset_names)):
        logger.info(f"Loading validation dataset: {name} from {val_path}")
        if val_path.endswith('.json'):
            dataset = load_dataset("json", data_files=val_path)["train"]
        else:
            dataset = load_from_disk(val_path)
        val_size_before = len(dataset)

        dataset = dataset.filter(partial(is_valid))

        val_size_after = len(dataset)

        # Log filtering statistics
        filtered = val_size_before - val_size_after
        logger.info(f"    {filtered}/{val_size_before} examples filtered out ({filtered/val_size_before*100:.2f}%)")
        logger.info(f"    {val_size_after}/{val_size_before} examples kept ({val_size_after/val_size_before*100:.2f}%)")

        eval_datasets[name] = dataset

        if "debug" in args.output_dir:
            eval_datasets[name] = eval_datasets[name].select(range(16))

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_torch_fused",
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        bf16_full_eval=True,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to=["tensorboard", "wandb"] if "debug"  not in args.output_dir else ["tensorboard"],
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model=f"eval_{dataset_names[0]}_crps",
    )

    data_collator = partial(
        dual_text_timeseries_collator,
        num_tokenizer=dual_pipeline.tokenizer,
        text_tokenizer=dual_pipeline.text_tokenizer,
        prompt=args.prompt_type,
    )

    trainer = StagedCustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=eval_datasets,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=dual_pipeline.tokenizer),
        ffn_lr_multiplier=args.ffn_lr_multiplier,
        text_lr_multiplier=args.text_lr_multiplier,  # NEW: separate LR for text encoder params
        callbacks=[AttentionRatioCallback()]
    )

    resume_ckpt = None
    if args.resume_from_checkpoint:
        resume_ckpt = args.resume_from_checkpoint
    else:
        try:
            last_ckpt = get_last_checkpoint(output_dir)
        except Exception as e:
            last_ckpt = None
            logger.warning(f"Could not scan for last checkpoint in {output_dir}: {e}")
        if last_ckpt:
            resume_ckpt = last_ckpt

    if resume_ckpt:
        logger.info(f"Resuming Trainer from checkpoint: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        logger.info("Starting training from scratch (no Trainer checkpoint resume).")
        trainer.train()

    if accelerator.is_main_process:
        try:
            # Define the final checkpoint folder name
            final_ckpt_name = f"checkpoint-{args.max_steps}"
            final_ckpt_path = os.path.join(output_dir, final_ckpt_name)
            final_target = os.path.join(output_dir, "final-checkpoint")

            # Rename the final checkpoint folder
            if os.path.exists(final_ckpt_path):
                shutil.move(final_ckpt_path, final_target)
            else:
                print(f"Final checkpoint folder {final_ckpt_name} does not exist.")

            # Retrieve the best checkpoint from the trainer state
            best_ckpt_path = trainer.state.best_model_checkpoint
            best_target = os.path.join(output_dir, "best-checkpoint")

            if best_ckpt_path:
                if os.path.abspath(best_ckpt_path) != os.path.abspath(os.path.join(output_dir, final_ckpt_name)):
                    shutil.move(best_ckpt_path, best_target)
                else:
                    os.symlink(final_target, best_target)
            else:
                print("Best checkpoint does not exist.")

            # Remove all other checkpoint folders
            for folder in os.listdir(output_dir):
                folder_path = os.path.join(output_dir, folder)
                if os.path.isdir(folder_path) and re.match(r"checkpoint-\d+", folder):
                    # Do not remove the renamed best or final checkpoints
                    if os.path.abspath(folder_path) not in [os.path.abspath(final_target), os.path.abspath(best_target)]:
                        shutil.rmtree(folder_path)

        except Exception as e:
            print(f"[Rank 0] Post-processing failed: {e}")

    # --- Ensure all ranks wait for Rank 0 to finish ---
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
