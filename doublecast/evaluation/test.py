import argparse
import gc
import json
import logging
import random
from functools import partial
from pathlib import Path

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import TrainingArguments, set_seed

from chronos import ChronosPipeline
from doublecast.evaluation.metrics import compute_metrics
from doublecast.models.doublecast import DoubleCastPipeline
from doublecast.models.utils import (
    CustomTrainer,
    dual_text_timeseries_collator,
    extract_future_from_response,
    is_valid,
    visualize_checkpoint_results,
)

set_seed(42)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def process_experiment(exp_dir, results_dir, dataset_test, max_prediction_length, args):
    """
    Process a single experiment directory
    """
    # Set minimal evaluation arguments
    training_args = TrainingArguments(
        output_dir=exp_dir,
        per_device_eval_batch_size=args.batch_size,
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=True,
    )

    # Define checkpoint directories
    best_checkpoint_dir = exp_dir / "best-checkpoint"
    final_checkpoint_dir = exp_dir / "final-checkpoint"

    # Find checkpoint-{step} directories
    step_checkpoints = sorted([d for d in exp_dir.iterdir()
                              if d.is_dir() and d.name.startswith("checkpoint-")],
                             key=lambda x: int(x.name.split("-")[1]))

    # Load baseline ChronosPipeline for comparison
    logger.info(f"Loading baseline ChronosPipeline from {args.chronos_path}...")
    chronos_pipeline = ChronosPipeline.from_pretrained(
        args.chronos_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    checkpoints_processed = 0

    # If specific checkpoint step is requested, process only that
    if args.checkpoint_step is not None:
        target_checkpoint = exp_dir / f"checkpoint-{args.checkpoint_step}"
        if target_checkpoint.exists():
            logger.info(f"Processing specific checkpoint: {target_checkpoint} (step {args.checkpoint_step})")
            process_checkpoint(
                checkpoint_path=target_checkpoint,
                checkpoint_type=f"step-{args.checkpoint_step}",
                chronos_pipeline=chronos_pipeline,
                dataset_test=dataset_test,
                max_prediction_length=max_prediction_length,
                results_dir=results_dir,
                training_args=training_args,
                args=args,
            )
            checkpoints_processed += 1
        else:
            logger.error(f"Checkpoint step {args.checkpoint_step} not found at {target_checkpoint}")
            return
    else:
        # Process best checkpoint if it exists
        if best_checkpoint_dir.exists():
            logger.info(f"Found best checkpoint at {best_checkpoint_dir}")
            process_checkpoint(
                checkpoint_path=best_checkpoint_dir,
                checkpoint_type="best",
                chronos_pipeline=chronos_pipeline,
                dataset_test=dataset_test,
                max_prediction_length=max_prediction_length,
                results_dir=results_dir,
                training_args=training_args,
                args=args,
            )
            checkpoints_processed += 1
            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()

        # Process final checkpoint if it exists
        if final_checkpoint_dir.exists():
            logger.info(f"Found final checkpoint at {final_checkpoint_dir}")
            process_checkpoint(
                checkpoint_path=final_checkpoint_dir,
                checkpoint_type="final",
                chronos_pipeline=chronos_pipeline,
                dataset_test=dataset_test,
                max_prediction_length=max_prediction_length,
                results_dir=results_dir,
                training_args=training_args,
                args=args,
            )
            checkpoints_processed += 1
            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()

        # If no best/final checkpoints found, process the latest step checkpoint
        if checkpoints_processed == 0 and step_checkpoints:
            latest_checkpoint = step_checkpoints[-1]  # Get the highest step number
            step_number = latest_checkpoint.name.split("-")[1]

            logger.info(f"No best/final checkpoints found. Processing latest checkpoint: {latest_checkpoint} (step {step_number})")
            process_checkpoint(
                checkpoint_path=latest_checkpoint,
                checkpoint_type=f"step-{step_number}",
                chronos_pipeline=chronos_pipeline,
                dataset_test=dataset_test,
                max_prediction_length=max_prediction_length,
                results_dir=results_dir,
                training_args=training_args,
                args=args,
            )
            checkpoints_processed += 1

    if checkpoints_processed == 0:
        logger.warning(f"No valid checkpoints found in {exp_dir}")

    # Release memory
    del chronos_pipeline
    gc.collect()
    torch.cuda.empty_cache()


def process_checkpoint(checkpoint_path, checkpoint_type, chronos_pipeline, dataset_test,
                      max_prediction_length, results_dir, training_args, args):
    """
    Process a single checkpoint (best or final)
    """
    logger.info(f"Processing {checkpoint_type} checkpoint from {checkpoint_path}...")

    # Load checkpoint pipeline
    pipeline = DoubleCastPipeline.from_pretrained(
        args.chronos_path,
        text_encoder_path=args.text_encoder_path,
        dual_block_placement=args.dual_block_placement,
        text_encoder_layer_index=args.text_encoder_layer_index,  # Add this line
        checkpoint_path=checkpoint_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model = pipeline.model

    # Set dynamic prediction length based on dataset
    pipeline.tokenizer.config.prediction_length = max_prediction_length
    pipeline.model.config.chronos_config['prediction_length'] = max_prediction_length
    logger.info(f"Set prediction length to {max_prediction_length}")

    # Define the data collator using tokenizers
    data_collator = partial(
        dual_text_timeseries_collator,
        num_tokenizer=pipeline.tokenizer,
        text_tokenizer=pipeline.text_tokenizer,
        prompt=args.prompt_type,
    )

    # Create CustomTrainer for evaluation
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        eval_dataset=dataset_test,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=pipeline.tokenizer),
        ffn_lr_multiplier=0.01
    )

    # Evaluate checkpoint
    logger.info(f"Evaluating {checkpoint_type} checkpoint...")
    metrics = trainer.evaluate(eval_dataset=dataset_test)

    # Save metrics
    metrics_file = results_dir / f"{checkpoint_type}_checkpoint_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "checkpoint": str(checkpoint_path),
            "test_on": args.test_dataset,
            **metrics
        }, f, indent=4)
    logger.info(f"{checkpoint_type.capitalize()} checkpoint metrics saved to {metrics_file}")

    # Visualize results
    sample_indices = random.sample(range(len(dataset_test)), min(10, len(dataset_test)))
    logger.info(f"Visualizing {len(sample_indices)} examples from {checkpoint_type} checkpoint...")

    visualize_checkpoint_results(
        dual_pipeline=pipeline,
        chronos_pipeline=chronos_pipeline,
        dataset_test=dataset_test,
        sample_indices=sample_indices,
        results_dir=results_dir,
        checkpoint_type=checkpoint_type,
        prompt_type=args.prompt_type
    )

    # Clean up
    del pipeline, model, trainer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_parent_dir", type=str, required=True,
                        help="Either a single experiment folder or a parent folder containing multiple experiment folders")
    parser.add_argument("--result_dir", type=str, default="results",
                        help="Directory name for results (relative to each experiment folder)")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to the test dataset (JSON format)")
    parser.add_argument("--chronos_path", type=str, default=None,
                        help="Path to the Chronos model (override run_config.json)")
    parser.add_argument("--text_encoder_path", type=str, default=None,
                        help="Path to the text encoder model (override run_config.json)")
    parser.add_argument("--dual_block_placement", nargs="+", default=None,
                        help="List of block indices for DualT5DecoderBlock, or 'all' (override run_config.json)")
    parser.add_argument("--text_encoder_layer_index", type=int, default=None,
                        help="Which layer of text encoder to use for embeddings (override run_config.json)")
    parser.add_argument("--prompt_type", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--checkpoint_step", type=int, default=None,
                        help="Specific checkpoint step to evaluate (e.g., 1000 for checkpoint-1000). If not specified, evaluates best/final or latest checkpoint.")
    args = parser.parse_args()

    # Determine if this is a single experiment or multiple experiments
    checkpoint_root = Path(args.checkpoint_parent_dir)

    # Function to check if a directory is a valid experiment directory
    def is_valid_experiment_dir(path):
        # Check for best/final checkpoints or run_config.json
        has_best_final = (path / "best-checkpoint").exists() or (path / "final-checkpoint").exists()
        has_config = (path / "run_config.json").exists()

        # Check for checkpoint-{step} directories
        has_step_checkpoints = any(d.name.startswith("checkpoint-") for d in path.iterdir() if d.is_dir())

        return has_best_final or has_config or has_step_checkpoints

    # Check if the provided directory is a single experiment
    if is_valid_experiment_dir(checkpoint_root):
        experiment_dirs = [checkpoint_root]
        logger.info(f"Processing single experiment directory: {checkpoint_root}")
    else:
        # Otherwise, look for subdirectories that are valid experiment directories
        experiment_dirs = [d for d in checkpoint_root.iterdir() if d.is_dir() and is_valid_experiment_dir(d)]
        if not experiment_dirs:
            raise ValueError(f"No valid experiment directories found in {checkpoint_root}")
        logger.info(f"Found {len(experiment_dirs)} experiment directories to process")

    # Load test dataset with enhanced approach
    logger.info(f"Loading test dataset from {args.test_dataset}")
    if args.test_dataset.endswith('.json'):
        dataset_test = load_dataset("json", data_files=args.test_dataset)["train"]
        print(f"Original dataset size: {len(dataset_test)}")
        dataset_test = dataset_test.filter(is_valid)
        print(f"Filtered dataset size: {len(dataset_test)}")
        max_prediction_length = max(
        [len(extract_future_from_response(sample["response"])[1]) for sample in dataset_test]
    )
    else:
        dataset_test = load_from_disk(args.test_dataset)
        print(f"Original dataset size: {len(dataset_test)}")
        dataset_test = dataset_test.filter(is_valid)
        print(f"Filtered dataset size: {len(dataset_test)}")
        max_prediction_length = max(len(sample["future_target"]) for sample in dataset_test)

    logger.info(f"Maximum prediction length in dataset: {max_prediction_length}")

    # Process each experiment directory
    for exp_dir in experiment_dirs:
        logger.info(f"\n{'='*80}\nProcessing experiment directory: {exp_dir}\n{'='*80}")

        # Create results directory for this experiment
        results_dir = exp_dir / args.result_dir
        results_dir.mkdir(exist_ok=True)

        # Load run_config.json if it exists
        config_path = exp_dir / "run_config.json"
        run_config = {}

        if config_path.exists():
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, "r") as f:
                run_config = json.load(f)

            # Log some key experiment info
            model_info = {
                "chronos_path": run_config.get("chronos_path", "N/A"),
                "text_encoder_path": run_config.get("text_encoder_path", "N/A"),
                "prompt_type": run_config.get("prompt_type", "N/A"),
                "trainable_parameters": run_config.get("trainable_parameters_formatted", "N/A"),
                "total_parameters": run_config.get("total_parameters_formatted", "N/A")
            }
            logger.info(f"Experiment configuration: {json.dumps(model_info, indent=2)}")
        else:
            logger.warning(f"No run_config.json found at {config_path}. Using command line arguments only.")

        # Use command line args or fall back to run_config values
        exp_args = argparse.Namespace(**vars(args))
        exp_args.chronos_path = args.chronos_path or run_config.get("chronos_path")
        exp_args.text_encoder_path = args.text_encoder_path or run_config.get("text_encoder_path")
        exp_args.dual_block_placement = args.dual_block_placement or run_config.get("dual_block_placement")
        exp_args.text_encoder_layer_index = args.text_encoder_layer_index or run_config.get("text_encoder_layer_index", -2)  # Add this line with default
        exp_args.prompt_type = args.prompt_type or run_config.get("prompt_type")

        # Ensure required parameters are available
        required_params = ["chronos_path", "text_encoder_path", "dual_block_placement", "prompt_type"]
        missing_params = [param for param in required_params if getattr(exp_args, param) is None]
        if missing_params:
            logger.error(f"Missing required parameters for {exp_dir}: {', '.join(missing_params)}")
            logger.error("Skipping this experiment directory")
            continue

        try:
            # Process this experiment
            process_experiment(
                exp_dir=exp_dir,
                results_dir=results_dir,
                dataset_test=dataset_test,
                max_prediction_length=max_prediction_length,
                args=exp_args
            )
            logger.info(f"Completed evaluation for {exp_dir}")
        except Exception as e:
            logger.error(f"Error evaluating {exp_dir}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue with next experiment instead of failing the entire run
            continue

    logger.info("All experiment evaluations complete.")


if __name__ == "__main__":
    main()
