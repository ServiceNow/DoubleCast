
from .crps import our_crps
import numpy as np
import torch


def compute_metrics(eval_pred, tokenizer, dataset_name=None):
    predictions = eval_pred.predictions  # This is your (logits, preds_with_scale) tuple
    labels = eval_pred.label_ids        # Ground truth labels

    original_logits, preds_with_scale = predictions  # Get both parts
    preds, future_targets, scale = preds_with_scale

    predictions_samples = tokenizer.output_transform(
        torch.tensor(preds), torch.tensor(scale)
    )

    # Per-sample normalization
    normalized_crps_scores = []
    unnormalized_crps_scores = []

    for i, (target, prediction_sample) in enumerate(zip(future_targets, predictions_samples)):
        valid_mask = ~np.isnan(target)
        masked_target = target[valid_mask]
        masked_prediction_sample = prediction_sample[:, valid_mask]
        score = our_crps(masked_target, masked_prediction_sample, compute_variance=False)['metric'] # (n_timesteps), (n_samples, n_timesteps)

        # Filter out NaN values and calculate mean absolute value
        if len(masked_target) > 0:
            mean_target = np.mean(np.abs(masked_target))
            if mean_target > 0:
                normalized_crps_scores.append(score / mean_target)
            else:
                normalized_crps_scores.append(score)
            unnormalized_crps_scores.append(score)
        else:
            normalized_crps_scores.append(score)
            unnormalized_crps_scores.append(score)

    mean_normalized_crps = np.nanmean(normalized_crps_scores)
    mean_unnormalized_crps = np.nanmean(unnormalized_crps_scores)

    # Return both metrics in the results dictionary
    metrics = {
        'crps': mean_normalized_crps,
        'unnormalized_crps': mean_unnormalized_crps,
    }

    if dataset_name:
        return {f"{dataset_name}_{k}": v for k, v in metrics.items()}

    return metrics


