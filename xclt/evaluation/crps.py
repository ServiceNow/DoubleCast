# Adapted from https://github.com/ServiceNow/context-is-key-forecasting/blob/main/scripts/compute_rcrps_with_hf_dataset.py
# This is ROI_CRPS but with no region of interest, and no metric scaling

import numpy as np
import pandas as pd
from .crps_covariance import weighted_sum_crps_variance

def crps(
    target: np.array,
    samples: np.array,
) -> np.array:
    """
    Compute the CRPS using the probability weighted moment form.
    See Eq ePWM from "Estimation of the Continuous Ranked Probability Score with
    Limited Information and Applications to Ensemble Weather Forecasts"
    https://link.springer.com/article/10.1007/s11004-017-9709-7

    This is a O(n log n) per variable exact implementation, without estimation bias.

    Parameters:
    -----------
    target: np.ndarray
        The target values. (variable dimensions)
    samples: np.ndarray
        The forecast values. (n_samples, variable dimensions)

    Returns:
    --------
    crps: np.ndarray
        The CRPS for each of the (variable dimensions)
    """
    assert (
        target.shape == samples.shape[1:]
    ), f"shapes mismatch between: {target.shape} and {samples.shape}"

    num_samples = samples.shape[0]
    num_dims = samples.ndim
    sorted_samples = np.sort(samples, axis=0)

    abs_diff = (
        np.abs(np.expand_dims(target, axis=0) - sorted_samples).sum(axis=0)
        / num_samples
    )

    beta0 = sorted_samples.sum(axis=0) / num_samples

    # An array from 0 to num_samples - 1, but expanded to allow broadcasting over the variable dimensions
    i_array = np.expand_dims(np.arange(num_samples), axis=tuple(range(1, num_dims)))
    beta1 = (i_array * sorted_samples).sum(axis=0) / (num_samples * (num_samples - 1))

    return abs_diff + beta0 - 2 * beta1

def mean_crps(target, samples):
    """
    The mean of the CRPS over all variables
    """
    if target.size > 0:
        return crps(target, samples).mean()
    else:
        raise RuntimeError(
            f"CRPS received an empty target. Shapes = {target.shape} and {samples.shape}"
        )

def our_crps(
    target: np.array,
    forecasts: np.array,
    compute_variance: bool = False,
) -> dict[str, float]:
    """
    Compute the Region-of-Interest CRPS for a single entry of the context-is-key Hugging Face dataset,
    for the given forecast.

    Parameters:
    ----------
    entry: dict
        A dictionary containing a single entry of the context-is-key Hugging Face dataset.
    forecast: np.array
        The forecast values. (n_samples, n_timesteps)

    Returns:
    --------
    result: dict[str, float]
        A dictionary containing the following entries:
        "metric": the final metric.
        "raw_metric": the metric before the log transformation.
        "scaling": the scaling factor applied to the CRPS and the violations.
        "crps": the weighted CRPS.
        "roi_crps": the CRPS only for the region of interest.
        "non_roi_crps": the CRPS only for the forecast not in the region of interest.
        "violation_mean": the average constraint violation over the samples.
        "violation_crps": the CRPS of the constraint violation.
        "metric_variance": an unbiased estimate of the variance of the metric.
    """
    variance_target = target
    variance_forecast = forecasts

    crps_value = mean_crps(target=target, samples=forecasts)
    raw_metric = 1.0 * crps_value
    metric = raw_metric

    # Computing the variance of the RCPRS is much more expensive,
    # especially when the number of samples is large.
    # So it can be commented out if not desired.
    if compute_variance:
        variance_weights = np.full(
            target.shape, fill_value=1.0 / len(target)
        )
        variance = weighted_sum_crps_variance(
            target=variance_target,
            samples=variance_forecast,
            weights=variance_weights,
        )
    else:
        variance = None

    return {
        "metric": metric,
        "variance": variance,
    }