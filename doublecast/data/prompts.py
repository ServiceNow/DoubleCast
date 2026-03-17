import pandas as pd


def prompt_naive(past_timestamps, future_timestamps, context, past_target, scale, freq=None):
    return (
        f"<context>\n{context}\n</context>\n\n"
    )


def prompt_structured(past_timestamps, future_timestamps, context, past_target, scale, freq=None):
    """
    A robust prompt that handles both DatetimeIndex and PeriodIndex timestamps.
    """
    # Get frequency if not provided
    freq_symbol = freq if freq else pd.infer_freq(future_timestamps)

    # Format timestamp consistently
    try:
        # Handle PeriodIndex by converting to timestamp first
        if isinstance(future_timestamps[0], pd.Period):
            start_date = future_timestamps[0].to_timestamp().strftime('%Y-%m-%d %H:%M:%S')
        else:
            start_date = future_timestamps[0].strftime('%Y-%m-%d %H:%M:%S')
    except (AttributeError, ValueError):
        start_date = str(future_timestamps[0])

    return (
        f"<info>\n"
        f"forecast_start_date={start_date}\n"
        f"frequency={freq_symbol}\n"
        f"scale_factor={scale.item():.4f}\n"
        f"</info>\n\n"
        f"<context>\n{context}\n</context>"
    )


PROMPTS = {
    "naive": prompt_naive,
    "structured": prompt_structured,
}
