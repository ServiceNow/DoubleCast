# DoubleCast
Code for the DoubleCast model architecture for context-aided time series forecasting, introduced in [Overcoming the Modality Gap in Context-Aided Forecasting](https://arxiv.org/abs/2603.12451).

This model was trained using the [CAF-7M](https://huggingface.co/datasets/ServiceNow/CAF_7M) semi-synthetic dataset, also introduced in the same paper. The trained weighted are available at [HuggingFace](https://huggingface.co/ServiceNow/DoubleCast).

A comparison of forecasts between DoubleCast and Chronos can be seen in the associated [demo](https://vincent-zheng.com/DoubleCast/).

Disclaimer: this code is not compatible with newer versions of the `transformers` library. This version is only known to be compatible with version `4.51.3` of said library.

## Usage example

```python
import torch
from functools import partial
import pandas as pd
import numpy as np

from doublecast.models.doublecast import DoubleCastPipeline
from doublecast.data.prompts import prompt_structured

# Load from HuggingFace Hub (downloads delta weights only;
# base Chronos and Qwen3-14B weights are fetched from their original repos)
pipeline = DoubleCastPipeline.from_pretrained(
    "ServiceNow/DoubleCast",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

past_timestamps = pd.date_range(start="2023-01-01", periods=20, freq="D")
future_timestamps = pd.date_range(start="2023-01-21", periods=10, freq="D")
freq = "D"
past_values = np.random.rand(20) + 0.2 * np.arange(20)
text_context = "The past values show a clear upward trend, with some fluctuations. However, afterwards a massive drop is expected in the values."

prompt_func = partial(
    prompt_structured,
    past_timestamps=past_timestamps,
    future_timestamps=future_timestamps,
    freq=freq
)

context = torch.tensor(
    past_values,
    dtype=torch.bfloat16,
).flatten()

dual_preds = pipeline.predict(
    context=context,
    text_context=text_context,
    prediction_length=len(future_timestamps),
    num_samples=20,
    limit_prediction_length=False,
    format_prompt=prompt_func,
)
```
