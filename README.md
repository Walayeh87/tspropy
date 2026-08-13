# tspropy

**Time Series Processing in Python**

`tspropy` is a Python library for processing, analyzing, and visualizing time-series data with a focus on practical data-processing workflows.

The library provides reusable tools built around **pandas, NumPy, and Plotly**, including controlled interpolation, boolean-mask and phase processing, time-series plotting, and other utilities for working with datetime-indexed data.

## Features

- **DataFrame processing**
  - Controlled interpolation of missing values
  - Per-column interpolation methods and gap limits
  - NaN and data-quality analysis

- **Time-series analysis**
  - Boolean mask and phase processing
  - Phase duration and phase-level calculations
  - Utilities for datetime-indexed data

- **Visualization**
  - Interactive Plotly time-series plots
  - Multiple Y-axes
  - Highlighted time ranges and operating phases
  - Comparison and annotation features

## Example

```python
from tspropy import interpolate_df

result = interpolate_df(
    df,
    methods_mapper={
        "PV Generation": "linear",
        "Grid Load": "time",
    },
)