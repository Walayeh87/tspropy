import pandas as pd

from src.plotters.plotly_plotters import plot_datetime_data

# Basic plotting features
# Plotting all dataset columns on one axes. Legends, x-label "Time" and grid are shown in default
df = pd.read_csv(
    r"C:\tspropy\examples\data\load_and_price.csv",
    index_col="datetime",
    parse_dates=True,
)

loads = df[["Total Site Load", "Grid Load", "PV Generation"]]
fig1 = plot_datetime_data(df=loads)
fig1.show()
