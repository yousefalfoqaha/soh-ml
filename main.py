import matplotlib.pyplot as plt
from asammdf import MDF

mdf = MDF("sample.mf4")

df = mdf.to_dataframe(
    channels=["U", "I"],
    raster="U",
    time_from_zero=True,
)

df = df.cumsum()
df.plot()
plt.show()
