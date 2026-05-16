from asammdf import MDF

mdf = MDF("sample.mf4")

df = mdf.to_dataframe(
    channels=["U", "I"],
    raster="U",
    time_from_zero=True,
)

print(df.info())
