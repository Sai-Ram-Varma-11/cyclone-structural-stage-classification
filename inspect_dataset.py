
import pandas as pd

df = pd.read_csv("data/raw/cyclone_dataset.csv")

print("DATASET SHAPE:", df.shape)

print("\nCYCLONE GRADES:")
print(df["grade"].value_counts(dropna=False))

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nCYCLONE BASINS:")
print(df["basin"].value_counts(dropna=False))

print("\nDATA TYPES:")
print(df.dtypes)
print("\nGRADES BY BASIN:")
print(pd.crosstab(df["basin"], df["grade"], dropna=False))