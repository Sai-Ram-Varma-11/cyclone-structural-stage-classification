import stormtrack as st

data = st.load()
df = data.observations

print("DATASET PREVIEW")
print(df.head(10).to_string())

print("\nDATASET INFORMATION")
print(df.info())

print("\nDATASET SHAPE")
print(df.shape)
df.to_csv("data/raw/cyclone_dataset.csv", index=False)
print("Dataset saved successfully!")