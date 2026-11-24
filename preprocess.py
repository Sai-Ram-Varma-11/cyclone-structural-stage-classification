
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
RAW_FILE = BASE_DIR / "data" / "raw" / "cyclone_dataset.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "cyclone_dataset.csv"
DATASET_FILE = OUTPUT_FILE

FEATURES = ["year", "month", "lat", "lon", "pressure", "basin"]
TARGET = "grade"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE)

    df = df[df["basin"].isin(["BOB", "ARB"])].copy()
    df = df.dropna(subset=[TARGET])

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    df["month"] = df["time"].dt.month

    df = df[["storm_id"] + FEATURES + [TARGET]]

    df.to_csv(OUTPUT_FILE, index=False)

    print("PREPROCESSING COMPLETED")
    print("Dataset shape:", df.shape)

    print("\nGRADE DISTRIBUTION:")
    print(df[TARGET].value_counts())

    print("\nMISSING VALUES:")
    print(df.isnull().sum())

    print("\nSAVED TO:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
