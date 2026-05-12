# src/preprocess.py

import os
import glob
import pandas as pd

RAW_FOLDER = "data"
PROCESSED_FOLDER = "data/processed"

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

files = glob.glob(os.path.join(RAW_FOLDER, "*.csv"))

if not files:
    print("No raw CSV files found.")
    exit()

for file in files:
    stock_name = os.path.basename(file).replace(".csv", "")
    print(f"Processing {stock_name}...")

    try:
        # Yahoo Finance CSV has 2 extra header rows
        df = pd.read_csv(file, skiprows=2)

        # Rename columns properly
        df.columns = [
            "Date",
            "Close",
            "High",
            "Low",
            "Open",
            "Volume"
        ]

        # Convert datatypes
        df["Date"] = pd.to_datetime(df["Date"])
        numeric_cols = ["Close", "High", "Low", "Open", "Volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clean data
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Save cleaned file
        output_file = os.path.join(
            PROCESSED_FOLDER,
            f"{stock_name}_cleaned.csv"
        )

        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

    except Exception as e:
        print(f"Error processing {stock_name}: {e}")

print("\nPreprocessing completed successfully.")