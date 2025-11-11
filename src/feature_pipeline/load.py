"""
Load & time-split the raw dataset

- Production writes to data/raw
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

def load_and_split_data(
  raw_path: str = "data/raw/untouched_raw_original.csv",
  output_dir: Path | str = DATA_DIR
):
  """Load raw dataset, split into train/valid/test by date and save to output_dir

  Args:
      raw_path (str, optional): _description_. Defaults to "data/raw/untouched_raw_original.csv".
      output_dir (Path | str, optional): _description_. Defaults to DATA_DIR.
  """
  df = pd.read_csv(raw_path)

  df['date'] = pd.to_datetime(df['date'])
  df = df.sort_values('date')

  # Cutoff
  date_valid = pd.Timestamp("2020-01-01")
  date_test = pd.Timestamp("2022-01-01")

  # Splits
  train_df = df[df['date'] < date_valid]
  valid_df = df[(df['date'] >= date_valid) & (df['date'] < date_test)]
  test_df = df[df['date'] >= date_test]

  # Save
  outdir = Path(output_dir)
  outdir.mkdir(parents=True, exist_ok=True)
  train_df.to_csv(outdir / "train.csv", index=False)
  valid_df.to_csv(outdir / "valid.csv", index=False)
  test_df.to_csv(outdir / "test.csv", index=False)

  print(f"Feature Pipeline #1: Load and split data completed, saved to {outdir}")
  print(f"Train: {train_df.shape} | Validation: {valid_df.shape} | Test: {test_df.shape}")

if __name__ == "__main__":
  load_and_split_data()