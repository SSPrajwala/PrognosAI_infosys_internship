import os
from load_data import load_dataset
from preprocess import compute_rul, clean_constant_sensors, scale_train_test
from create_sequences import generate_sequences
from utils import ensure_dir
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\kesir\Desktop\PrognosAI\data"
OUTPUT_DIR = r"C:\Users\kesir\Desktop\PrognosAI\outputs"

SEQUENCE_LENGTH = 30

fds = ["FD001", "FD002", "FD003", "FD004"]

for fd in fds:
    print(f"\n========== Processing {fd} ==========")

    out_dir = f"{OUTPUT_DIR}/{fd}"
    ensure_dir(out_dir)

    # Load raw data
    train, test, rul = load_dataset(DATA_DIR, fd)

    # Compute RUL for train
    train = compute_rul(train)

    # Assign RUL to test sequences
    max_test_cycles = test.groupby("unit")["cycle"].max().reset_index()
    test = test.merge(max_test_cycles, on="unit", suffixes=("", "_max"))
    test["RUL"] = test["cycle_max"] - test["cycle"]
    test.drop(columns=["cycle_max"], inplace=True)

    # Remove constant sensors
    train, removed = clean_constant_sensors(train)
    test = test.drop(columns=removed)

    feature_cols = [c for c in train.columns if c not in ["unit", "cycle", "RUL"]]

    # Scale
    train, test, scaler = scale_train_test(train, test, feature_cols)

    # Create sequences
    X_train, y_train = generate_sequences(train, SEQUENCE_LENGTH, feature_cols)

    # Save everything
    train.to_csv(f"{out_dir}/cleaned_train.csv", index=False)
    test.to_csv(f"{out_dir}/cleaned_test.csv", index=False)

    np.save(f"{out_dir}/sequences_train.npy", X_train)
    np.save(f"{out_dir}/labels_train.npy", y_train)

    # Documentation file
    with open(f"{out_dir}/data_preparation_{fd}.md", "w") as f:
        f.write(f"# Data Preparation Summary for {fd}\n")
        f.write(f"- Removed sensors: {removed}\n")
        f.write(f"- Sequence length: {SEQUENCE_LENGTH}\n")
        f.write(f"- Final features: {feature_cols}\n")

    print(f"Done {fd}. Saved files in {out_dir}")
