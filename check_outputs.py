import pandas as pd
import numpy as np
import os

BASE = r"C:\Users\kesir\Desktop\PrognosAI\outputs"

fds = ["FD001", "FD002", "FD003", "FD004"]

for fd in fds:
    print("\n==============================")
    print(f"Summary for {fd}")

    path = os.path.join(BASE, fd)

    # Load CSVs
    train = pd.read_csv(f"{path}/cleaned_train.csv")
    test = pd.read_csv(f"{path}/cleaned_test.csv")

    print("\nTrain shape:", train.shape)
    print("Test shape:", test.shape)

    print("\nColumns:", list(train.columns))

    # Check sequences
    X = np.load(f"{path}/sequences_train.npy")
    y = np.load(f"{path}/labels_train.npy")

    print("\nSequences shape:", X.shape)
    print("Labels shape:", y.shape)

    print("Example sequence:", X[0][0][:5])  # first 5 sensor values
    print("Example label:", y[0])
