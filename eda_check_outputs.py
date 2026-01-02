import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = r"C:\Users\kesir\Desktop\PrognosAI\outputs"

fds = ["FD001", "FD002", "FD003", "FD004"]

for fd in fds:
    print(f"\n===== Checking {fd} =====")

    fd_path = f"{OUTPUT_DIR}/{fd}"

    train = pd.read_csv(f"{fd_path}/cleaned_train.csv")
    test = pd.read_csv(f"{fd_path}/cleaned_test.csv")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Columns:", train.columns.tolist())

    # Check missing values
    print("\nMissing values in train:\n", train.isna().sum().sum())
    print("Missing values in test:\n", test.isna().sum().sum())

    # Check RUL distribution
    plt.figure()
    sns.histplot(train["RUL"], bins=50)
    plt.title(f"RUL Distribution - {fd}")
    plt.show()

    # Check sample sequences
    X = np.load(f"{fd_path}/sequences_train.npy")
    y = np.load(f"{fd_path}/labels_train.npy")

    print("Sequences shape:", X.shape)
    print("Labels shape:", y.shape)

    print("Example label:", y[0])
    print("Example sequence first row:", X[0][0])
