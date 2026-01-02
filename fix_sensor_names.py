import pandas as pd
import os

BASE_DIR = "outputs"

def normalize_sensor_names(df):
    rename_map = {}
    for col in df.columns:
        if col.startswith("sensor_"):
            num = col.split("_")[1]
            rename_map[col] = f"s{num}"
    return df.rename(columns=rename_map)

for fd in ["FD001", "FD002", "FD003", "FD004"]:
    path = os.path.join(BASE_DIR, fd, "cleaned_test.csv")
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)
    df = normalize_sensor_names(df)
    df.to_csv(path, index=False)
    print(f"✅ Fixed sensor names for {fd}")
