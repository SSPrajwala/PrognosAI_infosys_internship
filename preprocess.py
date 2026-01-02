import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def compute_rul(train_df):
    """
    Compute RUL for each engine
    """
    max_cycles = train_df.groupby("unit")["cycle"].max()
    train_df["RUL"] = train_df.apply(
        lambda row: max_cycles[row["unit"]] - row["cycle"],
        axis=1
    )
    return train_df


def clean_constant_sensors(df):
    """
    Remove sensors with no variation
    """
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=constant_cols)
    return df, constant_cols


def scale_train_test(train_df, test_df, feature_cols):
    """
    Normalize features with MinMaxScaler
    """
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df, scaler
