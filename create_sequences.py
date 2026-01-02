import numpy as np

def generate_sequences(df, sequence_length, feature_cols):
    """
    Creates 3D sequences for LSTM
    Shape: (samples, seq_len, features)
    """
    sequences = []
    labels = []

    units = df["unit"].unique()

    for unit in units:
        unit_df = df[df["unit"] == unit].reset_index(drop=True)
        data = unit_df[feature_cols].values
        rul = unit_df["RUL"].values

        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
            labels.append(rul[i + sequence_length])

    return np.array(sequences), np.array(labels)
