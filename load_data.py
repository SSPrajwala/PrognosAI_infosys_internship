import pandas as pd

def read_cmapss(path):
    """
    Reads any CMAPSS file (train or test)
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = [
        "unit", "cycle",
        "op1", "op2", "op3",
        *[f"sensor_{i}" for i in range(1, 22)]
    ]
    return df


def load_dataset(data_dir, fd):
    """
    Loads train, test, and RUL files for FD00X dataset
    """
    train = read_cmapss(f"{data_dir}/train_{fd}.txt")
    test = read_cmapss(f"{data_dir}/test_{fd}.txt")
    rul = pd.read_csv(f"{data_dir}/RUL_{fd}.txt", header=None)

    return train, test, rul
