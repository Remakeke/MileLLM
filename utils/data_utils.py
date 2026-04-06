import pandas as pd


def load_csv(path):
    """Load dataset from CSV file."""
    return pd.read_csv(path)