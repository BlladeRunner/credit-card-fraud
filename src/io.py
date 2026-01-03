from __future__ import annotations
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
