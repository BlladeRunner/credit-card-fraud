from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_col],
    )
    return train_df, test_df
