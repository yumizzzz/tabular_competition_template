import numpy as np
import pandas as pd

from src.features.base import AbstractBaseBlock
from src.utils.cv_method import Method, select_cv_method


class TargetEncodingBlock(AbstractBaseBlock):
    """ターゲットエンコーディングを行うBlock"""

    def __init__(
        self,
        column: str,
        target_column: str,
        method: Method,
        n_splits: int,
        shuffle: bool = True,
        seed: int = 42,
        group_col: str | None = None,
    ):
        self.column = column
        self.target_column = target_column
        self.method = method
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed
        self.group_col = group_col

    def fit(self, input_df: pd.DataFrame):
        self.target_map = input_df.groupby(self.column)[self.target_column].mean()

        output_encoded = np.zeros(len(input_df))
        cv = select_cv_method(method=self.method, n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)

        x = input_df.drop(self.target_column, axis=1)
        y = input_df[self.target_column].values
        if self.group_col is not None:
            group = input_df[self.group_col].values
        else:
            group = None

        for train_idx, valid_idx in cv.split(x, y, group):
            _map = input_df.iloc[train_idx].groupby(self.column)[self.target_column].mean().to_dict()
            output_encoded[valid_idx] = input_df.iloc[valid_idx][self.column].map(_map)

        output_df = pd.DataFrame(output_encoded, columns=[self.column]).add_prefix("te_")
        return output_df

    def transform(self, input_df: pd.DataFrame):
        output_encoded = input_df[self.column].map(self.target_map).values
        output_df = pd.DataFrame(output_encoded, columns=[self.column]).add_prefix("te_")
        return output_df
