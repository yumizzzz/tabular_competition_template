from typing import Literal

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.features.base import AbstractBaseBlock


class LabelEncodingBlock(AbstractBaseBlock):
    """ラベルエンコーディングを行う"""

    def __init__(
        self,
        columns: list[str],
        source_df: pd.DataFrame | None = None,
        type: Literal["category", "int"] = "category",  # catboostの場合int型にする必要あり
    ):
        self.columns = columns
        self.source_df = source_df
        self.type = type

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """学習用データに対し, ラベルエンコーディングを行う

        未知の値は-1に変換. Nanは-999にする
        """
        self.oe = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-999,
        )

        if self.source_df is not None:
            self.oe.fit(self.source_df[self.columns])
        else:
            self.oe.fit(input_df[self.columns])

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """学習時にfitしたラベルエンコーディングを行う"""
        output_df = input_df[self.columns].copy()
        output_df = self.oe.transform(output_df)
        output_df = pd.DataFrame(output_df, columns=self.columns)

        if self.type == "category":
            return output_df.add_prefix("le_").astype("category")
        elif self.type == "int":
            return output_df.add_prefix("le_").astype(int)
        else:
            raise NotImplementedError
