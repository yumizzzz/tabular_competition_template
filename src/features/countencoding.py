import pandas as pd
from category_encoders import CountEncoder

from src.features.base import AbstractBaseBlock


class CountEncodingBlock(AbstractBaseBlock):
    """カウントエンコーディングを行う"""

    def __init__(self, columns: list[str], source_df: pd.DataFrame | None = None):
        self.columns = columns
        self.source_df = source_df

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """学習用データに対し, カウントエンコーディングを行う. NaNの数もカウントする"""

        self.ce = CountEncoder(cols=self.columns)
        if self.source_df is not None:
            self.ce.fit(self.source_df[self.columns])
        else:
            self.ce.fit(input_df[self.columns])

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """学習時にfitしたカウントエンコーディングを行う. NaNの数もカウントする"""
        output_df = input_df[self.columns].copy()
        output_df = self.ce.transform(output_df)

        return output_df.add_prefix("ce_")
