import pandas as pd

from src.features.base import AbstractBaseBlock


class GroupBlock(AbstractBaseBlock):
    """GroupKfold用のBlock

    Args:
        column (str): カラム名
    """

    def __init__(self, column: str):
        self.column = column

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """目的変数をそのまま返す. なお型はpd.DataFrameであることに注意

        Args:
            input_df (pd.DataFrame): 変数

        Returns:
            pd.DataFrame: カラムをそのまま返す
        """
        return input_df[[self.column]]
