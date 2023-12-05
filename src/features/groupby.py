import pandas as pd

from src.features.base import AbstractBaseBlock


class GroupbyBlock(AbstractBaseBlock):
    """Groupby特徴量を作成するBlock"""

    def __init__(
        self,
        groupby_column: str,
        columns: list[str],
        source_df: pd.DataFrame | None = None,
        agg_funcs: list[str] = ["mean", "std", "max", "min"],
    ):
        self.groupby_column = groupby_column
        self.columns = columns
        self.source_df = source_df
        self.agg_funcs = agg_funcs

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        # groupbyの対象となるDataFrameを設定
        if self.source_df is None:
            self.source_df = input_df.copy()

        self.dfs = []
        for c in self.columns:
            groupby_df = (
                self.source_df.groupby(self.groupby_column)
                .agg({c: self.agg_funcs})
                .reset_index()
                .set_index(self.groupby_column)
            )
            groupby_df.columns = [f"agg_{agg_func}_{c}_gropuby_{self.groupby_column}" for agg_func in self.agg_funcs]
            self.dfs.append(groupby_df)

        self.group_df = pd.concat(self.dfs, axis=1).reset_index()

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.merge(input_df[self.groupby_column], self.group_df, on=self.groupby_column, how="left")
        output_df = output_df.drop(self.groupby_column, axis=1).reset_index(drop=True)

        return output_df
