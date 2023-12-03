from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig


class BaseModel(metaclass=ABCMeta):
    """モデル作成の基底クラス"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def train(
        self,
        tr_x: pd.DataFrame,
        tr_y: pd.DataFrame,
        va_x: pd.DataFrame | None = None,
        va_y: pd.DataFrame | None = None,
    ) -> None:
        """学習を実施

        Args:
            tr_x (pd.DataFrame): 学習データの説明変数
            tr_y (pd.DataFrame): 学習データの目的変数
            va_x (pd.DataFrame | None, optional): 検証データの説明変数. Defaults to None.
            va_y (pd.DataFrame | None, optional): 検証データの目的変数. Defaults to None.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """予測を実施

        Args:
            X (pd.DataFrame): 予測データの説明変数

        Returns:
            np.ndarray: 予測結果
        """
        pass

    @abstractmethod
    def save_model(self, save_dir: str | Path, run_name: str, fold: int) -> None:
        """モデルを保存する

        Args:
            save_dir (str | Path): 保存先のディレクトリ
            run_name (str): 実行名
            fold (int): fold番号
        """
        pass
