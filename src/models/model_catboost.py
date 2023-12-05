import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoost, Pool
from omegaconf import DictConfig, OmegaConf

from src.models.base import BaseModel


class CatBoostModel(BaseModel):
    """CatBoostモデル"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def train(
        self,
        tr_x: pd.DataFrame,
        tr_y: pd.DataFrame,
        va_x: pd.DataFrame | None = None,
        va_y: pd.DataFrame | None = None,
    ) -> None:
        """CatBoostモデルの学習を行う"""

        tr_pool = Pool(tr_x, tr_y, cat_features=self.cfg.categorical_features)
        va_pool = Pool(va_x, va_y, cat_features=self.cfg.categorical_features)
        self.model = CatBoost(OmegaConf.to_container(self.cfg.params))
        self.model.fit(
            tr_pool,
            eval_set=[va_pool],
            use_best_model=True,
            plot=False,
            verbose=self.cfg.verbose,
            early_stopping_rounds=self.cfg.early_stopping_rounds,
        )

    def predict(self, X: pd.DataFrame) -> Any:
        """CatBoostモデルを使って予測を行う"""
        return self.model.predict(X)

    def save_model(self, save_dir: str | Path, run_name: str, fold: int) -> None:
        """モデルを保存する"""
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, f"{run_name}_model-{fold}.pkl")

        with open(save_path, mode="wb") as f:
            pickle.dump(self.model, f)

    def feature_importance(self) -> pd.DataFrame:
        """feature_importanceを出力

        Returns:
            pd.DataFrame: feature_importance
        """
        feature_importance = self.model.get_feature_importance(type="PredictionValuesChange")
        return pd.DataFrame(
            {
                "feature_name": self.model.feature_names_,
                "feature_importance": feature_importance / feature_importance.sum(),
            }
        )
