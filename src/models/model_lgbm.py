import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from wandb.lightgbm import log_summary, wandb_callback

from src.models.base import BaseModel


class LightGBM(BaseModel):
    """LightGBMモデル"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def train(
        self,
        tr_x: pd.DataFrame,
        tr_y: pd.DataFrame,
        va_x: pd.DataFrame | None = None,
        va_y: pd.DataFrame | None = None,
    ) -> None:
        """LightGBMモデルの学習を行う"""

        lgb_train = lgb.Dataset(tr_x, tr_y, categorical_feature=self.cfg.categorical_feature)
        lgb_val = lgb.Dataset(va_x, va_y, reference=lgb_train, categorical_feature=self.cfg.categorical_feature)

        callbacks = [
            lgb.early_stopping(stopping_rounds=self.cfg.early_stopping_rounds),
            lgb.log_evaluation(period=self.cfg.log_evaluation_period),
            wandb_callback(),
        ]

        self.model = lgb.train(
            OmegaConf.to_container(self.cfg.params),  # type: ignore
            lgb_train,
            num_boost_round=self.cfg.num_boost_round,
            valid_sets=[lgb_train, lgb_val],
            callbacks=callbacks,  # type: ignore
        )

        log_summary(self.model, save_model_checkpoint=False)

    def predict(self, X: pd.DataFrame) -> Any:
        """LightGBMモデルを使って予測を行う"""
        return self.model.predict(X, num_iteration=self.model.best_iteration)

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
        feature_importance = self.model.feature_importance(importance_type="gain")
        return pd.DataFrame(
            {
                "feature_name": self.model.feature_name(),
                "feature_importance": feature_importance / feature_importance.sum(),
            }
        )
