from omegaconf import DictConfig

from .model_catboost import CatBoostModel
from .model_lgbm import LightGBMModel

MODEL = LightGBMModel | CatBoostModel


def build_model(cfg: DictConfig) -> MODEL:
    """モデルを作成する関数

    Args:
        cfg (DictConfig): hydraで読み込んだ設定ファイル

    Returns:
        model (BaseModel): モデル
    """
    if cfg.model.model_name == "LightGBMModel":
        return LightGBMModel(cfg.model)
    elif cfg.model.model_name == "CatBoostModel":
        return CatBoostModel(cfg.model)
    else:
        raise NotImplementedError
