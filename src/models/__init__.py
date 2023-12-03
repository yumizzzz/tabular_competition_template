from omegaconf import DictConfig

from .model_lgbm import LightGBM

MODEL = LightGBM


def build_model(cfg: DictConfig) -> MODEL:
    """モデルを作成する関数

    Args:
        cfg (DictConfig): hydraで読み込んだ設定ファイル

    Returns:
        model (BaseModel): モデル
    """
    if cfg.model.model_name == "LightGBM":
        model = LightGBM(cfg.model)
    else:
        raise NotImplementedError
    return model
