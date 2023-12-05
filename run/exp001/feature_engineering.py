import hydra
import pandas as pd
from omegaconf import DictConfig

from src.features import CountEncodingBlock, GroupBlock, IdentityBlock, LabelEncodingBlock, TargetBlock
from src.features.base import AbstractBaseBlock
from src.features.run_blocks import run_blocks
from src.utils.utils import seed_everything


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """特徴量作成の実行関数

    Args:
        cfg (DictConfig): hydraで読み込んだ設定ファイル
    """
    seed_everything(cfg.setting.seed)

    train_df = pd.read_csv(cfg.dir.train_csv_path)
    test_df = pd.read_csv(cfg.dir.test_csv_path)
    all_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    # 特徴量変換関数の定義
    feature_run_blocks: list[AbstractBaseBlock] = [
        IdentityBlock(cfg.setting.numerical_features),
        LabelEncodingBlock(cfg.setting.categorical_features, all_df),
        CountEncodingBlock(cfg.setting.categorical_features, all_df),
    ]

    # 特徴量作成及び保存
    run_blocks(train_df, feature_run_blocks, cfg.dir.features_dir, is_train=True)
    run_blocks(test_df, feature_run_blocks, cfg.dir.features_dir, is_train=False)

    # 目的変数の保存
    target_run_blocks: list[AbstractBaseBlock] = [TargetBlock(cfg.setting.target_col)]
    run_blocks(train_df, target_run_blocks, cfg.dir.features_dir, is_train=True)

    # GroupKFold用のカラム保存
    if cfg.setting.group_col is not None:
        group_run_blocks: list[AbstractBaseBlock] = [GroupBlock(cfg.setting.group_col)]
        run_blocks(train_df, group_run_blocks, cfg.dir.features_dir, is_train=True)


if __name__ == "__main__":
    main()
