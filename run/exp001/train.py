import shutil
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

import wandb
from src.models import build_model
from src.utils.cv_method import select_cv_method
from src.utils.load_data import load_group, load_submission, load_x_test, load_x_train, load_y_train
from src.utils.metric import metric
from src.utils.utils import seed_everything
from src.utils.visualizer import create_feature_importance


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """学習の実行関数

    Args:
        cfg (DictConfig): hydraで読み込んだ設定ファイル
    """
    seed_everything(cfg.setting.seed)

    wandb.init(
        project=cfg.setting.project_name,
        name=cfg.setting.run_name,
        mode="online",
        config=OmegaConf.to_container(cfg),  # type: ignore
    )

    # 各データの読み込み
    x_train = load_x_train(cfg.dir.features_dir, cfg.model.features)
    x_test = load_x_test(cfg.dir.features_dir, cfg.model.features)
    y_train = load_y_train(cfg.dir.features_dir)
    sub_df = load_submission(cfg.dir.submission_csv_path)
    if cfg.setting.group_col is not None:
        group = load_group(cfg.dir.features_dir)
    else:
        group = None

    # CVの設定
    cv = select_cv_method(cfg.model.cv_method, cfg.model.n_splits, cfg.model.shuffle, cfg.setting.seed)

    oof = np.zeros(x_train.shape[0])
    test_preds: list[np.ndarray] = []
    feature_importance_list: list[pd.DataFrame] = []

    for i_fold, (train_idx, valid_idx) in enumerate(cv.split(x_train, y_train, group)):
        print(f"================== 🚀 Start training: Fold{i_fold} ==================")

        tr_x, tr_y = x_train.iloc[train_idx], y_train.iloc[train_idx]
        va_x, va_y = x_train.iloc[valid_idx], y_train.iloc[valid_idx]

        # モデルの学習
        model = build_model(cfg)
        model.train(tr_x, tr_y, va_x, va_y)
        model.save_model(cfg.dir.models_dir, cfg.setting.run_name, i_fold)

        # モデルの推論
        tr_pred = model.predict(tr_x)
        va_pred = model.predict(va_x)
        te_pred = model.predict(x_test)

        # 評価指標の計算
        tr_score = metric(cfg.model.metric, tr_y, tr_pred)
        va_score = metric(cfg.model.metric, va_y, va_pred)

        # logの出力
        wandb.log({f"Fold{i_fold} train's {cfg.model.metric}": tr_score})
        wandb.log({f"Fold{i_fold} valid's {cfg.model.metric}": va_score})
        print(f"training's {cfg.model.metric}: {tr_score:.4f} valid's {cfg.model.metric}: {va_score:.4f}")

        # 結果を保持
        oof[valid_idx] = va_pred
        test_preds.append(te_pred)

        # feature importanceの取得
        if hasattr(model, "feature_importance"):
            feature_importance_list.append(model.feature_importance())

    print("======================== 🎉 Finish!! ========================")

    # 全体スコアを計算
    oof_score = metric(cfg.model.metric, y_train, oof)
    wandb.log({f"oof's {cfg.model.metric}": oof_score})
    print(f"oof's {cfg.model.metric}: {oof_score:.4f}")

    # feature importanceの可視化保存
    if len(feature_importance_list) > 0:
        fig = create_feature_importance(feature_importance_list)
        wandb.log({"feature importance": wandb.Image(fig)})
        Path(cfg.dir.figs_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(cfg.dir.figs_dir, f"{cfg.setting.run_name}_feature_importance.png"))

    # oof保存
    oof_df = pd.DataFrame({"score": oof})
    Path(cfg.dir.oof_dir).mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(Path(cfg.dir.oof_dir, f"{cfg.setting.run_name}_oof.csv"), index=False)

    # submission作成
    sub_df[cfg.setting.target_col] = np.mean(test_preds, axis=0)
    Path(cfg.dir.submission_dir).mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(Path(cfg.dir.submission_dir, f"{cfg.setting.run_name}_submission.csv"), index=False)

    # wandbの終了処理
    wandb.finish()

    # wandbの取得したlogを指定したディレクトリに保存
    Path(cfg.dir.logs_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy("wandb/latest-run/files/output.log", Path(cfg.dir.logs_dir, f"{cfg.setting.run_name}.log"))


if __name__ == "__main__":
    main()
