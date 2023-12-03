from pathlib import Path

import pandas as pd


def load_x_train(features_dir: str | Path, features: list[str]) -> pd.DataFrame:
    """学習用データの説明変数の読み込み

    Args:
        features_dir (str | Path): 特徴量が保存されているディレクトリ
        features (list[str]): 読み込む特徴量の名前のリスト

    Returns:
        pd.DataFrame: 与えられた特徴量を列とする学習データのDataFrame
    """
    dfs = [pd.read_pickle(Path(features_dir, f"{feat}_train.pkl")) for feat in features]
    return pd.concat(dfs, axis=1)


def load_x_test(features_dir: str | Path, features: list[str]) -> pd.DataFrame:
    """テストデータの説明変数の読み込み

    Args:
        features_dir (str | Path): 特徴量が保存されているディレクトリ
        features (list[str]): 読み込む特徴量の名前のリスト

    Returns:
        pd.DataFrame: 与えられた特徴量を列とするテストデータのDataFrame
    """
    dfs = [pd.read_pickle(Path(features_dir, f"{feat}_test.pkl")) for feat in features]
    return pd.concat(dfs, axis=1)


def load_y_train(features_dir: str | Path) -> pd.DataFrame:
    """学習用データの目的変数の読み込み

    Args:
        features_dir (str | Path): 目的変数が保存されているディレクトリ

    Returns:
        pd.DataFrame: 目的変数のDataFrame
    """
    return pd.read_pickle(Path(features_dir, "TargetBlock_train.pkl"))


def load_group(features_dir: str | Path) -> pd.DataFrame:
    """GroupKFold用カラムの読み込み

    Args:
        features_dir (str | Path): グループのカラムが保存されているディレクトリ

    Returns:
        pd.DataFrame: グループのDataFrame
    """
    return pd.read_pickle(Path(features_dir, "GroupBlock_train.pkl"))


def load_submission(submission_csv_path: str | Path) -> pd.DataFrame:
    """submissionファイルの読み込み

    Args:
        submission_csv_path (str | Path): submissionファイルのパス

    Returns:
        pd.DataFrame: submissionファイルのDataFrame
    """
    return pd.read_csv(submission_csv_path)
