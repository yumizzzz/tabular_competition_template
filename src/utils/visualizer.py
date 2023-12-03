import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_feature_importance(f_importances_list: list[pd.DataFrame]) -> plt.Figure:
    """特徴量の重要度を保存する"""

    feature_importance_df = pd.concat(f_importances_list)
    order = (
        feature_importance_df.groupby("feature_name")
        .sum()[["feature_importance"]]
        .sort_values("feature_importance", ascending=False)
        .index[:50]
    )
    fig, ax = plt.subplots(figsize=(12, max(4, len(order) * 0.2)))
    sns.boxenplot(
        data=feature_importance_df,
        y="feature_name",
        x="feature_importance",
        order=order,
        ax=ax,
        hue="feature_name",
        palette="viridis",
        legend=False,
    )
    fig.tight_layout()
    ax.grid()
    ax.set_title("feature importance")
    fig.tight_layout()

    return fig
