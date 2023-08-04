"""Functions to visualize"""
# =================
# ==== IMPORTS ====
# =================

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# Options
sns.set_style("whitegrid")


# ===================
# ==== FUNCTIONS ====
# ===================

def plot_distribution_feat_cat(df: pd.DataFrame, feat_cat: list[str]) -> None:
    """Plot the distribution for the categorical features

    Args:
        df: DataFrame with the categorical features
        feat_cat: list of the categorical features
    """
    n_rows = len(feat_cat) // 3 + int((len(feat_cat) % 3) > 0)
    _, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(feat_cat):
        ax = axes[i]
        # n_unique = df[feat].nunique()
        sns.histplot(df[feat], ax=ax)
        # df[feat].hist(bins=n_unique).plot(ax=ax)
        ax.set_title(f'Distribution of the {feat}')
        ax.set_xlabel(f'{feat}')
        ax.set_ylabel('Number per category')
    plt.show();


def plot_distribution_feat_cont(df: pd.DataFrame, feat_cont: list[str]) -> None:
    """
    """
    n_rows = len(feat_cont) // 3 + int((len(feat_cont) % 3) > 0)
    _, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(feat_cont):
        ax = axes[i]
        # n_unique = df[feat].nunique()
        # sns.histplot(df[feat], ax=ax)
        df[feat].hist(bins=500).plot(ax=ax)
        ax.set_title(f'Distribution of the {feat}')
        ax.set_xlabel(f'{feat}')
        ax.set_ylabel('Number per category')
    plt.show();
