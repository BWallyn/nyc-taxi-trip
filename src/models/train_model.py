"""Train a model"""
# =================
# ==== IMPORTS ====
# =================

import os
import pandas as pd

from catboost import Pool, CatBoostRegressor


# ===================
# ==== FUNCTIONS ====
# ===================

def train_catboost_model(
    df_train: pd.DataFrame, df_eval:pd.DataFrame, y_train: pd.Series, y_eval: pd.Series,
    feat_cat: list[str], verbose: int=100, plot: bool=False,
    **params
) -> CatBoostRegressor:
    """Train a CatBoostRegressor model to predict the duration of the trip

    Args:
        df_train: Train set
        df_eval: Evaluation dataset
        y_train: Targets of the train set
        y_eval: Targets of the evaluation set
        feat_cat: List of the categorical features
        verbose: Verbose of Catboost
    """
    # Create pool
    pool_train = Pool(data=df_train, label=y_train, cat_features=feat_cat)
    pool_eval = Pool(data=df_eval, label=y_eval, cat_features=feat_cat)
    # Train catboost model
    model = CatBoostRegressor(**params)
    model.fit(
        pool_train,
        eval_set=pool_eval,
        verbose=verbose,
        plot=plot
    )
    return model


def save_model(model: CatBoostRegressor, feat: list[str], feat_cat: list[str]) -> None:
    """
    """


# =============
# ==== RUN ====
# =============

if __name__ == "__main__":
    # Options
    path = os.path.join('data', "interim")
    name_file = "yellow_trip_2023-1_2023-5_duration_feat"
    # Load data
    df_train = pd.read_parquet(os.path.join(path, f"{name_file}_train.parquet"))
    df_valid = pd.read_parquet(os.path.join(path, f"{name_file}_valid.parquet"))
    y_train = pd.read_csv(os.path.join(path, f"target_train.csv"))
    y_valid = pd.read_csv(os.path.join(path, f"target_valid.csv"))
    # Parameters
    params = {
        'iterations': 1000,
        'depth': 17,
        'early_stopping_rounds': 100,
        'use_best_model': True
    }
    feat_cat = ["VendorID", "RatecodeID", "store_and_fwd_flag", "PULocationID", "DOLocationID", "payment_type", "improvement_surcharge"]
    # Run
    model = train_catboost_model(
        df_train=df_train, df_eval=df_valid, y_train=y_train, y_eval=y_valid,
        feat_cat=feat_cat, verbose=100, plot=False
    )