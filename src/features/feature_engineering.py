"""Functions for feature engineering"""
# =================
# ==== IMPORTS ====
# =================

import os

import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# Options
set_config(transform_output="pandas")


# ===================
# ==== FUNCTIONS ====
# ===================

def merge_airport_fee(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the two features airport_fee and drop the one "Airport_fee"

    Args:
        df: DataFrame with the two features airport_fee and Airport_fee
    Returns:
        df: Dataframe with the features airport_fee merged
    """
    df.loc[df["Airport_fee"].isna(), "Airport_fee"] = df.loc[df["Airport_fee"].isna(), "airport_fee"]
    df.drop(columns="airport_fee", inplace=True)
    return df


def pipeline_feature_engineering(feat_num: list[str], feat_cat: list[str]) -> pd.DataFrame:
    """
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, feat_num),
            ("categorical", categorical_transformer, feat_cat)
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor)
        ]
    )
    return pipeline


def feature_engineering(
    df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame,
    feat_num: list[str], feat_cat: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    """
    # Merge Airport fee features
    df_train = merge_airport_fee(df_train)
    df_valid = merge_airport_fee(df_valid)
    df_test = merge_airport_fee(df_test)
    # Pipeline
    pipeline = pipeline_feature_engineering(feat_num=feat_num, feat_cat=feat_cat)
    pipeline.fit(df_train)
    df_train = pipeline.transform(df_train)
    df_valid = pipeline.transform(df_valid)
    df_test = pipeline.transform(df_test)
    return df_train, df_valid, df_test


# =============
# ==== RUN ====
# =============

if __name__ == "__main__":
    # Options
    path = os.path.join('data', 'interim')
    name_file = "yellow_trip_2023-1_2023-5_duration"
    feat_cat = [
        "VendorID", "RatecodeID", "store_and_fwd_flag", "PULocationID", "DOLocationID", "payment_type", "improvement_surcharge"
    ]
    feat_num = [
        "passenger_count",
        # "trip_distance",
        "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount", "total_amount", "congestion_surcharge", "Airport_fee",
    ]
    # Load data
    df_train = pd.read_parquet(os.path.join(path, f"{name_file}_train.parquet"))
    df_valid = pd.read_parquet(os.path.join(path, f"{name_file}_valid.parquet"))
    df_test = pd.read_parquet(os.path.join(path, f"{name_file}_test.parquet"))
    # Run
    df_train, df_valid, df_test = feature_engineering(df_train, df_valid, df_test, feat_num, feat_cat)
    # Save data
    df_train.to_parquet(os.path.join(path, f"{name_file}_feat_train.parquet"))
    df_valid.to_parquet(os.path.join(path, f"{name_file}_feat_valid.parquet"))
    df_test.to_parquet(os.path.join(path, f"{name_file}_feat_test.parquet"))
