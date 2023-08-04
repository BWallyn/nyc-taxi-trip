"""Functions for feature engineering"""
# =================
# ==== IMPORTS ====
# =================

import pandas as pd


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
    df.drop(columns="Airport_fee", inplace=True)
    return df