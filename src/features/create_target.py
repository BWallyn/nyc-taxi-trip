"""Create the target duration on the dataset"""
# =================
# ==== IMPORTS ====
# =================

import os
import pandas as pd


# ===================
# ==== FUNCTIONS ====
# ===================

def compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Add duration time (in s) to the dataset using pickup and dropoff times.

    Args:
        df: DataFrame with pick-up and drop-off times
    Returns:
        df: DataFrame with the duration feature
    """
    # Convert datetime
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    # Compute duration
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()
    return df


# =============
# ==== RUN ====
# =============

if __name__ == "__main__":
    # Options
    yyyy_start = 2023
    yyyy_end = 2023
    mm_start = 1
    mm_end = 5
    # Run
    df = pd.read_parquet(os.path.join("data", "external", f'yellow_trip_{yyyy_start}-{mm_start}_{yyyy_end}-{mm_end}.parquet'))
    df = compute_duration(df)
    df.to_parquet(path=os.path.join("data", "interim", f'yellow_trip_{yyyy_start}-{mm_start}_{yyyy_end}-{mm_end}_duration.parquet'))