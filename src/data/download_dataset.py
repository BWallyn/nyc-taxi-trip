"""Script to download data from NYC TLC"""
# =================
# ==== IMPORTS ====
# =================

import gc
import os
from tqdm import tqdm
from urllib.error import HTTPError

import pandas as pd


# ===================
# ==== FUNCTIONS ====
# ===================

def download_dataset_yymm(link: str, year_month: str) -> pd.DataFrame:
    """Download parquet dataset with the link and specific month

    Args:
        link: Link to the website containing the dataset
        year_month: Specific month of the data to download. Written as yyyy-mm
    Returns:
        df: DataFrame of the NYC taxis
    """
    path = f'{link}{year_month}.parquet'
    try:
        df = pd.read_parquet(path=path)
    except HTTPError as err:
        if err.code == 403:
            print(f'Dataset {year_month} not found')
            df = pd.DataFrame()
    return df


def download_all_data(link: str, yyyy_start: int, yyyy_end: int, mm_start: int, mm_end: int) -> pd.DataFrame:
    """
    """
    # Create list of months
    if yyyy_start == yyyy_end:
        list_months = [
            str(yyyy_start) + '-0' + str(month) if len(str(month)) == 1 else str(yyyy_start) + '-' + str(month) for month in range(mm_start, mm_end+1)
        ]
    else:
        n_year_diff = yyyy_end - yyyy_start
        for i in range(n_year_diff):
            list_months = [
                str(yyyy_start+i) + '-0' + str(month) if len(str(month)) == 1 else str(yyyy_start+i) + '-' + str(month) for month in range(mm_start, 13)
            ]
        list_months += [
            str(yyyy_start) + '-0' + str(month) if len(str(month)) == 1 else str(yyyy_start) + '-' + str(month) for month in range(mm_start, mm_end+1)
        ]
    # download dataset of each month and merge
    df = pd.DataFrame()
    for year_month in tqdm(list_months):
        df_tmp = download_dataset_yymm(link=link, year_month=year_month)
        df = pd.concat([df, df_tmp])
        del df_tmp
        gc.collect()
    return df



# =============
# ==== RUN ====
# =============

if __name__ == "__main__":
    # Options
    path = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_"
    yyyy_start = 2023
    yyyy_end = 2023
    mm_start = 1
    mm_end = 5
    # Run
    df = download_all_data(link=path, yyyy_start=yyyy_start, yyyy_end=yyyy_end, mm_start=mm_start, mm_end=mm_end)
    # Save results
    df.to_parquet(os.path.join('data', 'external', f"yellow_trip_{yyyy_start}-{mm_start}_{yyyy_end}-{mm_end}.parquet"))
