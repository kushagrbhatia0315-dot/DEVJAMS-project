import sqlite3
import pandas as pd
import os
import glob

def extract_wildfires(db_path: str, limit: int, features: list, target: str) -> pd.DataFrame:
    print("--> [1/5] Extracting Wildfire Database...")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Missing Database: {db_path}")
    columns = ", ".join(features + [target])
    query = f"SELECT {columns} FROM Fires WHERE FIRE_YEAR >= 1996 ORDER BY RANDOM() LIMIT {limit};"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    print(f"    Successfully built dataset with {len(df)} wildfire records.")
    return df
def extract_storms(storms_dir: str, features: list) -> pd.DataFrame:
    print("--> [2/5] Extracting and Combining Storms CSVs...")
    if not os.path.exists(storms_dir):
        raise FileNotFoundError(f"Missing Folder: {storms_dir}")
    all_files = glob.glob(os.path.join(storms_dir, "*.csv"))
    if len(all_files) == 0:
        raise ValueError(f"No CSV files found in the {storms_dir} folder!")
    print(f"    Found {len(all_files)} storm files. Stitching them together...")
    df_list = []
    for file in all_files:
        df = pd.read_csv(file, usecols=features, low_memory=False)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)
