import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

STATE_MAP = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
    "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA",
    "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY", "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN",
    "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY"
}

def merge_and_engineer(wf_df: pd.DataFrame, storm_df: pd.DataFrame, target_col: str):
    print("--> [3/5] Merging Datasets & Engineering Binary Features...")
    
    invalid_causes = ['Missing/Undefined', 'Miscellaneous']
    wf_df = wf_df[~wf_df[target_col].isin(invalid_causes)].dropna().copy()
    
    # THE BIG FIX: Binary Classification (Natural vs Human)
    cause_map = {
        'Lightning': 'Natural (Weather)',
        'Arson': 'Human-Caused',
        'Debris Burning': 'Human-Caused',
        'Equipment Use': 'Human-Caused',
        'Powerline': 'Human-Caused',
        'Structure': 'Human-Caused',
        'Railroad': 'Human-Caused',
        'Campfire': 'Human-Caused',
        'Smoking': 'Human-Caused',
        'Children': 'Human-Caused',
        'Fireworks': 'Human-Caused'
    }
    wf_df[target_col] = wf_df[target_col].map(cause_map)
    wf_df = wf_df.dropna(subset=[target_col])
    
    wf_df['MONTH'] = pd.to_datetime(
        wf_df['FIRE_YEAR'] * 1000 + wf_df['DISCOVERY_DOY'], 
        format='%Y%j'
    ).dt.month
    
    wf_df['DOY_SIN'] = np.sin(2 * np.pi * wf_df['DISCOVERY_DOY'] / 365.25)
    wf_df['DOY_COS'] = np.cos(2 * np.pi * wf_df['DISCOVERY_DOY'] / 365.25)
    wf_df['MONTH_SIN'] = np.sin(2 * np.pi * wf_df['MONTH'] / 12.0)
    wf_df['MONTH_COS'] = np.cos(2 * np.pi * wf_df['MONTH'] / 12.0)

    storm_df['STATE'] = storm_df['STATE'].str.upper().map(STATE_MAP)
    month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
                 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    storm_df['MONTH'] = storm_df['MONTH_NAME'].str.title().map(month_map)
    storm_counts = storm_df.groupby(['STATE', 'YEAR', 'MONTH']).size().reset_index(name='MONTHLY_STORMS')
    
    merged_df = pd.merge(
        wf_df, storm_counts, 
        left_on=['STATE', 'FIRE_YEAR', 'MONTH'], 
        right_on=['STATE', 'YEAR', 'MONTH'], 
        how='left'
    )
    merged_df['MONTHLY_STORMS'] = merged_df['MONTHLY_STORMS'].fillna(0)

    merged_df['LAT_ROUNDED'] = merged_df['LATITUDE'].round(2)
    merged_df['LON_ROUNDED'] = merged_df['LONGITUDE'].round(2)
    merged_df['SPATIAL_INTERACT'] = merged_df['LATITUDE'] * merged_df['LONGITUDE']
    merged_df['LOG_FIRE_SIZE'] = np.log1p(merged_df['FIRE_SIZE'])
    merged_df['STORM_FIRE_INTERACTION'] = merged_df['LOG_FIRE_SIZE'] * np.log1p(merged_df['MONTHLY_STORMS'])
    
    cols_to_drop = ['DISCOVERY_DOY', 'YEAR', 'FIRE_YEAR', 'LATITUDE', 'LONGITUDE', 'FIRE_SIZE', 'MONTH']
    merged_df = merged_df.drop(columns=cols_to_drop, errors='ignore')
    
    return merged_df

def prepare_for_ml(df: pd.DataFrame, target_col: str, test_size: float):
    y = df[target_col]
    X_raw = df.drop(columns=[target_col])
    X = pd.get_dummies(X_raw)
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
