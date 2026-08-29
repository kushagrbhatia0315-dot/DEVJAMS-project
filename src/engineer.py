import pandas as pd
from sklearn.model_selection import train_test_split

# Dictionary to translate Storm states to Wildfire abbreviations
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
    """Aligns calendars, counts storm events, and merges the datasets."""
    print("--> [3/5] Merging Datasets & Engineering Features...")
    
    # Clean Wildfires
    wf_df = wf_df[wf_df[target_col] != 'Missing/Undefined'].dropna()
    
    # Extract Month from Julian Day in Wildfires
    wf_df['MONTH'] = (wf_df['DISCOVERY_DOY'] / 30.5).astype(int) + 1
    wf_df['MONTH'] = wf_df['MONTH'].clip(1, 12)
    wf_df = wf_df.drop(columns=['DISCOVERY_DOY'])

    # Standardize Storms Dataset
    storm_df['STATE'] = storm_df['STATE'].str.upper().map(STATE_MAP)
    month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
                 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    storm_df['MONTH'] = storm_df['MONTH_NAME'].str.title().map(month_map)
    
    # Count storms per State, Year, and Month
    storm_counts = storm_df.groupby(['STATE', 'YEAR', 'MONTH']).size().reset_index(name='MONTHLY_STORMS')
    
    # Merge datasets: Matches the wildfire to the total storms in that state during that month
    merged_df = pd.merge(
        wf_df, storm_counts, 
        left_on=['STATE', 'FIRE_YEAR', 'MONTH'], 
        right_on=['STATE', 'YEAR', 'MONTH'], 
        how='left'
    )
    merged_df['MONTHLY_STORMS'] = merged_df['MONTHLY_STORMS'].fillna(0)
    merged_df = merged_df.drop(columns=['YEAR']) # Drop duplicate year column
    
    return merged_df

def prepare_for_ml(df: pd.DataFrame, target_col: str, test_size: float, random_seed: int):
    """One-Hot Encodes text and splits into Train/Test chunks."""
    y = df[target_col]
    X_raw = df.drop(columns=[target_col])
    
    # Convert 'STATE' into numbers
    X = pd.get_dummies(X_raw)
    
    return train_test_split(X, y, test_size=test_size, random_state=random_seed)
