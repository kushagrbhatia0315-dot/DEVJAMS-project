import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset Paths
WILDFIRE_DB = os.path.join(BASE_DIR, 'FPA_FOD_20170508.sqlite')
STORMS_DIR = os.path.join(BASE_DIR, 'data', 'Storms 1996-2019') 

# Output Paths
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'rf_model_v2.pkl')
PLOT_SAVE_PATH = os.path.join(BASE_DIR, 'outputs', 'confusion_matrix.png')

# Model Settings
ROW_LIMIT = 50000 
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Columns
WF_FEATURES = ['FIRE_YEAR', 'DISCOVERY_DOY', 'STATE', 'FIRE_SIZE']
WF_TARGET = 'STAT_CAUSE_DESCR'
STORM_FEATURES = ['YEAR', 'MONTH_NAME', 'STATE', 'EVENT_TYPE']
