import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
WILDFIRE_DB = os.path.join(BASE_DIR, 'FPA_FOD_20170508.sqlite')
STORMS_DIR = os.path.join(BASE_DIR, 'data', 'Storms 1996-2019') 
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'rf_model_v2.pkl')
PLOT_SAVE_PATH = os.path.join(BASE_DIR, 'outputs', 'confusion_matrix.png')
BAR_GRAPH_SAVE_PATH = os.path.join(BASE_DIR, 'outputs', 'annual_comparison.png')
ROW_LIMIT = 640000 
TEST_SIZE = 0.2
WF_FEATURES = ['FIRE_YEAR', 'DISCOVERY_DOY', 'STATE', 'FIRE_SIZE', 'LATITUDE', 'LONGITUDE']
WF_TARGET = 'STAT_CAUSE_DESCR'
STORM_FEATURES = ['YEAR', 'MONTH_NAME', 'STATE', 'EVENT_TYPE', 'BEGIN_LAT', 'BEGIN_LON']
