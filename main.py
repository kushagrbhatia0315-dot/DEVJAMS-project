import sys
from src import config
from src.ingestor import extract_wildfires, extract_storms
from src.engineer import merge_and_engineer, prepare_for_ml
from src.modeler import train_and_predict
from src.evaluator import evaluate_and_plot

def main():
  print("=" * 55)
  print("STARTING WILDFIRE & STORM PREDICTION PIPELINE")
  print("=" * 55)
  try:
    wf_df = extract_wildfires(config.WILDFIRE_DB, config.ROW_LIMIT, config.WF_FEATURES, config.WF_TARGET)
    storm_df = extract_storms(config.STORMS_DIR, config.STORM_FEATURES)
    
