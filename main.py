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

    merged_df = merge_and_engineer(wf_df, storm_df, config.WF_TARGET)
    
    X_train, X_test, y_train, y_test = prepare_for_ml(df=merged_df, target_col=config.WF_TARGET, test_size=config.TEST_SIZE, random_seed=config.RANDOM_SEED)
    
    evaluate_and_plot(y_test, predictions, config.PLOT_SAVE_PATH)
  except Exception as e:
        print(f"\n[!] PIPELINE FAILED: {e}")
        sys.exit(1)
if _name_ == "_main_":
   main()
