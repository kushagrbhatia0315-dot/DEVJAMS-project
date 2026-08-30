import sys
import pandas as pd
from src import config
from src.ingestor import extract_wildfires, extract_storms
from src.engineer import merge_and_engineer, prepare_for_ml
from src.modeler import train_and_predict, predict_fall_2026_scenario
from src.evaluator import evaluate_and_plot, plot_wildfires_vs_storms_per_year

def main():
    print("=" * 55)
    print(" STARTING WILDFIRE & STORM PREDICTION PIPELINE")
    print("=" * 55)
    try:
        wf_df = extract_wildfires(config.WILDFIRE_DB, config.ROW_LIMIT, config.WF_FEATURES, config.WF_TARGET)
        storm_df = extract_storms(config.STORMS_DIR, config.STORM_FEATURES)
        merged_df = merge_and_engineer(wf_df, storm_df, config.WF_TARGET)
        wf_df.to_csv("api_disaster_data.csv", index=False)
        
        X_train, X_test, y_train, y_test = prepare_for_ml(
            df=merged_df, target_col=config.WF_TARGET, test_size=config.TEST_SIZE
        )
        predictions = train_and_predict(
            X_train=X_train, y_train=y_train, X_test=X_test, save_path=config.MODEL_SAVE_PATH
        )
        forecast_df = predict_fall_2026_scenario(config.MODEL_SAVE_PATH, X_train.columns, merged_df)
        forecast_df.to_csv("api_2026_forecast.csv", index=False)
        print("--> [API EXTRACT] Saved api_2026_forecast.csv for the server...")
        
        results_df = pd.DataFrame({'y_test': y_test, 'predictions': predictions})
        results_df.to_csv("api_predictions.csv", index=False)
        
        metrics, _, _ = evaluate_and_plot(y_test, predictions, config.PLOT_SAVE_PATH)
        print(f"--> [MODEL METRICS] Accuracy: {metrics['accuracy'] * 100:.2f}% | "
              f"Precision: {metrics['precision'] * 100:.2f}% | "
              f"Recall: {metrics['recall'] * 100:.2f}% | "
              f"F1 Score: {metrics['f1'] * 100:.2f}%")
              
        plot_wildfires_vs_storms_per_year(wf_df, storm_df, config.BAR_GRAPH_SAVE_PATH)
        print("--> [PIPELINE SUCCESS] All tasks completed successfully.")
        
    except Exception as e:
        print(f"\n[!] PIPELINE FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
