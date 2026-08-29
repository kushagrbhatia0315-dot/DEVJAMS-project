import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_and_plot(y_test, predictions, save_path: str):
    """Calculates accuracy and generates the visual heatmap."""
    print("--> [5/5] Evaluating Multi-Dataset Model...\n")
    acc = accuracy_score(y_test, predictions)
    print(f"*** FINAL ACCURACY: {round(acc * 100, 2)}% ***\n")
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Oranges', cbar=False)
    plt.title('Wildfire & Storm Pipeline Prediction Matrix', pad=20, fontsize=14)
    plt.xlabel('Predicted Fire Cause', fontsize=12)
    plt.ylabel('Actual Fire Cause', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
