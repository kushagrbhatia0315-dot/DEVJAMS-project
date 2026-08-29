import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_and_plot(y_test, predictions, save_path: str):
    
    acc = accuracy_score(y_test, predictions)
    
    matrix = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax)
    
    ax.set_title('Wildfire & Storm Pipeline Prediction Matrix', pad=20, fontsize=14)
    ax.set_xlabel('Predicted Fire Cause', fontsize=12)
    ax.set_ylabel('Actual Fire Cause', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)

    return acc, fig, matrix
