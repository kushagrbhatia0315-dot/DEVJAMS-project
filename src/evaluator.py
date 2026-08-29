import os
import pandas as pd
import numpy as np
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
    plt.close(fig)
    return acc, fig, matrix
def plot_wildfires_vs_storms_per_year(wf_df: pd.DataFrame, storm_df: pd.DataFrame, save_path: str):
    wf_counts = wf_df.groupby('FIRE_YEAR').size().reset_index(name='Wildfires')
    wf_counts = wf_counts.rename(columns={'FIRE_YEAR': 'Year'})
    storm_counts = storm_df.groupby('YEAR').size().reset_index(name='Storms')
    storm_counts = storm_counts.rename(columns={'YEAR': 'Year'})
    merged = pd.merge(wf_counts, storm_counts, on='Year', how='inner').sort_values('Year')
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(merged['Year']))
    width = 0.35 
    ax.bar(x - width/2, merged['Wildfires'], width, label='Wildfires', color='#d95f02', edgecolor='black')
    ax.bar(x + width/2, merged['Storms'], width, label='Storms', color='#1f78b4', edgecolor='black')
    ax.set_title('Annual Occurrences: Wildfires vs. Storms', pad=20, fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12, labelpad=10)
    ax.set_ylabel('Total Events Count', fontsize=12, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(merged['Year'], rotation=45)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig) 
    
    return fig
