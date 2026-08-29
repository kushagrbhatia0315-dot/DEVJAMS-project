import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_and_plot(y_test, predictions, save_path: str):
    acc = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    labels = sorted(y_test.unique())
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt='d', 
        cmap='Oranges', 
        cbar=False, 
        ax=ax,
        xticklabels=labels,
        yticklabels=labels
    )
    
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
    ax.bar(x + width/2, merged['Storms'], width, label='Storms', color='#17a3f4', edgecolor='black')
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

def plot_us_disaster_map(wf_df: pd.DataFrame, storm_df: pd.DataFrame, selected_year: int):
 
    wf_filtered = wf_df[wf_df['FIRE_YEAR'] == selected_year].copy()
    storm_filtered = storm_df[storm_df['YEAR'] == selected_year].copy()

    wf_map = pd.DataFrame({
        'lat': wf_filtered['LATITUDE'],
        'lon': wf_filtered['LONGITUDE'],
        'Type': 'Wildfire',
        'Details': wf_filtered['STAT_CAUSE_DESCR']
    }).dropna()

    storm_map = pd.DataFrame({
        'lat': storm_filtered['BEGIN_LAT'],
        'lon': storm_filtered['BEGIN_LON'],
        'Type': 'Storm',
        'Details': storm_filtered['EVENT_TYPE']
    }).dropna()

    combined_map_df = pd.concat([wf_map, storm_map], ignore_index=True)

    fig = px.scatter_geo(
        combined_map_df,
        lat='lat',
        lon='lon',
        color='Type',
        hover_data=['Details'],
        color_discrete_map={'Wildfire': 'red', 'Storm': 'blue'},
        scope='usa',
        title=f'US Disaster Map ({selected_year}): Red = Wildfire | Blue = Storm'
    )
    
    fig.update_layout(
        geo=dict(bgcolor='rgba(15,23,42,1)', landcolor='rgba(30,41,59,1)', showlakes=True, lakecolor='rgba(15,23,42,1)'),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig
