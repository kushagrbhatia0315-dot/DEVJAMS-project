import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix

def create_gauge(value, max_val, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#102A43'}},
        number={'font': {'color': '#102A43'}},
        gauge={
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_val*0.33], 'color': 'rgba(0,255,0,0.1)'},
                {'range': [max_val*0.33, max_val*0.66], 'color': 'rgba(255,255,0,0.1)'},
                {'range': [max_val*0.66, max_val], 'color': 'rgba(255,0,0,0.1)'}
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=15, r=15, t=40, b=15)
    )
    return fig

def evaluate_and_plot(y_test, predictions, save_path: str):
    acc = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    labels = sorted(y_test.unique())
    
    fig = px.imshow(
        matrix, 
        text_auto=True, 
        color_continuous_scale='Teal',
        x=labels, 
        y=labels,
        labels=dict(x="Predicted Cause", y="Actual Cause", color="Count")
    )
    
    fig.update_layout(
        title=dict(text="🎯 AI Prediction Confusion Matrix", font=dict(size=26, color='#102A43')),
        template="plotly_white",
        plot_bgcolor='rgba(255,255,255,0.8)',
        paper_bgcolor='rgba(255,255,255,0.8)',
        margin=dict(l=40, r=40, t=80, b=40),
        font=dict(size=14, color='black')
    )
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        fig.write_image(save_path, scale=2)
    except Exception:
        pass
        
    return acc, fig, matrix

def plot_wildfires_vs_storms_per_year(wf_df: pd.DataFrame, storm_df: pd.DataFrame, save_path: str):
    wf_counts = wf_df.groupby('FIRE_YEAR').size().reset_index(name='Wildfires')
    wf_counts = wf_counts.rename(columns={'FIRE_YEAR': 'Year'})
    
    storm_counts = storm_df.groupby('YEAR').size().reset_index(name='Storms')
    storm_counts = storm_counts.rename(columns={'YEAR': 'Year'})
    
    merged = pd.merge(wf_counts, storm_counts, on='Year', how='inner').sort_values('Year')
    
    fig = px.bar(
        merged, 
        x='Year', 
        y=['Wildfires', 'Storms'],
        barmode='group',
        color_discrete_map={'Wildfires': '#ff4b4b', 'Storms': '#17a3f4'},
        title="Annual Occurrences: Wildfires vs. Storms"
    )
    
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis_title="Total Events Count",
        xaxis_title="Year",
        hovermode="x unified",
        legend_title_text="Disaster Type",
        font=dict(color='#102A43')
    )
    return fig

def plot_us_disaster_map(wf_df: pd.DataFrame, storm_df: pd.DataFrame, selected_year: int):
    wf_filtered = wf_df[wf_df['FIRE_YEAR'] == selected_year]
    storm_filtered = storm_df[storm_df['YEAR'] == selected_year]

    fig = go.Figure()

    fig.add_trace(go.Scattermap(
        lon=storm_filtered['BEGIN_LON'], lat=storm_filtered['BEGIN_LAT'],
        text="Storm: " + storm_filtered['EVENT_TYPE'],
        mode='markers',
        marker=dict(size=7, color='#17a3f4', opacity=0.8),
        name='Storms',
        hoverinfo='text',
        cluster=dict(
            enabled=True,
            size=[15, 25, 40],
            step=[10, 50, 100]
        )
    ))

    fig.add_trace(go.Scattermap(
        lon=wf_filtered['LONGITUDE'], lat=wf_filtered['LATITUDE'],
        text="Fire: " + wf_filtered['STAT_CAUSE_DESCR'],
        mode='markers',
        marker=dict(size=7, color='#ff4b4b', opacity=0.8),
        name='Wildfires',
        hoverinfo='text',
        cluster=dict(
            enabled=True,
            size=[15, 25, 40],
            step=[10, 50, 100]
        )
    ))

    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lon=-98.5795, lat=39.8283),
            zoom=3.5
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=f"Dynamic Clustered Distribution Map ({selected_year})", font=dict(color='#102A43')),
        legend=dict(
            bgcolor='rgba(255,255,255,0.7)',
            yanchor="top", y=0.99, xanchor="left", x=0.01
        )
    )
    return fig
