from confluent_kafka import Consumer, KafkaException, KafkaError
import pandas as pd
import json
from collections import deque
import numpy as np
import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import threading

# Kafka Consumer configuration
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'nyc_violation_consumer',
    'auto.offset.reset': 'earliest'
}

# Initialize the Kafka consumer
consumer = Consumer(conf)
consumer.subscribe(['nyc_violations'])

# Data structures
batch_size = 1000
data_window = deque(maxlen=batch_size)
stats_df = pd.DataFrame(columns=['YearMonth', 'Violation County', 'mean', 'std', 'median', 'max', 'min'])

def process_batch():
    global stats_df
    # Create a DataFrame from the collected batch
    df = pd.DataFrame(data_window)
    
    # Convert 'Issue Date' to datetime and extract year and month
    df['Issue Date'] = pd.to_datetime(df['Issue Date'], errors='coerce')
    df['YearMonth'] = df['Issue Date'].dt.to_period('M').astype(str)
    
    # Filter out rows where 'Vehicle Year' is empty, None, or 0
    df = df[df['Vehicle Year'].notna() & (df['Vehicle Year'] != '') & (df['Vehicle Year'] != '0') & (df['Vehicle Year'] != 0)]
    
    # Convert 'Vehicle Year' to numeric (in case it was read as string)
    df['Vehicle Year'] = pd.to_numeric(df['Vehicle Year'], errors='coerce')
    
    # Filter data where Date year > 2012
    df = df[df['Issue Date'].dt.year > 2012]
    
    # Aggregate statistics by year, month, and borough
    grouped = df.groupby(['YearMonth', 'Violation County']).agg({
        'Vehicle Year': ['mean', 'std', 'median', 'max', 'min']
    })
    
    # Flatten MultiIndex columns
    grouped.columns = grouped.columns.droplevel()
    grouped = grouped.reset_index()

    # Append aggregated results to stats_df
    stats_df = pd.concat([stats_df, grouped], ignore_index=True)
    
    # Drop duplicates to ensure unique YearMonth and Violation County
    stats_df = stats_df.groupby(['YearMonth', 'Violation County']).agg({
        'mean': 'mean',
        'std': 'mean',
        'median': 'mean',
        'max': 'mean',
        'min': 'mean'
    }).reset_index()

def process_message(msg):
    record = json.loads(msg.value())
    data_window.append(record)
    
    # Process the batch if it reaches the batch size
    if len(data_window) >= batch_size:
        process_batch()
        data_window.clear()

def consume_messages():
    while True:
        try:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    # raise KafkaException(msg.error())
                    print(msg.error())
            else:
                process_message(msg)
        except KeyboardInterrupt:
            break

# Start the Kafka consumer in a separate thread
consumer_thread = threading.Thread(target=consume_messages, daemon=True)
consumer_thread.start()

# Create Dash application
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])

index_page = html.Div([
    html.Div([
        dcc.Link('Statistics by Borough', href='/boroughs'),
        html.Br(),
        dcc.Link('Overall Statistics', href='/'),
    ]),
    html.Div([
        dcc.Graph(id='mean-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='std-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='median-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='max-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='min-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # Update every second
        n_intervals=0
    )
])

boroughs_page = html.Div([
    html.Div([
        dcc.Link('Overall Statistics', href='/'),
        html.Br(),
        dcc.Link('Statistics by Borough', href='/boroughs'),
    ]),
    html.Div([
        dcc.Graph(id='borough-mean-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='borough-std-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='borough-median-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='borough-max-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='borough-min-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    dcc.Interval(
        id='borough-interval-component',
        interval=1*1000,  # Update every second
        n_intervals=0
    )
])

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/boroughs':
        return boroughs_page
    else:
        return index_page

@app.callback(
    [Output('mean-graph', 'figure'),
     Output('std-graph', 'figure'),
     Output('median-graph', 'figure'),
     Output('max-graph', 'figure'),
     Output('min-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n):
    if stats_df.empty:
        empty_fig = go.Figure()
        return [empty_fig] * 5

    # Group by YearMonth and compute statistics
    grouped = stats_df.groupby('YearMonth').agg({
        'mean': 'mean',
        'std': 'mean',
        'median': 'mean',
        'max': 'mean',
        'min': 'mean'
    }).reset_index()

    # Sort by YearMonth
    grouped = grouped.sort_values(by='YearMonth')

    dates = grouped['YearMonth']
    means = grouped['mean']
    stds = grouped['std']
    medians = grouped['median']
    maxs = grouped['max']
    mins = grouped['min']

    mean_fig = go.Figure()
    mean_fig.add_trace(go.Scatter(x=dates, y=means, mode='lines', name='Mean'))
    mean_fig.update_layout(
        title='Mean Vehicle Year',
        xaxis_title='Year-Month',
        yaxis_title='Mean Vehicle Year',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    std_fig = go.Figure()
    std_fig.add_trace(go.Scatter(x=dates, y=stds, mode='lines', name='Std Dev'))
    std_fig.update_layout(
        title='Standard Deviation of Vehicle Year',
        xaxis_title='Year-Month',
        yaxis_title='Standard Deviation',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    median_fig = go.Figure()
    median_fig.add_trace(go.Scatter(x=dates, y=medians, mode='lines', name='Median'))
    median_fig.update_layout(
        title='Median Vehicle Year',
        xaxis_title='Year-Month',
        yaxis_title='Median Vehicle Year',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    max_fig = go.Figure()
    max_fig.add_trace(go.Scatter(x=dates, y=maxs, mode='lines', name='Max'))
    max_fig.update_layout(
        title='Maximum Vehicle Year',
        xaxis_title='Year-Month',
        yaxis_title='Maximum Vehicle Year',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    min_fig = go.Figure()
    min_fig.add_trace(go.Scatter(x=dates, y=mins, mode='lines', name='Min'))
    min_fig.update_layout(
        title='Minimum Vehicle Year',
        xaxis_title='Year-Month',
        yaxis_title='Minimum Vehicle Year',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    return [mean_fig, std_fig, median_fig, max_fig, min_fig]

@app.callback(
    [Output('borough-mean-graph', 'figure'),
     Output('borough-std-graph', 'figure'),
     Output('borough-median-graph', 'figure'),
     Output('borough-max-graph', 'figure'),
     Output('borough-min-graph', 'figure')],
    [Input('borough-interval-component', 'n_intervals')]
)
def update_borough_graphs(n):
    if stats_df.empty:
        empty_fig = go.Figure()
        return [empty_fig] * 5

    # Sort and plot borough-wise statistics
    grouped = stats_df.sort_values(by=['YearMonth', 'Violation County'])

    dates = grouped['YearMonth'].unique()
    boroughs = grouped['Violation County'].unique()

    # Initialize figures
    mean_fig = go.Figure()
    std_fig = go.Figure()
    median_fig = go.Figure()
    max_fig = go.Figure()
    min_fig = go.Figure()

    for borough in boroughs:
        borough_data = grouped[grouped['Violation County'] == borough]
        
        mean_fig.add_trace(go.Scatter(x=borough_data['YearMonth'], y=borough_data['mean'], mode='lines', name=f'Mean - {borough}'))
        std_fig.add_trace(go.Scatter(x=borough_data['YearMonth'], y=borough_data['std'], mode='lines', name=f'Std Dev - {borough}'))
        median_fig.add_trace(go.Scatter(x=borough_data['YearMonth'], y=borough_data['median'], mode='lines', name=f'Median - {borough}'))
        max_fig.add_trace(go.Scatter(x=borough_data['YearMonth'], y=borough_data['max'], mode='lines', name=f'Max - {borough}'))
        min_fig.add_trace(go.Scatter(x=borough_data['YearMonth'], y=borough_data['min'], mode='lines', name=f'Min - {borough}'))

    mean_fig.update_layout(
        title='Mean Vehicle Year by Borough',
        xaxis_title='Year-Month',
        yaxis_title='Mean Vehicle Year',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    std_fig.update_layout(
        title='Standard Deviation of Vehicle Year by Borough',
        xaxis_title='Year-Month',
        yaxis_title='Standard Deviation',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    median_fig.update_layout(
        title='Median Vehicle Year by Borough',
        xaxis_title='Year-Month',
        yaxis_title='Median Vehicle Year',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    max_fig.update_layout(
        title='Maximum Vehicle Year by Borough',
        xaxis_title='Year-Month',
        yaxis_title='Maximum Vehicle Year',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    min_fig.update_layout(
        title='Minimum Vehicle Year by Borough',
        xaxis_title='Year-Month',
        yaxis_title='Minimum Vehicle Year',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    return [mean_fig, std_fig, median_fig, max_fig, min_fig]

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
