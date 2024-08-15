from confluent_kafka import Consumer, KafkaException, KafkaError
import pandas as pd
import json
from collections import deque
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
top_streets_df = pd.DataFrame(columns=['YearMonth', 'Street', 'mean', 'std', 'median', 'max', 'min'])

borough_map = {
    "BX": "BX",
    "BRONX": "BX",
    "BK": "BK",
    "BROOKLYN": "BK",
    "K": "BK",
    "KINGS": "BK",
    "MN": "MN",
    "MANHATTAN": "MN",
    "Q": "QS",
    "QS": "QS",
    "QN": "QS",
    "QNS": "QS",
    "QUEENS": "QS",
    "SI": "SI",
    "ST": "SI",
    "STATEN ISLAND": "SI",
    "NY": "TOTAL",
    "": "TOTAL",
    "R": "TOTAL",
}

# Additional setup for Top 10 Streets
street_codes = {
    "44750-10575-10575": "RICHMOND AVE",
    "34370-10810-10910": "WEST 30 ST",
    "34430-10410-13610": "WEST 33 ST",
    "34330-10510-10610": "WEST 28 ST",
    "34310-10510-10610": "WEST 27 ST",
    "34110-10610-10810": "WEST 17 ST",
    "77150-47520-57720": "WHITE PLAINS RD",
    "34290-10510-10610": "WEST 26 ST",
    "34430-10510-10610": "WEST 33 ST",
    "34310-13610-10510": "WEST 27 ST",
}

def process_batch():
    global stats_df, top_streets_df
    # Create a DataFrame from the collected batch
    df = pd.DataFrame(data_window)

    # Remap boroughs
    df['Violation County'] = df['Violation County'].apply(lambda x: borough_map.get(x, x))

    # Extract and map Street Code
    df['Street Code Combo'] = df[['Street Code1', 'Street Code2', 'Street Code3']].astype(str).agg('-'.join, axis=1)
    df['Street'] = df['Street Code Combo'].map(street_codes)

    # Process for top 10 streets
    df_top_streets = df[df['Street Code Combo'].isin(street_codes.keys())]

    # The rest of the batch processing logic remains the same for both boroughs and top streets...
    
    # Convert 'Issue Date' to datetime and extract year and month
    df['Issue Date'] = pd.to_datetime(df['Issue Date'], errors='coerce')
    df['YearMonth'] = df['Issue Date'].dt.to_period('M').astype(str)

    df_top_streets['Issue Date'] = pd.to_datetime(df_top_streets['Issue Date'], errors='coerce')
    df_top_streets['YearMonth'] = df_top_streets['Issue Date'].dt.to_period('M').astype(str)

    # Filter out rows where 'Vehicle Year' is empty, None, or 0
    df = df[df['Vehicle Year'].notna() & (df['Vehicle Year'] != '') & (df['Vehicle Year'] != '0') & (df['Vehicle Year'] != 0)]
    df_top_streets = df_top_streets[df_top_streets['Vehicle Year'].notna() & (df_top_streets['Vehicle Year'] != '') & (df_top_streets['Vehicle Year'] != '0') & (df_top_streets['Vehicle Year'] != 0)]
    
    # Convert 'Vehicle Year' to numeric (in case it was read as string)
    df['Vehicle Year'] = pd.to_numeric(df['Vehicle Year'], errors='coerce')
    df_top_streets['Vehicle Year'] = pd.to_numeric(df_top_streets['Vehicle Year'], errors='coerce')
    
    # Filter data where Date year > 2012
    df = df[df['Issue Date'].dt.year > 2012]
    df_top_streets = df_top_streets[df_top_streets['Issue Date'].dt.year > 2012]

    # Aggregate statistics by year, month, and borough
    grouped = df.groupby(['YearMonth', 'Violation County']).agg({
        'Vehicle Year': ['mean', 'std', 'median', 'max', 'min']
    })

    grouped_top_streets = df_top_streets.groupby(['YearMonth', 'Street']).agg({
        'Vehicle Year': ['mean', 'std', 'median', 'max', 'min']
    })

    # Flatten MultiIndex columns
    grouped.columns = grouped.columns.droplevel()
    grouped = grouped.reset_index()

    grouped_top_streets.columns = grouped_top_streets.columns.droplevel()
    grouped_top_streets = grouped_top_streets.reset_index()

    # Append aggregated results to stats_df
    stats_df = pd.concat([stats_df, grouped], ignore_index=True)
    top_streets_df = pd.concat([top_streets_df, grouped_top_streets], ignore_index=True)
    
    # Drop duplicates to ensure unique YearMonth and Violation County
    stats_df = stats_df.groupby(['YearMonth', 'Violation County']).agg({
        'mean': 'mean',
        'std': 'mean',
        'median': 'mean',
        'max': 'mean',
        'min': 'mean'
    }).reset_index()

    top_streets_df = top_streets_df.groupby(['YearMonth', 'Street']).agg({
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
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])

index_page = html.Div([
    html.Div([
        dcc.Link('Overall Statistics', href='/'),
        html.Br(),
        dcc.Link('Statistics by Borough', href='/boroughs'),
        html.Br(),
        dcc.Link('Top 10 Streets Statistics', href='/top-10-streets'),
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
        html.Br(),
        dcc.Link('Top 10 Streets Statistics', href='/top-10-streets'),
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

# Create a layout for the new Top 10 Streets page
top_streets_page = html.Div([
    html.Div([
        dcc.Link('Overall Statistics', href='/'),
        html.Br(),
        dcc.Link('Statistics by Borough', href='/boroughs'),
        html.Br(),
        dcc.Link('Top 10 Streets Statistics', href='/top-10-streets'),
    ]),
    html.Div([
        dcc.Graph(id='street-mean-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='street-std-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='street-median-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='street-max-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='street-min-graph'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    dcc.Interval(
        id='street-interval-component',
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
    elif pathname == '/top-10-streets':
        return top_streets_page
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

@app.callback(
    [Output('street-mean-graph', 'figure'),
     Output('street-std-graph', 'figure'),
     Output('street-median-graph', 'figure'),
     Output('street-max-graph', 'figure'),
     Output('street-min-graph', 'figure')],
    [Input('street-interval-component', 'n_intervals')]
)
def update_street_graphs(n):
    if top_streets_df.empty:
        empty_fig = go.Figure()
        return [empty_fig] * 5

    # Sort and plot street-wise statistics
    grouped = top_streets_df.sort_values(by=['YearMonth', 'Street'])

    streets = grouped['Street'].unique()

    # Initialize figures
    mean_fig = go.Figure()
    std_fig = go.Figure()
    median_fig = go.Figure()
    max_fig = go.Figure()
    min_fig = go.Figure()

    for street in streets:
        street_data = grouped[grouped['Street'] == street]
        
        mean_fig.add_trace(go.Scatter(x=street_data['YearMonth'], y=street_data['mean'], mode='lines', name=f'Mean - {street}'))
        std_fig.add_trace(go.Scatter(x=street_data['YearMonth'], y=street_data['std'], mode='lines', name=f'Std Dev - {street}'))
        median_fig.add_trace(go.Scatter(x=street_data['YearMonth'], y=street_data['median'], mode='lines', name=f'Median - {street}'))
        max_fig.add_trace(go.Scatter(x=street_data['YearMonth'], y=street_data['max'], mode='lines', name=f'Max - {street}'))
        min_fig.add_trace(go.Scatter(x=street_data['YearMonth'], y=street_data['min'], mode='lines', name=f'Min - {street}'))

    mean_fig.update_layout(
        title='Mean Vehicle Year by Street',
        xaxis_title='Year-Month',
        yaxis_title='Mean Vehicle Year',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    std_fig.update_layout(
        title='Standard Deviation of Vehicle Year by Street',
        xaxis_title='Year-Month',
        yaxis_title='Standard Deviation',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    median_fig.update_layout(
        title='Median Vehicle Year by Street',
        xaxis_title='Year-Month',
        yaxis_title='Median Vehicle Year',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    max_fig.update_layout(
        title='Maximum Vehicle Year by Street',
        xaxis_title='Year-Month',
        yaxis_title='Maximum Vehicle Year',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    min_fig.update_layout(
        title='Minimum Vehicle Year by Street',
        xaxis_title='Year-Month',
        yaxis_title='Minimum Vehicle Year',
        xaxis=dict(tickformat='%Y-%m', tickmode='auto')
    )

    return mean_fig, std_fig, median_fig, max_fig, min_fig

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
