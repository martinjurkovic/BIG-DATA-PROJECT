from confluent_kafka import Consumer, KafkaException, KafkaError
import pandas as pd
import json
from collections import deque
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
import threading
from bigdata.utils import county_map

# Kafka Consumer configuration
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'nyc_violation_clustering_consumer',
    'auto.offset.reset': 'earliest'
}

# Initialize the Kafka consumer
consumer = Consumer(conf)
consumer.subscribe(['nyc_violations'])

# Data structures
batch_size = 10000
data_window = deque(maxlen=batch_size)

# Initialize the MiniBatchKMeans algorithm
n_clusters = 2
clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=batch_size)
scaler = StandardScaler()
label_encoders = {}
fitted = False  # Flag to check if the clusterer has been fitted

# Variables to keep track of accuracy
true_labels = []
predicted_labels = []

# Custom LabelEncoder to handle unseen labels
class CustomLabelEncoder:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.classes_ = set()
        self.default_label = -1

    def fit(self, data):
        self.label_encoder.fit(data)
        self.classes_ = set(self.label_encoder.classes_)
        return self

    def transform(self, data):
        return [self.label_encoder.transform([x])[0] if x in self.classes_ else self.default_label for x in data]

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

def process_batch():
    global clusterer, scaler, label_encoders, true_labels, predicted_labels, fitted
    # Create a DataFrame from the collected batch
    df = pd.DataFrame(data_window)

    numericals = ['Vehicle Year', 'Feet From Curb']

    categoricals = [
        "Plate Type",
        "Violation Code",
        "Vehicle Body Type",
        "Vehicle Make",
        "Issuing Agency",
        "Violation County",
        # "Violation Legal Code",
        # "Unregistered Vehicle?",
    ]

    df = df.dropna(subset=numericals+categoricals+['Registration State'])

    # Remap boroughs
    df['Violation County'] = df['Violation County'].apply(lambda x: county_map.get(x, x))
    
    # Convert 'Issue Date' to datetime and extract year and month
    df['Issue Date'] = pd.to_datetime(df['Issue Date'], errors='coerce')
    df['YearMonth'] = df['Issue Date'].dt.to_period('M').astype(str)
    
    # Filter out rows where 'Vehicle Year' is empty, None, or 0
    df = df[df['Vehicle Year'].notna() & (df['Vehicle Year'] != '') & (df['Vehicle Year'] != '0') & (df['Vehicle Year'] != 0)]
    
    # Convert 'Vehicle Year' to numeric (in case it was read as string)
    df['Vehicle Year'] = pd.to_numeric(df['Vehicle Year'], errors='coerce')
    
    # Filter data where Date year > 2012
    df = df[df['Issue Date'].dt.year > 2012]

    # Select relevant features for clustering
    
    features = df[numericals+categoricals]

    # print which features are NA
    # print(features.isna().sum())

    y = (df['Registration State'] == "NY").astype(int)

    # print(features.shape)
    # print(y.shape)

    if not features.empty:
        # Encode categorical features
        for column in categoricals:
            if column not in label_encoders:
                label_encoders[column] = CustomLabelEncoder()
                features.loc[:, column] = label_encoders[column].fit_transform(features[column])

            else:
                features.loc[:, column] = label_encoders[column].transform(features[column])

        # Scale the features
        scaled_features = scaler.fit_transform(features)

        # Predict the cluster for each sample in the batch
        cluster_labels = clusterer.predict(scaled_features) if fitted else [0] * len(scaled_features)

        # Train the model on the current batch
        clusterer.partial_fit(scaled_features)
        fitted = True

        # Store true labels and predicted labels for accuracy calculation
        true_labels.extend(y)
        predicted_labels.extend(cluster_labels)

        # Calculate accuracy
        if len(true_labels) >= batch_size:
            accuracy = accuracy_score(true_labels[:batch_size], predicted_labels[:batch_size])
            print(f'Accuracy: {accuracy}')
            true_labels = true_labels[batch_size:]
            predicted_labels = predicted_labels[batch_size:]

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
                    print(msg.error())
            else:
                process_message(msg)
        except KeyboardInterrupt:
            break

# Start the Kafka consumer in a separate thread
consumer_thread = threading.Thread(target=consume_messages, daemon=True)
consumer_thread.start()

# Keep the main thread alive to allow background processing
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Stopped by user")
