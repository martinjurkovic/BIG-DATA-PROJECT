from confluent_kafka import Producer
import pandas as pd

# Kafka Producer configuration
conf = {
    'bootstrap.servers': 'localhost:9092'
}

# Initialize the Kafka producer
producer = Producer(conf)

# Callback function when a message is delivered
def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed for record {msg.key()}: {err}")
    else:
        # print(f"Record {msg.key()} successfully produced to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
        pass

# Read the Parquet file
df = pd.read_parquet('../../data/parquet/2014.parquet')

# Iterate over rows and send to Kafka
print("Started producing messages to Kafka...")
for _, row in df.iterrows():
    record_value = row.to_json()
    producer.produce('nyc_violations', key=str(row['Summons Number']), value=record_value, callback=delivery_report)
    producer.poll(0)

# Wait for any outstanding messages to be delivered and delivery reports to be received
producer.flush()
