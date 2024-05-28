import os
import socket
import json
from argparse import ArgumentParser
from confluent_kafka import Producer


def file_reader(file_path):
    with open(file_path, 'r') as file:
        header_line = file.readline()
        keys = header_line.strip().split(',')
        for line in file:
            items = line.strip().split(',')
            yield {key: value for key, value in zip(keys, items)}

if __name__ == '__main__':
    # Parse the command line.
    # parser = ArgumentParser()
    # parser.add_argument('station', type=int, default=0, help='The station number to produce data for.')
    # args = parser.parse_args()

    # Create Producer instance
    # producer = Producer(config)
    producer = Producer({
        'bootstrap.servers': "localhost:29092",
        'client.id': socket.gethostname()})

    # Message delivery callback (triggered by poll() or flush())
    # when a message has been successfully delivered or permanently
    # failed delivery (after retries).
    def delivery_callback(err, msg):
        if err:
            print('ERROR: Message failed delivery: {}'.format(err))
        else:
            print("Produced event to topic {topic}: key = {key:12} value = {value:12}".format(
                topic=msg.topic(), key=msg.key().decode('utf-8'), value=msg.value().decode('utf-8')))

    DATA_PATH = 'data/CSV'
    data_files = os.listdir(DATA_PATH)
    for file in data_files:
        if not file.endswith('.csv'):
            continue
        filepath = os.path.join(DATA_PATH, file)
        count = 0
        for row_data in file_reader(filepath):
            row_data_bytes = json.dumps(row_data).encode('utf-8')
            producer.produce(topic="nyc_tickets", key=row_data['Violation County'], value=row_data_bytes, callback=delivery_callback)
            count += 1
            if count % 10000 == 0:
                producer.poll(10000)
                producer.flush()

    # Block until the messages are sent.
    producer.poll(10000)
    producer.flush()