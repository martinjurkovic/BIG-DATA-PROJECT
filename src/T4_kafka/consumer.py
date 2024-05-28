import sys
import json
from argparse import ArgumentParser, FileType

import pandas as pd
from confluent_kafka import Consumer, OFFSET_BEGINNING
import socket

if __name__ == '__main__':
    config = {'bootstrap.servers': "localhost:29092",
                            'client.id': socket.gethostname(),
                             'group.id': 'test_group', 
                             'auto.offset.reset': 'earliest'}
    # # Create Consumer instance
    consumer = Consumer(config)

    # Set up a callback to reset consumer offsets on assignment
    def reset_offset(consumer, partitions):
        for p in partitions:
            p.offset = OFFSET_BEGINNING
        consumer.assign(partitions)

    # Subscribe to topic
    topics = ["nyc_tickets"]
    consumer.subscribe(topics, on_assign=reset_offset)

    # Poll for new messages from Kafka and print them.
    # create a dataframe indexed by date from 01/01/2024 to 12/31/2024 with columns for each NYC borough
    data = pd.DataFrame(index=pd.date_range(start='2014-01-01', end='2024-12-31', freq='D'),
                        columns=['BX', 'BK', 'MN', 'QS', 'SI', 'TOTAL'])
    borough_map = {
        'BX': 'BX',
        'BRONX': 'BX',
        'BK': 'BK',
        'BROOKLYN': 'BK',
        'K': 'BK',
        'KINGS': 'BK',
        'MN': 'MN',
        'MANHATTAN': 'MN',
        'Q': 'QS',
        'QS': 'QS',
        'QN': 'QS',
        'QNS': 'QS',
        'QUEENS': 'QS',
        'SI': 'SI',
        'ST': 'SI',
        'STATEN ISLAND': 'SI',
        'NY': 'TOTAL',
        '' : 'TOTAL',
        'R': 'TOTAL',
    }

    data = data.fillna(0.)
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                # Initial message consumption may take up to
                # `session.timeout.ms` for the consumer group to
                # rebalance and start consuming
                print("Waiting...")
            elif msg.error():
                print("ERROR: %s".format(msg.error()))
            else:
                # Extract the (optional) key and value, and print.
                msg_data = json.loads(msg.value().decode('utf-8'))
                key = msg.key().decode('utf-8')
                boroug = borough_map[key.upper()]
                #get only the date without the time
                date = pd.to_datetime(msg_data['Issue Date'])
                try:
                    data.loc[date, boroug] += 1
                except KeyError as e:
                    print(f"KeyError: {e}")
                    continue
                if boroug != 'TOTAL':
                    data.loc[date, 'TOTAL'] += 1
                print(f"{msg.topic()}: {boroug} - {date} - {data.loc[date, boroug]}")
    except KeyboardInterrupt:
        pass
    finally:
        # Leave group and commit final offsets
        consumer.close()