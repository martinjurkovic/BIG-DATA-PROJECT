# How to run Kafka processing?
```bash
cd src/T4_kafka
```

First docker:
    
```bash
docker-compose up -d
```

Then in separate terminals run the producer and consumer:

```bash
python producer.py
```
For statistical dashboard:
```bash
python dash_consumer.py
```

For clustering:
```bash
python clustering_consumer.py
```