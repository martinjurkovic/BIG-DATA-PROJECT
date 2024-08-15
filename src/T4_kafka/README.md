# How to run Kafka processing?
```bash
cd src/kafka
```

First docker:
    
```bash
docker-compose up -d
```

Then in separate terminals run the producer and consumer:

```bash
python producer.py
```

```bash
python consumer.py
```