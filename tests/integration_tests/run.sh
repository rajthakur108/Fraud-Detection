#!/bin/bash

docker build \
    -f tests/integration_tests/Dockerfile \
    -t fraud-prediction-service:v2 \
    .

docker run -d \
    -p 9696:9696 \
    --name fraud-test \
    fraud-prediction-service:v2

sleep 10

uv run pytest tests/integration_tests/test.py

docker stop fraud-test
docker rm fraud-test