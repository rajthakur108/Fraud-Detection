#!/usr/bin/env bash

LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`

export LOCAL_IMAGE_NAME="fraud-detection-service:${LOCAL_TAG}"

docker build \
    -f tests/integration_tests/Dockerfile \
    -t ${LOCAL_IMAGE_NAME} \
    .

docker run -d \
    -p 9696:9696 \
    --name fraud-test \
    ${LOCAL_IMAGE_NAME}

sleep 10

uv run pytest tests/integration_tests/test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker logs --tail 50 fraud-test
fi

docker stop fraud-test
docker rm fraud-test

exit ${ERROR_CODE}