#!/bin/bash

docker run -p 8080:8080 \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  flask-serving:latest
