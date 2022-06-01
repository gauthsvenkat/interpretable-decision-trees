#!/bin/bash

PARENT=python
TAG=3.5.10-slim

IMAGE_NAME=iai
VERSION=v1.0.0

docker build --build-arg PARENT_IMAGE=${PARENT}:${TAG} -t ${IMAGE_NAME}:${VERSION} . -f docker/Dockerfile
docker tag ${IMAGE_NAME}:${VERSION}  ${IMAGE_NAME}:latest