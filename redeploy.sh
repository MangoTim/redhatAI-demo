#!/bin/bash

echo "Stopping current deployment..."
./stop.sh

echo "Starting new deployment..."
./deploy.sh

echo "Redeployment complete."
