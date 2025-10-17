#!/bin/bash

# Clear previous logs
> flask.log
> index.log

# Start Flask app and log output
echo "Starting Flask app..."
nohup python app.py > flask.log 2>&1 &

# Start static server and log output
echo "Starting static server on port 8000..."
nohup python -m http.server 8000 -b 0.0.0.0 > index.log 2>&1 &

echo "Deployment started. Logs: flask.log and index.log"
