#!/bin/bash

# Kill Flask app on port 5000
FLASK_PID=$(lsof -ti tcp:5000)
if [ -n "$FLASK_PID" ]; then
    kill "$FLASK_PID" && echo "Flask app on port 5000 stopped."
else
    echo "No Flask app found on port 5000."
fi

# Kill static server on port 8000
STATIC_PID=$(lsof -ti tcp:8000)
if [ -n "$STATIC_PID" ]; then
    kill "$STATIC_PID" && echo "Static server on port 8000 stopped."
else
    echo "No static server found on port 8000."
fi

echo "Stop script completed."
