#!/bin/bash
source setup_hailo_env.sh

# Set the bind address and port
BIND_ADDRESS="0.0.0.0"
BIND_PORT="5000"

# Set the path to the virtual environment (if applicable)
VENV_PATH=".venv"

# Activate the virtual environment (if applicable)
if [ -n "$VENV_PATH" ]; then
  source "$VENV_PATH/bin/activate"
fi

# Start Gunicorn
gunicorn -b "$BIND_ADDRESS:$BIND_PORT" --worker-class eventlet -w 1 app:app

# Deactivate the virtual environment (if applicable)
if [ -n "$VENV_PATH" ]; then
  deactivate
fi