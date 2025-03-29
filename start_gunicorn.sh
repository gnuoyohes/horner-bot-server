#!/bin/bash

# Set the application module and variable name
APP_MODULE="app"
APP_VARIABLE="app"

# Set the bind address and port
BIND_ADDRESS="0.0.0.0"
BIND_PORT="5000"

# Set the number of worker processes
WORKERS="5"

# Set the path to the virtual environment (if applicable)
VENV_PATH="/venv"

# Activate the virtual environment (if applicable)
if [ -n "$VENV_PATH" ]; then
  source "$VENV_PATH/bin/activate"
fi

# Start Gunicorn
gunicorn -b "$BIND_ADDRESS:$BIND_PORT" -w "$WORKERS" "$APP_MODULE:$APP_VARIABLE"

# Deactivate the virtual environment (if applicable)
if [ -n "$VENV_PATH" ]; then
  deactivate
fi