#!/bin/bash
# Script to run starter.py with xvfb (virtual display)
# This allows headless=False to work on servers without a display

# Check if xvfb is installed
if ! command -v xvfb-run &> /dev/null; then
    echo "xvfb-run not found. Installing xvfb..."
    sudo apt-get update && sudo apt-get install -y xvfb
fi

# Run with xvfb
xvfb-run -a python starter.py

