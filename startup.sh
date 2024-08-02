#!/bin/bash
echo "Current directory: $(pwd)"
echo "Contents of current directory:"
ls -la
echo "Contents of /home/site/wwwroot:"
ls -la /home/site/wwwroot
echo "Starting Gunicorn..."
gunicorn --bind=0.0.0.0 --timeout 600 app:app