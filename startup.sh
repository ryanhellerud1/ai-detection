#!/bin/bash
echo "Current directory: $(pwd)"
echo "Contents of current directory:"
ls -la
echo "Contents of models directory:"
ls -la models/
gunicorn --bind=0.0.0.0 --timeout 600 app:app