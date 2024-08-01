import sys
import os
import logging
from flask import Flask, request, render_template
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.debug(f"Python version: {sys.version}")
logger.debug(f"Python path: {sys.path}")
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Directory contents: {os.listdir('.')}")

try:
    import numpy as np
    logger.debug(f"NumPy version: {np.__version__}")
except ImportError as e:
    logger.error(f"Failed to import NumPy: {str(e)}")

try:
    import torch
    logger.debug(f"PyTorch version: {torch.__version__}")
except ImportError as e:
    logger.error(f"Failed to import PyTorch: {str(e)}")

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def home():
    logger.debug("Home route accessed")
    return "Hello, World!"

if __name__ == '__main__':
    logger.debug("Starting Flask app")
    app.run(debug=True, host='0.0.0.0', port=8000)