try:
    import torch
    import numpy as np
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
except ImportError as e:
    print(f"Error importing required libraries: {str(e)}")
    # Handle the error appropriately, maybe set a flag or use alternative libraries

import sys
import os
import logging
from flask import Flask, request, render_template
from flask_cors import CORS
from detect_ai import calculate_perplexity, load_model_and_tokenizer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.debug(f"Python version: {sys.version}")
logger.debug(f"Python path: {sys.path}")
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Directory contents: {os.listdir('.')}")

app = Flask(__name__)
CORS(app)

# Load model and tokenizer once when the app starts
model, tokenizer = load_model_and_tokenizer()

@app.route('/', methods=['GET', 'POST'])
def home():
    logger.debug("Home route accessed")
    result = None
    if request.method == 'POST':
        text = request.form['text']
        perplexity = calculate_perplexity(text, model, tokenizer)
        if perplexity is not None:
            result = {
                'text': 'Likely AI-generated' if perplexity < 50 else 'Likely human-written',
                'perplexity': f'{perplexity:.2f}'
            }
        else:
            result = {
                'text': 'Error calculating perplexity',
                'perplexity': 'N/A'
            }
    return render_template('index.html', result=result)

if __name__ == '__main__':
    logger.debug("Starting Flask app")
    app.run(debug=True, host='0.0.0.0', port=8000)