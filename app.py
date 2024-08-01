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

try:
    import torch
    logger.debug(f"PyTorch version: {torch.__version__}")
    logger.debug(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    logger.error(f"Failed to import PyTorch: {str(e)}")

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