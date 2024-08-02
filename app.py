from flask import Flask, request, jsonify
from flask_cors import CORS
from detect_ai import download_model_and_tokenizer, analyze_text
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Download model and tokenizer during app initialization
logger.debug("Downloading model and tokenizer...")
download_model_and_tokenizer()
logger.debug("Model and tokenizer downloaded successfully")

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    logger.debug(f"Analyzing text: {text[:50]}...")  # Log first 50 characters
    perplexity = analyze_text(text)
    logger.debug(f"Perplexity: {perplexity}")
    return jsonify({'perplexity': perplexity})

if __name__ == '__main__':
    app.run(debug=True)