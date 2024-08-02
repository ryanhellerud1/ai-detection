import os
import logging
from flask import Flask, request, jsonify
from detect_ai import analyze_text, load_model_and_tokenizer

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@app.before_first_request
def startup():
    logging.info("Starting up the application")
    logging.info(f"Current directory: {os.getcwd()}")
    logging.info(f"Directory contents: {os.listdir()}")
    try:
        import numpy
        logging.info(f"NumPy version: {numpy.__version__}")
    except ImportError as e:
        logging.error(f"Failed to import NumPy: {e}")

@app.route('/')
def home():
    logging.info("Home route accessed")
    return "Hello, World!"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    logging.info(f"Analyzing text: {text[:50]}...")  # Log first 50 characters
    try:
        perplexity = analyze_text(text)
        logging.info(f"Perplexity: {perplexity}")
        return jsonify({'perplexity': perplexity})
    except Exception as e:
        logging.error(f"Error analyzing text: {str(e)}")
        return jsonify({'error': 'An error occurred during analysis'}), 500

if __name__ == '__main__':
    logging.info("Application starting")
    app.run(debug=True)