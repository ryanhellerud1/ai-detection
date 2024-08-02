import os
import logging
from flask import Flask, request, jsonify, render_template
from detect_ai import analyze_text, load_model_and_tokenizer

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# Load the model when the app starts
load_model_and_tokenizer()

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

@app.route('/', methods=['GET', 'POST'])
def home():
    logging.info("Home route accessed")
    if request.method == 'POST':
        text = request.form.get('text', '')
        if not text:
            logging.warning("No text provided in request")
            return render_template('index.html', error='No text provided')
        
        logging.info(f"Analyzing text: {text[:50]}...")  # Log first 50 characters
        try:
            logging.info("Starting text analysis")
            perplexity = analyze_text(text)
            logging.info(f"Analysis complete. Perplexity: {perplexity}")
            result_text = "Likely AI-generated" if perplexity < 50 else "Likely human-written"
            return render_template('index.html', result={'text': result_text, 'perplexity': perplexity})
        except Exception as e:
            logging.error(f"Error analyzing text: {str(e)}", exc_info=True)
            return render_template('index.html', error='An error occurred during analysis')
    
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    logging.info("Analyze route accessed")
    data = request.json
    text = data.get('text', '')
    if not text:
        logging.warning("No text provided in request")
        return jsonify({'error': 'No text provided'}), 400
    
    logging.info(f"Analyzing text: {text[:50]}...")  # Log first 50 characters
    try:
        logging.info("Starting text analysis")
        perplexity = analyze_text(text)
        logging.info(f"Analysis complete. Perplexity: {perplexity}")
        return jsonify({'perplexity': perplexity})
    except Exception as e:
        logging.error(f"Error analyzing text: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred during analysis'}), 500

if __name__ == '__main__':
    logging.info("Application starting")
    app.run(debug=True)