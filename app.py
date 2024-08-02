from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from detect_ai import analyze_text
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        logger.debug(f"Analyzing text: {text[:50]}...")  # Log first 50 characters
        try:
            perplexity = analyze_text(text)
            logger.debug(f"Perplexity: {perplexity}")
            result = {
                'text': 'Likely AI-generated' if perplexity < 50 else 'Likely human-written',
                'perplexity': f'{perplexity:.2f}'
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            result = {
                'text': 'Error calculating perplexity',
                'perplexity': 'N/A'
            }
    return render_template('index.html', result=result)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    logger.debug(f"Analyzing text: {text[:50]}...")  # Log first 50 characters
    try:
        perplexity = analyze_text(text)
        logger.debug(f"Perplexity: {perplexity}")
        return jsonify({'perplexity': perplexity})
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return jsonify({'error': 'An error occurred during analysis'}), 500

if __name__ == '__main__':
    app.run(debug=True)