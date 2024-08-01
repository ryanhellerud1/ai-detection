import sys
import os
import logging
from flask import Flask, request, render_template
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting application...")

# Add python_packages to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python_packages')))
logger.info(f"Python path: {sys.path}")

app = Flask(__name__)
CORS(app)

logger.info("Flask app created")

try:
    logger.info("Importing detect_ai module...")
    from detect_ai import analyze_text
    logger.info("Successfully imported detect_ai module")
except Exception as e:
    logger.error(f"Error importing detect_ai module: {str(e)}", exc_info=True)
    raise

@app.route('/', methods=['GET', 'POST'])
def home():
    logger.info("Received request to home route")
    result = None
    if request.method == 'POST':
        text = request.form['text']
        logger.info(f"Analyzing text: {text[:50]}...")  # Log first 50 characters of input
        try:
            perplexity = analyze_text(text)
            logger.info(f"Calculated perplexity: {perplexity}")
            
            if perplexity < 50:
                result = " likely AI-generated"
                confidence = "high"
            else:
                result = " likely human-written"
                confidence = "high"
            
            result = {
                "text": result,
                "confidence": confidence,
                "perplexity": f"{perplexity:.2f}"
            }
        except Exception as e:
            logger.error(f"Error during text analysis: {str(e)}", exc_info=True)
            result = {"error": "An error occurred during analysis"}
    
    return render_template('index.html', result=result)

logger.info("Application startup complete")

if __name__ == '__main__':
    app.run(debug=True)