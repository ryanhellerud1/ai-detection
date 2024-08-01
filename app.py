from flask import Flask, request, render_template
from detect_ai import analyze_text
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        perplexity = analyze_text(text)
        
        logging.info(f"Calculated perplexity: {perplexity}")
        
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
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)