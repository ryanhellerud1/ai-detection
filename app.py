from flask import Flask, request, jsonify
from detect_ai import analyze_text
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    perplexity = analyze_text(text)
    
    if perplexity < 25:
        result = "very likely AI-generated"
        confidence = "high"
    elif perplexity < 50:
        result = "likely AI-generated"
        confidence = "moderate"
    elif perplexity < 75:
        result = "possibly AI-generated"
        confidence = "low"
    elif perplexity < 100:
        result = "uncertain"
        confidence = "very low"
    else:
        result = "likely human-written"
        confidence = "moderate"
    
    return jsonify({
        'classification': result,
        'confidence': confidence,
        'perplexity': f"{perplexity:.2f}"
    })

if __name__ == '__main__':
    app.run(debug=True)