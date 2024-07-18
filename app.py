from flask import Flask, render_template, request
from detect_ai import analyze_text

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']
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
        
        result = {
            'classification': result,
            'confidence': confidence,
            'perplexity': f"{perplexity:.2f}"
        }
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)