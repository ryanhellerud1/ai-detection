# AI Text Detection Project

This project is a Flask-based web application that uses a GPT-2 model to analyze text and determine whether it's likely to be AI-generated or human-written.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

## Installation

Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
Install the required packages:
   ```
   pip install -r requirements.txt
   ```
## Running the Application

Start the Flask application:
   ```
   flask run
   ```
Open a web browser and navigate to `http://127.0.0.1:5000/`

## API Usage

You can also use the `/analyze` endpoint to analyze text programmatically:
