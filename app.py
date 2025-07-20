from flask import Flask, render_template, request
from V1 import process_image_url  # Import your function
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_link = request.form.get('image_link')
        entity_name = request.form.get('entity_name')
        prediction = process_image_url(image_link, entity_name)  # Make sure this function is imported
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        print(f"Error in /predict: {e}")  # For server logs
        return f"Error: {str(e)}"          # For user feedback


if __name__ == "__main__":
    # Use environment variable for port (required for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
