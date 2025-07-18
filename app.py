from flask import Flask, render_template, request
from V1 import process_image_url  # Import your function
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_link = request.form['image_link']
    entity_name = request.form['entity_name']
    prediction = process_image_url(image_link, entity_name)
    return render_template('result.html', image_link=image_link, prediction=prediction)

if __name__ == "__main__":
    # Use environment variable for port (required for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
