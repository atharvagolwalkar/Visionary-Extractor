# Use a base image with Python 3.9 on Debian 11 (Bullseye)
# Bullseye is a more recent stable release, ensuring repository availability.
FROM python:3.9-slim-bullseye

# Install system dependencies: Tesseract OCR and its language data, plus other necessary libs
# libgl1-mesa-glx is often needed for OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models
# This command runs Python code to initialize EasyOCR, which triggers model download.
# We set verbose=False to keep build logs cleaner.
# Ensure 'en' is the correct language you need.
RUN python -c "import easyocr; reader = easyocr.Reader(['en'], verbose=False); print('EasyOCR English model pre-downloaded.')"

# Copy the rest of your application code
COPY . .

# Expose the port your Flask app will run on (Render will use this)
EXPOSE 5000

# Use Gunicorn to run the Flask application in production
# -w: number of worker processes (adjust based on your Render plan's CPU cores, 2-4 is a good start)
# -b: bind to all interfaces on the port provided by Render's environment variable (default 5000)
# app:app refers to the 'app' Flask instance inside 'app.py'
CMD gunicorn -w 4 -b 0.0.0.0:${PORT:-5000} app:app
