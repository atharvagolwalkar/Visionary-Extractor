import cv2
import pytesseract
import re
import requests
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance
import easyocr
import pandas as pd
import os
import logging

# Configure logging for V1.py
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"), # Log to the same file as app.py
                        logging.StreamHandler()
                    ])

# --- IMPORTANT: Explicitly set the path to the Tesseract executable ---
# This is crucial for pytesseract to find the Tesseract OCR engine inside the Docker container.
# The standard installation path on Debian-based systems (like python:3.9-slim-bullseye) is /usr/bin/tesseract.
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
logging.info(f"Pytesseract command set to: {pytesseract.pytesseract.tesseract_cmd}")


# --- IMPORTANT: Initialize EasyOCR Reader ONCE globally ---
# This will download models on first run if not present, but only once per app startup.
try:
    # Set gpu=True if you have a CUDA-enabled GPU and the necessary drivers/libraries installed
    # Otherwise, set gpu=False for CPU-only processing.
    global_easyocr_reader = easyocr.Reader(['en'], gpu=False)
    logging.info("EasyOCR Reader initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize EasyOCR Reader: {e}", exc_info=True)
    # Handle this critical error, perhaps by exiting or disabling OCR features.
    global_easyocr_reader = None # Set to None to indicate failure


def download_image(url):
    """
    Downloads an image from a given URL.
    Args:
        url (str): The URL of the image.
    Returns:
        PIL.Image.Image: The downloaded image.
    Raises:
        ValueError: If the URL is invalid or the file is not an image.
        requests.exceptions.RequestException: For network-related errors.
    """
    try:
        # Add a timeout to prevent hanging indefinitely
        response = requests.get(url, timeout=15)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        if 'image' not in response.headers.get('Content-Type', ''):
            raise ValueError(f"URL does not point to an image file: {url}")

        return Image.open(BytesIO(response.content))
    except requests.exceptions.Timeout:
        logging.error(f"Image download timed out for URL: {url}")
        raise ValueError("Image download timed out.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error during image download for URL {url}: {e}")
        raise ValueError(f"Network error during image download: {e}")
    except Exception as e:
        logging.error(f"Error opening image from URL {url}: {e}", exc_info=True)
        raise ValueError(f"File downloaded but couldn't open as image: {e}")

def preprocess_image(image):
    """
    Preprocesses the image for better OCR results.
    Converts to RGB, enhances contrast and sharpness, and applies denoising.
    Args:
        image (PIL.Image.Image): The input image.
    Returns:
        PIL.Image.Image: The processed image.
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)

        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2)

        np_image = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(np_image, None, 10, 10, 7, 21)

        processed_image = Image.fromarray(denoised)
        return processed_image
    except Exception as e:
        logging.error(f"Error in preprocess_image: {e}", exc_info=True)
        raise # Re-raise the exception to be caught by the caller

def extract_text(image):
    """
    Extracts text from the image using both Tesseract and EasyOCR.
    Args:
        image (PIL.Image.Image): The preprocessed image.
    Returns:
        str: Combined text from both OCR engines.
    """
    tesseract_text = ""
    easyocr_text = ""

    # Tesseract OCR
    try:
        tesseract_text = pytesseract.image_to_string(image, config='--psm 6')
        logging.debug(f"Tesseract extracted: {tesseract_text[:100]}...")
    except Exception as e:
        logging.warning(f"Error with Tesseract OCR: {e}")

    # EasyOCR
    if global_easyocr_reader:
        try:
            easyocr_result = global_easyocr_reader.readtext(np.array(image))
            easyocr_text = ' '.join([detection[1] for detection in easyocr_result])
            logging.debug(f"EasyOCR extracted: {easyocr_text[:100]}...")
        except Exception as e:
            logging.warning(f"Error with EasyOCR: {e}")
    else:
        logging.warning("EasyOCR Reader not initialized. Skipping EasyOCR extraction.")

    return f"{tesseract_text} {easyocr_text}"

def extract_entity_data(text, entity_name):
    """
    Extracts numerical value and unit for a specific entity from text.
    Args:
        text (str): The combined text from OCR.
        entity_name (str): The name of the entity to extract (e.g., 'width', 'voltage').
    Returns:
        tuple: (value, unit) or (None, None) if not found.
    """
    patterns = {
        "width": r'(\d+(?:\.\d+)?)\s*(inch|in|inches|cm|centimeter|centimeters|centimetre|centimetres|mm|millimeter|millimeters|millimetre|millimetres)',
        "depth": r'(\d+(?:\.\d+)?)\s*(inch|in|inches|cm|centimeter|centimeters|centimetre|centimetres|mm|millimeter|millimeters|millimetre|millimetres)',
        "height": r'(\d+(?:\.\d+)?)\s*(inch|in|inches|cm|centimeter|centimeters|centimetre|centimetres|mm|millimeter|millimeters|millimetre|millimetres)',
        "item_volume": r'(\d+(?:\.\d+)?)\s*(cup|cups|ml|millilitre|millilitres|milliliter|milliliters|fluid ounce|fluid ounces)',
        "item_weight": r'(\d+(?:\.\d+)?)\s*(pound|lb|pounds|kg|kilogram|kilograms|g|gram|grams|mg|milligram|milligrams|oz|ounce|ounces)',
        "maximum_weight_recommendation": r'(\d+(?:\.\d+)?)\s*(pound|lb|pounds|kg|kilogram|kilograms|g|gram|grams|mg|milligram|milligrams|oz|ounce|ounces)',
        "voltage": r'(\d+(?:\.\d+)?)\s*(volt|v)',
        "wattage": r'(\d+(?:\.\d+)?)\s*(watt|w)'
    }

    pattern = patterns.get(entity_name.lower())
    if pattern:
        matches = re.findall(pattern, text, re.IGNORECASE)
        logging.debug(f"Entity '{entity_name}' - Matches found: {matches}")
        if matches:
            match = matches[0]
            if len(match) >= 2:
                value, unit = match[:2]
                return float(value), unit

    return None, None

def normalize_unit(value, unit, entity_name):
    """
    Normalizes the extracted unit to a standard form.
    Args:
        value (float): The extracted numerical value.
        unit (str): The extracted unit.
        entity_name (str): The name of the entity.
    Returns:
        tuple: (normalized_value, normalized_unit).
    """
    unit = unit.lower()
    if entity_name in ["width", "depth", "height"]:
        if unit in ['in', 'inch', 'inches']:
            return value, 'inch'
        elif unit in ['cm', 'centimeter', 'centimeters', 'centimetre', 'centimetres']:
            return value, 'centimetre'
        elif unit in ['mm', 'millimeter', 'millimeters', 'millimetre', 'millimetres']:
            return value, 'millimetre'
    elif entity_name == "item_volume":
        if unit in ['cup', 'cups']:
            return value, 'cup'
        elif unit in ['ml', 'millilitre', 'millilitres', 'milliliter', 'milliliters']:
            return value, 'millilitre'
        elif unit in ['fluid ounce', 'fluid ounces']:
            return value, 'fluid ounce'
    elif entity_name in ["item_weight", "maximum_weight_recommendation"]:
        if unit in ['lb', 'pound', 'pounds']:
            return value, 'pound'
        elif unit in ['kg', 'kilogram', 'kilograms']:
            return value, 'kilogram'
        elif unit in ['g', 'gram', 'grams']:
            return value, 'gram'
        elif unit in ['mg', 'milligram', 'milligrams']:
            return value, 'milligram'
        elif unit in ['oz', 'ounce', 'ounces']:
            return value, 'ounce'
    elif entity_name == "voltage":
        return value, 'volt'
    elif entity_name == "wattage":
        return value, 'watt'
    return value, unit # Return original if no normalization rule applies

def process_image_url(url, entity_name):
    """
    Main function to process an image URL and extract a specific entity.
    Args:
        url (str): The URL of the image.
        entity_name (str): The name of the entity to extract.
    Returns:
        str: The extracted and normalized entity value (e.g., "10.50 inch") or "Not found".
    """
    try:
        image = download_image(url)
        preprocessed_image = preprocess_image(image)
        extracted_text = extract_text(preprocessed_image)
        value, unit = extract_entity_data(extracted_text, entity_name)

        if value is not None and unit:
            normalized_value, normalized_unit = normalize_unit(value, unit, entity_name)
            return f"{normalized_value:.2f} {normalized_unit}"
        else:
            return 'Not found'
    except Exception as e:
        logging.error(f"Error in process_image_url for {url} and {entity_name}: {e}", exc_info=True)
        # Re-raise the exception so Flask can catch it and return an error message
        raise


# import cv2
# import pytesseract
# import re
# import requests
# import numpy as np
# from io import BytesIO
# from PIL import Image, ImageEnhance
# import easyocr
# import pandas as pd
# import os

# def download_image(url):
#     response = requests.get(url)
#     # Check if the request succeeded
#     if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
#         try:
#             return Image.open(BytesIO(response.content))
#         except Exception as e:
#             raise ValueError("File downloaded but couldn't open as image: " + str(e))
#     else:
#         raise ValueError('Invalid image URL or not an image file')

# def preprocess_image(image):
#     try:
#         # Convert the image to RGB (if it's not already)
#         if image.mode != 'RGB':
#             image = image.convert('RGB')

#         # Enhance the image contrast
#         enhancer = ImageEnhance.Contrast(image)
#         image = enhancer.enhance(2)  # Increase contrast by a factor of 2

#         # Apply sharpening to make the text clearer
#         enhancer = ImageEnhance.Sharpness(image)
#         image = enhancer.enhance(2)  # Sharpen the image by a factor of 2

#         # Convert PIL image to a NumPy array for additional OpenCV processing
#         np_image = np.array(image)

#         # Optional: Apply further denoising (useful for images with noise)
#         denoised = cv2.fastNlMeansDenoisingColored(np_image, None, 10, 10, 7, 21)

#         # Convert back to PIL Image format for Tesseract
#         processed_image = Image.fromarray(denoised)
#         return processed_image
#     except Exception as e:
#         print(f"Error in preprocess_image: {e}")
#         return None

# def extract_text(image):
#     tesseract_text = pytesseract.image_to_string(image, config='--psm 6')
#     reader = easyocr.Reader(['en'])
#     easyocr_result = reader.readtext(np.array(image))  # Convert PIL image to numpy array
#     easyocr_text = ' '.join([detection[1] for detection in easyocr_result])
#     return f"{tesseract_text} {easyocr_text}"

# def extract_entity_data(text, entity_name):
#     patterns = {
#         "width": r'(\d+(?:\.\d+)?)\s*(inch|in|inches|cm|centimeter|centimeters|centimetre|centimetres|mm|millimeter|millimeters|millimetre|millimetres)',
#         "depth": r'(\d+(?:\.\d+)?)\s*(inch|in|inches|cm|centimeter|centimeters|centimetre|centimetres|mm|millimeter|millimeters|millimetre|millimetres)',
#         "height": r'(\d+(?:\.\d+)?)\s*(inch|in|inches|cm|centimeter|centimeters|centimetre|centimetres|mm|millimeter|millimeters|millimetre|millimetres)',
#         "item_volume": r'(\d+(?:\.\d+)?)\s*(cup|cups|ml|millilitre|millilitres|milliliter|milliliters|fluid ounce|fluid ounces)',
#         "item_weight": r'(\d+(?:\.\d+)?)\s*(pound|lb|pounds|kg|kilogram|kilograms|g|gram|grams|mg|milligram|milligrams|oz|ounce|ounces)',
#         "maximum_weight_recommendation": r'(\d+(?:\.\d+)?)\s*(pound|lb|pounds|kg|kilogram|kilograms|g|gram|grams|mg|milligram|milligrams|oz|ounce|ounces)',
#         "voltage": r'(\d+(?:\.\d+)?)\s*(volt|v)',
#         "wattage": r'(\d+(?:\.\d+)?)\s*(watt|w)'
#     }

#     pattern = patterns.get(entity_name.lower())
#     if pattern:
#         matches = re.findall(pattern, text, re.IGNORECASE)
#         print(f"Matches found: {matches}")  # Debugging line
#         if matches:
#             match = matches[0]
#             if len(match) >= 2:
#                 value, unit = match[:2]
#                 return float(value), unit

#     return None, None

# def normalize_unit(value, unit, entity_name):
#     unit = unit.lower()
#     if entity_name in ["width", "depth", "height"]:
#         if unit in ['in', 'inch', 'inches']:
#             return value, 'inch'
#         elif unit in ['cm', 'centimeter', 'centimeters', 'centimetre', 'centimetres']:
#             return value, 'centimetre'
#         elif unit in ['mm', 'millimeter', 'millimeters', 'millimetre', 'millimetres']:
#             return value, 'millimetre'
#     elif entity_name == "item_volume":
#         if unit in ['cup', 'cups']:
#             return value, 'cup'
#         elif unit in ['ml', 'millilitre', 'millilitres', 'milliliter', 'milliliters']:
#             return value, 'millilitre'
#         elif unit in ['fluid ounce', 'fluid ounces']:
#             return value, 'fluid ounce'
#     elif entity_name in ["item_weight", "maximum_weight_recommendation"]:
#         if unit in ['lb', 'pound', 'pounds']:
#             return value, 'pound'
#         elif unit in ['kg', 'kilogram', 'kilograms']:
#             return value, 'kilogram'
#         elif unit in ['g', 'gram', 'grams']:
#             return value, 'gram'
#         elif unit in ['mg', 'milligram', 'milligrams']:
#             return value, 'milligram'
#         elif unit in ['oz', 'ounce', 'ounces']:
#             return value, 'ounce'
#     elif entity_name == "voltage":
#         return value, 'volt'
#     elif entity_name == "wattage":
#         return value, 'watt'
#     return value, unit

# def process_image_url(url, entity_name):
#     image = download_image(url)
#     if image is None:
#         return None

#     preprocessed_image = preprocess_image(image)
#     if preprocessed_image is None:
#         return None

#     extracted_text = extract_text(preprocessed_image)
#     value, unit = extract_entity_data(extracted_text, entity_name)

#     if value is not None and unit:
#         normalized_value, normalized_unit = normalize_unit(value, unit, entity_name)
#         return f"{normalized_value:.2f} {normalized_unit}"

#     return 'Not found'


# def modelmain(dataset):
#     results = []
#     for _, row in dataset.iterrows():
#         print(f"Processing {row['image_link']} for {row['entity_name']}...")
#         entity_value = process_image_url(row['image_link'], row['entity_name'])
#         if entity_value == 'Not found':
#             entity_value = ""


#         results.append({
#             'index': row['index'],
#             # 'image_link': row['image_link'],
#             # 'group_id': row['group_id'],
#             # 'entity_name': row['entity_name'],
#             'prediction': entity_value
#         })

#     results_df = pd.DataFrame(results)
#     output_file = 'extraction_results.csv'
    
#     # Check if the file already exists
#     if os.path.exists(output_file):
#         # Append to the existing file
#         results_df.to_csv(output_file, mode='a', header=False, index=False)
#     else:
#         # Create a new file and write the header
#         results_df.to_csv(output_file, index=False, header=['index', 'prediction'])
    
#     print("\nExtraction complete. Results saved to 'extraction_results.csv'.")
#     return results



# def predictor(image_link, entity_name,index,group_id):
#     '''
#     Call your model/approach here
#     '''
    
#     # Create a temporary dataset with a single row
#     temp_df = pd.DataFrame({
#         'index': [index],
#         'image_link': [image_link],
#         'group_id': [group_id],  # Assuming you don't need it for a single prediction
#         'entity_name': [entity_name]
#     })

#     # Process the temporary dataset
#     results = modelmain(temp_df)
    
#     # Get the result for the single image
#     result = results[0]['prediction']
#     return result

# if __name__ == "__main__":
#     # Use current directory for deployment compatibility
#     DATASET_FOLDER = os.path.dirname(os.path.abspath(__file__))
    
#     test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample.csv'))
    
#     test['prediction'] = test.apply(
#         lambda row: predictor(row['image_link'], row['entity_name'],row['index'],row['group_id']), axis=1)
    
#     output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
#     test[['index', 'prediction']].to_csv(output_filename, index=False)