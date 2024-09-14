import requests
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import pytesseract
import pandas as pd
import csv
import re

# Load allowed units
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

allowed_units = {unit for entity in entity_unit_map for unit in entity_unit_map[entity]}

# Function to download images from URLs
def download_image(url):
    try:
        print(f"Downloading image from {url}")
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

# Function to preprocess image for better OCR results
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # Apply a filter to reduce noise
    image = image.filter(ImageFilter.MedianFilter())
    return image

# Function to apply OCR and extract text
def extract_text_from_image(image):
    try:
        image = preprocess_image(image)
        text = pytesseract.image_to_string(image)
        print(f"OCR Text: {text}")
        return text
    except Exception as e:
        print(f"Error applying OCR: {e}")
        return ""

# Function to clean and extract value + unit
def extract_value_and_unit(ocr_text, entity_name):
    entity_units = entity_unit_map.get(entity_name, allowed_units)
    print(f"Extracting value and unit from: {ocr_text}")

    # Use regex to find potential values and units
    for unit in entity_units:
        # Simple regex to match numbers followed by the unit
        pattern = re.compile(r'\b(\d+[\.,]?\d*)\s?(' + re.escape(unit) + r')\b', re.IGNORECASE)
        match = pattern.search(ocr_text)
        if match:
            value = match.group(1).replace(',', '.')
            unit = match.group(2).lower()
            return f"{value} {unit}"
    
    # If no match, try to extract numeric values directly
    numeric_pattern = re.compile(r'\b(\d+[\.,]?\d*)\b')
    number_match = numeric_pattern.search(ocr_text)
    if number_match:
        number = number_match.group(1).replace(',', '.')
        # Check for units around the number
        for unit in entity_units:
            if unit in ocr_text:
                return f"{number} {unit}"
    
    return ""

# Main prediction function
def generate_predictions(test_file, output_file):
    try:
        with open(test_file, 'r') as test_f, open(output_file, 'w') as out_f:
            test_reader = pd.read_csv(test_f)
            fieldnames = ['index', 'prediction']
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()

            for _, row in test_reader.iterrows():
                index = row['index']
                image_url = row['image_link']
                entity_name = row['entity_name']
                
                print(f"Processing index {index} with entity {entity_name}")
                # Download image
                image = download_image(image_url)
                if image is None:
                    writer.writerow({'index': index, 'prediction': ''})
                    continue

                # Apply OCR to extract text from image
                ocr_text = extract_text_from_image(image)

                # Use OCR to extract value + unit
                prediction = extract_value_and_unit(ocr_text, entity_name)
                
                writer.writerow({'index': index, 'prediction': prediction})

    except Exception as e:
        print(f"Error processing CSV files: {e}")

# Usage
test_file_path = 'dataset/test.csv'
output_file_path = 'output.csv'
generate_predictions(test_file_path, output_file_path)
