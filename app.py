import os
import csv
import requests
import pytesseract
from PIL import Image
from io import BytesIO
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load constants
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
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    return None

# Function to apply OCR and extract text
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Function to clean and extract value + unit
def extract_value_and_unit(ocr_text, entity_name):
    entity_units = entity_unit_map.get(entity_name, allowed_units)
    # Search for numbers + units in OCR text
    for unit in entity_units:
        if unit in ocr_text:
            # Extract the value before the unit
            try:
                value = float(ocr_text.split(unit)[0].strip())
                return f"{value} {unit}"
            except ValueError:
                continue
    return ""

# Hugging Face model pipeline for image captioning (optional)
def image_captioning(image):
    caption_pipeline = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    caption = caption_pipeline(image)[0]['generated_text']
    return caption

# Initialize GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

# GPT-2 doesn't have a pad_token by default, so we add one
tokenizer.pad_token = tokenizer.eos_token  # Use the eos_token as the pad_token

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Text generation function
def generate_entity_extraction(text, entity_name):
    # Prepare the input with attention mask
    inputs = tokenizer(
        f"Extract the {entity_name} from the following text:\n\n{text}\n\n",
        return_tensors="pt",  # Return as PyTorch tensors
        padding=True,         # Explicitly enable padding
        truncation=True,      # Truncate if the text is too long
        add_special_tokens=True  # Add special tokens if necessary
    )

    # Add attention mask to avoid warning
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate response
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # Pass attention mask
        max_new_tokens=50,              # Set max tokens to control length
        do_sample=True,                 # Sampling for diverse outputs
        temperature=0.7                 # Adjust temperature for creativity
    )

    # Decode the generated tokens
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Main prediction function
def generate_predictions(test_file, output_file):
    # Read the test data
    with open(test_file, 'r') as test_f, open(output_file, 'w') as out_f:
        test_reader = csv.DictReader(test_f)
        fieldnames = ['index', 'prediction']
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for row in test_reader:
            index = row['index']
            image_url = row['image_link']
            entity_name = row['entity_name']
            
            # Download image
            image = download_image(image_url)
            if image is None:
                writer.writerow({'index': index, 'prediction': ''})
                continue

            # Apply OCR to extract text from image
            ocr_text = extract_text_from_image(image)

            # Optional: Generate image caption (if needed)
            caption = image_captioning(image)
            ocr_text += " " + caption  # Append caption to OCR results

            # Use OCR to extract value + unit
            prediction = extract_value_and_unit(ocr_text, entity_name)
            
            # If no prediction from OCR, attempt using local GPT-Neo model
            if not prediction:
                prediction = generate_entity_extraction(ocr_text, entity_name)

            writer.writerow({'index': index, 'prediction': prediction})

# Usage
test_file_path = 'dataset/test.csv'
output_file_path = 'output.csv'
generate_predictions(test_file_path, output_file_path)
