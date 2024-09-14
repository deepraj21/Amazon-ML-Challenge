import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import os
from sklearn.linear_model import LinearRegression
import numpy as np

# Define the entity unit map
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 
                    'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 
                    'millilitre', 'pint', 'quart'}
}

# Create a set of allowed units
allowed_units = {unit for entity in entity_unit_map for unit in entity_unit_map[entity]}

# Function to download images
def download_images(image_links, limit=100):
    for i, link in enumerate(image_links[:limit]):
        response = requests.get(link)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(f"images/{i}.jpg")  # Save images locally
        else:
            print(f"Failed to download image {i}: {link}")

# Prepare transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Load test data
test_df = pd.read_csv('dataset/test.csv')
os.makedirs('images', exist_ok=True)  # Create directory for images
download_images(test_df['image_link'], limit=100)

# Function to format predictions
def format_prediction(value, unit):
    if unit in allowed_units:
        return f"{value} {unit}"
    return ""

# Placeholder for predictions
predictions = []

# Extract features from training data to train the regression model
def extract_features(image_links):
    features = []
    for link in image_links:
        response = requests.get(link)
        img = Image.open(BytesIO(response.content))
        input_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(input_tensor)  # Forward pass
        features.append(output.numpy())
    return np.array(features)

# Load training data (Assuming you have a CSV with entity values)
train_df = pd.read_csv('dataset/train.csv')
train_image_links = train_df['image_link'].tolist()  # Use the image links directly
train_features = extract_features(train_image_links)

# Train a regression model (e.g., Linear Regression)
y_train = train_df['entity_value'].apply(lambda x: float(x.split()[0])).values  # Extract numeric value
unit_train = train_df['entity_value'].apply(lambda x: x.split()[1]).values  # Extract units

# Train the model
regressor = LinearRegression()
regressor.fit(train_features, y_train)

# Loop through each image and make predictions
for index in range(min(100, len(test_df))):  # Process only the first 100 images
    img_path = f"images/{index}.jpg"
    img = Image.open(img_path)
    input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)  # Forward pass
        predicted_value = regressor.predict(output.numpy().reshape(1, -1))[0]  # Predict the entity value
        predicted_unit = 'gram'  # Replace with appropriate logic for unit selection

    # Format the prediction
    predictions.append(format_prediction(predicted_value, predicted_unit))

# Create output DataFrame
output_df = pd.DataFrame({
    'index': test_df['index'][:100],  # Only include the first 100 indices
    'prediction': predictions
})

# Save output to CSV
output_df.to_csv('output.csv', index=False)

# Check if the output file is formatted correctly using the sanity checker
# Uncomment the following line to run the sanity checker
# os.system('python src/sanity.py output.csv')