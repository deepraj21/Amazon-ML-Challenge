import pandas as pd
import numpy as np
import cv2
import os
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Constants
IMAGE_SIZE = (224, 224)  # Adjust based on your model's input
BATCH_SIZE = 32
EPOCHS = 10  # Increase as needed
IMAGE_DIR = 'images/'  # Directory to save downloaded images

# Create image directory if it doesn't exist
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load Data
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

# Function to download and preprocess images
def download_image(image_link, filename):
    response = requests.get(image_link)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download image from {image_link}")

def load_and_preprocess_image(image_link):
    filename = os.path.join(IMAGE_DIR, image_link.split('/')[-1])  # Save image with its original name
    if not os.path.exists(filename):
        download_image(image_link, filename)
    
    image = cv2.imread(filename)
    if image is None:
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)  # Handle missing images
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return image

# Load training images and labels
X_train = []
y_train = []

for index, row in train_df.iterrows():
    image = load_and_preprocess_image(row['image_link'])
    X_train.append(image)
    y_train.append(row['entity_value'])  # Keep the raw values for now

X_train = np.array(X_train)

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Split the data
X_train, X_val, y_train_encoded, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')  # Adjust based on unique labels
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Prepare test data
X_test = []

for index, row in test_df.iterrows():
    image = load_and_preprocess_image(row['image_link'])
    X_test.append(image)

X_test = np.array(X_test)

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Prepare output
output_df = pd.DataFrame({
    'index': test_df['index'],
    'prediction': [le.inverse_transform([pred])[0] if pred < len(le.classes_) else "" for pred in predicted_classes]
})

# Format predictions to match the output requirements
def format_prediction(prediction):
    # This function assumes the format "x unit" is desired
    if prediction:
        # Split the value from the unit
        value, unit = prediction.split()[:-1], prediction.split()[-1]
        formatted_value = " ".join(value) + " " + unit
        return formatted_value
    return ""

output_df['prediction'] = output_df['prediction'].apply(format_prediction)

# Save predictions
output_df.to_csv("model_train_output.csv", index=False)