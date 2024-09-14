import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import requests
import re
from torchvision import models
import torch.nn as nn
import torch.optim as optim

# Define entity type mapping
entity_type_mapping = {
    'width': 0,
    'depth': 1,
    'height': 2,
    'item_weight': 3,
    'maximum_weight_recommendation': 4,
    'voltage': 5,
    'wattage': 6,
    'item_volume': 7
}

# Dataset class for loading image and text data
class ImageTextDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_url = self.data_frame.iloc[idx, 0]
        entity_type = self.data_frame.iloc[idx, 2]
        entity_value = self.data_frame.iloc[idx, 3]

        # Load image
        image = Image.open(requests.get(img_url, stream=True).raw)

        if self.transform:
            image = self.transform(image)

        # Convert entity_type to numerical ID
        entity_type_id = torch.tensor(entity_type_mapping.get(entity_type, -1))

        # Extract numeric value from entity_value
        match = re.search(r'\d+(\.\d+)?', entity_value)  # Match float or integer numbers
        if match:
            value = float(match.group())  # Extract the numeric part as a float
        else:
            raise ValueError(f"No numeric value found in entity_value: {entity_value}")

        # Return image, entity_type_id, and entity_value
        return image, entity_type_id, torch.tensor(value)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
train_dataset = ImageTextDataset(csv_file='training_data.csv', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the model
class EntityAwareModel(nn.Module):
    def __init__(self, num_entity_types):
        super(EntityAwareModel, self).__init__()
        self.image_model = models.resnet18(pretrained=True)
        self.image_model.fc = nn.Linear(512, 256)  # Adjust according to the output size of ResNet
        self.entity_embedding = nn.Embedding(num_entity_types, 10)  # Example: embedding size 10
        self.fc = nn.Linear(256 + 10, 1)  # Adjust input size based on image_model output and entity embedding

    def forward(self, image, entity_type):
        image_features = self.image_model(image)
        entity_embedding = self.entity_embedding(entity_type)
        combined_features = torch.cat((image_features, entity_embedding), dim=1)
        output = self.fc(combined_features)
        return output

# Initialize model, criterion, and optimizer
num_entity_types = len(entity_type_mapping)
model = EntityAwareModel(num_entity_types)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop

# for epoch in range(10):  # Example: Train for 10 epochs
#     model.train()  # Set model to training mode
#     for images, entity_types, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(images, entity_types)
#         loss = criterion(outputs.squeeze(), labels)
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')

# Inference: Predicting on a new image
model.eval()  # Set model to evaluation mode
image = Image.open('images/73.jpg')
image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

# Dummy entity type for inference
entity_type = 'height'  # Example entity type
entity_type_id = torch.tensor([entity_type_mapping.get(entity_type, -1)])

with torch.no_grad():
    prediction = model(image, entity_type_id)
print(f'Predicted Value: {prediction.item()}')
