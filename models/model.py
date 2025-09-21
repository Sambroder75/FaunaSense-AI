import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from torchvision.models import MobileNet_V2_Weights

# Load the model
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()  # Set the model to evaluation mode

# Preprocess the image
def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),         # Resize image to 224x224
        transforms.ToTensor(),                 # Convert image to tensor
        transforms.Normalize(                  # Normalize image
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(image_path).convert('RGB')  # Ensure the image is RGB
    img_tensor = transform(img).unsqueeze(0)     # Add batch dimension
    return img_tensor

# Predict the animal
def predict_animal(image_path):
    try:
        # Preprocess the image
        processed_image = prepare_image(image_path)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prediction = torch.argmax(probabilities).item()

        # Load ImageNet labels
        labels_path = "imagenet_classes.txt"
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file '{labels_path}' not found.")

        with open(labels_path) as f:
            labels = [line.strip() for line in f.readlines()]
        
        # Get the predicted animal name
        animal_name = labels[top_prediction].lower()  # Normalize to lowercase
        print("Predicted animal label:", animal_name)  # Debugging log

        return animal_name
    except Exception as e:
        print(f"Error during prediction: {str(e)}")  # Log the error
        return None

# Animal information database with synonyms
animal_info = {
    "cat": {
        "classification": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora",
        "venomous": "No",
        "dangerous": "No",
        "how_to_get_rid_of": "Use humane traps to relocate the cat."
    },
    "lion": {
        "classification": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora",
        "venomous": "No",
        "dangerous": "Yes",
        "how_to_get_rid_of": "Contact a local wildlife authority. Do not approach."
    },
    "cobra": {
        "classification": "Kingdom: Animalia, Phylum: Chordata, Class: Reptilia, Order: Squamata",
        "venomous": "Yes",
        "dangerous": "Yes",
        "how_to_get_rid_of": "Call animal control immediately. Do not attempt to handle the snake."
    }
}

# Synonyms mapping
animal_synonyms = {
    "egyptian_cat": "cat",
    "tabby_cat": "cat",
    "tabby, tabby cat": "cat",
    "lion, king of beasts, panthera leo": "lion"
}

# Retrieve animal information
def get_animal_info(animal_name):
    try:
        # Handle synonyms
        normalized_name = animal_synonyms.get(animal_name, animal_name)
        info = animal_info.get(normalized_name)

        if not info:
            print(f"No information found for animal: {animal_name}")
        return info
    except Exception as e:
        print(f"Error retrieving animal info: {str(e)}")
        return None

