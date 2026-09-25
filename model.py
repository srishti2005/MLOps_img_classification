import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import os

class SimpleImageClassifier(nn.Module):
    """
    Convolutional Neural Network for lightweight image classification.
    Optimized for fast CPU inference in CI environments.
    """
    def __init__(self, num_classes=10):
        super(SimpleImageClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 x 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 8 x 8
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self.classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ImageClassificationService:
    def __init__(self, model_path=None):
        self.device = torch.device("cpu")
        self.model = SimpleImageClassifier(num_classes=10).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

    def preprocess(self, image_input):
        """
        Accepts either a file path, PIL Image, or raw byte stream.
        """
        if isinstance(image_input, (str, bytes, io.BytesIO)):
            if isinstance(image_input, bytes):
                image_input = io.BytesIO(image_input)
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input type.")
            
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image_input):
        """
        Runs inference and returns predicted label, class index, and confidence score.
        """
        tensor = self.preprocess(image_input)
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        idx = predicted_idx.item()
        return {
            "class_index": idx,
            "label": self.model.classes[idx],
            "confidence": round(confidence.item(), 4),
            "all_probabilities": {
                self.model.classes[i]: round(prob.item(), 4)
                for i, prob in enumerate(probabilities)
            }
        }