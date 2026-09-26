"""Image preprocessing for cervical cancer H&E images."""

from torchvision import transforms
from PIL import Image


IMAGE_SIZE = 224


def get_training_transforms():
    """Prepare images for model training."""

    return transforms.Compose([
        # Make every image the same size
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        # Data augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),

        # Slightly change colour/brightness
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),

        # Convert image into numbers PyTorch understands
        transforms.ToTensor(),

        # Normalize for pretrained EfficientNet
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_validation_transforms():
    """Prepare images for validation and prediction."""

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(image_path):
    """Load one image and prepare it for EfficientNet."""

    # Open image and ensure it has RGB colour channels
    image = Image.open(image_path).convert("RGB")

    # Use validation transforms because we are predicting,
    # not training.
    transform = get_validation_transforms()

    image = transform(image)

    # Before: [3, 224, 224]
    # After:  [1, 3, 224, 224]
    # The 1 means we currently have one image in the batch.
    image = image.unsqueeze(0)

    return image
"""Original H&E image
        ↓
 Resize to 224 × 224
        ↓
Data augmentation (training only)
        ↓
 Convert pixels to numbers
        ↓
      Normalize
        ↓
 [1, 3, 224, 224]
        ↓
   EfficientNet"""
