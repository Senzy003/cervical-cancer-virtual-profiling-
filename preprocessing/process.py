"""Image preprocessing for cervical cancer H&E patches.

IMPORTANT: these transforms are for PATCHES (e.g. 256 x 256 px tiles cut
from a whole-slide image at 20x), not whole slides. Resizing a whole slide
or a large region down to 224 x 224 destroys the cell-level detail the model
needs. Whole slides are tiled first (CLAM / TRIDENT), then each patch goes
through these transforms.

Pipeline:
    Whole-slide image
        -> tile into 256 x 256 patches at 20x (tissue only)
        -> resize / crop to 224 x 224
        -> H&E-safe augmentation (training only)
        -> tensor + normalize
        -> [N, 3, 224, 224] -> encoder
"""

import io
import random

import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


IMAGE_SIZE = 224

# ImageNet statistics: what EfficientNet, UNI and Phikon expect.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class RandomRotate90:
    """Rotate by 0, 90, 180 or 270 degrees.

    Tissue has no 'up', so right-angle rotations are free augmentation and,
    unlike RandomRotation(10), they leave no black corners.
    """

    def __call__(self, image):
        return TF.rotate(image, random.choice([0, 90, 180, 270]))


def get_training_transforms():
    """H&E-suited augmentation: flips, right-angle rotations and mild
    stain (colour) variation to mimic lab-to-lab staining differences."""

    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotate90(),
        # Hue matters for H&E: pink/purple shifts between labs and scanners
        transforms.ColorJitter(brightness=0.1, contrast=0.1,
                               saturation=0.1, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_validation_transforms():
    """Deterministic transforms for validation and prediction."""

    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def tissue_fraction(image, white_threshold=220):
    """Fraction of pixels that look like tissue rather than white glass."""

    grey = np.asarray(image.convert("L"))
    return float((grey < white_threshold).mean())


def _prepare(image, min_tissue):
    image = image.convert("RGB")
    fraction = tissue_fraction(image)
    if fraction < min_tissue:
        raise ValueError(
            f"Image is only {fraction:.0%} tissue; please upload an H&E "
            f"patch with at least {min_tissue:.0%} tissue."
        )
    # [3, 224, 224] -> [1, 3, 224, 224]: a batch of one image
    return get_validation_transforms()(image).unsqueeze(0)


def preprocess_image(image_path, min_tissue=0.25):
    """Load one patch from disk and prepare it for the model."""

    return _prepare(Image.open(image_path), min_tissue)


def preprocess_image_bytes(data, min_tissue=0.25):
    """Prepare an uploaded patch (raw bytes from the API) for the model."""

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as error:  # PIL raises several error types
        raise ValueError("File is not a readable image.") from error
    return _prepare(image, min_tissue)
