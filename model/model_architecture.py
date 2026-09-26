"""The CNN/model definition."""
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
#EfficientNet looks at the pixels and gradually converts them into useful numerical features.

class ProteinExpressionModel(nn.Module):

    def __init__(self, num_proteins=3):
        super().__init__()

        # Load EfficientNet trained on ImageNet
        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)

        # Number of features entering the final classifier
        input_features = self.model.classifier[1].in_features

        # Replace original ImageNet classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(input_features, num_proteins)
        )

    def forward(self, x):
        return self.model(x)
    
"""Original H&E image
        ↓
convert RGB
        ↓
resize 224 × 224
        ↓
convert pixels to numbers
        ↓
normalize
        ↓
PyTorch tensor
convert to something a nueral network can understand"""