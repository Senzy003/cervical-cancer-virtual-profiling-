"""FastAPI app — the API endpoint(s)."""
"""FastAPI backend for the Virtual Protein Profiling project."""

from fastapi import FastAPI
from model.model_architecture import ProteinExpressionModel


# Create the web application/server
app = FastAPI(
    title="Cervical Cancer Virtual Protein Profiling API",
    description="Research prototype for analysing cervical cancer H&E images.",
    version="0.1.0"
)


# Create the neural network
model = ProteinExpressionModel()

# Prediction mode rather than training mode
model.eval()


@app.get("/")
def home():
    """Check that the API is running."""

    return {
        "message": "Virtual Protein Profiling API is running"
    }


@app.get("/health")
def health():
    """Check that the API and model are available."""

    return {
        "status": "healthy",
        "model": "EfficientNet-B0",
        "targets": [
            "PD-L1",
            "p16",
            "Ki-67"
        ],
        "trained": False
    }