from .random_classifier import RandomClassifier
from .simple_cnn_classifier import SimpleCNNClassifier

def get_model(model_name):
    if model_name == "random":
        return RandomClassifier()
    elif model_name == "simple_cnn":
        return SimpleCNNClassifier()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
