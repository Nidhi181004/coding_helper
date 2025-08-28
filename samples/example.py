import math

def process_data(xs):
    """Normalize a list and compute z-scores."""
    if not xs:
        return []
    mean = sum(xs) / len(xs)
    var = sum((x - mean)**2 for x in xs) / max(1, len(xs)-1)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(x - mean) / std for x in xs]

def train_model(data):
    """Fake training: returns a 'model' that predicts the mean."""
    if not data:
        raise ValueError("Empty dataset")
    mean = sum(data) / len(data)
    def model(x):
        return mean
    return model
