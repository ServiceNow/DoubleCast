from .train import main as train
from .train_staged import main as train_staged


__all__ = [
    "train",
    "train_staged",
]
