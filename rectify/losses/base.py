class BaseLossFunction:
    """Base class for loss functions."""
    def __init__(self):
        pass

    def compute(loss, prediction, target):
        raise NotImplementedError("This method should be overridden by subclasses.")