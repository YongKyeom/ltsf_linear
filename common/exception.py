class DataLoaderException(Exception):
    """Exception raised for errors in the data loading process."""

    pass


class ModelTrainingException(Exception):
    """Exception raised for errors during model training."""

    pass


class HyperoptException(Exception):
    """Exception raised for errors during hyperparameter optimization."""

    pass
