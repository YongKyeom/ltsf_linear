import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, predictions: dict):
    plt.figure(figsize=(14, 7))
    plt.plot(train_data, label="Train")
    plt.plot(val_data, label="Validation")
    plt.plot(test_data, label="Test")

    for model_name, preds in predictions.items():
        plt.plot(preds, label=f"{model_name} Predictions")

    plt.legend()
    plt.show()
