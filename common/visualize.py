import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(
    date_list: list,
    train_list: list,
    true_list: list,
    predictions: dict,
    window_size: int = 336, 
    forecast_size: int = 96
):
    """
    Plots the true values, train values, and predictions for each window.

    Parameters:
    date_list (list): List of lists containing dates for each window.
    train_list (list): List of lists containing training values for each window.
    true_list (list): List of lists containing true values for each window.
    predictions (dict): Dictionary where keys are model names and values are lists of lists containing predicted values for each window.
    window_size (int): The size of the input window.
    forecast_size (int): The size of the forecast window.

    The function creates a subplot for each window and plots the true values, train values, and predictions for each model.
    """
    num_windows = len(date_list)
    fig, axes = plt.subplots(num_windows, 1, figsize=(10, 5 * num_windows), sharex=False)
    
    if num_windows == 1:
        axes = [axes]
    
    palette = sns.color_palette("tab10", len(predictions))
    model_colors = {model: color for model, color in zip(predictions.keys(), palette)}
    
    for i in range(num_windows):
        ax = axes[i]
        ax.plot(date_list[i][:window_size], train_list[i], label='Train', color='gray')
        ax.plot(date_list[i][-forecast_size:], true_list[i], label='True', color='black')
        
        for model, pred_list in predictions.items():
            ax.plot(date_list[i][-forecast_size:], pred_list[i], label=f'{model}', color=model_colors[model])
        
        ax.legend()
        ax.set_title(f'Window {i+1}')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        
        # Set x-ticks to show only a subset of dates
        ax.set_xticks(date_list[i][::int(len(date_list[i]) / 10)])
    
    plt.xlabel('Date')
    plt.tight_layout(pad=3.0)
    plt.show()
