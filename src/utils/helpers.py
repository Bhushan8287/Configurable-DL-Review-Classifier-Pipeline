from matplotlib import pyplot as plt
from src.utils.logger import logger_for_helper_functions
import json

def plot_and_save_model_performance(training_acc, val_acc, traning_loss, val_loss, plot_save_path, model_name):
    """
    Plot and save training/validation accuracy and loss curves for a given model.

    Parameters
    ----------
    training_acc : list or array-like
        Accuracy values recorded during training over epochs.
    val_acc : list or array-like
        Accuracy values recorded during validation over epochs.
    traning_loss : list or array-like
        Loss values recorded during training over epochs.
    val_loss : list or array-like
        Loss values recorded during validation over epochs.
    plot_save_path : str
        File path where the resulting performance plot should be saved.
    model_name : str
        Name of the model being plotted, used in the plot titles.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If plotting or saving fails, an error is logged and raised.
    """
    try:
        plt.figure(figsize=(12, 4))  # Create a figure with specified width and height

        # Subplot 1: Training vs Validation Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(training_acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title(f'Accuracy Over Epochs {model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Subplot 2: Training vs Validation Loss
        plt.subplot(1, 2, 2)
        plt.plot(traning_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title(f'Loss Over Epochs {model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        

        # Save the plot to the specified file path
        plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
        logger_for_helper_functions.info('Plot saved')

    except Exception as plt_e:
        # Log and raise any exceptions encountered during plotting/saving
        logger_for_helper_functions.debug(f'Error encountered during plotting/saving. error: {plt_e}')
        raise


def save_training_metrics_json(training_acc, val_acc, traning_loss, val_loss, save_metrics_path):
    """
    Save training and validation metrics to a JSON file.

    Parameters
    ----------
    training_acc : list or array-like
        Accuracy values recorded during training over epochs.
    val_acc : list or array-like
        Accuracy values recorded during validation over epochs.
    traning_loss : list or array-like
        Loss values recorded during training over epochs.
    val_loss : list or array-like
        Loss values recorded during validation over epochs.
    save_metrics_path : str
        File path where the training metrics JSON should be saved.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If saving to JSON fails, an error is logged and raised.
    """
    try:
        # Construct a dictionary with metrics to be saved
        training_metrics_dict = {
            "training_acc": training_acc,
            "validation_acc": val_acc,
            "training_loss": traning_loss,
            "validation_loss": val_loss,
        }

        # Write the dictionary to a JSON file
        with open(save_metrics_path, 'w') as f:
            json.dump(training_metrics_dict, f, indent=4)

        logger_for_helper_functions.info('Metrics saved')

    except Exception as metrics_e:
        # Log and raise any exceptions encountered during JSON file saving
        logger_for_helper_functions.debug(f'Error encountered during saving training metrics. error: {metrics_e}')
        raise
