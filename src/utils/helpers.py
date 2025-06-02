from matplotlib import pyplot as plt
from src.utils.logger import logger_for_helper_functions
import json

def plot_and_save_model_performance(training_acc, val_acc, traning_loss, val_loss, plot_save_path, model_name):
    try:
        plt.figure(figsize=(12, 4))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(training_acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title(f'Accuracy Over Epochs {model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(traning_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title(f'Loss Over Epochs {model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
        logger_for_helper_functions.info('Plot saved')

    except Exception as plt_e:
        logger_for_helper_functions.debug(f'Error encountered during plotting/saving. error: {plt_e}')
        raise


def save_training_metrics_json(training_acc, val_acc, traning_loss, val_loss, save_metrics_path):
    try:
        training_metrics_dict = {
            "training_acc": training_acc,
            "validation_acc": val_acc,
            "training_loss": traning_loss,
            "validation_loss": val_loss,
            }
        with open(save_metrics_path, 'w') as f:
            json.dump(training_metrics_dict, f, indent=4)
        logger_for_helper_functions.info('Metrics saved')
    
    except Exception as metrics_e:
        logger_for_helper_functions.debug(f'Error encountered during saving training metrics. error: {metrics_e}')
        raise