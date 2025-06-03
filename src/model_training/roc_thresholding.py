from src.utils.logger import logger_for_threshold_finder, log_component_start, log_component_end
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from src.config.config_loader import get_config
from matplotlib import pyplot as plt
import numpy as np
import os, json

def roc_threshold_metrics(xtest_padded_sequences, ytest, model_name, best_model):
    """
    Compute ROC curve, find the optimal classification threshold, evaluate model metrics using that threshold, 
    and save the results including metrics, ROC curve image, and optimal threshold value.

    Parameters
    ----------
    xtest_padded_sequences : array-like
        The padded input sequences used for testing the model.
    
    ytest : array-like
        The true binary labels for the test set.

    model_name : str
        Name of the model, used for naming saved files.

    best_model : keras.Model
        Trained Keras model used to make predictions.

    Returns
    -------
    None
        The function performs file I/O operations (saving plots and metrics) and does not return any value.

    Raises
    ------
    Exception
        Logs and raises any exceptions encountered during the thresholding and evaluation process.
    """
    try:
        # Log the start of the ROC thresholding component
        log_component_start(logger_for_threshold_finder, 'ROC Thresholding Component')

        # Load paths and config values from configuration
        config = get_config()
        save_plot_path = config['roc_curve_plot_savepath']
        save_optimal_threshold_path = config['optimal_threshold_value_path']
        save_metrics_path = config['classification_metrics_save_path']

        # Generate prediction probabilities for the test set
        logger_for_threshold_finder.info(f'Getting predictions using the trained {model_name} model')
        y_pred_proba = best_model.predict(xtest_padded_sequences, batch_size=128, verbose=0)
        logger_for_threshold_finder.info('Prediction process completed succesfully')

        # Compute false positive rate (fpr), true positive rate (tpr), and thresholds for ROC
        logger_for_threshold_finder.info('Computing (fpr, tpr and thresholds)')
        fpr, tpr, thresholds = roc_curve(ytest, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot and save the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend()
        plot_file_name = f'{model_name}_roc_curve.png'
        full_plot_path = os.path.join(save_plot_path, plot_file_name)
        plt.savefig(full_plot_path, dpi=300, bbox_inches="tight")
        logger_for_threshold_finder.info('ROC curve plot saved succesfully')
        
        # Calculate the optimal threshold using Youden’s J statistic
        logger_for_threshold_finder.info('Finding optimal threshold')
        j_scores = tpr - fpr
        best_threshold = float(thresholds[np.argmax(j_scores)])
        threshold_file_name = f'{model_name}_optimal_threshold.json'
        threshold_save_path = os.path.join(save_optimal_threshold_path, threshold_file_name)
        with open(threshold_save_path, 'w') as file:
            json.dump(best_threshold, file)
        logger_for_threshold_finder.info('Threshold value saved succesfully')

        # Apply the optimal threshold and evaluate classification metrics
        logger_for_threshold_finder.info('Using the found optimal threshold to classify predictions and calculating metrics')
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        accuracy = accuracy_score(ytest, y_pred)
        precision = precision_score(ytest, y_pred)
        recall = recall_score(ytest, y_pred)
        f1 = f1_score(ytest, y_pred)
        classif_report = classification_report(ytest, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(ytest, y_pred)
        logger_for_threshold_finder.info('Metrics computed succesfully')

        # Structure all metrics into a dictionary for export
        classification_metrics_dict = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_score": f1,
            "Classification_report": classif_report,
            "Confusion_matrix": conf_matrix.tolist()  # Convert numpy array to list for JSON compatibility
        }

        # Save metrics to a JSON file
        classif_metrics_file_name = f'Classification_metrics_for_{model_name}.json'
        metrics_save_path = os.path.join(save_metrics_path, classif_metrics_file_name)
        with open(metrics_save_path, 'w') as file:
            json.dump(classification_metrics_dict, file)
        logger_for_threshold_finder.info('Mertics saved succesfully')

        # Log the end of the component
        log_component_end(logger_for_threshold_finder, 'ROC Thresholding Component')

    except Exception as rc_e:
        # Log and raise any exception encountered
        logger_for_threshold_finder.debug(f'Error encountered during finding optimal threshold. error: {rc_e}')
        log_component_end(logger_for_threshold_finder, 'ROC Thresholding Component')
        raise
