from src.utils.logger import logger_for_evaluater, log_component_start, log_component_end
from src.config.config_loader import get_config
from tensorflow.keras.models import load_model
import json
import os

def evaluate_model(xtest_padded_sequences, ytest, model_name):
    """
    Evaluate a trained Keras model on test data and save the evaluation metrics to a JSON file.

    This function loads a model specified in the configuration, evaluates it on the provided
    test data (sequences and labels), logs the evaluation process, and saves the metrics
    (accuracy and loss) as a JSON file. It returns the original inputs along with the
    loaded model object for downstream usage.

    Parameters
    ----------
    xtest_padded_sequences : array-like
        Preprocessed and padded input test sequences.
    ytest : array-like
        True labels for the input test data.
    model_name : str
        Name of the model used to construct the evaluation metrics file name.

    Returns
    -------
    xtest_padded_sequences : array-like
        Input test sequences, returned unchanged for pipeline consistency.
    ytest : array-like
        Ground truth labels, returned unchanged for pipeline consistency.
    model_name : str
        Model name used in the evaluation context, returned unchanged.
    best_model : keras.Model
        The loaded and evaluated Keras model instance.

    Raises
    ------
    Exception
        Any exception that occurs during evaluation is logged and re-raised.
    """
    try:
        # Log the start of the model evaluation component
        log_component_start(logger_for_evaluater, 'Model Evaluater Component')

        # Load configuration parameters
        config = get_config()
        save_metrics_path = config['model_evaluation_metrics_path']
        model_to_load = config['model_to_load_for_evaluation']
        batch_size = config['batch_size_to_use_for_evaluation']

        # Load the pre-trained Keras model specified in the config
        best_model = load_model(model_to_load)

        # Evaluate the model on the provided test data
        test_loss, test_accuracy = best_model.evaluate(
            xtest_padded_sequences, ytest, batch_size=batch_size, verbose=0)

        logger_for_evaluater.info('Model evaluated')

        # Create a dictionary of evaluation metrics
        evaluation_metrics = {
            'Test_accuracy': test_accuracy,
            'Test_loss': test_loss
        }

        # Define path and filename to store the metrics JSON
        metrics_file_name = f'{model_name}_evaluation_metrics_using_default_threshold.json'
        full_metrics_path = os.path.join(save_metrics_path, metrics_file_name)

        # Write metrics to a JSON file
        with open(full_metrics_path, 'w') as file:
            json.dump(evaluation_metrics, file)

        logger_for_evaluater.info(f'Evaluation metrics saved for model: {model_name}')

        # Log the end of the model evaluation component
        log_component_end(logger_for_evaluater, 'Model Evaluater Component')

        # Return original inputs and the evaluated model
        return xtest_padded_sequences, ytest, model_name, best_model

    except Exception as evl_e:
        # Log and re-raise any exception encountered during evaluation
        logger_for_evaluater.debug(f'Error encountered during evaluating model. error: {evl_e}')
        log_component_end(logger_for_evaluater, 'Model Evaluater Component')
        raise
