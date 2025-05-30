import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging

# Suppress verbose logs from TensorFlow, h5py, and matplotlib
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('h5py').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Logging config function
def get_logging_config():
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(r'C:\Users\BW\Desktop\DL movie review classifier\logs\pipeline_logs.log'),
            logging.StreamHandler()
        ]
    )
    return logging

logging_config = get_logging_config()

def log_component_start(logger, component_name: str):
    """
    Logs a standardized start message for a pipeline component.
    """
    logger.info(f"{'='*10} Starting {component_name} {'='*10}")

def log_component_end(logger, component_name: str):
    """
    Logs a standardized end message for a pipeline component, with spacing.
    """
    logger.info(f"{'='*10} Finished {component_name} {'='*10}\n")


logger_for_loading_data = logging_config.getLogger('Dataset_loading_component')
logger_for_loading_data.setLevel(get_logging_config().DEBUG)

logger_for_config_file = logging_config.getLogger('Loading_config_file_component')
logger_for_config_file.setLevel(get_logging_config().DEBUG)

logger_for_data_cleaning = logging_config.getLogger('Data_cleaning_component')
logger_for_data_cleaning.setLevel(get_logging_config().DEBUG)

logger_for_data_splitting = logging_config.getLogger('Data_splitting_component')
logger_for_data_splitting.setLevel(get_logging_config().DEBUG)

logger_for_tokenizer_text = logging_config.getLogger('Tokenize_text_component')
logger_for_tokenizer_text.setLevel(get_logging_config().DEBUG)

logger_for_model_trainer = logging_config.getLogger('Model_trainer_component')
logger_for_model_trainer.setLevel(get_logging_config().DEBUG)

logger_for_evaluater = logging_config.getLogger('Evaluater_component')
logger_for_evaluater.setLevel(get_logging_config().DEBUG)

logger_for_threshold_finder = logging_config.getLogger('Threshold_finder_component')
logger_for_threshold_finder.setLevel(get_logging_config().DEBUG)

# logger_for_hypertuning = get_logging_config().getLogger('Hyperparameter_tuning_component')
# logger_for_hypertuning.setLevel(get_logging_config().DEBUG)

logger_for_pipeline_code = logging_config.getLogger('Pipeline_component')
logger_for_pipeline_code.setLevel(get_logging_config().DEBUG)