import pandas as pd
from src.config.config_loader import load_config_file, get_config
from src.utils.logger import logger_for_loading_data, log_component_start, log_component_end


def load_data():
    """
    Loads the dataset for the pipeline.

    This function retrieves the dataset file path from the project's configuration file (`config.yaml`) using the `get_config` function.
    It then reads the dataset into a pandas DataFrame using `pd.read_csv()` 
    and returns the DataFrame.

    Logging is used to track the status and shape of the loaded dataset.

    Returns:
        pd.DataFrame: The dataset loaded from the file specified in the configuration.

    Raises:
        Exception: If the dataset cannot be loaded due to issues such as an invalid path, missing file, or file format error,
        the exception is logged and re-raised.
    """
    try: 
        # Log success and shape of the dataset
        log_component_start(logger_for_loading_data, 'Loading Data Component')

        # Load the configuration file
        config = get_config()

        # Log that the dataset loading process has started
        logger_for_loading_data.info('The dataset is being loaded')

        # Get the dataset path from the config file
        dataset_path = config['dataset']['path']

        # Read the CSV dataset into a pandas DataFrame
        data_df = pd.read_csv(dataset_path)

        logger_for_loading_data.info('Dataset has been loaded and is now sent for cleaning')
        logger_for_loading_data.info(f'The dataset shape is: {data_df.shape}')
        log_component_end(logger_for_loading_data, 'Loading Data Component')

        # Return the loaded DataFrame
        return data_df
    
    except Exception as e:
        # Log the exception and halt pipeline execution
        logger_for_loading_data.debug(f'There was error: {e} when trying to load the dataset')
        log_component_end(logger_for_loading_data, 'Loading Data Component')
        raise