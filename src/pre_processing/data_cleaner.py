import re
from src.utils.logger import logger_for_data_cleaning, log_component_start, log_component_end
from src.config.config_loader import get_config

def clean_text(dataset):
    """
    Cleans the text data in the dataset for NLP tasks by applying multiple preprocessing steps such as 
    lowercasing, removing HTML tags, URLs, mentions, hashtags, special characters, and extra whitespace.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The input dataset containing a text column to be cleaned. The column name is fetched from the config file.

    Returns
    -------
    pandas.DataFrame
        The cleaned dataset with the same structure as the input but with the text column processed.

    Raises
    ------
    Exception
        If any error occurs during the cleaning process, it is logged and raised for debugging purposes.

    Notes
    -----
    - The function relies on external configuration from a YAML file using `get_config()`.
    - Logging is used extensively to track each step of the cleaning process.
    - The cleaned dataset is saved to a file path defined in the configuration.
    """
    config = get_config()
    dataset_text_column = config['dataset']['text_column']
    cleaned_dataset_save_path = config['dataset']['cleaned_dataset_save_path']
    try:
        log_component_start(logger_for_data_cleaning, 'Data Cleaning Component')

        # Lowercase
        dataset[dataset_text_column] = dataset[dataset_text_column].apply(lambda text: text.lower())
        logger_for_data_cleaning.info('Text converted to lower case')

        # Remove HTML tags
        dataset[dataset_text_column] = dataset[dataset_text_column].apply(lambda text: re.sub(r'<.*?>', '', text))
        logger_for_data_cleaning.info('HTML tags removed from text')

        # Remove URLs
        dataset[dataset_text_column] = dataset[dataset_text_column].apply(lambda text: re.sub(r'http\S+|www\.\S+', '', text))
        logger_for_data_cleaning.info('URls removed from text')

        # Remove @mentions
        dataset[dataset_text_column] = dataset[dataset_text_column].apply(lambda text: re.sub(r'@\w+', '', text))
        logger_for_data_cleaning.info('Mentions removed form text')

        # Remove hashtags (keep the word)
        dataset[dataset_text_column] = dataset[dataset_text_column].apply(lambda text: re.sub(r'#', '', text))
        logger_for_data_cleaning.info('Hastags sign removed from text')

        # Remove special characters (non-alphabetic, except space)
        dataset[dataset_text_column] = dataset[dataset_text_column].apply(lambda text: re.sub(r'[^a-zA-Z\s]', '', text))
        logger_for_data_cleaning.info('Special characters removed form text')

        # Remove extra whitespace
        dataset[dataset_text_column] = dataset[dataset_text_column].apply(lambda text: re.sub(r'\s+', ' ', text).strip())
        logger_for_data_cleaning.info('Extra white spaces removed from the text')

        dataset.to_csv(cleaned_dataset_save_path, index=False)
        logger_for_data_cleaning.info('Cleaned dataset file saved')

        log_component_end(logger_for_data_cleaning, 'Data Cleaning Component')

        return dataset
    
    except Exception as dcl_e:
        logger_for_data_cleaning.debug(f'Error encountered during data cleaning. error:{dcl_e}')
        log_component_end(logger_for_data_cleaning, 'Data Cleaning Component')
        raise
