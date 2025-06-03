import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from src.config.config_loader import get_config
import joblib, json
from src.utils.logger import logger_for_tokenizer_text, log_component_start, log_component_end
import pandas as pd

def text_tokenizer(xtrain, ytrain, xtest, ytest):
    """
    Tokenizes and pads text data using TensorFlow's Tokenizer.

    This function:
    - Loads tokenizer and padding configuration from config file.
    - Fits a tokenizer on training data.
    - Transforms both train and test text data into padded sequences.
    - Saves the tokenizer object and max length used for padding.

    Parameters
    ----------
    xtrain : list of str
        List of raw training text data.
    ytrain : list or array-like
        Corresponding training labels.
    xtest : list of str
        List of raw testing text data.
    ytest : list or array-like
        Corresponding testing labels.

    Returns
    -------
    xtrain_padded_sequences : ndarray
        Padded tokenized training sequences.
    ytrain : array-like
        Unchanged training labels.
    xtest : ndarray
        Padded tokenized testing sequences.
    ytest : array-like
        Unchanged testing labels.
    
    Raises
    ------
    Exception
        If any error occurs during tokenization, it is logged and raised.
    """
    try:
        # Start logging for tokenization component
        log_component_start(logger_for_tokenizer_text, 'Tokenize Text Component')
        
        # Load tokenization-related config values
        config = get_config()
        text_column = config['dataset']['text_column']
        num_of_words = config['tokenization']['num_of_words']
        oov_token = config['tokenization']['oov_token']
        max_length = config['tokenization']['max_length']
        padding = config['tokenization']['padding']
        turncating = config['tokenization']['truncating']
        save_tokenizer_obj = config['tokenization']['save_tokenizer_path']
        save_max_length_obj = config['tokenization']['save_max_length']
        
        logger_for_tokenizer_text.info('Text is being tokenized')

        xtrain_texts = xtrain[text_column].astype(str).tolist()
        xtest_texts = xtest[text_column].astype(str).tolist()

        # Create and fit tokenizer
        tokenizer = Tokenizer(num_words=num_of_words, oov_token=oov_token)
        tokenizer.fit_on_texts(xtrain_texts)

        # Convert text to sequences
        xtrain_sequences = tokenizer.texts_to_sequences(xtrain_texts)
        xtest_sequences = tokenizer.texts_to_sequences(xtest_texts)

        # Save tokenizer object
        with open(save_tokenizer_obj, 'wb') as file:
            joblib.dump(tokenizer, file)
        logger_for_tokenizer_text.info('tokenizer object saved')
        
        # Pad sequences to max_length
        xtrain_padded_sequences = pad_sequences(xtrain_sequences, maxlen=max_length, padding=padding, truncating=turncating)
        xtest_padded_sequences = pad_sequences(xtest_sequences, maxlen=max_length, padding=padding, truncating=turncating)

        logger_for_tokenizer_text.info('padding done')

        # Save max_length used for padding
        with open(save_max_length_obj, 'w') as file:
            json.dump(max_length, file, indent=4)
        logger_for_tokenizer_text.info('max_length object saved')
        logger_for_tokenizer_text.info('Tokenization of text completed')

        # End logging for tokenization component
        log_component_end(logger_for_tokenizer_text, 'Tokenize Text Component')

        return xtrain_padded_sequences, ytrain, xtest_padded_sequences, ytest
    
    except Exception as tokenizer_e:
        # Log error and re-raise
        logger_for_tokenizer_text.debug(f'Error encountered during tokenization. error: {tokenizer_e}')
        log_component_end(logger_for_tokenizer_text, 'Tokenize Text Component')
        raise
