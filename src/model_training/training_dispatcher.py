from src.utils.logger import logger_for_dispatch_training, log_component_start, log_component_end

def dispatch_training(xtrain_padded_sequences, ytrain, xtest_padded_sequences, ytest, model_to_use):
    """
    Dispatch and train the selected model (BiLSTM, LSTM, or GRU) for text classification.

    Parameters
    ----------
    xtrain_padded_sequences : array-like
        Padded input sequences for training.
    ytrain : array-like
        Labels corresponding to the training data.
    xtest_padded_sequences : array-like
        Padded input sequences for testing.
    ytest : array-like
        Labels corresponding to the test data.
    model_to_use : str
        Identifier for the model to use. Expected values are:
        - 'bilstm'
        - 'lstm'
        - 'gru_rnn'

    Returns
    -------
    xtest_padded_sequences : array-like
        Unchanged test sequences, returned for evaluation or further processing.
    ytest : array-like
        Unchanged test labels, returned for evaluation or further processing.

    Raises
    ------
    Exception
        If an error occurs during model dispatching or training.
    """
    try:
        # Log the start of the model dispatching component
        log_component_start(logger_for_dispatch_training, 'Dispatcher Training Component')

        # If user selected BiLSTM model
        if model_to_use == 'bilstm':
            from src.model_training.model_definitions.bilstm_model import bilstm
            xtest_padded_sequences, ytest = bilstm(xtrain_padded_sequences, ytrain, xtest_padded_sequences, ytest)
            logger_for_dispatch_training.info('Bilstm model was trained and test splits were returned for evaluation')
            log_component_end(logger_for_dispatch_training, 'Dispatcher Training Component')
            return xtest_padded_sequences, ytest

        # If user selected LSTM model
        elif model_to_use == 'lstm':
            from src.model_training.model_definitions.lstm_model import lstm
            xtest_padded_sequences, ytest = lstm(xtrain_padded_sequences, ytrain, xtest_padded_sequences, ytest)
            logger_for_dispatch_training.info('Lstm model was trained and test splits were returned for evaluation')
            log_component_end(logger_for_dispatch_training, 'Dispatcher Training Component')
            return xtest_padded_sequences, ytest

        # If user selected GRU model
        elif model_to_use == 'gru_rnn':
            from src.model_training.model_definitions.gru_model import gru
            xtest_padded_sequences, ytest = gru(xtrain_padded_sequences, ytrain, xtest_padded_sequences, ytest)
            logger_for_dispatch_training.info('Gru rnn model was trained and test splits were returned for evaluation')
            log_component_end(logger_for_dispatch_training, 'Dispatcher Training Component')
            return xtest_padded_sequences, ytest

    except Exception as dispatch_er:
        # Log any error encountered during model dispatch
        logger_for_dispatch_training.debug(f'Error encountered during dispatching. error: {dispatch_er}')
        log_component_end(logger_for_dispatch_training, 'Dispatcher Training Component')
