from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from src.utils.helpers import plot_and_save_model_performance, save_training_metrics_json
from src.utils.logger import logger_for_gru_model, log_component_start, log_component_end
from src.config.config_loader import get_config

def gru(xtrain_padded_sequences, ytrain, xtest_padded_sequences, ytest):
    """
    Constructs and trains a Gated Recurrent Unit (GRU) based RNN model for binary text classification.

    This function builds a sequential neural network model with embedding and GRU layers, trains it using
    the specified hyperparameters from the configuration file, and logs the process including model
    performance plotting and metric saving.

    Parameters
    ----------
    xtrain_padded_sequences : array-like
        Tokenized and padded input sequences for training.
    ytrain : array-like
        Corresponding binary labels (0 or 1) for the training sequences.
    xtest_padded_sequences : array-like
        Tokenized and padded input sequences for testing.
    ytest : array-like
        Corresponding binary labels for the testing sequences.

    Returns
    -------
    xtest_padded_sequences : array-like
        Test input sequences (padded), returned for consistency and further evaluation.
    ytest : array-like
        Ground truth test labels, returned for consistency and further evaluation.
    """
    try:
        # Log the start of the GRU component
        log_component_start(logger_for_gru_model, 'GRU RNN Model Component')

        # Load all model and training hyperparameters from configuration
        config = get_config()
        embedding_dimensions = config['gru_rnn_model']['embedding_dim']
        vocab = config['tokenization']['num_of_words']
        max_length = config['tokenization']['max_length']
        units_first = config['gru_rnn_model']['grurnn_units_first_layer']
        units_second = config['gru_rnn_model']['grurnn_units_second_layer']
        activation_func = config['gru_rnn_model']['activation_mid_layer']
        dropout_prob = config['gru_rnn_model']['dropout']
        adam_optimizer_learning_rate = config['gru_rnn_model']['adam_optimizer_learning_rate']
        early_stopping_moniter = config['gru_rnn_model']['early_stopping_moniter']
        early_stopping_patience = config['gru_rnn_model']['patience']
        model_chckpoint_moniter = config['gru_rnn_model']['model_checkpoint_moniter']
        model_chckpint_mode = config['gru_rnn_model']['mode']
        epochs = config['gru_rnn_model']['epochs']
        batch_size = config['gru_rnn_model']['batch_size']
        validation_split = config['gru_rnn_model']['validation_split']
        verbose = config['gru_rnn_model']['verbose']
        model_save_path = config['gru_rnn_model']['save_model_path']
        plot_save_path = config['gru_rnn_model']['save_plot_path']
        save_metrics_path = config['gru_rnn_model']['save_metrics_path']

        # Build the GRU-based Sequential model
        model = Sequential()
        model.add(Embedding(input_dim=vocab, output_dim=embedding_dimensions, input_length=max_length))
        model.add(GRU(units=units_first, return_sequences=False))  # GRU layer with no return sequences
        model.add(Dropout(dropout_prob))  # Dropout to reduce overfitting
        model.add(Dense(units=units_second, activation=activation_func))  # Dense hidden layer
        model.add(Dropout(dropout_prob))  # Another dropout layer
        model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

        # Compile the model with Adam optimizer and binary crossentropy loss
        optimizer = Adam(learning_rate=adam_optimizer_learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Define callbacks: EarlyStopping and ModelCheckpoint
        early_stopping = EarlyStopping(monitor=early_stopping_moniter, patience=early_stopping_patience, restore_best_weights=True)
        checkpoint = ModelCheckpoint(filepath=model_save_path, save_best_only=True, monitor=model_chckpoint_moniter, mode=model_chckpint_mode)

        # Train the model (restricted to first 1000 samples for speed or debugging)
        logger_for_gru_model.info('GRU model training started')
        history = model.fit(
            xtrain_padded_sequences[:1000], ytrain[:1000], 
            epochs=epochs, batch_size=batch_size, 
            validation_split=validation_split, 
            callbacks=[early_stopping, checkpoint], 
            verbose=verbose
        )
        logger_for_gru_model.info('Model training finished')
        logger_for_gru_model.info('GRU model saved')

        # Plot and save the training and validation metrics
        plot_and_save_model_performance(training_acc=history.history['accuracy'],
                                        val_acc=history.history['val_accuracy'],
                                        traning_loss=history.history['loss'],
                                        val_loss=history.history['val_loss'], 
                                        plot_save_path=plot_save_path,
                                        model_name='GRU-RNN')
        
        # Save training metrics in JSON format
        save_training_metrics_json(training_acc=history.history['accuracy'],
                                   val_acc=history.history['val_accuracy'],
                                   traning_loss=history.history['loss'],
                                   val_loss=history.history['val_loss'],
                                   save_metrics_path=save_metrics_path)

        # Log the end of the GRU component
        log_component_end(logger_for_gru_model, 'GRU RNN Model Component')
        return xtest_padded_sequences, ytest

    except Exception as gru_e:
        # Log the exception details and safely terminate component
        logger_for_gru_model.debug(f'Error encountered during GRU model training. error: {gru_e}')
        log_component_end(logger_for_gru_model, 'GRU RNN Model Component')
        raise
