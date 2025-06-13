from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from src.utils.helpers import plot_and_save_model_performance, save_training_metrics_json
from src.utils.logger import logger_for_lstm_model, log_component_start, log_component_end
from src.config.config_loader import get_config

def lstm(xtrain_padded_sequences, ytrain, xtest_padded_sequences, ytest):
    """
    Builds, trains, and saves an LSTM-based neural network model for binary text classification.

    Parameters
    ----------
    xtrain_padded_sequences : array-like
        Padded input sequences for training data.
    ytrain : array-like
        Corresponding binary labels for the training data.
    xtest_padded_sequences : array-like
        Padded input sequences for testing data.
    ytest : array-like
        Corresponding binary labels for the test data.

    Returns
    -------
    xtest_padded_sequences : array-like
        Returned unchanged for consistency with pipeline interfaces.
    ytest : array-like
        Returned unchanged for consistency with pipeline interfaces.
    """
    try:
        # Log component start
        log_component_start(logger_for_lstm_model, 'LSTM Model Componet')

        # Load configuration values
        config = get_config()
        embedding_dimensions = config['lstm_model']['embedding_dim']
        vocab = config['tokenization']['num_of_words']
        max_length = config['tokenization']['max_length']
        units_first = config['lstm_model']['lstm_units_first_layer']
        units_second = config['lstm_model']['lstm_units_second_layer']
        activation_func = config['lstm_model']['activation_mid_layer']
        dropout_prob = config['lstm_model']['dropout']
        adam_optimizer_learning_rate = config['lstm_model']['adam_optimizer_learning_rate']
        early_stopping_moniter = config['lstm_model']['early_stopping_moniter']
        early_stopping_patience = config['lstm_model']['patience']
        model_chckpoint_moniter = config['lstm_model']['model_checkpoint_moniter']
        model_chckpint_mode = config['lstm_model']['mode']
        epochs = config['lstm_model']['epochs']
        batch_size = config['lstm_model']['batch_size']
        validation_split = config['lstm_model']['validation_split']
        verbose = config['lstm_model']['verbose']
        model_save_path = config['lstm_model']['save_model_path']
        plot_save_path = config['lstm_model']['save_plot_path']
        save_metrics_path = config['lstm_model']['save_metrics_path']

        # Define the Sequential LSTM model architecture
        model = Sequential()
        model.add(Embedding(input_dim=vocab, output_dim=embedding_dimensions, input_length=max_length))
        model.add(LSTM(units=units_first, return_sequences=False))
        model.add(Dropout(dropout_prob))
        model.add(Dense(units=units_second, activation=activation_func))
        model.add(Dropout(dropout_prob))
        model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

        # Compile the model with Adam optimizer and binary crossentropy loss
        optimizer = Adam(learning_rate=adam_optimizer_learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Define callbacks: early stopping and model checkpointing
        early_stopping = EarlyStopping(monitor=early_stopping_moniter, patience=early_stopping_patience, restore_best_weights=True)
        checkpoint_path = model_save_path
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor=model_chckpoint_moniter, mode=model_chckpint_mode)

        # Start training the model
        logger_for_lstm_model.info('LSTM model training started')
        history = model.fit(
            xtrain_padded_sequences, ytrain, 
            epochs=epochs, batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, checkpoint],
            verbose=verbose
        )
        logger_for_lstm_model.info('Model training finished')
        logger_for_lstm_model.info('Lstm model saved')

        # Save performance plots and metrics
        plot_and_save_model_performance(training_acc=history.history['accuracy'],
                                        val_acc=history.history['val_accuracy'],
                                        traning_loss=history.history['loss'],
                                        val_loss=history.history['val_loss'],
                                        plot_save_path=plot_save_path,
                                        model_name='LSTM')

        save_training_metrics_json(training_acc=history.history['accuracy'],
                                   val_acc=history.history['val_accuracy'],
                                   traning_loss=history.history['loss'],
                                   val_loss=history.history['val_loss'],
                                   save_metrics_path=save_metrics_path)

        # Log component completion
        log_component_end(logger_for_lstm_model, 'LSTM Model Componet')
        return xtest_padded_sequences, ytest

    except Exception as bilstm_e:
        # Log error during model training
        logger_for_lstm_model.debug(f'Error encountered during lstm model training. error: {bilstm_e}')
        log_component_end(logger_for_lstm_model, 'LSTM Model Componet')
        raise
