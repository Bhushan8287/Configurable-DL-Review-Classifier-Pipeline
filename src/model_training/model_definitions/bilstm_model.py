from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM
from tensorflow.keras.regularizers import l2
from src.utils.logger import logger_for_bilstm_model, log_component_start, log_component_end
from src.utils.helpers import plot_and_save_model_performance, save_training_metrics_json
from src.config.config_loader import get_config

def bilstm(xtrain_padded_sequences, ytrain, xtest_padded_sequences, ytest):
    """
    Build and train a Bidirectional LSTM (BiLSTM) model for binary text classification.

    Parameters
    ----------
    xtrain_padded_sequences : array-like
        Padded training input sequences.
    ytrain : array-like
        Binary training labels.
    xtest_padded_sequences : array-like
        Padded test input sequences.
    ytest : array-like
        Binary test labels.

    Returns
    -------
    xtest_padded_sequences : array-like
        Unmodified test sequences, returned for consistency in pipeline.
    ytest : array-like
        Unmodified test labels, returned for consistency in pipeline.
    """
    try:
        # Log start of BiLSTM component execution
        log_component_start(logger_for_bilstm_model, 'BILSTM Model Componet')

        # Load model and preprocessing configuration values from config file
        config = get_config()
        embedding_dimensions = config['bilstm_model']['embedding_dim']
        vocab = config['tokenization']['num_of_words']
        max_length = config['tokenization']['max_length']
        units_first = config['bilstm_model']['bilstm_units_first_layer']
        units_second = config['bilstm_model']['bilstm_units_second_layer']
        activation_func = config['bilstm_model']['activation_mid_layer']
        dropout_prob = config['bilstm_model']['dropout']
        adam_optimizer_learning_rate = config['bilstm_model']['adam_optimizer_learning_rate']
        early_stopping_moniter = config['bilstm_model']['early_stopping_moniter']
        early_stopping_patience = config['bilstm_model']['patience']
        model_chckpoint_moniter = config['bilstm_model']['model_checkpoint_moniter']
        model_chckpint_mode = config['bilstm_model']['mode']
        epochs = config['bilstm_model']['epochs']
        batch_size = config['bilstm_model']['batch_size']
        validation_split = config['bilstm_model']['validation_split']
        verbose = config['bilstm_model']['verbose']
        model_save_path = config['bilstm_model']['save_model_path']
        plot_save_path = config['bilstm_model']['save_plot_path']
        save_metrics_path = config['bilstm_model']['save_metrics_path']

        # Define the architecture of the BiLSTM model
        model = Sequential()
        model.add(Embedding(input_dim=vocab, output_dim=embedding_dimensions, input_length=max_length))
        model.add(Bidirectional(LSTM(units=units_first, return_sequences=False, kernel_regularizer=l2(0.001))))
        model.add(Dropout(dropout_prob))
        model.add(Dense(units=units_second, activation=activation_func, kernel_regularizer=l2(0.001)))
        model.add(Dropout(dropout_prob))
        model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

        # Compile the model with binary crossentropy and Adam optimizer
        optimizer = Adam(learning_rate=adam_optimizer_learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Define training callbacks: EarlyStopping and ModelCheckpoint
        early_stopping = EarlyStopping(monitor=early_stopping_moniter, patience=early_stopping_patience, restore_best_weights=True)
        checkpoint_path = model_save_path
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor=model_chckpoint_moniter, mode=model_chckpint_mode)

        # Start training the BiLSTM model
        logger_for_bilstm_model.info('Bilstm model training started')
        history = model.fit(
            xtrain_padded_sequences, ytrain, 
            epochs=epochs, batch_size=batch_size, 
            validation_split=validation_split, 
            callbacks=[early_stopping, checkpoint], 
            verbose=verbose
        )
        logger_for_bilstm_model.info('Model training finished')
        logger_for_bilstm_model.info('Bilstm model saved')

        # Plot and save model training and validation performance metrics
        plot_and_save_model_performance(training_acc=history.history['accuracy'],
                                        val_acc=history.history['val_accuracy'],
                                        traning_loss=history.history['loss'],
                                        val_loss=history.history['val_loss'], 
                                        plot_save_path=plot_save_path,
                                        model_name='BI-LSTM')
        
        # Save training metrics to a JSON file
        save_training_metrics_json(training_acc=history.history['accuracy'],
                                   val_acc=history.history['val_accuracy'],
                                   traning_loss=history.history['loss'],
                                   val_loss=history.history['val_loss'],
                                   save_metrics_path=save_metrics_path)

        # Log successful completion of BiLSTM component
        log_component_end(logger_for_bilstm_model, 'BILSTM Model Componet')
        return xtest_padded_sequences, ytest

    except Exception as bilstm_e:
        # Log any error that occurs during model training
        logger_for_bilstm_model.debug(f'Error encountered during Bilstm model training. error: {bilstm_e}')
        log_component_end(logger_for_bilstm_model, 'BILSTM Model Componet')
        raise
