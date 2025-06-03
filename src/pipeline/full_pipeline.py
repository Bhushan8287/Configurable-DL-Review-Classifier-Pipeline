from src.data.data_loader import load_data
from src.pre_processing.data_cleaner import clean_text
from src.pre_processing.data_splitter import split_dataset
from src.pre_processing.tokenize_text import text_tokenizer
from src.model_training.training_dispatcher import dispatch_training
from src.model_training.evaluater import evaluate_model
from src.model_training.roc_thresholding import roc_threshold_metrics
from src.utils.logger import logger_for_pipeline_code
from src.config.config_loader import get_config


def Pipeline():
    """
    Executes the complete machine learning pipeline for text classification.

    The pipeline includes the following steps:
        1. Load raw dataset.
        2. Clean text data.
        3. Split data into training and test sets.
        4. Tokenize and pad text sequences.
        5. Train the model based on the specified architecture.
        6. Evaluate the trained model.
        7. Perform ROC thresholding and compute classification metrics.

    Logging is performed at each stage to track progress and potential issues.

    Raises
    ------
    Exception
        If any component in the pipeline fails, the error is logged and re-raised.
    """
    config = get_config()
    model_to_train = config['model_to_train']
    
    try:
        logger_for_pipeline_code.info('Pipeline execution has started')

        # Step 1: Load the raw dataset from the source
        dataset_loaded = load_data()

        # Step 2: Clean the dataset using predefined text cleaning logic
        dataset_text_cleaned = clean_text(dataset_loaded)

        # Step 3: Split the cleaned dataset into training and testing sets
        X_train, y_train, X_test, y_test = split_dataset(dataset_text_cleaned)

        # Step 4: Tokenize the text and convert to padded sequences
        X_train_padded_sequences, y_train, X_test_padded_sequences, y_test = text_tokenizer(
            xtrain=X_train, 
            ytrain=y_train, 
            xtest=X_test, 
            ytest=y_test)

        # Step 5: Train the model using the specified architecture
        X_test_padded_sequences, y_test, model_to_use = dispatch_training(
            xtrain_padded_sequences=X_train_padded_sequences,
            ytrain=y_train, 
            xtest_padded_sequences=X_test_padded_sequences, 
            ytest=y_test,
            model_to_use=model_to_train)
        
        # Step 6: Evaluate the model performance and get best model and its name
        X_test_padded_sequences, y_test, model_name, best_model = evaluate_model(
            xtest_padded_sequences=X_test_padded_sequences, 
            ytest=y_test, 
            model_name=model_to_use)
        
        # Step 7: Perform ROC analysis and save classification metrics
        roc_threshold_metrics(
            xtest_padded_sequences=X_test_padded_sequences, 
            ytest=y_test, 
            model_name=model_name, 
            best_model=best_model)

        logger_for_pipeline_code.info('Pipeline execution has finished')
    
    except Exception as pl_e:
        # Log if any step in the pipeline fails
        logger_for_pipeline_code.debug(f'Error encountered during execution of the pipeline. error {pl_e}')
        raise
