from src.data.data_loader import load_data
from src.pre_processing.data_cleaner import clean_text
from src.pre_processing.data_splitter import split_dataset
from src.pre_processing.tokenize_text import text_tokenizer
from src.model_training.training_dispatcher import dispatch_training
from src.utils.logger import logger_for_pipeline_code
from src.config.config_loader import get_config


def Pipeline():

    config = get_config()
    # update the below to use different model
    model_to_train = config['model_to_train']
    
    try:
        logger_for_pipeline_code.info('Pipeline execution has started')

        dataset_loaded = load_data()

        dataset_text_cleaned = clean_text(dataset_loaded)

        X_train, y_train, X_test, y_test = split_dataset(dataset_text_cleaned)

        X_train_padded_sequences, y_train, X_test_padded_sequences, y_test = text_tokenizer(
            xtrain=X_train, 
            ytrain=y_train, 
            xtest=X_test, 
            ytest=y_test)

        xtest_padded_sequences, ytest = dispatch_training(
            xtrain_padded_sequences=X_train_padded_sequences,
            ytrain=y_train, 
            xtest_padded_sequences=X_test_padded_sequences, 
            ytest=y_test,
            model_to_use=model_to_train)
        


        logger_for_pipeline_code.info('Pipeline execution has finished')
    except Exception as pl_e:
        # Log if any step in the pipeline fails
        logger_for_pipeline_code.debug(f'Error encountered during execution of the pipeline. error {pl_e}')