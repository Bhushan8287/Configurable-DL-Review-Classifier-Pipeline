from sklearn.model_selection import train_test_split
from src.config.config_loader import get_config
from src.utils.logger import log_component_start, log_component_end, logger_for_data_splitting


def split_dataset(cleaned_datadf):
    """
    Splits the input dataset into training and testing sets for features (X) and target (y).

    The function performs the following:
        1. Loads the test size and random state from `config.yaml` via `get_config()`.
        2. Splits the input DataFrame into:
           - Features (X): All columns except 'Daily_Revenue'.
           - Target (y): The 'Daily_Revenue' column.
        3. Performs a train-test split using scikit-learn's `train_test_split`.

    Logging tracks each stage of the splitting process including the shapes of the resulting splits.

    Args:
        loaded_datadf (pd.DataFrame): The raw dataset loaded from CSV.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) — all as pandas DataFrames/Series.

    Raises:
        Exception: If the split fails, logs the error and re-raises the exception.
    """
    try:
        log_component_start(logger_for_data_splitting, 'Splitting Data Component')
        # Load config values for splitting
        config = get_config()
        test_size = config['dataset_splitting']['test_size']
        random_state = config['dataset_splitting']['random_state']
        target_feature_name = config['dataset']['label_column']
        
        logger_for_data_splitting.info('The dataset is now being split into "X", "y" and "train" and "test" splits')

        # Separate features and target
        X = cleaned_datadf.drop(columns=target_feature_name)
        y = cleaned_datadf[target_feature_name]

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        logger_for_data_splitting.info(f'Split shapes X_train: {X_train.shape}, X_test: {X_test.shape}, 'f'y_train: {y_train.shape}, y_test: {y_test.shape}')
        logger_for_data_splitting.info('The dataset has been successfully split into training and testing sets and are returned')

        log_component_end(logger_for_data_splitting, 'Splitting Data Component')
        return X_train, y_train, X_test, y_test
    
    # if error is encountered it will be logged and the pipeline flow will halt(stop)
    except Exception as split_e:
        logger_for_data_splitting.debug(f'Error encountered when tried to split the dataset. error {split_e}')
        log_component_end(logger_for_data_splitting, 'Splitting Data Component')
        raise