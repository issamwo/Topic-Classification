from Classifier.constants import *
from Classifier.utils.common import read_yaml, create_directories
from Classifier.entity.config_entity import DataIngestionConfig, DataCleaningConfig,\
      DataPreprocessingConfig, ModelTrainingConfig, ModelEvaluationConfig

import os
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

# Now you can access the variables using os.getenv.
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            storage_bucket_name=config.storage_bucket_name,
            source_blob_name=config.source_blob_name,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    def get_data_cleaning_config(self) -> DataCleaningConfig:
        config = self.config.data_cleaning
        
        create_directories([config.root_dir])

        data_cleaning_config = DataCleaningConfig(
            root_dir=Path(config.root_dir),
            unpreprocessed_data_path=Path(config.unpreprocessed_data_path),
            column_text=config.column_text,
            column_topic=config.column_topic,
            cleaned_data_path=Path(config.cleaned_data_path)
        )

        return data_cleaning_config
    
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        
        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            cleaned_data_path=Path(config.cleaned_data_path),
            preprocessed_spilitted_data_path=Path(config.preprocessed_spilitted_data_path),
            max_words=self.params.MAX_WORDS,
            test_size=self.params.TEST_SIZE,
            topic_names=self.params.TOPIC_NAMES
        )

        return data_preprocessing_config
    

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        
        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            preprocessed_spilitted_data_path=Path(config.preprocessed_spilitted_data_path),
            model_path=Path(config.model_path),
            batch_size=self.params.BATCH_SIZE,
            epochs=self.params.EPOCHS,
            max_words=self.params.MAX_WORDS,
            validation_split=self.params.VALIDATION_SPLIT,
            learning_rate=self.params.LEARNING_RATE,
            beta_1=self.params.BETA_1,
            beta_2=self.params.BETA_2
        )

        return model_training_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_training

        model_evaluation_config = ModelEvaluationConfig(
            path_model=Path(config.model_path),
            preprocessed_spilitted_data_path=Path(config.preprocessed_spilitted_data_path),
            mlflow_uri=MLFLOW_TRACKING_URI,
            model_params=dict(
                batch_size=self.params.BATCH_SIZE,
                epochs=self.params.EPOCHS,
                max_words=self.params.MAX_WORDS,
                validation_split=self.params.VALIDATION_SPLIT,
                learning_rate=self.params.LEARNING_RATE,
                beta_1=self.params.BETA_1,
                beta_2=self.params.BETA_2
            ),
        )
        return model_evaluation_config