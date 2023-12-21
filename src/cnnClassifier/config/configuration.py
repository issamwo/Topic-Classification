from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig, DataCleaningConfig

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
            preprocessed_data_path=Path(config.preprocessed_data_path)
        )

        return data_cleaning_config