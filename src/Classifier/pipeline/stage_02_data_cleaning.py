from Classifier.config.configuration import ConfigurationManager
from Classifier.components.data_cleaning import DataCleaning
from Classifier import logger

STAGE_NAME = "Data Cleaning Stage"

class DataCleaningTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_cleaning_config = config.get_data_cleaning_config()
        data_cleaning = DataCleaning(config=data_cleaning_config)
        data_cleaning.clean_data(data_cleaning.read_data(data_cleaning.detect_last_file()))


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataCleaningTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e