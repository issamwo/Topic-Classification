from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_preprocessing import DataPreprocessing
from cnnClassifier import logger

STAGE_NAME = "Data Preprocessing Stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):

        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.preprocess_data(data_preprocessing.read_data(data_preprocessing.detect_last_file()))



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e