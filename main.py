from Classifier import logger
from Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Classifier.pipeline.stage_02_data_cleaning import DataCleaningTrainingPipeline



STAGE_NAME = "Data Ingestion Stage"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Data Cleaning Stage"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataCleaningTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e