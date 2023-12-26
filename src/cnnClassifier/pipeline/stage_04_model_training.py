from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import ModelTraining
from cnnClassifier import logger

STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):

        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(config=model_training_config)
        model_training.train_model(model_training.read_pickle_files())



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e