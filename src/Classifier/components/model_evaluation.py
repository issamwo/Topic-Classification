import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model
from urllib.parse import urlparse
import pickle
import json
import os
from Classifier.constants import *
from Classifier.entity.config_entity import ModelEvaluationConfig



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig): 
        self.config = config


    def read_pickle_files(self):
        data = {}
        for filename in os.listdir(self.config.preprocessed_spilitted_data_path):
            if filename.endswith('.pickle'):
                with open(os.path.join(self.config.preprocessed_spilitted_data_path, filename), 'rb') as f:
                    data[filename] = pickle.load(f)
        return data
    

    def evaluate_model(self, data: dict):
        """

        Args:
            model (tf.keras.Model): _description_
            data (dict): _description_
        """
        # Access test data
        x_test = data['X_test_preprocessed.pickle']
        y_test = data['y_test_preprocessed.pickle']
        # Load model
        model = load_model(os.path.join(self.config.path_model, 'model.h5'))
        # Store model:
        self.model = model
        # Evaluate model
        evaluation = model.evaluate(x_test, y_test, batch_size=self.config.model_params['batch_size'], verbose=1)
        # add score
        self.score = evaluation
        # Create directory to store evaluation metrics
        os.makedirs("artifacts/model_evaluation", exist_ok=True)
        # Create a dictionary that contains the loss and metrics
        evaluation_dict = {'loss': evaluation[0], 'accuracy': evaluation[1]}
        # Save the dictionary as a JSON file
        with open(os.path.join('artifacts/model_evaluation', 'evaluation.json'), 'w') as f:
            json.dump(evaluation_dict, f)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.model_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="MLP")
            else:
                mlflow.keras.log_model(self.model, "model")
        




