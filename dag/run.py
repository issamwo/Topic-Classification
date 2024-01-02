from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta

from Classifier.config.configuration import ConfigurationManager
from Classifier.components.data_ingestion import DataIngestion
from Classifier.components.data_cleaning import DataCleaning
from Classifier.components.data_preprocessing import DataPreprocessing
from Classifier.components.model_training import ModelTraining
from Classifier.components.model_evaluation import ModelEvaluation
from Classifier import logger

# Define default arguments for your DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 2),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define your DAG
dag = DAG('Tickets_labelling',
          default_args=default_args,
          description='A simple DAG to run the pipelines from ingestion to model evaluation',
          schedule_interval=timedelta(days=1))


class TicketClassification:
    def __init__(self) -> None:
        pass
    
    def ingestion(self):

        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
    

    def cleaning(self):
        config = ConfigurationManager()
        data_cleaning_config = config.get_data_cleaning_config()
        data_cleaning = DataCleaning(config=data_cleaning_config)
        data_cleaning.clean_data(data_cleaning.read_data(data_cleaning.detect_last_file()))
    

    def preprocessing(self):

        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.preprocess_data(data_preprocessing.read_data(data_preprocessing.detect_last_file()))
    

    def training(self):

        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(config=model_training_config)
        model_training.train_model(model_training.read_pickle_files())
    

    def evaluation(self):

        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate_model(model_evaluation.read_pickle_files())
        model_evaluation.log_into_mlflow()


# Create Airflow tasks for each pipeline
t1 = PythonOperator(task_id='data_ingestion', python_callable=TicketClassification.ingestion, dag=dag)
t2 = PythonOperator(task_id='data_cleaning', python_callable=TicketClassification.cleaning, dag=dag)
t3 = PythonOperator(task_id='data_preprocessing', python_callable=TicketClassification.preprocessing, dag=dag)
t4 = PythonOperator(task_id='model_training', python_callable=TicketClassification.training, dag=dag)
t5 = PythonOperator(task_id='model_evaluation', python_callable=TicketClassification.evaluation, dag=dag)

# Define dependencies (if any)
t1 >> t2 >> t3 >> t4 >> t5   # This sets up a linear dependency, modify as needed
