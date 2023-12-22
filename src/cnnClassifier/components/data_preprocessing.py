import os
import pandas as pd
import json
import numpy as np
from cnnClassifier import logger
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle

from cnnClassifier.entity.config_entity import DataPreprocessingConfig
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories, write_to_pickle

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig): 
        self.config = config
    
    def detect_last_file(self) -> Path:
        """
        get last unzip files from the ingestion pipeline
        """
        logger.info("Looking for all JSON cleaned files to Select latest one created")
        directory = self.config.cleaned_data_path
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        max_element = None
        max_output = float('-inf')

        for element in json_files:
            output = os.path.getctime(os.path.join(os.getcwd(), directory,element))
            if output > max_output:
                max_output = output
                max_element = element
        logger.info(f"latest cleaned file detected: {max_element}")
        return os.path.join(os.getcwd(), directory,max_element)
    

    def read_data(self, file_path: Path) -> pd.DataFrame:
        """
        read and subset the data
        """
        logger.info("Started reading cleaned data file")
        # Load the JSON file as a string
        with open(file_path) as f:
            data = json.load(f)
        # Normalize the JSON data and create a DataFrame
        df = pd.json_normalize(data)
        # Subset of data
        logger.info("File read is completed")
        return df
    
    
    def vector_classes_info_to_json(self, vector_class, output_file_name: str, output_path: Path):
        """_summary_

        Args:
            vector_class (_type_): _description_
        """
        # count the number of each class
        class_counts = Counter(vector_class)
        
        # calculate the total number of classes
        total_classes = sum(class_counts.values())
        
        # calculate the percentage of each class
        class_percentages = {cls: count / total_classes * 100 for cls, count in class_counts.items()}
        
        # create a dictionary with the counts and percentages
        class_info = {
            'counts': class_counts,
            'percentages': class_percentages
                    }

        # write the dictionary to a JSON file
        with open(os.path.join(output_path,f"{output_file_name}.json"), 'w') as f:
            json.dump(class_info, f)


    def preprocess_data(self, df: pd.DataFrame):
        """Split data and apply preprocessing

        Args:
            df (pd.DataFrame): _description_
        """
        df['Topic'] = df['Topic'].map(self.config.topic_names)

        X_train, X_test, y_train, y_test = train_test_split(
            df.text, df.Topic, test_size=0.25, random_state=42
            )
        
        self.vector_classes_info_to_json(y_train,'metadata_info_train',self.config.preprocessed_spilitted_data_path)
        self.vector_classes_info_to_json(y_test,'metadata_info_test',self.config.preprocessed_spilitted_data_path)
        
        # vector class to binary class matrix 
        num_classes = np.max(y_train) + 1

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        
        # Tokenizer init:
        tokenizer = Tokenizer(num_words=self.config.max_words)
        tokenizer.fit_on_texts(X_train)

        # Save trained Tokenizer 
        with open(os.path.join(self.config.preprocessed_spilitted_data_path,'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(Tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Convert the text data into sequences of integers
        X_train_sequences = tokenizer.texts_to_sequences(X_train)
        X_test_sequences = tokenizer.texts_to_sequences(X_test)
        # # Transform the word vector to tf-idf
        x_train_tfidf = tokenizer.sequences_to_matrix(X_train_sequences, mode="tfidf")
        X_test_tfidf = tokenizer.sequences_to_matrix(X_test_sequences, mode="tfidf")

        # how to store this data : pickle
        write_to_pickle(x_train_tfidf, 'X_train_preprocessed',self.config.preprocessed_spilitted_data_path)
        write_to_pickle(X_test_tfidf, 'X_test_preprocessed', self.config.preprocessed_spilitted_data_path)

        write_to_pickle(y_train, 'y_train_preprocessed', self.config.preprocessed_spilitted_data_path)
        write_to_pickle(y_test, 'y_test_preprocessed', self.config.preprocessed_spilitted_data_path)
        

        

