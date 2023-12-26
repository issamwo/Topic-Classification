import os
import json
from cnnClassifier import logger
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from keras import optimizers
import pickle

from cnnClassifier.entity.config_entity import ModelTrainingConfig
from cnnClassifier.constants import *

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig): 
        self.config = config


    def read_pickle_files(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        data = {}
        for filename in os.listdir(self.config.preprocessed_spilitted_data_path):
            if filename.endswith('.pickle'):
                with open(os.path.join(self.config.preprocessed_spilitted_data_path, filename), 'rb') as f:
                    data[filename] = pickle.load(f)
        return data
    

    def train_model(self, data: dict):
        """_summary_

        Args:
            data (dict): _description_
        """

        x_train = data['X_train_preprocessed.pickle']
        y_train = data['y_train_preprocessed.pickle']
        x_test = data['X_test_preprocessed.pickle']
        y_test = data['y_test_preprocessed.pickle']

        num_classes = y_train.shape[1]

        # logger

        model = Sequential()
        model.add(Dense(512, input_shape=(self.config.max_words,)))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))

        
        # create an optimizer instance
        adam = optimizers.Adam(learning_rate=self.config.learning_rate, beta_1=self.config.beta_1,\
                                beta_2=self.config.beta_2, epsilon=1e-08, decay=0.0, amsgrad=False)

        # compile your model with the optimizer
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        # 
        model.fit(x_train, y_train, batch_size=self.config.batch_size,\
                   epochs=self.config.epochs, verbose=1, validation_split=self.config.validation_split)
        
        model.save(os.path.join(self.config.model_path, 'model.h5'))
        
        # Model evaluation
        evaluation = model.evaluate(x_test, y_test, batch_size=self.config.batch_size, verbose=1)
        # Create a dictionary that contains the loss and metrics
        evaluation_dict = {'loss': evaluation[0], 'accuracy': evaluation[1]}
        # save it
        with open(os.path.join(self.config.model_path, 'evaluation.json'), 'w') as f:
            json.dump(evaluation_dict, f)
