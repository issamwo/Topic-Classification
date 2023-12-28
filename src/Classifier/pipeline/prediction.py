import os
import numpy as np
from tensorflow.keras.models import load_model
import pickle

class PredictionPipeline:
    def __init__(self) -> None:
        self.model = load_model(os.path.join('artifacts/model_training/', 'model.h5'))
        with open('artifacts/data_preprocessing/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
    

    def predict(self, text):

        # load model
        model = self.model
        # Load tokenizer
        tokenizer = self.tokenizer
        # Target class
        target_names = ["Bank Account services", "Credit card or prepaid card",\
                            "Others", "Theft/Dispute Reporting", "Mortgage/Loan"]
        # Preprocessing
        text_sequences = tokenizer.texts_to_sequences([text])
        text_tfidf = tokenizer.sequences_to_matrix(text_sequences, mode="tfidf")
        predicted = model.predict(text_tfidf)

        return {
                'class': target_names[np.argmax(predicted[0])]
                }
            
