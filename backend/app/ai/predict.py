"""
File: predict.py

Purpose:
Loads the trained ASL model and predicts
hand signs from MediaPipe landmarks.
"""

import numpy as np
import json
from tensorflow import keras

class ASLPredictor:
    def __init__(self):
        print("Loading trained ASL model...")
        self.model=keras.models.load_model(
            "../models/asl_model.keras"
        )
        with open("../models/labels.json", "r") as file:
            self.label_map=json.load(file)

        self.index_to_label={
            index: letter
            for letter, index in self.label_map.items()
        }
        print("model loaded successfully")

    def predict(self, sample):
        sample=np.array(sample, dtype=np.float32)#convert to a numpy array
        sample=sample.reshape(1, 63)#reshaped to 1 sample 63 numbers

        prediction=self.model.predict(sample, verbose=0)
        predict_class=int(np.argmax(prediction))#finds largest prediction
        confidence=float(prediction[0][predict_class])
        predicted_letter=self.index_to_label[predict_class]
        return predicted_letter,confidence

if __name__=="__main__":
    predictor=ASLPredictor()