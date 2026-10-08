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
        self.model=keras.models.load_model(#load trained neural network
            "../models/asl_model.keras"#keras file contains neural network architecture and tranining configuration
        )
        #model predicts numbers
        with open("../models/labels.json", "r") as file:
            self.label_map=json.load(file)
        #convert model output back to letters
        self.index_to_label={
            index: letter
            for letter, index in self.label_map.items()
        }
        print("model loaded successfully")

    def predict(self, sample):
        sample=np.array(sample, dtype=np.float32)#neural network needs numerical range, use numpy arrays
        sample=sample.reshape(1, 63)#reshaped to 1 sample 63 features

        prediction=self.model.predict(sample, verbose=0)
        predict_class=int(np.argmax(prediction))#finds largest prediction/ highest probability
        confidence=float(prediction[0][predict_class])
        predicted_letter=self.index_to_label[predict_class]#convert index to letter
        return predicted_letter,confidence

if __name__=="__main__":
    predictor=ASLPredictor()
