"""
File: predict_gesture.py

Purpose:
Loads the trained dynamic ASL gesture model and predicts
ASL gestures from a sequence of MediaPipe hand landmarks.
"""

import numpy as np
import json
from tensorflow import keras

class GesturePredictor:
    def __init__(self):
        print("Loading trained dynamic gestre model...")
        self.model=keras.models.load_model(
            "../models/gesture_model.keras"
        )
        with open("../models/gesture_labels.json", "r") as file:
            self.label_map=json.load(file)

        self.index_to_label={
            index: gesture
            for gesture, index in self.label_map.items()
        }
        print("Dynamic gesture model loaded successfully")

    def predict(self, sample):
        sample=np.array(sample, dtype=np.float32)#convert to a numpy array
        sample=sample.flatten()
        sample=sample.reshape(1, 3780)

        prediction=self.model.predict(sample, verbose=0)
        predict_class=int(np.argmax(prediction))#finds largest prediction
        confidence=float(prediction[0][predict_class])
        predicted_gesture=self.index_to_label[predict_class]
        return predicted_gesture,confidence

if __name__=="__main__":
    predictor=GesturePredictor()
    sample=np.load("../dataset/dynamic/YES/sample001.npy")
    prediction, confidence=predictor.predict(sample)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence*100:2f}%")