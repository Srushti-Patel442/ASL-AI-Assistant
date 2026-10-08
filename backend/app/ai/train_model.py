"""
File: train_model.py

Purpose:
Loads the collected ASL landmark dataset and trains
a TensorFlow model to recognize ASL letters.
"""

import os
import csv
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def train_model():
    print("Loading dataset....")
    print("TensorFlow Version:", tf.__version__)

    x=[]#landmark values(input)
    y=[]#correct answer (a, b, c)

    dataset_path="../dataset/static"

    letters=sorted([
        folder
        for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
    ])

    label_map={
        letter: index
        for index, letter in enumerate(letters)
    }

    print(label_map)

    for letter, label in label_map.items():
        csv_path=f"../dataset/{letter}/{letter}.csv"
        print(f"Loading {csv_path}...")

        with open(csv_path, "r") as file:
            reader=csv.reader(file)
            for row in reader:
                sample=[float(value) for value in row]
                x.append(sample)
                y.append(label)

    print(f"Loaded {len(x)} samples.")
    x=np.array(x, dtype=np.float32)#telling NumPy to store every value as a 32-bit floating point number because tensorflow does calculation using floating point values
    y=np.array(y)
    print ("Feature shape:", x.shape)
    print("Label shape:", y.shape)

    x_train, x_val, y_train, y_val=train_test_split(
        x,
        y,
        test_size=0.20,# 80% for training 20% for validating
        random_state=42,
        stratify=y
    )
    print("Training samples:", len(x_train))
    print("Validation samples:", len(x_val))

    model=keras.Sequential([
        keras.layers.Input(shape=(63,)),
        keras.layers.Dense(128, activation="relu"),#neurons for specific patterns
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(len(label_map), activation="softmax")#update number for mroe letters
    ])

    model.compile(
        optimizer="adam",#optimizer is the algorithm that adjusts the neural network's weights after each batch of training data
        loss="sparse_categorical_crossentropy",#loss function to measure how wrong the model is
        metrics=["accuracy"]#tells tensorflow to report something easy to understand like an acccuracy %
    )

    history=model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=50,#the model sees all samples once
        batch_size=16#tensorflow trains on 16 smaples at a time
    )

    os.makedirs("../models", exist_ok=True)

    with open("../models/labels.json", "w") as file:
        json.dump(label_map, file, indent=4)#save the label mapping
    model.save("../models/asl_model.keras")#brain of the trained ai, sores neural network structure, learned weights, and training configuration
    print("Model saved successfully!")

if __name__=="__main__":
    train_model()
