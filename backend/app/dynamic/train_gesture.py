"""
File: train_gesture.py

Purpose:
Loads the collected dynamic ASL gesture dataset
and trains a TensorFlow model to recognize
dynamic ASL gestures.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def train_gesture():
    print("Loading dynamic dataset....")
    print("TensorFlow Version:", tf.__version__)

    x=[]#landmark values
    y=[]#correct answer (a, b, c)

    dataset_path="../dataset/dynamic"

    gestures=sorted([
        folder
        for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
    ])

    label_map={
        gesture: index
        for index, gesture in enumerate(gestures)
    }

    print(label_map)

    for gesture,label in label_map.items():
        gesture_folder=os.path.join(dataset_path, gesture)
        for file in os.listdir(gesture_folder):
            if file.endswith(".npy"):
                file_path=os.path.join(gesture_folder, file)
                print(f"Loading {file_path}...")
                sample=np.load(file_path)
                sample=sample.flatten()
                x.append(sample)
                y.append(label)

    print(f"Loaded {len(x)} samples.")
    x=np.array(x, dtype=np.float32)#telling NumPy to store every value as a 32-bit floating point number because tensorflow does calculation using floating point values
    y=np.array(y)
    print ("Feature shape:", x.shape)
    print("Label shape:", y.shape)
    print("Unique labels:", np.unique(y))

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
        keras.layers.Input(shape=(3780,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(len(label_map), activation="softmax")######### update number for mroe letters
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
        epochs=30,#epoch means the model sees all 308 samples once
        batch_size=8#tensorflow trains on 16 smaples at a time
    )

    os.makedirs("../models", exist_ok=True)

    with open("../models/gesture_labels.json", "w") as file:
        json.dump(label_map, file, indent=4)
    model.save("../models/gesture_model.keras")#brain of the trained ai
    print("Model saved successfully!")

if __name__=="__main__":
    train_gesture()
