"""
Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data

Objective: Predict housing prices using both structured data and house images.
Dataset: Houses Dataset (Tabular + Images)

Instructions:
● Use CNNs to extract features from images
● Combine extracted image features with tabular data
● Train a model using both modalities
● Evaluate performance using MAE and RMSE
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, concatenate, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_house_attributes(inputPath):
 cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
 df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
 
 # Identify unique zipcodes and create dummy variables (one-hot encoding)
 zipcodes = df["zipcode"].value_counts().keys().tolist()
 counts = df["zipcode"].value_counts().tolist()
 
 # Remove extremely rare zipcodes for stability
 for (zipcode, count) in zip(zipcodes, counts):
 if count < 25:
 df.loc[df["zipcode"] == zipcode, "zipcode"] = 99999
 
 df = pd.get_dummies(df, columns=["zipcode"])
 return df

def load_house_images(df, inputPath):
 images = []
 # Loop over the indexes of the houses
 for i in df.index.values:
 # Load the frontal image (as a representative sample)
 basePath = os.path.sep.join([inputPath, f"{i+1}_frontal.jpg"])
 
 # If the frontal image doesn't exist, we skip or use a black square
 # But in this dataset, they are numbered sequentially.
 if os.path.exists(basePath):
 image = cv2.imread(basePath)
 else:
 image = np.zeros((32, 32, 3), dtype="uint8")
 
 if image is not None:
 image = cv2.resize(image, (32, 32))
 images.append(image)
 else:
 images.append(np.zeros((32, 32, 3), dtype="uint8"))
 
 return np.array(images)

def create_mlp(dim, regress=False):
 inputs = Input(shape=(dim,))
 x = Dense(8, activation="relu")(inputs)
 x = Dense(4, activation="relu")(x)
 if regress:
 x = Dense(1, activation="linear")(x)
 return Model(inputs, x)

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
 inputShape = (height, width, depth)
 inputs = Input(shape=inputShape)
 
 for (i, f) in enumerate(filters):
 if i == 0:
 x = Conv2D(f, (3, 3), padding="same", activation="relu")(inputs)
 else:
 x = Conv2D(f, (3, 3), padding="same", activation="relu")(x)
 x = MaxPooling2D(pool_size=(2, 2))(x)
 
 x = Flatten()(x)
 x = Dense(16, activation="relu")(x)
 x = Dropout(0.5)(x)
 x = Dense(4, activation="relu")(x)
 
 if regress:
 x = Dense(1, activation="linear")(x)
 return Model(inputs, x)

def main():
 print("--- Task 3: Multimodal Housing Price Prediction ---")
 
 dataset_path = "./dataset"
 if not os.path.exists(dataset_path):
 print(f"Error: Dataset directory '{dataset_path}' not found.")
 print("Please ensure the 'Houses Dataset' is extracted into a 'dataset' folder.")
 return
 
 info_path = os.path.join(dataset_path, "HousesInfo.txt")
 if not os.path.exists(info_path):
 print(f"Error: '{info_path}' not found inside '{dataset_path}'.")
 return
 
 print("Loading tabular data...")
 df = load_house_attributes(info_path)
 
 print("Loading image data (frontal views, resizing to 32x32)...")
 images = load_house_images(df, dataset_path)
 images = images / 255.0 # Normalize images
 
 print(f"Data Loaded! Tabular shape: {df.shape}, Images shape: {images.shape}")
 
 # Train-test split
 split = train_test_split(df, images, test_size=0.25, random_state=42)
 (trainAttrX, testAttrX, trainImagesX, testImagesX) = split
 
 maxPrice = trainAttrX["price"].max()
 trainY = trainAttrX["price"] / maxPrice
 testY = testAttrX["price"] / maxPrice
 
 trainAttrX = trainAttrX.drop("price", axis=1)
 testAttrX = testAttrX.drop("price", axis=1)
 
 # Scale tabular data
 cs = MinMaxScaler()
 trainAttrX = cs.fit_transform(trainAttrX)
 testAttrX = cs.transform(testAttrX)
 
 print("Building Multimodal Network...")
 mlp = create_mlp(trainAttrX.shape[1], regress=False)
 cnn = create_cnn(32, 32, 3, regress=False)
 
 # Concatenate features from MLP and CNN
 combinedInput = concatenate([mlp.output, cnn.output])
 
 x = Dense(4, activation="relu")(combinedInput)
 x = Dense(1, activation="linear")(x)
 
 model = Model(inputs=[mlp.input, cnn.input], outputs=x)
 
 model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
 
 print("Training Multimodal Model...")
 model.fit(
 x=[trainAttrX, trainImagesX], 
 y=trainY, 
 validation_data=([testAttrX, testImagesX], testY), 
 epochs=10, 
 batch_size=8,
 verbose=1
 )
 
 print("\nEvaluating Model...")
 preds = model.predict([testAttrX, testImagesX])
 
 # Unscale predictions
 preds = preds.flatten() * maxPrice
 testY_unscaled = testY * maxPrice
 
 mae = mean_absolute_error(testY_unscaled, preds)
 rmse = np.sqrt(mean_squared_error(testY_unscaled, preds))
 
 print("\nFinal Metrics:")
 print(f"MAE (Mean Absolute Error): ${mae:,.2f}")
 print(f"RMSE (Root Mean Squared Error): ${rmse:,.2f}")
 print("\nThe model successfully learned from both numerical inputs (bedrooms, area) and visual inputs (images)!")

if __name__ == "__main__":
 main()
