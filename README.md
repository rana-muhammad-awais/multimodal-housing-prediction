# Task 3: Multimodal Housing Price Prediction

## Objective
Predict housing prices using a multimodal deep learning approach that combines structured tabular data (bedrooms, bathrooms, area, zipcode) with unstructured visual data (house images).

## Methodology / Approach
- **Data Loading**: We load the tabular data from `HousesInfo.txt` and map each row to its corresponding `frontal.jpg` image.
- **Preprocessing**: 
 - Zip codes are one-hot encoded and numerical features are scaled using `MinMaxScaler`.
 - Images are resized to `32x32` pixels and normalized to `[0, 1]` ranges.
- **Architecture (Late Fusion)**:
 - **Tabular Branch**: A Multi-Layer Perceptron (MLP) processes the numerical and categorical data.
 - **Image Branch**: A Convolutional Neural Network (CNN) extracts visual features (like style or condition) from the house images.
 - **Fusion**: The output vectors from the MLP and CNN are concatenated into a single dense layer, which predicts the final continuous housing price.
- **Evaluation**: The model is evaluated on a 25% holdout set, and metrics include Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to quantify the prediction accuracy in actual dollar amounts.

## How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy opencv-python scikit-learn tensorflow
```

### 2. Dataset Setup
Ensure that the Eman Ahmed `Houses Dataset` has been placed in the `dataset` folder within this directory. The directory structure should look like:
```
dataset/
├── HousesInfo.txt
├── 1_frontal.jpg
├── ...
```

### 3. Run the Model
Execute the Python script to build, train, and evaluate the multimodal network:
```bash
python task3_multimodal.py
```
