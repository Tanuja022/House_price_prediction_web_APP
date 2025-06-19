import joblib
import numpy as np
import os

def test_model_exists():
    assert os.path.exists("house_price_model.pkl"), "Model file not found."

def test_model_prediction_accuracy():
    model = joblib.load("house_price_model.pkl")
    input_data = np.array([[1000, 2]])  # input from training data
    prediction = model.predict(input_data)[0]
    
    expected_price = 300000
    tolerance = 0.2 * expected_price  # ±20% tolerance
    assert abs(prediction - expected_price) <= tolerance, f"Prediction {prediction} out of expected range"
