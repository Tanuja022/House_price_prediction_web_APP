import joblib
import numpy as np

def test_model_prediction():
    model = joblib.load("house_price_model.pkl")
    
    # Test with known input
    test_input = np.array([[1000, 2]])
    predicted_price = model.predict(test_input)[0]

    # Check if the prediction is in expected range (±20%)
    expected_price = 300000
    tolerance = 0.2 * expected_price
    assert abs(predicted_price - expected_price) <= tolerance, f"Prediction {predicted_price} is not within range"
