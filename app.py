from flask import Flask,request,jsonify
from flask_cors import CORS
import joblib

app=Flask(__name__)
CORS(app)
model=joblib.load('house_price_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    area = data['area']
    bedrooms = data['bedrooms']
    prediction = model.predict([[area, bedrooms]])
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
