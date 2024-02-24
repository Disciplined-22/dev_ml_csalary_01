from flask import Flask, request, jsonify
import joblib

# Import the trained model from train.py
from train import model

app = Flask(__name__)

# Define API Endpoints:

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(data) 
    return jsonify({'prediction': prediction})

# Run the Flask Application:

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

