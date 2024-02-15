import json
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the machine learning model
with open("model.pkl", "rb") as f:
    clf = pickle.load(f)
    clf.feature_names_in_ = None

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.json

    # Check if JSON data follows the expected pattern
    expected_keys = ["attendance_score", "assignment_score", "assessment_score"]
    if not set(expected_keys) == set(data.keys()):
        # Return 0 if the pattern is not matched
        return jsonify({"prediction": 0})

    # Process the data for prediction
    data_array = np.array([data[key] for key in expected_keys]).reshape(1, -1)
    prediction = clf.predict(data_array).astype(int)

    # Return data_array along with prediction
    return jsonify({
        "data_array": data_array.tolist(),
        "prediction": prediction.tolist()
    })

if __name__ == '__main__':
    app.run()

