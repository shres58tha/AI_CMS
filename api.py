import json
import numpy as np
import pandas as pd
import pickle
import gzip  # Import gzip module for reading compressed files
from flask import Flask, request, jsonify
import random
from remarks import remarks_dict  # Importing the remarks dictionary

app = Flask(__name__)

# Load the compressed machine learning model
def load_compressed_model(file_path):
    with gzip.open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

compressed_model_path = "model.pkl.gz"
clf = load_compressed_model(compressed_model_path)
clf.feature_names_in_ = None

# Function to select a random remark from each group based on the prediction value
def select_random_remark(prediction):
    return random.choice(remarks_dict[prediction])

# Route for prediction
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

    # Select a random remark based on the prediction value
    selected_remark = select_random_remark(prediction[0])

    # Return data_array along with prediction and selected remark
    return jsonify({
        "data_array": data_array.tolist(),
        "prediction": prediction.tolist(),
        "selected_remark": selected_remark
    })

if __name__ == '__main__':
    app.run()

