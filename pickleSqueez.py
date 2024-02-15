import gzip
import pickle

# Path to the original model.pkl file
original_file_path = "model.pkl"

# Path to the compressed model file
compressed_file_path = "model.pkl.gz"

# Load the model
with open(original_file_path, 'rb') as f:
    model = pickle.load(f)

# Compress the model and save it to a gzip-compressed file
with gzip.open(compressed_file_path, 'wb') as f:
    pickle.dump(model, f)

print("Model compressed and saved to:", compressed_file_path)

