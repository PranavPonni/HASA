import pickle
import numpy as np

# Set your file path (update to your actual episode folder and file)
file_path = "/home/handlingteam2/HASA/user/pranav/example/data/new/2.0_0/episode0/timestep20.pkl"

# Load the pickle file
with open(file_path, "rb") as f:
    data = pickle.load(f)

print("== Tactile values after offset subtraction ==")
for key in data:
    if 'tactile' in key:
        arr = np.array(data[key])
        print(f"{key}: mean = {np.mean(arr):.2f}, max = {np.max(arr):.2f}, min = {np.min(arr):.2f}")
