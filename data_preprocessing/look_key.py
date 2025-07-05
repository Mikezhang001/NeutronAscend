import numpy as np

if __name__ == "__main__":
    # Set dataset name
    dataset_name = "Cora"  # Options: Cora, Citeseer, Pubmed, mycielskian18, reddit, products
    dataset_path = f"../data/npz/{dataset_name}.npz"

    # Load the NPZ file
    data = np.load(dataset_path, allow_pickle=True)

    # Print keys in the NPZ file
    print("Keys in the NPZ file:", data.files)

    # Loop through each key and print details
    for key in data.files:
        array = data[key]  # Get the array corresponding to the key
        print(f"Key: {key}, Data Type: {array.dtype}, Array Shape: {array.shape}")