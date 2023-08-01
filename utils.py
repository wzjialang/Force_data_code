import torch
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameters
batch_size_train = 64
batch_size_val = 1 # 1 as we evaluate with the whole sequence
lr = 0.0001
epochs = 100


def normalize_data(data_dir):

    # List all CSV files in the folder
    csv_files = [file for file in os.listdir(data_dir) if file.endswith(".csv")]

    # Initialize an empty list to store DataFrames for each CSV file
    dataframes = []

    # Read each CSV file and store them in the dataframes list
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(file_path).iloc[:, 1] # keep only force data
        dataframes.append(df)


    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(dataframes, axis=0, ignore_index=True)

    # Standardize the second column using StandardScaler on the concatenated DataFrame
    scaler = StandardScaler()
    standard_scaler = scaler.fit(merged_df.values.reshape(-1, 1))
    
    return standard_scaler