import torch
from sklearn.preprocessing import StandardScaler
import random

def gaussian_transform(data):
    
    # Specify the mean and standard deviation of the Gaussian noise
    mean = 0.0
    std_dev = 0.1  # You can adjust this value to control the noise level

    # Generate Gaussian noise
    noise = torch.randn_like(data) * std_dev + mean

    # Add the noise to the tensor
    out = data + noise
    
    return out


def temporal_swifting_transform(data):    
    
    # Create a 1D PyTorch tensor of length 300
    original_tensor = data

    # Define the number of pieces to split the tensor into
    num_pieces = 30

    # Calculate the length of each piece
    piece_length = original_tensor.shape[1] // num_pieces

    # Split the original tensor into pieces
    tensor_pieces = [original_tensor[i:i + piece_length] for i in range(0, len(original_tensor), piece_length)]

    # Shuffle the pieces
    random.shuffle(tensor_pieces)

    # Concatenate the shuffled pieces back into a single tensor
    shuffled_tensor = torch.cat(tensor_pieces)
    
    return shuffled_tensor