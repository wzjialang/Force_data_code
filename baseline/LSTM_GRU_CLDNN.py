import torch
import torch.nn as nn

class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiLayerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, hidden_size))
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to (batch, seq_len, features)
        out, _ = self.lstm(x)
        # out = self.avg_pool(out).squeeze(1)  # Average Pooling
        # out = self.fc(out)  # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out

class MultiLayerGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiLayerGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, hidden_size))
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to (batch, seq_len, features)
        out, _ = self.gru(x)
        # out = self.avg_pool(out).squeeze(1)  # Average Pooling
        # out = self.fc(out)  # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out
    
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 2 * hidden_size))
        self.fc = nn.Linear(2 * hidden_size, output_size)  # 2 * hidden_size due to bidirectional

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to (batch, seq_len, features)
        out, _ = self.lstm(x)
        # out = self.avg_pool(out).squeeze(1)  # Average Pooling
        # out = self.fc(out)  # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out
    
class CLDNN(nn.Module):
    def __init__(self, input_channels, lstm_hidden_size, lstm_num_layers, num_classes):
        super(CLDNN, self).__init__()
        
        # Convolutional Layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # LSTM Layers
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, lstm_hidden_size))
        self.lstm = nn.LSTM(128, lstm_hidden_size, lstm_num_layers, batch_first=True)
        
        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Convolutional Layers
        conv_out = self.conv_layers(x)
        
        # Reshape for LSTM
        lstm_in = conv_out.permute(0, 2, 1)  # Reshape to (batch, seq_len, features)
        
        # LSTM Layers
        lstm_out, _ = self.lstm(lstm_in)

        # lstm_out = self.avg_pool(lstm_out).squeeze(1)  # Average Pooling
        
        # Fully Connected Layers
        # lstm_out = lstm_out[:, -1, :]  # Get output from the last time step
        # output = self.fc_layers(lstm_out)
        output = self.fc_layers(lstm_out[:, -1, :])
        
        return output