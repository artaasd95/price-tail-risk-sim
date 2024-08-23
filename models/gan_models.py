import torch
import torch.nn as nn
import torch.nn.functional as F

class MainGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob, output_size):
        super(MainGenerator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM layer
        out, _ = self.lstm(x)
        
        # Use only the output from the last time step
        out = self.fc1(out)
        
        return out


class NoiseGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(NoiseGenerator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.drop_out = nn.Dropout(dropout_prob)

    def forward(self, z2):
        x = torch.relu(self.fc1(z2))
        x = self.drop_out(x)
        x = torch.relu(self.fc2(x))
        x = self.drop_out(x)
        x = torch.relu(self.fc3(x))
        x = self.drop_out(x)
        x = torch.relu(self.fc4(x))
        x = self.drop_out(x)
        x = torch.relu(self.fc5(x))
        x = self.drop_out(x)
        x = self.fc6(x)
        return x



class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(Discriminator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM layer
        out, _ = self.lstm(x)
        
        # Fully connected layers
        out = self.leaky_relu(self.fc1(out))
        out = self.fc2(out)
        #out = self.sigmoid(self.fc2(out))
        return out



