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
        x = x.unsqueeze(1)
        # LSTM layer
        out, _ = self.lstm(x)
        
        # Use only the output from the last time step
        out = self.fc1(out[:, -1, :])
        
        return out



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
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        # LSTM layer
        out, _ = self.lstm(x)
        
        # Fully connected layers
        out = self.leaky_relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        #out = self.sigmoid(self.fc2(out))
        return out

class NoiseGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(NoiseGenerator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=256,
                            num_layers=3, batch_first=True, dropout=dropout_prob)
        self.fc2 = nn.Linear(256, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.drop_out1 = nn.Dropout(dropout_prob)
        self.drop_out2 = nn.Dropout(dropout_prob)
        self.drop_out3 = nn.Dropout(dropout_prob)
        self.drop_out4 = nn.Dropout(dropout_prob)


    def forward(self, z2):
        x = torch.relu(self.fc1(z2))
        x = self.drop_out1(x)
        x = x.unsqueeze(1)
        # LSTM layer
        out, _ = self.lstm(x)
        # Fully connected layers
        x = torch.relu(out[:, -1, :])
        x = torch.relu(self.fc2(x))
        x = self.drop_out2(x)
        x = torch.relu(self.fc3(x))
        x = self.drop_out3(x)
        x = torch.relu(self.fc4(x))
        x = self.drop_out4(x)
        x = F.tanh(self.fc5(x))
        return x
    

class NoiseDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(NoiseDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.drop_out1 = nn.Dropout(dropout_prob)
        self.drop_out2 = nn.Dropout(dropout_prob)
        self.drop_out3 = nn.Dropout(dropout_prob)
        self.drop_out4 = nn.Dropout(dropout_prob)

    def forward(self, z2):
        x = torch.relu(self.fc1(z2))
        x = self.drop_out1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop_out2(x)
        x = torch.relu(self.fc3(x))
        x = self.drop_out3(x)
        x = torch.relu(self.fc4(x))
        x = self.drop_out4(x)
        x = F.sigmoid(self.fc5(x))
        return x