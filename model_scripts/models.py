# Imports
import torch.nn as nn


# Classes
class CropClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=2):
        super(CropClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, batchX):
        out, (ht, ct) = self.lstm(batchX)
        return self.fc1(out[:, -1, :])
