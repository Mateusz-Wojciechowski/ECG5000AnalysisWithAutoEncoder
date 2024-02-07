import torch.nn as nn
import torch.functional as fun


class AutoEncoder(nn.Module):
    def __init__(self, d_data):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_data, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 140),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_encoder = self.encoder(x)
        x_decoder = self.decoder(x_encoder)
        return x_decoder
