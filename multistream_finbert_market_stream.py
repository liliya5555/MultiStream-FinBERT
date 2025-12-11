# multistream_finbert_market_stream.py
import torch
import torch.nn as nn

class MarketSignalStream(nn.Module):
    def __init__(self, input_channels=18, seq_len=60):
        super(MarketSignalStream, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1, dilation=1),
            nn.ELU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=2, dilation=2),
            nn.ELU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
            nn.ELU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * seq_len, 32)

    def forward(self, x):
        x = self.network(x)
        x = self.flatten(x)
        return self.fc(x)
