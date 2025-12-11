# multistream_finbert_accounting_stream.py
import torch
import torch.nn as nn

class AccountingStream(nn.Module):
    def __init__(self, input_dim=24):
        super(AccountingStream, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.model(x)
