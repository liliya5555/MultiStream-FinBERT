# multistream_finbert_fusion_classifier.py
import torch
import torch.nn as nn

class FusionClassifier(nn.Module):
    def __init__(self, input_dim=96):
        super(FusionClassifier, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        x, _ = self.attn(x, x, x)
        x = self.fusion(x.mean(dim=1))
        return self.classifier(x)
