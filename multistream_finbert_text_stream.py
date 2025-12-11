# multistream_finbert_text_stream.py
from transformers import BertModel, BertTokenizer
import torch.nn as nn

class TextStream(nn.Module):
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        super(TextStream, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.attention = nn.Sequential(
            nn.Linear(768, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )
        self.projection = nn.Linear(768, 32)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        weighted_output = (outputs.last_hidden_state * self.attention(outputs.last_hidden_state)).sum(dim=1)
        return self.projection(weighted_output)
