# models/lstm_decoder.py

import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    """
    2-layer bidirectional LSTM mapping envelope sequence → character logits.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 29,  # 26 letters + blank + space + pad
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x : (B, T, input_dim)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out  # (B, T, num_classes)


def build_lstm_decoder(device: str = "cpu") -> LSTMDecoder:
    model = LSTMDecoder()
    return model.to(device)
