import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.size(0) == 0:
            raise ValueError("Empty batch received")

        mask = (x.sum(dim=-1) != 0).long()
        lengths = mask.sum(dim=1).cpu()

        lengths = torch.clamp(lengths, min=1)

        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]

        packed_x = rnn_utils.pack_padded_sequence(x, lengths.long(), batch_first=True)
        lstm_out, (h_n, c_n) = self.lstm(packed_x)

        lstm_out, _ = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)

        _, unsort_idx = sort_idx.sort(0)
        lstm_out = lstm_out[unsort_idx]

        batch_size = lstm_out.size(0)
        last_output = lstm_out[torch.arange(batch_size), lengths[unsort_idx] - 1]

        x = self.layer_norm(last_output)
        x = self.fc1(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        output = self.fc2(x)
        return output
