from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, device):
        # Initialization here ...
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.device = device

        # Input modulation gate (g)
        self.Wgx = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Wgh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Input gate (i)
        self.Wix = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Wih = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Forget gate (f)
        self.Wfx = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Wfh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output gate (o)
        self.Wox = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Woh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output layer
        self.Why = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        batch_size, input_length = x.size(0), x.size(1)

        h_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        c_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        for t in range(input_length):
            x_t = x[:, t, :]
            g_t = torch.tanh(self.Wgx(x_t) + self.Wgh(h_t))
            i_t = torch.sigmoid(self.Wix(x_t) + self.Wih(h_t))
            f_t = torch.sigmoid(self.Wfx(x_t) + self.Wfh(h_t))
            o_t = torch.sigmoid(self.Wox(x_t) + self.Woh(h_t))
            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

        p = self.Why(h_t)
        y = torch.softmax(p, dim=-1)
        return y