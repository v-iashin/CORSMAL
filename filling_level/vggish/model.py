import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

class RNN(torch.nn.Module):
    """Some Information about RNN"""

    def __init__(self, model_type, input_dim, hidden_dim, n_layers, drop_p, output_dim, bi_dir):
        super(RNN, self).__init__()
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_p = drop_p
        self.output_dim = output_dim
        self.bi_dir = bi_dir
        if bi_dir:
            self.num_dir = 2
        else:
            self.num_dir = 1

        if self.model_type == 'GRU':
            self.rnn = torch.nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_p)
        elif self.model_type == 'LSTM':
            self.rnn = torch.nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_p)
        else:
            raise NotImplementedError

        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        hiddens_0 = self.init_hidden(x)
        out, (hiddens) = self.rnn(x, hiddens_0)
        out = self.fc(self.relu(out[:, -1]))
        # (T, out_dim), (3, 3, 512)
        return out, hiddens

    def init_hidden(self, x):
        size = (self.n_layers*self.num_dir, len(x), self.hidden_dim)
        h = torch.zeros(size).float().to(x.device)
        if self.model_type == 'LSTM':
            c = torch.zeros(size).float().to(x.device)
            return h, c
        else:
            return h


if __name__ == "__main__":
    seq_1 = torch.randn((1, 512))
    seq_2 = torch.randn((5, 512))
    seq_3 = torch.randn((2, 512))

    padded_seq = pad_sequence([seq_1, seq_2, seq_3], batch_first=True, padding_value=0)
    # lengths = [len(item) for item in [seq_1, seq_2, seq_3]]
    # packed_seq = pack_padded_sequence(padded_seq, lengths, batch_first=True, enforce_sorted=False)

    rnn = RNN('GRU', 512, 512, 3, 0.0, 10, False)
    # out, hiddens = rnn(packed_seq)
    out, hiddens = rnn(padded_seq)
    print(out.shape, hiddens.shape)
    _, preds = torch.max(out, 1)
    print(torch.nn.functional.softmax(out, dim=-1))
