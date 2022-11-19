import torch
from torch import nn
from torch.nn import functional as F


def elu():
    return nn.ELU(inplace=True)


def activation(name):
    def _apply_activation(x):
        if name == "relu":
            return F.relu(x)
        elif name == "selu":
            return F.selu(x)
        elif name == "sigmoid":
            return torch.sigmoid(x)
        elif name is None:
            return x
        else:
            raise ValueError()

    return _apply_activation


class FullyConnectedLayer(nn.Module):
    # TODO: look into ordering of dropout, batch norm, activation.

    def __init__(
        self,
        input_dim,
        hidden_dim,
        activation,
        dropout_prob=None,
        batch_norm=False
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.norm_layer = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_prob or 0.)

    def forward(self, x):
        h = self.linear(x)
        h = activation(self.activation)(h)
        h = self.dropout(h)
        return self.norm_layer(h)


class MLP(nn.Module):

    def __init__(
        self,
        hidden_layers,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_activation="relu",
        output_activation=None,
        dropout_prob=None,
        batch_norm=False,
        squeeze=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.squeeze_out = squeeze
        self.hidden_layers = nn.Sequential(*[
            FullyConnectedLayer(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                activation=hidden_activation,
                dropout_prob=self.dropout_prob or 0.,
                batch_norm=batch_norm)
            for i in range(hidden_layers)
        ])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, *args, **kwargs):
        h = self.hidden_layers(x)
        out = self.fc_out(h)
        out = activation(self.output_activation)(out)
        if self.squeeze_out and self.output_dim == 1:
            return out.squeeze(-1)
        else:
            return out


class eDICEBlock(nn.Module):

    """Transformer-style block without layer normalization, used in eDICE."""

    def __init__(self, model_dim, num_heads, ffn_embed_dim, dropout=0.1, ffn_dropout=0.):
        super().__init__()
        self.mha = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.ffn = MLP(1, model_dim, ffn_embed_dim, model_dim, dropout_prob=ffn_dropout)  # should have 1 hidden layer

        self.dropout = dropout
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, x, mask=None):
        # mask is B, L
        # Q. difference between attn_mask and key_padding_mask?
        # https://stackoverflow.com/questions/62629644/what-the-difference-between-att-mask-and-key-padding-mask-in-multiheadattnetion
        attn_output, attn_weights = self.mha(x, x, x, key_padding_mask=mask.squeeze(-1))
        attn_output = self.dropout1(attn_output)
        out1 = x + attn_output

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = out1 + ffn_output
        return out2
