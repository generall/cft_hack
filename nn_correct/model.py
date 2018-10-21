import itertools

import torch
from torch import nn


def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    :param iterable:
    :return:
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class CorrectorModel(nn.Module):

    def __init__(
            self,
            embedding_size,
            conv_sizes,
            out_size,
            dropout=0.1,
            window=3
    ):
        super(CorrectorModel, self).__init__()

        activation = nn.LeakyReLU

        self.dropout = dropout
        self.window = window
        self.out_size = out_size
        self.conv_sizes = conv_sizes
        self.embedding_size = embedding_size

        convolutions = [
            nn.Conv1d(self.embedding_size, self.conv_sizes[0], self.window),
            activation(),
            nn.Dropout(self.dropout)
        ]

        for from_size, to_size in pairwise(self.conv_sizes):
            convolutions += [
                torch.nn.Conv1d(from_size, to_size, self.window, padding=self.window-1),
                activation(),
                torch.nn.Dropout(self.dropout)
            ]

        self.convolution = nn.Sequential(*convolutions)

        self.final_layer = nn.Linear(self.conv_sizes[-1], self.out_size)
        self.final_activation = nn.LogSoftmax(dim=2)

    def forward(self, names, lengths):
        names = names.transpose(1, 2)
        conv_names = self.convolution(names)
        conv_names = conv_names.transpose(1, 2)

        diffs = self.final_activation(self.final_layer(conv_names)).transpose(1, 2)

        return diffs

