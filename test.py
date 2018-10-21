import torch

from nn_correct.loader import FIOLoader
from nn_correct.model import CorrectorModel
from nn_correct.vectorizer import ru_idx

if __name__ == '__main__':

    loader = FIOLoader('./data/train_ru.csv', 10, 1, alphabet=ru_idx)
    reader = loader.read_batch()

    batch1 = next(reader)

    name_mtx, lengths, diff_mtx = loader.vectorize(batch1)

    model = CorrectorModel(
        embedding_size=loader.vectorizer.length,
        conv_sizes=[20, 20, 20],
        out_size=loader.diff_vectorizer.length,
        dropout=0.1,
        window=3,
        lstm_layers=2,
        lstm_size=20
    )

    model.forward(torch.from_numpy(name_mtx), lengths)

