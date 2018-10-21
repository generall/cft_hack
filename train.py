import argparse
import datetime

import os

import numpy as np
import torch

import torch.optim as optim
from torchlite.torch.learner import Learner
from torchlite.torch.learner.cores import ClassifierCore
from torchlite.torch.metrics import Metric
from torchlite.torch.train_callbacks import TensorboardVisualizerCallback, ModelSaverCallback, ReduceLROnPlateau

from nn_correct.loader import FIOLoader
from nn_correct.model import CorrectorModel
from nn_correct.vectorizer import ru_idx

parser = argparse.ArgumentParser(description='Train CNN-LSTM for CFT contest')

parser.add_argument('--train-data', dest='train_data', help='path to train data', default="")
parser.add_argument('--valid-data', dest='valid_data', help='path to valid data', default="")

parser.add_argument('--restore-model', dest='restore_model',
                    help='path to saved model')  # default=os.path.join(MODELS_DIR, 'Siames_epoch-150.pth'))

parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--save-every', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--parallel', type=int, default=0)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--run', default='none', help='name of current run for tensorboard')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--torch-seed', type=int, default=42)

args = parser.parse_args()

train_loader = FIOLoader(args.train_data, args.batch_size, args.parallel, ru_idx)
valid_loader = FIOLoader(args.valid_data, args.batch_size, args.parallel, ru_idx)

loss = torch.nn.NLLLoss()

model = CorrectorModel(
    embedding_size=train_loader.vectorizer.length,
    conv_sizes=[50, 50],
    out_size=train_loader.diff_vectorizer.length,
    dropout=0.1,
    window=3
)

if args.restore_model:
    ModelSaverCallback.restore_model_from_file(model, args.restore_model, load_with_cpu=(not args.cuda))

optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)

run_name = args.run + '-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

tb_dir = os.path.join('./data/tensorboard', run_name)
if not os.path.exists(tb_dir):
    os.mkdir(tb_dir)


class AccuracyMetric(Metric):
    @property
    def get_name(self):
        return "accuracy"

    def __call__(self, y_pred, y_true):
        y_true = y_true.cpu().numpy()
        y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()

        errors = np.sum(np.abs(y_pred - y_true), axis=1)

        return np.sum(errors > 0) / errors.shape[0]


metrics = [
    AccuracyMetric()
]


class MyReduceLROnPlateau(ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        step = logs["step"]
        if step == 'validation':
            batch_logs = logs.get('batch_logs', {})
            epoch_loss = batch_logs.get('loss')
            if epoch_loss is not None:
                print('reduce lr num_bad_epochs: ', self.lr_sch.num_bad_epochs)
                self.lr_sch.step(epoch_loss, epoch)


callbacks = [
    TensorboardVisualizerCallback(tb_dir),
    ModelSaverCallback('./data/models', epochs=args.epoch, every_n_epoch=args.save_every),
    MyReduceLROnPlateau(optimizer, loss_step="valid", factor=0.5, verbose=True, patience=args.patience)
]

learner = Learner(ClassifierCore(model, optimizer, loss), use_cuda=args.cuda)
learner.train(args.epoch, metrics, train_loader, valid_loader, callbacks=callbacks)
