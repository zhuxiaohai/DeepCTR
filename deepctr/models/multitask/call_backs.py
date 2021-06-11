from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.models import save_model
from tensorflow.python.keras import backend as K

import numpy as np


class MyEarlyStopping(Callback):
    def __init__(self, monitor, patience=0, savepath=None, coef_of_balance=0.2, direction='maximize'):
        super(MyEarlyStopping, self).__init__()
        self.patience = patience
        self.savepath = savepath
        self.best_weights = None
        self.coef_of_balance = coef_of_balance
        self.monitor = monitor
        self.direction = direction
        self.wait = 0
        self.current_epoch = 0
        self.best = None
        self.best_epoch = None
        self.best_monitor = None

    def _compare_op(self, x, y):
        if self.direction == 'maximize':
            return x > y
        else:
            return x < y

    def _save_model(self):
        self.model.save_weights(self.savepath + '\model-{:02d}-{:.5f}'.format(
            self.current_epoch + 1, K.get_value(self.model.optimizer.lr)))
        print('\n')
        print("Restoring model weights from the end of the best epoch: %05d" % (self.best_epoch + 1))
        self.model.set_weights(self.best_weights)
        save_model(self.model, self.savepath + '\\best_model_epoch{}_{}{:.4f}.h5'.format(
            self.best_epoch + 1, self.monitor, self.best_monitor))

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.current_epoch = 0
        if self.direction == 'maximize':
            self.best = -np.inf
        else:
            self.best = np.inf
        self.best_epoch = -1
        self.best_monitor = None

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch
        if self.direction == 'maximize':
            current_metric = logs[self.monitor] - \
                             self.coef_of_balance * (abs(logs[self.monitor] - logs[self.monitor[4:]]))
        else:
            current_metric = logs[self.monitor] + \
                             self.coef_of_balance * (abs(logs[self.monitor] - logs[self.monitor[4:]]))

        if self._compare_op(current_metric, self.best):
            self.best = current_metric
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.best_monitor = logs[self.monitor]
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.save_weights(self.savepath + '\model-{:02d}-{:.5f}'.format(
                    self.current_epoch + 1, K.get_value(self.model.optimizer.lr)))
                self.model.stop_training = True
                print('\n')
                print("Restoring model weights from the end of the best epoch: %05d" % (self.best_epoch + 1))
                self.model.set_weights(self.best_weights)
                save_model(self.model, self.savepath + '\\best_model_epoch{}_{}{:.4f}.h5'.format(
                    self.best_epoch + 1, self.monitor, self.best_monitor))

    def on_train_end(self, logs=None):
        if self.current_epoch == self.best_epoch:
            self.model.save_weights(self.savepath + '\model-{:02d}-{:.5f}'.format(
                self.current_epoch + 1, K.get_value(self.model.optimizer.lr)))
            print('\n')
            print("Restoring model weights from the end of the best epoch: %05d" % (self.best_epoch + 1))
            self.model.set_weights(self.best_weights)
            save_model(self.model, self.savepath + '\\best_model_epoch{}_{}{:.4f}.h5'.format(
                self.best_epoch + 1, self.monitor, self.best_monitor))
        else:
            print("Epoch %05d: early stopping" % (self.current_epoch + 1))