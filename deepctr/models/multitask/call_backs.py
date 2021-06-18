import platform

import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.models import save_model, Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Lambda

import numpy as np


class MyRecorder(Callback):
    def __init__(self, log_dir, data, gradient_freq=1, experts_freq=1, batch_size=256):
        self.gradient_freq = gradient_freq
        self.batch_size = batch_size
        self.experts_freq = experts_freq
        if platform.system() == 'Windows':
            self.joint_symbol = '\\'
        else:
            self.joint_symbol = '/'
        self.writer = tf.summary.create_file_writer(self.joint_symbol.join([log_dir, 'callback']))
        self.data = data
        super(MyRecorder, self).__init__()

    def _log_experts(self, epoch):
        if len(self.data) == 3:
            x, y, sample_weight = self.data
        else:
            x, y = self.data
        outputs = {layer.name: layer.gates_output
                   for layer in self.model.layers
                   if (layer.name.find('cgc_layer') >= 0) or (layer.name.find('mmoe') >= 0)}
        intermediate_model = Model(self.model.input, outputs=Lambda(lambda fn: outputs)(self.model.input))
        intermediate_results = intermediate_model(x, training=False)
        with self.writer.as_default():
            for layer_name, layer_outputs in intermediate_results.items():
                for gate_name, gate_output in layer_outputs.items():
                    for i in range(gate_output.shape[-1]):
                        tf.summary.scalar(layer_name + '_gate_' + gate_name + '_expert_' + str(i),
                                          data=tf.reduce_mean(gate_output[:, i]),
                                          step=epoch)
        self.writer.flush()

    def _log_gradients(self, epoch):
        if len(self.data) == 3:
            x, y, sample_weight = self.data
        else:
            x, y = self.data
            sample_weight = {task_name: tf.constant(1.0) for task_name in y.keys()}

        with tf.GradientTape(persistent=True) as tape:
            with tf.GradientTape(persistent=True) as tape_weight_grad:
                y_pred = self.model(x, training=False)
                total_loss = 0
                if self.model.gradnorm_config is not None:
                    # renormalize the dynamic_weights to make them add up to the number of tasks
                    normalize_coeff = len(self.model.dynamic_weights) / \
                                      tf.reshape(tf.reduce_sum(list(self.model.dynamic_weights.values())), (1,))
                    task_losses = {}
                    weighted_task_losses = {}
                for task_name, loss_fn in self.model.loss_fns.items():
                    if loss_fn is not None:
                        # loss = loss_fn(y[task_name], y_pred[task_name], sample_weight=sample_weight[task_name])
                        masks = tf.cast(sample_weight[task_name], tf.float32)
                        loss = tf.reduce_sum(tf.multiply(loss_fn(y[task_name], y_pred[task_name]), masks)) / \
                               tf.reduce_sum(masks)
                        if self.model.uncertainty and (self.model.gradnorm_config is None):
                            precision = K.exp(-self.model.log_vars[task_name])
                            loss = precision * loss + self.model.log_vars[task_name]
                            total_loss += loss
                        elif self.model.gradnorm_config is not None:
                            self.model.dynamic_weights[task_name].assign(
                                tf.multiply(self.model.dynamic_weights[task_name], normalize_coeff))
                            task_losses[task_name] = loss
                            if self.model.optimizers.iterations == 0:
                                self.model.initial_task_losses[task_name] = loss
                            loss *= self.model.dynamic_weights[task_name]
                            weighted_task_losses[task_name] = loss
                            total_loss += loss
                        else:
                            loss *= self.model.loss_weights[task_name]
                            total_loss += loss
                total_loss += sum(self.model.losses)
            if self.model.gradnorm_config is not None:
                # last_shared_weights = [weight for weight in self.get_layer('cgc_layer2').trainable_weights
                #                        if weight.name.find('share') >= 0]
                weight_grad_norms = {}
                task_loss_ratios = {}
                for task_name in weighted_task_losses.keys():
                    weight_grad = tape_weight_grad.gradient(weighted_task_losses[task_name], self.model.last_shared_weights)
                    weight_grad_norm = tf.reduce_sum([tf.norm(i, ord=2) for i in weight_grad])
                    weight_grad_norms[task_name] = weight_grad_norm
                    task_loss_ratios[task_name] = task_losses[task_name] / self.model.initial_task_losses[task_name]
                mean_grad_norm = tf.reduce_mean(list(weight_grad_norms.values()))
                inverse_task_loss_ratios = tf.stack(list(task_loss_ratios.values())) \
                                           / tf.reduce_mean(list(task_loss_ratios.values()))
                target_grad_norms = tf.stop_gradient(mean_grad_norm * (inverse_task_loss_ratios ** self.model.alpha))
                grad_norms_loss = tf.norm(tf.stack(list(weight_grad_norms.values())) - target_grad_norms, ord=1)

        with self.writer.as_default():
            if self.model.gradnorm_config is not None:
                original_weights = [trainable_weight for trainable_weight in self.model.trainable_weights
                                    if trainable_weight.name.find('dyna_weight') < 0]
                grads = tape.gradient(total_loss, original_weights)
                dynamic_weights = list(self.model.dynamic_weights.values())
                dynamic_weights_grad = tape.gradient(grad_norms_loss, dynamic_weights)
                # In eager mode, grads does not have name, so we get names from model.trainable_weights
                for weights, grad in zip(original_weights+dynamic_weights, grads+dynamic_weights_grad):
                    tf.summary.histogram(
                        weights.name.replace(':', '_') + '_grads', data=grad, step=epoch)
            else:
                grads = tape.gradient(total_loss, self.model.trainable_weights)
                # In eager mode, grads does not have name, so we get names from model.trainable_weights
                for weights, grad in zip(self.model.trainable_weights, grads):
                    tf.summary.histogram(
                        weights.name.replace(':', '_') + '_grads', data=grad, step=epoch)
        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        if self.gradient_freq and epoch % self.gradient_freq == 0:
            self._log_gradients(epoch)
        if self.experts_freq and epoch % self.experts_freq == 0:
            self._log_experts(epoch)


class MyEarlyStopping(Callback):
    def __init__(self, monitor, patience=0, savepath=None, coef_of_balance=0.2, direction='maximize'):
        super(MyEarlyStopping, self).__init__()
        self.patience = patience
        self.savepath = savepath
        if platform.system() == 'Windows':
            self.joint_symbol = '\\'
        else:
            self.joint_symbol = '/'
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
        self.model.save_weights(self.joint_symbol.join([self.savepath, 'model-{:02d}-{:.5f}']).format(
            self.current_epoch + 1, K.get_value(self.model.optimizer.lr)))
        print('\n')
        print("Restoring model weights from the end of the best epoch: %05d" % (self.best_epoch + 1))
        self.model.set_weights(self.best_weights)
        save_model(self.model, self.joint_symbol.join([self.savepath, 'best_model_epoch{}_{}{:.4f}.h5']).format(
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
                self.model.save_weights(self.joint_symbol.join([self.savepath, 'model-{:02d}-{:.5f}']).format(
                    self.current_epoch + 1, K.get_value(self.model.optimizer.lr)))
                self.model.stop_training = True
                print('\n')
                print("Restoring model weights from the end of the best epoch: %05d" % (self.best_epoch + 1))
                self.model.set_weights(self.best_weights)
                save_model(self.model, self.joint_symbol.join([self.savepath, 'best_model_epoch{}_{}{:.4f}.h5']).format(
                    self.best_epoch + 1, self.monitor, self.best_monitor))

    def on_train_end(self, logs=None):
        if self.current_epoch == self.best_epoch:
            self.model.save_weights(self.joint_symbol.join([self.savepath, 'model-{:02d}-{:.5f}']).format(
                self.current_epoch + 1, K.get_value(self.model.optimizer.lr)))
            print('\n')
            print("Restoring model weights from the end of the best epoch: %05d" % (self.best_epoch + 1))
            self.model.set_weights(self.best_weights)
            save_model(self.model, self.joint_symbol.join([self.savepath, 'best_model_epoch{}_{}{:.4f}.h5']).format(
                self.best_epoch + 1, self.monitor, self.best_monitor))
        else:
            print("Epoch %05d: early stopping" % (self.current_epoch + 1))