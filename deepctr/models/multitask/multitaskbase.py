import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Constant


# class MultiTaskModelBase(keras.Model):
#     def compile(self, optimizers, loss_fns, metrics_logger, loss_weights=None, uncertainly=False, run_eagerly=False):
#         if not isinstance(loss_fns, dict):
#             raise ValueError("loss_fns must be a dict")
#         if not isinstance(metrics_logger, dict):
#             raise ValueError("metrics_logger must be a dict")
#         self.uncertainty = uncertainly
#         if loss_weights is None:
#             self.loss_weights = {task_name: 1.0 for task_name in loss_fns.keys()}
#         else:
#             if not isinstance(loss_weights, dict):
#                 raise ValueError("loss_weights must be a dict")
#             self.loss_weights = loss_weights
#         super(MultiTaskModelBase, self).compile(run_eagerly=run_eagerly)
#         self.optimizers = optimizers
#         self.loss_fns = loss_fns
#         self.total_loss_logger = keras.metrics.Mean(name="total_loss")
#         self.loss_logger = {task_name: keras.metrics.Mean(name=task_name + "_loss") for
#                             task_name in self.loss_fns.keys() if self.loss_fns[task_name] is not None}
#         self.metrics_logger = metrics_logger
#         if self.uncertainty:
#             print('loss weights wont be used')
#             self.log_vars_logger = {task_name: keras.metrics.Mean(name="exp(-log_var)_" + task_name) for
#                                     task_name in self.loss_fns.keys() if self.loss_fns[task_name] is not None}
#             self.log_vars = {}
#             for task_name in self.log_vars_logger.keys():
#                 self.log_vars[task_name] = self.add_weight(name='log_var_' + task_name, shape=(1,),
#                                                            initializer=Constant(0.), trainable=True)
#
#     def build(self, input_shape):
#         super(MultiTaskModelBase, self).build(input_shape)
#
#     def train_step(self, data):
#         # Unpack the data. Its structure depends on your model and
#         # on what you pass to `fit()`.
#         if len(data) == 3:
#             x, y, sample_weight = data
#         else:
#             x, y = data
#             sample_weight = {task_name: tf.constant(1.0) for task_name in y.keys()}
#
#
#         with tf.GradientTape() as tape:
#             y_pred = self(x, training=True)
#             total_loss = 0
#             for task_name, loss_fn in self.loss_fns.items():
#                 if loss_fn is not None:
#                     # loss = loss_fn(y[task_name], y_pred[task_name], sample_weight=sample_weight[task_name])
#                     weight = tf.cast(sample_weight[task_name], tf.float32)
#                     loss = tf.reduce_sum(tf.multiply(loss_fn(y[task_name], y_pred[task_name]), weight)) / tf.reduce_sum(weight)
#                     if self.uncertainty:
#                         precision = K.exp(-self.log_vars[task_name])
#                         loss = precision * loss + self.log_vars[task_name]
#                         total_loss += loss
#                         self.log_vars_logger[task_name].update_state(precision)
#                     else:
#                         loss *= self.loss_weights[task_name]
#                         total_loss += loss
#                     self.loss_logger[task_name].update_state(loss)
#                 self.metrics_logger[task_name].update_state(y[task_name], y_pred[task_name],
#                                                             sample_weight=sample_weight[task_name])
#             total_loss += sum(self.losses)
#         self.total_loss_logger.update_state(total_loss)
#         grads = tape.gradient(total_loss, self.trainable_weights)
#         self.optimizers.apply_gradients(zip(grads, self.trainable_weights))
#
#         return {m.name: m.result() for m in self.metrics}
#
#     @property
#     def metrics(self):
#         # We list our `Metric` objects here so that `reset_states()` can be
#         # called automatically at the start of each epoch
#         # or at the start of `evaluate()`.
#         # If you don't implement this property, you have to call
#         # `reset_states()` yourself at the time of your choosing.
#         if self.uncertainty:
#             return [self.total_loss_logger] + \
#                    list(self.metrics_logger.values()) + \
#                    list(self.loss_logger.values()) + \
#                    list(self.log_vars_logger.values())
#         else:
#             return [self.total_loss_logger] + \
#                    list(self.metrics_logger.values()) + \
#                    list(self.loss_logger.values())
#
#     def test_step(self, data):
#         if len(data) == 3:
#             x, y, sample_weight = data
#         else:
#             x, y = data
#             sample_weight = {task_name: tf.constant(1.0) for task_name in y.keys()}
#
#         y_pred = self(x, training=False)
#         total_loss = 0
#         for task_name, loss_fn in self.loss_fns.items():
#             if loss_fn is not None:
#                 # loss = loss_fn(y[task_name], y_pred[task_name], sample_weight=sample_weight[task_name])
#                 weight = tf.cast(sample_weight[task_name], tf.float32)
#                 loss = tf.reduce_sum(tf.multiply(loss_fn(y[task_name], y_pred[task_name]), weight)) / tf.reduce_sum(weight)
#                 if self.uncertainty:
#                     precision = K.exp(-self.log_vars[task_name])
#                     loss = precision * loss + self.log_vars[task_name]
#                     total_loss += loss
#                     self.log_vars_logger[task_name].update_state(precision)
#                 else:
#                     loss *= self.loss_weights[task_name]
#                     total_loss += loss
#                 self.loss_logger[task_name].update_state(loss)
#             self.metrics_logger[task_name].update_state(y[task_name], y_pred[task_name],
#                                                         sample_weight=sample_weight[task_name])
#         total_loss += sum(self.losses)
#         self.total_loss_logger.update_state(total_loss)
#
#         return {m.name: m.result() for m in self.metrics}

class MultiTaskModelBase(keras.Model):
    def compile(self, optimizers, loss_fns, metrics_logger, loss_weights=None,
                uncertainly=False, gradnorm_config=None, run_eagerly=False):
        if not isinstance(loss_fns, dict):
            raise ValueError("loss_fns must be a dict")
        if not isinstance(metrics_logger, dict):
            raise ValueError("metrics_logger must be a dict")
        self.uncertainty = uncertainly
        self.gradnorm_config = gradnorm_config
        if loss_weights is None:
            self.loss_weights = {task_name: 1.0 for task_name in loss_fns.keys()}
        else:
            if not isinstance(loss_weights, dict):
                raise ValueError("loss_weights must be a dict")
            self.loss_weights = loss_weights
        super(MultiTaskModelBase, self).compile(run_eagerly=run_eagerly)
        self.optimizers = optimizers
        self.loss_fns = loss_fns
        self.total_loss_logger = keras.metrics.Mean(name="total_loss")
        self.loss_logger = {task_name: keras.metrics.Mean(name=task_name + "_loss") for
                            task_name in self.loss_fns.keys() if self.loss_fns[task_name] is not None}
        self.metrics_logger = metrics_logger
        if self.uncertainty and (gradnorm_config is None):
            print('uncertainty: loss weights wont be used')
            self.log_vars_logger = {task_name: keras.metrics.Mean(name="exp(-log_var)_" + task_name) for
                                    task_name in self.loss_fns.keys() if self.loss_fns[task_name] is not None}
            self.log_vars = {}
            for task_name in self.log_vars_logger.keys():
                self.log_vars[task_name] = self.add_weight(name='log_var_' + task_name, shape=(1,),
                                                           initializer=Constant(0.), trainable=True)
        elif gradnorm_config is not None:
            print('gradnorm: loss weights wont be used')
            self.grad_norms_loss_logger = keras.metrics.Mean(name="grad_norms_loss")
            # self.last_shared_weights = gradnorm_config['last_shared_weights']

            self.initial_task_losses = []
            self.kick_off_flag = True
            self.dynamic_weights_logger = {task_name: keras.metrics.Mean(name="dyna_weight_logger_" + task_name) for
                                           task_name in self.loss_fns.keys() if self.loss_fns[task_name] is not None}
            self.weight_grad_norm_logger = {task_name: keras.metrics.Mean(name="weight_grad_norm_logger_" + task_name) for
                                            task_name in self.loss_fns.keys() if self.loss_fns[task_name] is not None}
            self.alpha = gradnorm_config['alpha']
            self.dynamic_weights = {}
            for task_name in self.dynamic_weights_logger.keys():
                self.dynamic_weights[task_name] = self.add_weight(name='dyna_weight_' + task_name, shape=(1,),
                                                                  initializer=Constant(1.), trainable=True)

    def build(self, input_shape):
        super(MultiTaskModelBase, self).build(input_shape)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = {task_name: tf.constant(1.0) for task_name in y.keys()}

        with tf.GradientTape(persistent=True) as tape:
            with tf.GradientTape(persistent=True) as tape_weight_grad:
                y_pred = self(x, training=True)
                total_loss = 0
                task_losses = []
                weighted_task_losses = []
                for task_name, loss_fn in self.loss_fns.items():
                    if loss_fn is not None:
                        # loss = loss_fn(y[task_name], y_pred[task_name], sample_weight=sample_weight[task_name])
                        masks = tf.cast(sample_weight[task_name], tf.float32)
                        loss = tf.reduce_sum(tf.multiply(loss_fn(y[task_name], y_pred[task_name]), masks)) / tf.reduce_sum(masks)
                        if self.uncertainty and (self.gradnorm_config is None):
                            precision = K.exp(-self.log_vars[task_name])
                            loss = precision * loss + self.log_vars[task_name]
                            total_loss += loss
                            self.log_vars_logger[task_name].update_state(precision)
                        elif self.gradnorm_config is not None:
                            task_losses.append(loss)
                            if self.kick_off_flag:
                                self.initial_task_losses.append(loss)
                            loss *= self.dynamic_weights[task_name]
                            weighted_task_losses.append(loss)
                            total_loss += loss
                        else:
                            loss *= self.loss_weights[task_name]
                            total_loss += loss
                        self.loss_logger[task_name].update_state(loss)
                    self.metrics_logger[task_name].update_state(y[task_name], y_pred[task_name],
                                                                sample_weight=sample_weight[task_name])
                total_loss += sum(self.losses)
            if self.gradnorm_config is not None:
                last_shared_weights = [weight for weight in self.get_layer('cgc_layer2').trainable_weights
                                       if weight.name.find('share') >= 0][0]
                weight_grad_norms = []
                for i, task_name in enumerate(self.loss_fns.keys()):
                    weight_grad = tape_weight_grad.gradient(weighted_task_losses[i], last_shared_weights)
                    weight_grad_norm = tf.norm(weight_grad, ord=2)
                    weight_grad_norms.append(weight_grad_norm)
                    self.weight_grad_norm_logger[task_name].update_state(weight_grad_norm)
                weight_grad_norms = tf.stack(weight_grad_norms)
                mean_grad_norm = tf.reduce_mean(weight_grad_norms)
                task_losses = tf.stack(task_losses)
                if self.kick_off_flag:
                    self.initial_task_losses = tf.stack(self.initial_task_losses)
                task_loss_ratios = task_losses / self.initial_task_losses
                inverse_task_loss_ratios = task_loss_ratios / tf.reduce_mean(task_loss_ratios)
                target_grad_norms = tf.stop_gradient(mean_grad_norm * (inverse_task_loss_ratios ** self.alpha))
                grad_norms_loss = tf.norm(weight_grad_norms - target_grad_norms, ord=1)
                self.grad_norms_loss_logger.update_state(grad_norms_loss)
        self.total_loss_logger.update_state(total_loss)
        if self.gradnorm_config is not None:
            trainable_weights = [trainable_weight for trainable_weight in self.trainable_weights
                                 if trainable_weight.name.find('dyna_weight') < 0]
            grads = tape.gradient(total_loss, trainable_weights)
            dynamic_weights = list(self.dynamic_weights.values())
            dynamic_weights_grad = tape.gradient(grad_norms_loss, dynamic_weights)
            self.optimizers.apply_gradients(zip(grads+dynamic_weights_grad, trainable_weights+dynamic_weights))
            # renormalize
            normalize_coeff = len(self.dynamic_weights) / tf.reduce_sum(list(self.dynamic_weights.values()))
            for task_name in self.dynamic_weights:
                self.dynamic_weights[task_name].assign(tf.multiply(self.dynamic_weights[task_name], normalize_coeff))
                self.dynamic_weights_logger[task_name].update_state(self.dynamic_weights[task_name])
            # self.dynamic_weights_logger[task_name].update_state(self.dynamic_weights[task_name])
            self.kick_off_flag = False
        else:
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizers.apply_gradients(zip(grads, self.trainable_weights))

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        if self.uncertainty and (self.gradnorm_config is None):
            return [self.total_loss_logger] + \
                   list(self.metrics_logger.values()) + \
                   list(self.loss_logger.values()) + \
                   list(self.log_vars_logger.values())
        elif self.gradnorm_config is not None:
            return [self.total_loss_logger] + \
                   list(self.metrics_logger.values()) + \
                   list(self.loss_logger.values()) + \
                   [self.grad_norms_loss_logger] + \
                   list(self.weight_grad_norm_logger.values()) + \
                   list(self.dynamic_weights_logger.values())
        else:
            return [self.total_loss_logger] + \
                   list(self.metrics_logger.values()) + \
                   list(self.loss_logger.values())

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = {task_name: tf.constant(1.0) for task_name in y.keys()}

        with tf.GradientTape(persistent=True) as tape_weight_grad:
            y_pred = self(x, training=False)
            total_loss = 0
            task_losses = []
            weight_task_losses = []
            for task_name, loss_fn in self.loss_fns.items():
                if loss_fn is not None:
                    # loss = loss_fn(y[task_name], y_pred[task_name], sample_weight=sample_weight[task_name])
                    masks = tf.cast(sample_weight[task_name], tf.float32)
                    loss = tf.reduce_sum(tf.multiply(loss_fn(y[task_name], y_pred[task_name]), masks)) / tf.reduce_sum(masks)
                    if self.uncertainty and (self.gradnorm_config is None):
                        precision = K.exp(-self.log_vars[task_name])
                        loss = precision * loss + self.log_vars[task_name]
                        total_loss += loss
                        self.log_vars_logger[task_name].update_state(precision)
                    elif self.gradnorm_config is not None:
                        task_losses.append(loss)
                        loss *= self.dynamic_weights[task_name]
                        weight_task_losses.append(loss)
                        total_loss += loss
                    else:
                        loss *= self.loss_weights[task_name]
                        total_loss += loss
                    self.loss_logger[task_name].update_state(loss)
                self.metrics_logger[task_name].update_state(y[task_name], y_pred[task_name],
                                                            sample_weight=sample_weight[task_name])
            total_loss += sum(self.losses)
        if self.gradnorm_config is not None:
            weight_grad_norms = []
            for i, task_name in enumerate(self.loss_fns.keys()):
                last_shared_weights = [weight for weight in self.get_layer('cgc_layer2').trainable_weights
                                       if weight.name.find('share') >= 0][0]
                weight_grad = tape_weight_grad.gradient(weight_task_losses[i], last_shared_weights)
                weight_grad_norm = tf.norm(weight_grad, ord=2)
                weight_grad_norms.append(weight_grad_norm)
                self.weight_grad_norm_logger[task_name].update_state(weight_grad_norm)
            weight_grad_norms = tf.stack(weight_grad_norms)
            mean_grad_norm = tf.reduce_mean(weight_grad_norms)
            task_losses = tf.stack(task_losses)
            task_loss_ratios = task_losses / self.initial_task_losses
            inverse_task_loss_ratios = task_loss_ratios / tf.reduce_mean(task_loss_ratios)
            target_grad_norms = tf.stop_gradient(mean_grad_norm * (inverse_task_loss_ratios ** self.alpha))
            grad_norms_loss = tf.norm(weight_grad_norms - target_grad_norms, ord=1)
            self.grad_norms_loss_logger.update_state(grad_norms_loss)
            for task_name in self.dynamic_weights:
                self.dynamic_weights_logger[task_name].update_state(self.dynamic_weights[task_name])
        self.total_loss_logger.update_state(total_loss)

        return {m.name: m.result() for m in self.metrics}
