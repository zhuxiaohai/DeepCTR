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
#
class MultiTaskModelBase(keras.Model):
    def compile(self, optimizers, loss_fns, metrics_logger, loss_weights=None,
                uncertainly=False, gradnorm_config=None, **kwargs):
        if not isinstance(loss_fns, dict):
            raise ValueError("loss_fns must be a dict")
        if not isinstance(metrics_logger, dict):
            raise ValueError("metrics_logger must be a dict")
        if loss_weights is None:
            self.loss_weights = {task_name: 1.0 for task_name in loss_fns.keys()}
        else:
            if not isinstance(loss_weights, dict):
                raise ValueError("loss_weights must be a dict")
            self.loss_weights = loss_weights
        if gradnorm_config is not None:
            kwargs['run_eagerly'] = True
        super(MultiTaskModelBase, self).compile(**kwargs)

        self.uncertainty = uncertainly
        self.gradnorm_config = gradnorm_config
        self.optimizers = optimizers
        self.loss_fns = loss_fns
        with tf.name_scope('metrics_logger'):
            with tf.name_scope('total_loss'):
                self.total_loss_logger = keras.metrics.Mean(name="total_loss")

            self.loss_logger = {}
            for task_name in self.loss_fns.keys():
                if self.loss_fns[task_name] is not None:
                    with tf.name_scope(task_name + "_loss"):
                        self.loss_logger[task_name] = keras.metrics.Mean(name=task_name + "_loss")

            self.metrics_logger = {}
            for task_name, metric in metrics_logger.items():
                with tf.name_scope(task_name + "_" + metric.__name__):
                    self.metrics_logger[task_name] = metric(name=task_name + "_" + metric.__name__)


        if self.uncertainty and (gradnorm_config is None):
            print('uncertainty: loss weights wont be used')

            self.log_vars = {}
            self.log_vars_logger = {}
            for task_name in self.loss_fns.keys():
                if self.loss_fns[task_name] is not None:
                    self.log_vars[task_name] = self.add_weight(name=task_name + '_log_var', shape=(1,),
                                                               initializer=Constant(0.), trainable=True)
                    with tf.name_scope('metrics_logger'):
                        with tf.name_scope(task_name + "_exp(-log_var)"):
                            self.log_vars_logger[task_name] = keras.metrics.Mean(name=task_name + "_exp(-log_var)")

        elif gradnorm_config is not None:
            print('gradnorm: loss weights wont be used')

            self.alpha = gradnorm_config['alpha']
            self.last_shared_weights = gradnorm_config['last_shared_weights']

            with tf.name_scope('metrics_logger'):
                with tf.name_scope('grad_norms_loss'):
                    self.grad_norms_loss_logger = keras.metrics.Mean(name="grad_norms_loss")

            self.dynamic_weights = {}
            self.initial_task_losses = {}
            self.dynamic_weights_logger = {}
            self.weight_grad_norm_logger = {}
            for task_name in self.loss_fns.keys():
                if self.loss_fns[task_name] is not None:
                    self.dynamic_weights[task_name] = self.add_weight(name=task_name + '_dyna_weight', shape=(1,),
                                                                      initializer=Constant(1.), trainable=True)
                    with tf.name_scope('metrics_logger'):
                        with tf.name_scope(task_name + "_dyna_weight"):
                            self.dynamic_weights_logger[task_name] = \
                                keras.metrics.Mean(name=task_name + "_dyna_weight")
                        with tf.name_scope(task_name + "_weight_grad_norm"):
                            self.weight_grad_norm_logger[task_name] = \
                                keras.metrics.Mean(name=task_name + "_weight_grad_norm")

    def build(self, input_shape):
        super(MultiTaskModelBase, self).build(input_shape)

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = {task_name: tf.ones_like(tf.reduce_sum(y[task_name], axis=-1)) for task_name in y.keys()}

        with tf.GradientTape(persistent=True) as tape:
            with tf.GradientTape(persistent=True) as tape_weight_grad:
                y_pred = self(x, training=True)
                total_loss = 0
                if self.gradnorm_config is not None:
                    # renormalize the dynamic_weights to make them add up to the number of tasks
                    normalize_coeff = len(self.dynamic_weights) / \
                                      tf.reshape(tf.reduce_sum(list(self.dynamic_weights.values())), (1,))
                    task_losses = {}
                    weighted_task_losses = {}
                for task_name, loss_fn in self.loss_fns.items():
                    self.metrics_logger[task_name].update_state(y[task_name], y_pred[task_name],
                                                                sample_weight=sample_weight[task_name])
                    if loss_fn is not None:
                        # loss = loss_fn(y[task_name], y_pred[task_name], sample_weight=sample_weight[task_name])
                        masks = tf.cast(sample_weight[task_name], tf.float32)
                        loss = tf.reduce_sum(tf.multiply(loss_fn(y[task_name], y_pred[task_name]), masks)) / \
                               tf.reduce_sum(masks)
                        if self.uncertainty and (self.gradnorm_config is None):
                            precision = K.exp(-self.log_vars[task_name])
                            loss = precision * loss + self.log_vars[task_name]
                            total_loss += loss
                            self.log_vars_logger[task_name].update_state(precision)
                        elif self.gradnorm_config is not None:
                            self.dynamic_weights[task_name].assign(
                                tf.multiply(self.dynamic_weights[task_name], normalize_coeff))
                            self.dynamic_weights_logger[task_name].update_state(self.dynamic_weights[task_name])
                            task_losses[task_name] = loss
                            if self.optimizers.iterations == 0:
                                self.initial_task_losses[task_name] = loss
                            loss *= self.dynamic_weights[task_name]
                            weighted_task_losses[task_name] = loss
                            total_loss += loss
                        else:
                            loss *= self.loss_weights[task_name]
                            total_loss += loss
                        self.loss_logger[task_name].update_state(loss)
                total_loss += sum(self.losses)
            if self.gradnorm_config is not None:
                # last_shared_weights = [weight for weight in self.get_layer('cgc_layer2').trainable_weights
                #                        if weight.name.find('share') >= 0]
                weight_grad_norms = {}
                task_loss_ratios = {}
                for task_name in weighted_task_losses.keys():
                    weight_grad = tape_weight_grad.gradient(weighted_task_losses[task_name], self.last_shared_weights)
                    weight_grad_norm = tf.reduce_sum([tf.norm(i, ord=2) for i in weight_grad])
                    weight_grad_norms[task_name] = weight_grad_norm
                    self.weight_grad_norm_logger[task_name].update_state(weight_grad_norm)
                    task_loss_ratios[task_name] = task_losses[task_name] / self.initial_task_losses[task_name]
                mean_grad_norm = tf.reduce_mean(list(weight_grad_norms.values()))
                inverse_task_loss_ratios = tf.stack(list(task_loss_ratios.values())) \
                                           / tf.reduce_mean(list(task_loss_ratios.values()))
                target_grad_norms = tf.stop_gradient(mean_grad_norm * (inverse_task_loss_ratios ** self.alpha))
                grad_norms_loss = tf.norm(tf.stack(list(weight_grad_norms.values())) - target_grad_norms, ord=1)
                self.grad_norms_loss_logger.update_state(grad_norms_loss)
        self.total_loss_logger.update_state(total_loss)

        if self.gradnorm_config is not None:
            original_weights = [trainable_weight for trainable_weight in self.trainable_weights
                                if trainable_weight.name.find('dyna_weight') < 0]
            grads = tape.gradient(total_loss, original_weights)
            dynamic_weights = list(self.dynamic_weights.values())
            dynamic_weights_grad = tape.gradient(grad_norms_loss, dynamic_weights)
            self.optimizers.apply_gradients(zip(grads+dynamic_weights_grad, original_weights+dynamic_weights))
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
            sample_weight = {task_name: tf.ones_like(tf.reduce_sum(y[task_name], axis=-1)) for task_name in y.keys()}

        with tf.GradientTape(persistent=True) as tape_weight_grad:
            y_pred = self(x, training=False)
            total_loss = 0
            if self.gradnorm_config is not None:
                # renormalize the dynamic_weights to make them add up to the number of tasks
                normalize_coeff = len(self.dynamic_weights) / \
                                  tf.reshape(tf.reduce_sum(list(self.dynamic_weights.values())), (1,))
                task_losses = {}
                weighted_task_losses = {}
            for task_name, loss_fn in self.loss_fns.items():
                self.metrics_logger[task_name].update_state(y[task_name], y_pred[task_name],
                                                            sample_weight=sample_weight[task_name])
                if loss_fn is not None:
                    # loss = loss_fn(y[task_name], y_pred[task_name], sample_weight=sample_weight[task_name])
                    masks = tf.cast(sample_weight[task_name], tf.float32)
                    loss = tf.reduce_sum(tf.multiply(loss_fn(y[task_name], y_pred[task_name]), masks)) / \
                           tf.reduce_sum(masks)
                    if self.uncertainty and (self.gradnorm_config is None):
                        precision = K.exp(-self.log_vars[task_name])
                        loss = precision * loss + self.log_vars[task_name]
                        total_loss += loss
                        self.log_vars_logger[task_name].update_state(precision)
                    elif self.gradnorm_config is not None:
                        self.dynamic_weights[task_name].assign(
                            tf.multiply(self.dynamic_weights[task_name], normalize_coeff))
                        self.dynamic_weights_logger[task_name].update_state(self.dynamic_weights[task_name])
                        task_losses[task_name] = loss
                        if self.optimizers.iterations == 0:
                            self.initial_task_losses[task_name] = loss
                        loss *= self.dynamic_weights[task_name]
                        weighted_task_losses[task_name] = loss
                        total_loss += loss
                    else:
                        loss *= self.loss_weights[task_name]
                        total_loss += loss
                    self.loss_logger[task_name].update_state(loss)
            total_loss += sum(self.losses)
        if self.gradnorm_config is not None:
            # last_shared_weights = [weight for weight in self.get_layer('cgc_layer2').trainable_weights
            #                        if weight.name.find('share') >= 0]
            weight_grad_norms = {}
            task_loss_ratios = {}
            for task_name in weighted_task_losses.keys():
                weight_grad = tape_weight_grad.gradient(weighted_task_losses[task_name], self.last_shared_weights)
                weight_grad_norm = tf.reduce_sum([tf.norm(i, ord=2) for i in weight_grad])
                weight_grad_norms[task_name] = weight_grad_norm
                self.weight_grad_norm_logger[task_name].update_state(weight_grad_norm)
                task_loss_ratios[task_name] = task_losses[task_name] / self.initial_task_losses[task_name]
            mean_grad_norm = tf.reduce_mean(list(weight_grad_norms.values()))
            inverse_task_loss_ratios = tf.stack(list(task_loss_ratios.values())) \
                                       / tf.reduce_mean(list(task_loss_ratios.values()))
            target_grad_norms = tf.stop_gradient(mean_grad_norm * (inverse_task_loss_ratios ** self.alpha))
            grad_norms_loss = tf.norm(tf.stack(list(weight_grad_norms.values())) - target_grad_norms, ord=1)
            self.grad_norms_loss_logger.update_state(grad_norms_loss)
        self.total_loss_logger.update_state(total_loss)

        return {m.name: m.result() for m in self.metrics}