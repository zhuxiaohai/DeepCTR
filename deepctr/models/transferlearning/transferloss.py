"""
Author:
    Xiaohai Zhu, zhuxiaohai_sast@163.com

"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import control_flow_util
from deepctr.layers.core import DNN


@tf.custom_gradient
def gradient_reversal(x, scale=tf.constant(1.0, dtype=tf.float32)):
    def grad(dy):
        return -dy * scale, None
    return x, grad


class GradientReversalLayer(Layer):
    def __init__(self, gamma=10.0, max_iter_num=1000, **kwargs):
        self.step = tf.Variable(0, trainable=False)
        self.gamma = gamma
        self.max_iter_num = max_iter_num
        super(GradientReversalLayer, self).__init__(**kwargs)

    def grl_lambda_schedule(self):
        process_ratio = self.step / self.max_iter_num
        self.step.assign_add(1)
        return tf.cast(2.0 / (1.0 + tf.exp(-self.gamma * process_ratio)) - 1.0, tf.float32)

    def call(self, x, training=True, **kwargs):
        true_branch = lambda: self.grl_lambda_schedule()
        false_branch = lambda: tf.constant(-1.0, dtype=tf.float32)
        grl_lambda = control_flow_util.smart_cond(training, true_branch, false_branch)
        return gradient_reversal(x, grl_lambda)

    def get_config(self):
        config = {'gamm': self.gamma,
                  'max_iter_num': self.max_iter_num
                  }
        base_config = super(GradientReversalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape, input_shape


class DomainAdversarialLoss(Layer):
    """
    calculate the loss between source domain and target domain using DANN
      Inputs
        - a list of 2 tensors: source, target
        with shape: ``(batch_size,units), (batch_size,units)

      Output
        - loss, scalar

    References
      - Domain-adversarial neural network (https://arxiv.org/pdf/1505.07818.pdf)
    """
    def __init__(self, gamma_grl=10.0, max_iter_grl=1000,
                 dnn_units=(8,), dnn_activation='relu', l2_reg_dnn=0.0, dnn_dropout=0.0,
                 use_bn=False, seed=1024, **kwargs):
        self.gamma_grl = gamma_grl
        self.max_iter_grl = max_iter_grl
        self.dnn_units = dnn_units
        self.dnn_activation = dnn_activation
        self.l2_reg_dnn = l2_reg_dnn
        self.dnn_dropout = dnn_dropout
        self.use_bn = use_bn
        self.seed = seed
        self.grl = GradientReversalLayer(gamma=gamma_grl, max_iter_num=max_iter_grl)
        if self.dnn_units is not None:
            self.domain_classifier = DNN(self.dnn_units, self.dnn_activation, self.l2_reg_dnn,
                                         self.dnn_dropout, self.use_bn, seed=self.seed, name='domain_classifier_dnn')
        self.domain_predictor = tf.keras.layers.Dense(2, use_bias=True, activation='softmax', name='domain_classifier_prediction')
        self.accuracy_fn = tf.keras.metrics.CategoricalAccuracy(name='domain_classifier_accuracy')
        self.conditional_capability = False
        super(DomainAdversarialLoss, self).__init__(**kwargs)

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def call(self, source, target, training=True, **kwargs):
        if (source is not None) & (target is not None):
            y_true = K.concatenate((tf.one_hot(tf.zeros_like(source, dtype=tf.int32)[:, 0], depth=2, axis=-1),
                                    tf.one_hot(tf.ones_like(target, dtype=tf.int32)[:, 0], depth=2, axis=-1)),
                                   0)
            inputs = K.concatenate((source, target), 0)
        elif source is not None:
            y_true = tf.one_hot(tf.zeros_like(source, dtype=tf.int32)[:, 0], depth=2, axis=-1)
            inputs = source
        elif target is not None:
            y_true = tf.one_hot(tf.ones_like(target, dtype=tf.int32)[:, 0], depth=2, axis=-1)
            inputs = target

        inputs = self.grl(inputs, training=training)
        if self.dnn_units is not None:
            inputs = self.domain_classifier(inputs)
        y_pred = self.domain_predictor(inputs)
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        loss = tf.reduce_mean(loss)
        # log metrics
        acc = self.accuracy_fn(y_true, y_pred)
        self.add_metric(acc, name="domain_classifier_accuracy")
        self.add_metric(loss, name="da_loss")
        return loss

    def get_config(self):
        config = {'gamm_grl': self.gamma_grl,
                  'max_iter_grl': self.max_iter_grl,
                  'dnn_units': self.dnn_units,
                  'dnn_activation': self.dnn_activation,
                  'l2_reg_dnn': self.l2_reg_dnn,
                  'dnn_dropout': self.dnn_dropout,
                  'use_bn': self.use_bn,
                  'seed': self.seed
                  }
        base_config = super(DomainAdversarialLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return ()


class MMDLoss(Layer):
    """
    calculate the mmd between source domain and target domain using DAN
      Inputs
        - a list of 2 tensors: source, target
        with shape: ``(batch_size,units), (batch_size,units)

      Output
        - loss, scalar

    References
      - Deep domain confusion network (https://arxiv.org/abs/1412.3474)
    """
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        self.conditional_capability = False
        super(MMDLoss, self).__init__(**kwargs)

    def guassian_kernel(self, x, y, kernel_mul, kernel_num, fix_sigma):
        n_samples = tf.cast(tf.shape(x)[0] + tf.shape(y)[0], x.dtype)
        total = tf.concat([x, y], axis=0)
        total0 = tf.expand_dims(total, 0)
        total1 = tf.expand_dims(total, 1)
        L2_distance = tf.reduce_sum((total0 - total1)**2, axis=2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = tf.reduce_sum(L2_distance) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul**(kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = tf.stack([tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list], axis=0)
        return tf.reduce_sum(kernel_val, axis=0)

    def linear_mmd(self, x, y):
        delta = tf.reduce_mean(x, axis=0) - tf.reduce_mean(y, axis=0)
        loss = tf.reduce_sum(tf.multiply(delta, delta))
        return loss

    def call(self, source, target, training=None, **kwargs):
        source_number = tf.shape(source)[0]
        if self.kernel_type == 'linear':
            loss = self.linear_mmd(source, target)
        elif self.kernel_type == 'rbf':
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = tf.reduce_mean(kernels[:source_number, :source_number])
            YY = tf.reduce_mean(kernels[source_number:, source_number:])
            XY = tf.reduce_mean(kernels[:source_number, source_number:])
            YX = tf.reduce_mean(kernels[source_number:, :source_number])
            loss = XX + YY - XY - YX
        else:
            ValueError('{} not supported'.format(self.kernel_type))
        self.add_metric(loss, name="mmd_distance")
        return loss

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def get_config(self):
        config = {'kernel_num': self.kernel_num,
                  'kernel_mul': self.kernel_mul,
                  'fix_sigma': self.fix_sigma,
                  'kernel_type': self.kernel_type,
                  }
        base_config = super(MMDLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return ()


class LMMDLoss(MMDLoss):
    """
    calculate the local mmd (conditional distribution) between source domain and target domain using DSAN
      Inputs
        - a list of 2 tensors: source, target
        with shape: ``(batch_size,units), (batch_size,units)

      Output
        - loss, scalar

    References
      - Zhu et al., Deep Subdomain Adaptation Network for Image Classification,
      IEEE Transactions on Neural Networks and Learning Systems, 2020b
    """
    def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None,
                 gamma=1.0, max_iter_num=1000, **kwargs):
        if kernel_type != 'rbf':
            raise NotImplementedError("{} is not supported yet.".format(kernel_type))
        super(LMMDLoss, self).__init__(kernel_type, kernel_mul, kernel_num, fix_sigma, **kwargs)
        self.gamma = gamma
        self.max_iter_num = max_iter_num
        self.num_class = num_class
        self.step = tf.Variable(0, trainable=False)
        self.conditional_capability = True

    def call(self, source, target, source_label, target_logits, training=None, **kwargs):
        source_size = tf.shape(source)[0]
        # (B, B)
        weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits)
        kernels = self.guassian_kernel(source, target,
                                       kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)

        SS = kernels[:source_size, :source_size]
        TT = kernels[source_size:, source_size:]
        ST = kernels[:source_size, source_size:]

        loss = tf.reduce_sum(weight_ss * SS) + tf.reduce_sum(weight_tt * TT) - 2 * tf.reduce_sum(weight_st * ST)

        # Dynamic weighting
        true_branch = lambda: self.lambda_schedule()
        false_branch = lambda: tf.constant(1.0, dtype=tf.float32)
        lamb = control_flow_util.smart_cond(training, true_branch, false_branch)
        loss = loss * lamb
        self.add_metric(loss, 'lmmd_distance')
        return loss

    def cal_weight(self, source_label, target_logits):
        source_size = tf.shape(source_label)[0]
        target_size = tf.shape(target_logits)[0]

        # source: should be one-hotted label
        source_label_onehot = source_label
        source_label_sum = tf.reduce_sum(source_label_onehot, axis=0, keepdims=True)
        source_label_sum = tf.where(source_label_sum == 0, 100.0, source_label_sum)
        source_label_onehot = source_label_onehot / source_label_sum

        # target: pseudo label. Note that target_logits should be multiclass-softmaxed beforehand
        target_label = tf.argmax(target_logits, 1)
        target_label_onehot = tf.one_hot(target_label, depth=self.num_class, axis=-1)
        target_logits_sum = tf.reduce_sum(target_logits, axis=0, keepdims=True)
        target_logits_sum = tf.where(target_logits_sum == 0, 100.0, target_logits_sum)
        target_logits = target_logits / target_logits_sum

        weight_ss = tf.zeros((source_size, source_size))
        weight_tt = tf.zeros((target_size, target_size))
        weight_st = tf.zeros((source_size, target_size))

        count = 0.0
        for i in range(self.num_class):
            if (tf.reduce_sum(source_label_onehot[:, i]) > 0) & (tf.reduce_sum(target_label_onehot[:, i]) > 0):
                # (B, 1)
                s_tvec = tf.reshape(source_label_onehot[:, i], (source_size, -1))
                t_tvec = tf.reshape(target_logits[:, i], (target_size, -1))

                # (B, B)
                ss = tf.matmul(s_tvec, tf.transpose(s_tvec))
                weight_ss += ss
                tt = tf.matmul(t_tvec, tf.transpose(t_tvec))
                weight_tt += tt
                st = tf.matmul(s_tvec, tf.transpose(t_tvec))
                weight_st += st
                count += 1.0

        if count > 0.0:
            weight_ss = weight_ss / count
            weight_tt = weight_tt / count
            weight_st = weight_st / count
        else:
            weight_ss = tf.constant(0.0)
            weight_tt = tf.constant(0.0)
            weight_st = tf.constant(0.0)
        return weight_ss, weight_tt, weight_st

    def lambda_schedule(self):
        process_ratio = self.step / self.max_iter_num
        self.step.assign_add(1)
        return tf.cast(2.0 / (1.0 + tf.exp(-self.gamma * process_ratio)) - 1.0, tf.float32)

    def get_config(self):
        config = {'gamm': self.gamma,
                  'max_iter_num': self.max_iter_num,
                  'num_class': self.num_class,
                  }
        base_config = super(LMMDLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return ()


