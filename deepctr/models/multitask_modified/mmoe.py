# -*- coding:utf-8 -*-
"""
Author:
    Xiaohai Zhu(zhuxiaohai_sast@163.com)

Reference:
    [1] [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in
    Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
"""
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.initializers import glorot_normal, Zeros
from tensorflow.python.keras import activations

from deepctr.feature_column import build_input_features, input_from_feature_columns, get_linear_logit
from deepctr.layers.utils import combined_dnn_input, add_func
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.models.multitask_modified.multitaskbase import MultiTaskModelBase


class MMOELayer(Layer):
    """
    The Multi-gate Mixture-of-Experts layer in MMOE model
      Input shape
        - 2D tensor with shape: ``(batch_size,units)``.

      Output shape
        - A list with **num_tasks** elements, which is a 2D tensor with shape: ``(batch_size, output_dim)`` .

      Arguments
        - **num_tasks**: integer, the number of tasks, equal to the number of outputs.
        - **num_experts**: integer, the number of experts.
        - **output_dim**: integer, the dimension of each output of MMOELayer.

    References
      - [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
    """
    def __init__(self, tasks, num_experts, output_dim, expert_activation='relu', seed=1024, **kwargs):
        self.tasks = tasks
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.expert_activation = expert_activation
        self.seed = seed
        self.gates = None
        self.gates_output = None
        super(MMOELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dimension = input_shape[-1]

        self.expert_kernel = self.add_weight(
            name='expert_kernel',
            shape=(input_dimension, self.output_dim * self.num_experts),
            dtype=tf.float32,
            initializer=glorot_normal(seed=self.seed),
        )

        self.expert_bias = self.add_weight(
            name='expert_bias',
            shape=(self.output_dim * self.num_experts),
            dtype=tf.float32,
            initializer=Zeros(),
        )

        self.gates = {}
        for task_name in self.tasks:
            self.gates[task_name] = {
                'gate_kernel_{}'.format(task_name):
                self.add_weight(name='gate_kernel_{}'.format(task_name),
                                shape=(input_dimension, self.num_experts),
                                dtype=tf.float32,
                                initializer=glorot_normal(seed=self.seed)
                                ),
                'gate_bias_{}'.format(task_name):
                self.add_weight(
                    name='gate_bias_{}'.format(task_name),
                    shape=(self.num_experts,),
                    dtype=tf.float32,
                    initializer=Zeros())}

        super(MMOELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = {}
        expert_out = tf.nn.bias_add(tf.tensordot(inputs, self.expert_kernel, axes=(-1, 0)), self.expert_bias)
        expert_out = tf.reshape(expert_out, [-1, self.output_dim, self.num_experts])
        if self.expert_activation is not None:
           expert_out = activations.get(self.expert_activation)(expert_out)
        self.gates_output = {}
        for task_name in self.tasks:
            gate_out = tf.tensordot(inputs, self.gates[task_name]['gate_kernel_{}'.format(task_name)], axes=(-1, 0))
            gate_out = tf.nn.bias_add(gate_out, self.gates[task_name]['gate_bias_{}'.format(task_name)])
            gate_out = tf.nn.softmax(gate_out)
            self.gates_output[task_name] = gate_out
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=1), [1, self.output_dim, 1])
            output = tf.reduce_sum(tf.multiply(expert_out, gate_out), axis=2)
            outputs[task_name] = output
        return outputs

    def get_config(self):
        config = {'tasks': self.tasks,
                  'num_experts': self.num_experts,
                  'output_dim': self.output_dim,
                  'expert_activation': self.expert_activation,
                  }
        base_config = super(MMOELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return {task_name: (input_shape[0], self.output_dim) for task_name in self.tasks}


# class MultiLossLayer(Layer):
#     def __init__(self, tasks, **kwargs):
#         self.num_tasks = len(tasks)
#         self.tasks = tasks
#         self.task_metrics_fn = {}
#         self.task_loss_fn = {}
#         for task_name, task_type in tasks.items():
#             with tf.name_scope(task_name):
#                 self.task_loss_fn[task_name] = Mean(name=task_name + '_loss')
#                 if task_type == 'binary':
#                     self.task_metrics_fn[task_name] = AUC(name=task_name + '_auc')
#         super(MultiLossLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # initialise log_vars
#         self.log_vars = []
#         for task_name in self.tasks.keys():
#             self.log_vars += [self.add_weight(name='log_var_' + task_name, shape=(1,),
#                                               initializer=Constant(0.), trainable=True)]
#         super(MultiLossLayer, self).build(input_shape)
#
#     def multi_loss(self, ys_true, ys_pred, tasks):
#         assert len(ys_true) == self.num_tasks and len(ys_pred) == self.num_tasks
#         total_loss = 0
#         for y_true, y_pred, task_name, task_type, log_var \
#                 in zip(ys_true, ys_pred, tasks.keys(), tasks.values(), self.log_vars):
#             precision = K.exp(-log_var)
#             if task_type == 'binary':
#                 loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
#                 self.add_metric(self.task_metrics_fn[task_name](y_true, y_pred), name=task_name + '_auc')
#             else:
#                 loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
#             self.add_metric(self.task_loss_fn[task_name](loss), name=task_name + '_loss')
#             total_loss += precision * loss + log_var
#         return tf.reduce_mean(total_loss)
#
#     def call(self, inputs):
#         ys_true = inputs[:self.num_tasks]
#         ys_pred = inputs[self.num_tasks:]
#         loss = self.multi_loss(ys_true, ys_pred, self.tasks)
#         self.add_loss(loss, inputs=inputs)
#         for task_name, log_var in zip(self.tasks.keys(), self.log_vars):
#             self.add_metric(tf.exp(-log_var), name='exp(-log_var)_' + task_name, aggregation='mean')
#         # We won't actually use the output.
#         return K.concatenate(inputs, -1)
#
#     def get_config(self):
#         config = {'tasks': self.tasks}
#         base_config = super(MultiLossLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


def MMOE(dnn_feature_columns, tasks, num_experts=4, expert_dim=8,
         bottom_shared_units=(128, 128), bottom_shared_use_bn=False, l2_reg_embedding=1e-5, l2_reg_dnn=0,
         dnn_dropout=0, dnn_activation='relu', task_dnn_units=None, task_use_bn=False, seed=1024):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param tasks: dict, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"``
    for regression loss. e.g. {'task1': 'binary', 'task2': 'regression'}
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param bottom_shared_units: list,list of positive integer or empty list, the layer number and units in each layer
    of shared-bottom DNN
    :param bottom_shared_use_bn: bool, whether to use batch normalization in bottom shared dnn
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer
    of task-specific DNN
    :param task_use_bn: whether to use batch normalization in task towers
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN

    :return: a Keras model instance.
    """
    num_tasks = len(tasks)
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    if bottom_shared_units is not None:
        dnn_input = DNN(bottom_shared_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                        bottom_shared_use_bn, seed=seed, name='bottom_shared_dnn')(dnn_input)

    mmoe_outs = MMOELayer(list(tasks.keys()), num_experts, expert_dim, dnn_activation, seed)(dnn_input)

    # if task_dnn_units != None:
    #     mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(mmoe_out)
    #                  for mmoe_out in mmoe_outs]

    task_outputs = {}
    for mmoe_out, task_name, task_type in zip(mmoe_outs.values(), tasks.keys(), tasks.values()):
        if task_dnn_units != None:
            mmoe_out = DNN(task_dnn_units,
                           dnn_activation,
                           l2_reg_dnn,
                           dnn_dropout,
                           task_use_bn,
                           seed=seed,
                           name=task_name+'_dnn')(mmoe_out)
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None, name=task_name+'_logit')(mmoe_out)
        output = PredictionLayer(task_type, name=task_name+'_prediction')(logit)
        task_outputs[task_name] = output

    # if use_uncertainty:
    #     ys_true = [Input(shape=(1,)) for _ in tasks]
    #     loss_layer_inputs = ys_true + task_outputs
    #     model_out = MultiLossLayer(tasks)(loss_layer_inputs)
    #     model_inputs = inputs_list + ys_true
    #     model = tf.keras.models.Model(model_inputs, model_out)
    #     return model
    # else:
    #     model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outputs)
    #     return model

    model = MultiTaskModelBase(inputs=inputs_list, outputs=task_outputs)
    return model


def MMOE_BIAS(dnn_feature_columns, tasks, bias_feature_columns_dict=None, bias_dropout_dict=None, bias_l2_reg=1e-5,
              num_experts=4, expert_dim=8,
              bottom_shared_units=(128, 128), bottom_shared_use_bn=False,
              l2_reg_embedding=1e-5, l2_reg_dnn=0,
              dnn_dropout=0, dnn_activation='relu',
              task_dnn_units=None, task_use_bn=False, seed=1024):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param tasks: dict, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"``
    for regression loss. e.g. {'task1': 'binary', 'task2': 'regression'}
    :param bias_feature_columns_dict: dict. e.g. {'task1': [feature_column1, 2...], 'task2': [feature_column1, 3...]}
    :param bias_dropout_dict: dict. e.g. {'task1': 0.2, 'task2': 0.3}
    :param bias_l2_reg: float.
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param bottom_shared_units: list,list of positive integer or empty list, the layer number and units in each layer
    of shared-bottom DNN
    :param bottom_shared_use_bn: bool, whether to use batch normalization in bottom shared dnn
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer
    of task-specific DNN
    :param task_use_bn: whether to use batch normalization in task towers
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN

    :return: a Keras model instance
    """
    num_tasks = len(tasks)
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")

    bias_feature_columns_list = []
    if bias_feature_columns_dict:
        for columns in bias_feature_columns_dict.values():
            bias_feature_columns_list += columns
    features = build_input_features(dnn_feature_columns + bias_feature_columns_list)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)


    # if bias_feature_columns:
    #     bias_features_dict = OrderedDict()
    #     bias_input = {}
    #     for task_name in bias_feature_columns.keys():
    #         bias_features = build_input_features(bias_feature_columns[task_name])
    #         bias_features_dict.update(bias_features)
    #         sparse_embedding_list, dense_value_list = input_from_feature_columns(bias_features, bias_feature_columns[task_name],
    #                                                                              l2_reg_embedding, seed)
    #         bias_input[task_name] = combined_dnn_input(sparse_embedding_list, dense_value_list)
    #     bias_inputs_list = list(bias_features.values())
    #     inputs_list += bias_inputs_list
    if bottom_shared_units is not None:
        dnn_input = DNN(bottom_shared_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                        bottom_shared_use_bn, seed=seed, name='bottom_shared_dnn')(dnn_input)

    mmoe_outs = MMOELayer(list(tasks.keys()), num_experts, expert_dim, dnn_activation, seed)(dnn_input)

    task_outputs = {}
    for mmoe_out, task_name, task_type in zip(mmoe_outs.values(), tasks.keys(), tasks.values()):
        if task_dnn_units != None:
            mmoe_out = DNN(task_dnn_units,
                           dnn_activation,
                           l2_reg_dnn,
                           dnn_dropout,
                           task_use_bn,
                           seed=seed,
                           name=task_name+'_dnn')(mmoe_out)
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None, name=task_name+'_logit')(mmoe_out)

        if bias_feature_columns_dict and bias_feature_columns_dict.get(task_name):
            bias_logit = get_linear_logit(features, bias_feature_columns_dict[task_name], seed=seed,
                                          prefix=task_name+'_bias_', l2_reg=bias_l2_reg)
            bias_logit = tf.reshape(bias_logit, (-1, logit.shape[-1]))
            bias_logit = tf.keras.layers.Dropout(bias_dropout_dict[task_name], seed=seed)(bias_logit)
            logit = add_func([logit, bias_logit])
        output = PredictionLayer(task_type, name=task_name+'_prediction')(logit)
        task_outputs[task_name] = output

    # if use_uncertainty:
    #     ys_true = [Input(shape=(1,)) for _ in tasks]
    #     loss_layer_inputs = ys_true + task_outputs
    #     model_out = MultiLossLayer(tasks)(loss_layer_inputs)
    #     model_inputs = inputs_list + ys_true
    #     model = tf.keras.models.Model(model_inputs, model_out)
    #     return model
    # else:
    #     model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outputs)
    #     return model
    model = MultiTaskModelBase(inputs=inputs_list, outputs=task_outputs)
    return model