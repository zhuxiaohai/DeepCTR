# -*- coding:utf-8 -*-
"""
Author:
    Xiaohai Zhu(zhuxiaohai_sast@163.com)

Reference:
    [1] Hongyan Tang, Ming Zhao. Progressive Layered Extraction (PLE):
    A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations
    (https://dl.acm.org/doi/abs/10.1145/3383313.3412236)
"""
import tensorflow as tf
from tensorflow.python.keras.layers import Layer

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.utils import combined_dnn_input
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.models.multitask.multitaskbase import MultiTaskModelBase


class CGC(Layer):
    def __init__(self, hidden_unit, tasks_config, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 seed=1024, output_share=True, **kwargs):
        self.hidden_unit = hidden_unit
        self.tasks_config = tasks_config
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_share = output_share
        num_all_experts = sum([num_experts for num_experts in self.tasks_config.values()])

        # newly created
        self.selected_experts_config = {}
        self.experts = {}
        self.gates = {}
        self.specific_task_names = []
        for index, (task_name, num_experts) in enumerate(self.tasks_config.items()):
            self.experts[task_name] = DNN((num_experts * self.hidden_unit,), activation, l2_reg, dropout_rate,
                                          use_bn, seed=seed, name=task_name+'_experts')
            if index != 0:
                self.specific_task_names.append(task_name)
                num_selected_experts = num_experts + self.tasks_config[self.shared_task_name]
                self.selected_experts_config[task_name] = num_selected_experts
                self.gates[task_name] = DNN((num_selected_experts,), activation, l2_reg, dropout_rate,
                                            use_bn, seed=seed, name=task_name+'_gate')
            else:
                self.shared_task_name = task_name
                num_selected_experts = num_all_experts
                self.selected_experts_config[task_name] = num_selected_experts
                if self.output_share:
                    self.gates[task_name] = DNN((num_selected_experts,), activation, l2_reg, 0,
                                                False, 'softmax', seed=seed, name=task_name+'_gate')
        super(CGC, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        # compute outputs of all the experts
        experts_output = {}
        for task_name in self.tasks_config.keys():
            output = self.experts[task_name](inputs[task_name])
            output = tf.reshape(output, (-1, self.hidden_unit, self.tasks_config[task_name]))
            experts_output[task_name] = output

        task_output = {}
        # shared task
        if self.output_share:
            selected_experts_output = tf.concat([experts_output[task_name] for task_name in self.tasks_config.keys()], -1)
            gate = self.gates[self.shared_task_name](inputs[self.shared_task_name])
            gate = tf.reshape(gate, (-1, 1, self.selected_experts_config[self.shared_task_name]))
            output = tf.reduce_sum(tf.multiply(selected_experts_output, gate), -1)
            task_output[self.shared_task_name] = output
        # specific tasks
        for task_name in self.specific_task_names:
            selected_experts_output = tf.concat([experts_output[task_name], experts_output[self.shared_task_name]], -1)
            gate = self.gates[task_name](inputs[task_name])
            gate = tf.reshape(gate, (-1, 1, self.selected_experts_config[task_name]))
            output = tf.reduce_sum(tf.multiply(selected_experts_output, gate), -1)
            task_output[task_name] = output

        return task_output

    def compute_output_shape(self, input_shape):
        if self.output_share:
            shape = {task_name: tuple(input_shape[task_name][:-1] + [self.hidden_unit])
                     for task_name in self.tasks_config.keys()}
        else:
            shape = {task_name: tuple(input_shape[task_name][:-1] + [self.hidden_unit])
                     for task_name in self.specific_task_names}
        return shape

    def get_config(self, ):
        config = {'tasks_config': self.tasks_config, 'activation': self.activation,
                  'hidden_unit': self.hidden_unit, 'l2_reg': self.l2_reg, 'use_bn': self.use_bn,
                  'dropout_rate': self.dropout_rate, 'output_share': self.output_share}
        base_config = super(CGC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PLE(dnn_feature_columns, tasks, config, bottom_shared_units=None, bottom_shared_use_bn=False,
        l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_dropout=0, dnn_activation='relu', cgc_use_bn=False,
        task_dnn_units=None, task_use_bn=False, seed=1024):
    """Instantiates the Progressive Layered Extraction.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param tasks: dict, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"``
    for regression loss. e.g. {'task1': 'binary', 'task2': 'regression'}
    :param config: dict, indicating the configuration of each cgc layer,
    e.g., {'layer1': {'hidden_unit': 3, tasks: {'share': 2, 'task1': 1, 'task2': 4}},
           'layer2': {'hidden_unit': 3, tasks: {'share': 2, 'task1': 1, 'task2': 4}},
           ......
           }
    :param bottom_shared_units: list,list of positive integer or empty list, the layer number and units in each layer
    of shared-bottom DNN
    :param bottom_shared_use_bn: bool, whether to use batch normalization in bottom shared dnn
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param cgc_use_bn: whether to use batch normalization in cgc
    :param task_dnn_units: list, list of positive integer or empty list, the layer number and units in each layer
    of task-specific DNN tower
    :param task_use_bn: bool, whether to use batch normalization in task towers.
    only available when task_dnn_units is not None
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN

    :return: a Keras model instance
    """
    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    if bottom_shared_units is not None:
        dnn_input = DNN(bottom_shared_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                        bottom_shared_use_bn, seed=seed, name='bottom_shared_dnn')(dnn_input)

    extraction_network_out = {task_name: dnn_input for task_name in config[list(config.keys())[0]]['tasks'].keys()}
    for index, (layer_name, layer_config) in enumerate(config.items()):
        if index == (len(config) - 1):
            output_share = False
        else:
            output_share = True
        extraction_network_out = CGC(layer_config['hidden_unit'],
                                     layer_config['tasks'],
                                     dnn_activation,
                                     l2_reg_dnn,
                                     dnn_dropout,
                                     cgc_use_bn,
                                     seed,
                                     output_share,
                                     name='cgc_'+layer_name)(extraction_network_out)

    task_outputs = {}
    for task_name, task_type in tasks.items():
        with tf.name_scope(task_name):
            task_output = extraction_network_out[task_name]
            if task_dnn_units != None:
                task_output = DNN(task_dnn_units,
                                  dnn_activation,
                                  l2_reg_dnn,
                                  dnn_dropout,
                                  task_use_bn,
                                  seed=seed,
                                  name=task_name+'_dnn')(task_output)
            task_output = tf.keras.layers.Dense(1, use_bias=False, activation=None, name=task_name+'_logit')(task_output)
            task_output = PredictionLayer(task_type, name=task_name+'_prediction')(task_output)
            task_outputs[task_name] = task_output

    model = MultiTaskModelBase(inputs=inputs_list, outputs=task_outputs)
    return model