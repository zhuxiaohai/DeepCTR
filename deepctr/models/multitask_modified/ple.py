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
from tensorflow.python.keras.initializers import glorot_normal, Zeros
from tensorflow.python.keras import activations

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.utils import combined_dnn_input
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.models.multitask_modified.multitaskbase import MultiTaskModelBase


class CGC(Layer):
    def __init__(self, hidden_unit, tasks_config, experts_activation='relu', output_share=True, seed=1024, **kwargs):
        self.hidden_unit = hidden_unit
        self.tasks_config = tasks_config
        self.experts_activation = experts_activation
        self.output_share = output_share
        self.seed = seed
        num_all_experts = sum([num_experts for num_experts in self.tasks_config.values()])

        self.selected_experts_config = {}
        self.specific_task_names = []
        for index, (task_name, num_experts) in enumerate(self.tasks_config.items()):
            if index != 0:
                self.specific_task_names.append(task_name)
                num_selected_experts = num_experts + self.tasks_config[self.shared_task_name]
                self.selected_experts_config[task_name] = num_selected_experts
            else:
                self.shared_task_name = task_name
                num_selected_experts = num_all_experts
                self.selected_experts_config[task_name] = num_selected_experts
        self.experts = None
        self.gates = None
        self.gates_output = None
        super(CGC, self).__init__(**kwargs)

    def build(self, input_shape):
        self.experts = {}
        self.gates = {}
        for task_name, num_experts in self.tasks_config.items():
            input_dimension = input_shape[task_name][-1]
            self.experts[task_name] = {
                'experts_kernel_{}'.format(task_name):
                self.add_weight(name='experts_kernel_{}'.format(task_name),
                                shape=(input_dimension, self.hidden_unit * num_experts),
                                dtype=tf.float32,
                                initializer=glorot_normal(seed=self.seed)
                                ),
                'experts_bias_{}'.format(task_name):
                self.add_weight(
                    name='experts_bias_{}'.format(task_name),
                    shape=(self.hidden_unit * num_experts,),
                    dtype=tf.float32,
                    initializer=Zeros())}

            if (task_name == self.shared_task_name) and (not self.output_share):
                continue
            else:
                num_selected_experts = self.selected_experts_config[task_name]
                self.gates[task_name] = {
                    'gate_kernel_{}'.format(task_name):
                    self.add_weight(name='gate_kernel_{}'.format(task_name),
                                    shape=(input_dimension, num_selected_experts),
                                    dtype=tf.float32,
                                    initializer=glorot_normal(seed=self.seed)
                                    ),
                    'gate_bias_{}'.format(task_name):
                    self.add_weight(
                        name='gate_bias_{}'.format(task_name),
                        shape=(num_selected_experts,),
                        dtype=tf.float32,
                        initializer=Zeros())}

        super(CGC, self).build(input_shape)

    def call(self, inputs, **kwargs):
        experts_output = {}
        for task_name in self.tasks_config.keys():
            output = tf.tensordot(inputs[task_name],
                                  self.experts[task_name]['experts_kernel_{}'.format(task_name)],
                                  axes=(-1, 0))
            output = tf.nn.bias_add(output, self.experts[task_name]['experts_bias_{}'.format(task_name)])
            output = tf.reshape(output, (-1, self.hidden_unit, self.tasks_config[task_name]))
            if self.experts_activation is not None:
                output = activations.get(self.experts_activation)(output)
            experts_output[task_name] = output

        task_output = {}
        self.gates_output = {}
        for task_name in [self.shared_task_name] + self.specific_task_names:
            if task_name == self.shared_task_name:
                if self.output_share:
                    selected_experts_output = tf.concat(
                        [experts_output[task_name] for task_name in self.tasks_config.keys()], -1)
                else:
                    continue
            else:
                selected_experts_output = tf.concat([experts_output[task_name],
                                                     experts_output[self.shared_task_name]], -1)
            gate = tf.tensordot(inputs[task_name],
                                self.gates[task_name]['gate_kernel_{}'.format(task_name)],
                                axes=(-1, 0))
            gate = tf.nn.bias_add(gate, self.gates[task_name]['gate_bias_{}'.format(task_name)])
            gate = tf.nn.softmax(gate)
            self.gates_output[task_name] = gate
            gate = tf.tile(tf.expand_dims(gate, axis=1), [1, self.hidden_unit, 1])
            output = tf.reduce_sum(tf.multiply(selected_experts_output, gate), axis=2)
            task_output[task_name] = output

        return task_output

    def compute_output_shape(self, input_shape):
        shape = {}
        for task_name in [self.shared_task_name] + self.specific_task_names:
            if task_name == self.shared_task_name:
                if self.output_share:
                    shape[task_name] = tuple(input_shape[task_name][:-1] + [self.hidden_unit])
                else:
                    continue
            else:
                shape[task_name] = tuple(input_shape[task_name][:-1] + [self.hidden_unit])
        return shape

    def get_config(self, ):
        config = {'hidden_unit': self.hidden_unit,
                  'tasks_config': self.tasks_config,
                  'experts_activation': self.experts_activation,
                  'output_share': self.output_share}
        base_config = super(CGC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PLE(dnn_feature_columns, tasks, config, bottom_shared_units=None, bottom_shared_use_bn=False,
        l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_dropout=0, dnn_activation='relu',
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
                                     output_share,
                                     seed,
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