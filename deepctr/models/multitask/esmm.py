# -*- coding:utf-8 -*-
"""
Author:
    xiaohai zhu(zhuxiaohai_sast@163.com)

Reference:
   [1] Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, Kun Gai.
   Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate
   (https://dl.acm.org/doi/10.1145/3209978.3210104)
"""
import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.utils import combined_dnn_input
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.models.multitask.multitaskbase import MultiTaskModelBase


def ESMM(dnn_feature_columns, tasks, dnn_hidden_units=(32, 16, 8), l2_reg_embedding=1e-5, l2_reg_dnn=0,
         seed=1024, dnn_dropout=0, dnn_activation='relu'):
    """Instantiates the Entire Space Multi-Task Model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param tasks: a list of tuples, each element being (task_name, task_type), e.g. ('ctr', 'binary').
      we choose this structure instead of dict to ensure dependency of tasks
    :param dnn_hidden_units: list,list of positive integer or empty list,
    the layer number and units in each layer of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN

    :return: a keras model instance.
    """
    num_tasks = len(tasks)
    if num_tasks != 3:
        raise ValueError("num_tasks must be 3, with the last being the product of the first two tasks")
    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    outputs = []
    for name, task_type in tasks[:-1]:
        output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     False, seed=seed)(dnn_input)
        output = tf.keras.layers.Dense(1, use_bias=False, activation=None)(output)
        output = PredictionLayer(task=task_type, name=name)(output)
        outputs.append(output)
    product_output = tf.keras.layers.Lambda(lambda x: x[0] * x[1])(outputs)

    model = MultiTaskModelBase(inputs=inputs_list, outputs={tasks[0][0]: outputs[0], tasks[-1][0]: product_output})
    return model

