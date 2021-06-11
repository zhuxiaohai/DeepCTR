# -*- coding:utf-8 -*-
"""
Author:
    Xiaohai Zhu
"""
import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.utils import combined_dnn_input
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.models.multitask.multitaskbase import MultiTaskModelBase


def SimpleDNN(dnn_feature_columns, tasks,
              dnn_hidden_units=(128, 128), l2_reg_embedding=1e-5, l2_reg_dnn=0,
              seed=1024, dnn_dropout=0, dnn_activation='relu'):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param tasks: dict, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"``
    for regression loss. e.g. {'task1': 'binary', 'task2': 'regression'}
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer
    of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN

    :return: If use_uncertainty is False, return a Keras model instance, otherwise,
    return a tuple (prediction_model, train_model).
    train_model should be compiled and fit data first, and then prediction_model is used for prediction.
    """
    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_outs = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                   False, seed=seed)(dnn_input)
    dnn_outs = [dnn_outs]
    task_outputs = {}
    for dnn_out, task_name, task_type in zip(dnn_outs, tasks.keys(), tasks.values()):
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(dnn_out)
        output = PredictionLayer(task_type, name=task_name)(logit)
        task_outputs[task_name] = output
    model = MultiTaskModelBase(inputs=inputs_list, outputs=task_outputs)
    return model