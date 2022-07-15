# -*- coding:utf-8 -*-
"""
Author:
    Xiaohai ZHU,zhuxioahai_sast@163.com
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input, Flatten
from tensorflow.keras.initializers import glorot_normal
from tensorflow.python.keras.layers import Conv1D, BatchNormalization

from deepctr.feature_column import DenseFeat, build_input_features
from deepctr.inputs import get_dense_input
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.utils import concat_func
from deepctr.layers.activation import activation_layer


def MCNN_block(x, filters, kernel_sizes=[3, 6, 12]):
    conv_list = []
    for kernel in kernel_sizes:
        conv = Conv1D(filters=filters, kernel_size=kernel, strides=1, activation='relu', padding='same')(x)
        conv_list.append(conv)
    x = tf.keras.layers.concatenate(conv_list, axis=-1)
    x = BatchNormalization()(x)
    x = activation_layer('relu')(x)
    return x


def MCNN(x, loop_num_list, filters_list, kernel_sizes_list, pool_size_list, strides_list):
    for loop_num, filters, kernel_sizes, pool_size, stride in zip(
            loop_num_list, filters_list, kernel_sizes_list, pool_size_list, strides_list):
        for i in range(loop_num):
            x = MCNN_block(x, filters, kernel_sizes)
        if (pool_size is not None) and (stride is not None):
            x = tf.keras.layers.MaxPool1D(pool_size, stride, padding='same')(x)
    x = tf.reduce_mean(x, axis=1)
    return x


def MCNNModel(constant_feature_columns, sequence_feature_columns,
              constant_dense_normalizer, sequence_dense_normalizer,
              loop_num_list, filters_list, kernel_sizes_list, pool_size_list, strides_list,
              dnn_hidden_units=(200, 80), dnn_activation='dice', dnn_use_bn=False,
              l2_reg_dnn=0, dnn_dropout=0, seed=1024, task='binary'):
    """
    :param constant_feature_columns: An iterable containing all constant features
    :param sequence_feature_columns: An iterable containing all sequence features
    :param constant_dense_normalizer: fitted keras Normalization layer to normalize constant dense features
    :param sequence_dense_normalizer: fitted keras Normalization layer to normalize sequence dense features
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    features = build_input_features(constant_feature_columns + sequence_feature_columns)
    features['seq_length'] = features.get('seq_length', Input((1,), name='seq_length', dtype='int32'))
    inputs_list = list(features.values())

    # prepare feature types
    constant_dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), constant_feature_columns) if constant_feature_columns else [])
    varlen_dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), sequence_feature_columns) if sequence_feature_columns else [])

    # sequence
    history_dense_value_list = get_dense_input(features, varlen_dense_feature_columns)
    history_dense = tf.keras.layers.Lambda(lambda x: tf.keras.backend.stack(x, axis=-1))(history_dense_value_list)
    history_dense = sequence_dense_normalizer(history_dense)
    history_dense = MCNN(history_dense, loop_num_list, filters_list, kernel_sizes_list, pool_size_list, strides_list)
    deep_input_emb = tf.keras.layers.Flatten()(history_dense)

    # concatenate dense
    deep_dense_value_list = get_dense_input(features, constant_dense_feature_columns)
    if len(deep_dense_value_list) > 0:
        dense_dnn_input = Flatten()(concat_func(deep_dense_value_list))
        dense_dnn_input = constant_dense_normalizer(dense_dnn_input)
        dnn_input = concat_func([deep_input_emb, dense_dnn_input])
    else:
        dnn_input = deep_input_emb

    # feed in mlp
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)

    # get logit
    final_logit = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)

    # get probability
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    return model