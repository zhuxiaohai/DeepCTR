# -*- coding:utf-8 -*-
"""
Author:
    zanshuxun, zanshuxun@aliyun.com
Reference:
    [1] Yu Y, Wang Z, Yuan B. An Input-aware Factorization Machine for Sparse Prediction[C]//IJCAI. 2019: 1466-1472.(https://www.ijcai.org/Proceedings/2019/0203.pdf)
"""

import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM, IFMSoftmax
from ..layers.utils import concat_func, add_func, combined_dnn_input
from deepctr.feature_column import SparseFeat, VarLenSparseFeat

def IFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128),
        l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
        dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the IFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    if not len(dnn_hidden_units) > 0:
        raise ValueError("dnn_hidden_units is null!")

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    sparse_feat_num = len(list(filter(lambda x: isinstance(x, SparseFeat) or isinstance(x, VarLenSparseFeat),
                                      dnn_feature_columns)))
    print('sparse_feat_num', sparse_feat_num)
    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    if not len(sparse_embedding_list) > 0:
        raise ValueError("there are no sparse features")

    dnn_input = combined_dnn_input(sparse_embedding_list, [])
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    # here, dnn_output is the m'_{x}
    dnn_output = tf.keras.layers.Dense(
        sparse_feat_num, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)
    # input_aware_factor m_{x,i}
    input_aware_factor = IFMSoftmax(sparse_feat_num)(dnn_output)

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear, sparse_feat_refine_weight=input_aware_factor)

    fm_input = concat_func(sparse_embedding_list, axis=1)
    refined_fm_input = tf.keras.layers.Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=-1))(
        [fm_input, input_aware_factor])
    fm_logit = FM()(refined_fm_input)

    final_logit = add_func([linear_logit, fm_logit])

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
