# -*- coding:utf-8 -*-
"""
Author:
    Xiaohai ZHU,zhuxioahai_sast@163.com

Reference:
    [1] Zhou G, Mou N, Fan Y, et al. Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018. (https://arxiv.org/pdf/1809.03672.pdf)
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input, Permute
from tensorflow.python.keras.initializers import glorot_normal, Zeros
from tensorflow.python.keras.layers import Layer

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features
from deepctr.inputs import create_embedding_matrix, embedding_lookup, get_dense_input
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.sequence import DynamicGRU
from deepctr.layers.utils import combined_dnn_input, softmax


class RnnAttentionalLayer(Layer):
    """The Attentional sequence pooling operation used in Rnn like structures.
      Input shape
        - A list of three tensor: [keys,keys_length]
        - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``
        - keys_length is a 2D tensor with shape: ``(batch_size, 1)``
      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
      Arguments
        - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.
        - **supports_masking**:If True,the input need to support masking.
      References
        - Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
    """
    def __init__(self, weight_normalization=False,
                 return_score=False,
                 attention_factor=4,
                 l2_reg_w=0,
                 seed=1024,
                 supports_masking=False, **kwargs):
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        self.supports_masking = supports_masking
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.seed = seed
        super(RnnAttentionalLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not self.supports_masking:
            if not isinstance(input_shape, list) or len(input_shape) != 2:
                raise ValueError('A `RnnAttentionalLayer` layer should be called '
                                 'on a list of 2 inputs')

            if len(input_shape[0]) != 3 or len(input_shape[1]) != 2:
                raise ValueError(
                    "Unexpected inputs dimensions,the 2 tensor dimensions are %d and %d , expect to be 3 and 2" % (
                        len(input_shape[0]), len(input_shape[1])))
        else:
            pass
        hidden_size = input_shape[0][-1].value
        self.attention_W = self.add_weight(shape=(hidden_size, self.attention_factor),
                                           initializer=glorot_normal(seed=self.seed),
                                           regularizer=l2(self.l2_reg_w),
                                           name="attention_W")
        self.attention_b = self.add_weight(shape=(self.attention_factor,),
                                           initializer=Zeros(),
                                           name="attention_b")
        self.projection_h = self.add_weight(shape=(self.attention_factor, 1),
                                            initializer=glorot_normal(seed=self.seed),
                                            name="projection_h")
        self.tensordot = tf.keras.layers.Lambda(lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))
        # self.att_weight = self.add_weight(shape=(hidden_size, 1),
        #                                   initializer=glorot_normal(seed=1024),
        #                                   name="att_weight")
        # self.att_bias = self.add_weight(shape=(1,), initializer=Zeros(), name="att_bias")
        # self.dense = tf.keras.layers.Lambda(lambda x: tf.nn.bias_add(tf.tensordot(
        #     x[0], x[1], axes=(-1, 0)), x[2]))
        super(RnnAttentionalLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None, training=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            keys = inputs
            key_masks = tf.expand_dims(mask[-1], axis=1)
        else:
            keys, keys_length = inputs
            hist_len = keys.get_shape()[1]
            key_masks = tf.sequence_mask(keys_length, hist_len)

        attention_temp = tf.nn.relu(tf.nn.bias_add(tf.tensordot(
            keys, self.attention_W, axes=(-1, 0)), self.attention_b))
        attention_score = self.tensordot([attention_temp, self.projection_h])
        # attention_score = self.dense([tf.tanh(keys), self.att_weight, self.att_bias])

        outputs = tf.transpose(attention_score, (0, 2, 1))

        if self.weight_normalization:
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(outputs)

        outputs = tf.where(key_masks, outputs, paddings)

        if self.weight_normalization:
            outputs = softmax(outputs)
        self.att_score = outputs

        if not self.return_score:
            outputs = tf.matmul(outputs, keys)

        if tf.__version__ < '1.13.0':
            outputs._uses_learning_phase = attention_score._uses_learning_phase
        else:
            outputs._uses_learning_phase = training is not None

        return outputs

    def compute_output_shape(self, input_shape):
        if self.return_score:
            return (None, 1, input_shape[1][1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'weight_normalization': self.weight_normalization,
                  'return_score': self.return_score,
                  'supports_masking': self.supports_masking,
                  'attention_factor': self.attention_factor,
                  'l2_reg_w': self.l2_reg_w,
                  'seed': self.seed}
        base_config = super(RnnAttentionalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def evolution(concat_behavior, user_behavior_length,
              embedding_size=8, gru_type="GRU",
              att_weight_normalization=False,
              attention_factor=4, l2_reg_w=1e-5):
    if gru_type == 'GRU':
        final_state = DynamicGRU(embedding_size, return_sequence=False,
                                 name="gru1")([concat_behavior, user_behavior_length])
    else:
        rnn_outputs = DynamicGRU(embedding_size, return_sequence=True,
                                 name="gru1")([concat_behavior, user_behavior_length])
        if gru_type == "AVGRU":
            final_state = RnnAttentionalLayer(weight_normalization=att_weight_normalization,
                                              return_score=False,
                                              attention_factor=attention_factor,
                                              l2_reg_w=l2_reg_w)([rnn_outputs, user_behavior_length])
        else:  # AUGRU
            scores = RnnAttentionalLayer(weight_normalization=att_weight_normalization,
                                         return_score=True,
                                         attention_factor=attention_factor,
                                         l2_reg_w=l2_reg_w)([rnn_outputs, user_behavior_length])
            final_state = DynamicGRU(embedding_size, gru_type=gru_type, return_sequence=False,
                                     name='gru2')([rnn_outputs, user_behavior_length, Permute([2, 1])(scores)])
    return final_state


def TimeSeries(constant_feature_columns, behavior_feature_columns, behavior_sparse_indicator,
        use_bn=False, dnn_hidden_units=(200, 80), dnn_activation='dice',
        l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024, task='binary'):
    """
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param behavior_sparse_indicator: list,to indicate  sequence sparse field
    :param gru_type: str,can be GRU AIGRU AUGRU AGRU
    :param use_negsampling: bool, whether or not use negtive sampling
    :param alpha: float ,weight of auxiliary_loss
    :param use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param att_hidden_units: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """
    features = build_input_features(constant_feature_columns + behavior_feature_columns)
    features['seq_length'] = Input((1,), name='seq_length', dtype='int32')
    user_behavior_length = features['seq_length']

    constant_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), constant_feature_columns) if constant_feature_columns else [])
    constant_dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), constant_feature_columns) if constant_feature_columns else [])
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), behavior_feature_columns) if behavior_feature_columns else [])
    varlen_dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), behavior_feature_columns) if behavior_feature_columns else [])

    history_fc_names = list(map(lambda x: "hist_" + x, behavior_sparse_indicator))

    inputs_list = list(features.values())

    embedding_dict = create_embedding_matrix(constant_feature_columns + behavior_feature_columns,
                                             l2_reg_embedding, seed, prefix="",
                                             seq_mask_zero=False)

    history_emb_list = embedding_lookup(embedding_dict, features, varlen_sparse_feature_columns,
                                        return_feat_list=history_fc_names, to_list=True)
    history_dense_value_list = get_dense_input(features, varlen_dense_feature_columns)
    # history_emb = concat_func(history_emb_list)
    history_dense = tf.keras.layers.Lambda(lambda x: tf.keras.backend.stack(x, axis=-1))(history_dense_value_list)
    # history_input = Concatenate()([history_emb, history_dense])
    history_input = history_dense

    dnn_input_emb_list = embedding_lookup(embedding_dict, features, constant_sparse_feature_columns,
                                          mask_feat_list=behavior_sparse_indicator, to_list=True)
    # deep_input_emb = concat_func(dnn_input_emb_list)
    deep_dense_value_list = get_dense_input(features, constant_dense_feature_columns)

    hist = evolution(history_input, user_behavior_length,
                     embedding_size=8, gru_type="AVGRU", att_weight_normalization=True,
                     attention_factor=4, l2_reg_w=0.1)

    # deep_input_emb = Concatenate()([deep_input_emb, hist])
    deep_input_emb = hist

    deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)

    dnn_input = combined_dnn_input([deep_input_emb], deep_dense_value_list)

    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, use_bn, seed)(dnn_input)
    final_logit = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    try:
        tf.keras.backend.get_session().run(tf.global_variables_initializer())
    except:
        tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())
    return model
