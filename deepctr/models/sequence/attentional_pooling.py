# -*- coding:utf-8 -*-
"""
Author:
    Xiaohai ZHU,zhuxioahai_sast@163.com

Reference:
    [1] Zhou G, Mou N, Fan Y, et al. Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018. (https://arxiv.org/pdf/1809.03672.pdf)
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input, Permute, Concatenate, Flatten
from tensorflow.keras.initializers import glorot_normal, Zeros
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Layer, Conv1D

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features
from deepctr.inputs import create_embedding_matrix, embedding_lookup, get_dense_input
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.utils import combined_dnn_input, softmax, concat_func


class SelfAttentionalPoolingLayer(Layer):
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
                 l2_reg_w=0.0,
                 seed=1024,
                 supports_masking=False, **kwargs):
        self.weight_normalization = weight_normalization
        # if return attention score for every time step
        self.return_score = return_score
        # Note that the function compute_mask
        # is only available to take effect when support masking is True.
        # If support masking = True, the fed in mask will be transferred to output
        # and if at the same time compute_mask is defined the newly defined mask will be transferred
        # to output.
        # If support masking = False, even if there is fed in mask and it is used,
        # no mask will be transferred out.
        self.supports_masking = True
        self.attention_factor = attention_factor
        self.supports_masking = supports_masking
        self.l2_reg_w = l2_reg_w
        self.seed = seed
        super(SelfAttentionalPoolingLayer, self).__init__(**kwargs)

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
        hidden_size = input_shape[0][-1]
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
        # self.att_weight = self.add_weight(shape=(hidden_size, 1),
        #                                   initializer=glorot_normal(seed=1024),
        #                                   name="att_weight")
        # self.att_bias = self.add_weight(shape=(1,), initializer=Zeros(), name="att_bias")
        # self.dense = tf.keras.layers.Lambda(lambda x: tf.nn.bias_add(tf.tensordot(
        #     x[0], x[1], axes=(-1, 0)), x[2]))
        super(SelfAttentionalPoolingLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None, training=None, **kwargs):
        rnn_input, sequence_length = inputs
        max_len = rnn_input.get_shape()[1]
        rnn_masks = tf.sequence_mask(sequence_length, max_len)
        if self.supports_masking:
            if mask is None:
                raise ValueError("When supports_masking=True,input must support masking")
            rnn_masks = tf.logical_and(rnn_masks, tf.expand_dims(mask, axis=1))

        attention_intermediate = tf.nn.relu(tf.nn.bias_add(tf.tensordot(
            rnn_input, self.attention_W, axes=(-1, 0)), self.attention_b))
        attention_score = tf.tensordot(attention_intermediate, self.projection_h, axes=(-1, 0))
        # attention_score = self.dense([tf.tanh(keys), self.att_weight, self.att_bias])

        outputs = tf.transpose(attention_score, (0, 2, 1))

        if self.weight_normalization:
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(outputs)

        outputs = tf.where(rnn_masks, outputs, paddings)

        if self.weight_normalization:
            outputs = softmax(outputs)
        self.att_score = outputs

        if not self.return_score:
            outputs = tf.matmul(outputs, rnn_input)

        if tf.__version__ < '1.13.0':
            outputs._uses_learning_phase = attention_score._uses_learning_phase
        else:
            outputs._uses_learning_phase = training is not None

        return outputs

    def compute_output_shape(self, input_shape):
        if self.return_score:
            return (None, 1, input_shape[0][1])
        else:
            return (None, 1, input_shape[0][-1])

    def get_config(self, ):
        config = {'weight_normalization': self.weight_normalization,
                  'return_score': self.return_score,
                  'supports_masking': self.supports_masking,
                  'attention_factor': self.attention_factor,
                  'l2_reg_w': self.l2_reg_w,
                  'seed': self.seed}
        base_config = super(SelfAttentionalPoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ModifiedDynamicGRU(Layer):
    def __init__(self, num_units=None, return_sequence=True, supports_masking=True, **kwargs):

        self.num_units = num_units
        self.return_sequence = return_sequence
        # Note that the function compute_mask
        # is only available to take effect when support masking is True.
        # If support masking = True, the fed in mask will be transferred to output
        # and if at the same time compute_mask is defined the newly defined mask will be transferred
        # to output.
        # If support masking = False, even if there is fed in mask and it is used,
        # no mask will be transferred out.
        self.supports_masking = supports_masking
        super(ModifiedDynamicGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_seq_shape = input_shape[0]
        if self.num_units is None:
            self.num_units = input_seq_shape.as_list()[-1]
        else:
            # self.gru_cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.num_units)
            self.gru_cell = tf.keras.layers.GRUCell(self.num_units)
        # Be sure to call this somewhere!
        super(ModifiedDynamicGRU, self).build(input_shape)

    def call(self, input_list, mask=None, training=None):
        """
        :param concated_embeds_value: None * field_size * embedding_size
        :return: None*1
        """
        rnn_input, sequence_length = input_list
        max_len = rnn_input.get_shape()[1]
        rnn_masks = tf.sequence_mask(tf.squeeze(sequence_length, -1), max_len)
        if self.supports_masking:
            if mask is None:
                raise ValueError("When supports_masking=True,input must support masking")
            rnn_masks = tf.logical_and(rnn_masks, mask)
        rnn_output, hidden_state = tf.keras.layers.RNN(self.gru_cell, return_state=True,
                                                       return_sequences=self.return_sequence)(
            rnn_input, mask=rnn_masks, training=training)
        # rnn_output, hidden_state = dynamic_rnn(self.gru_cell, inputs=rnn_input, att_scores=None,
        #                                        sequence_length=tf.squeeze(sequence_length,
        #                                                                   ), dtype=tf.float32, scope=self.name)
        if self.return_sequence:
            return rnn_output
        else:
            return tf.expand_dims(hidden_state, axis=1)

    def compute_output_shape(self, input_shape):
        rnn_input_shape = input_shape[0]
        if self.return_sequence:
            return rnn_input_shape
        else:
            return (None, 1, rnn_input_shape[2])

    def get_config(self, ):
        config = {'num_units': self.num_units, 'return_sequence': self.return_sequence}
        base_config = super(ModifiedDynamicGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_sequence_pooling(sequence_input, sequence_length,
                         embedding_size=8, gru_type="GRU",
                         att_weight_normalization=False,
                         attention_factor=4, l2_reg_w=1e-5):
    if gru_type == 'GRU':
        final_state = ModifiedDynamicGRU(embedding_size, return_sequence=False,
                                         supports_masking=False,
                                         name="dynamic_gru")([sequence_input, sequence_length])
        return final_state
    else:
        rnn_outputs = ModifiedDynamicGRU(embedding_size, return_sequence=True,
                                         supports_masking=False,
                                         name="dynamic_gru")([sequence_input, sequence_length])
        pooling_result = SelfAttentionalPoolingLayer(weight_normalization=att_weight_normalization,
                                                     return_score=False,
                                                     supports_masking=False,
                                                     attention_factor=attention_factor,
                                                     l2_reg_w=l2_reg_w)([rnn_outputs, sequence_length])
        return pooling_result


def AttentionalPooling(constant_feature_columns, behavior_feature_columns, behavior_sparse_indicator,
                       constant_dense_normalizer, behavior_dense_normalizer,
                       gru_type="AVGRU", gru_hidden_size=8, gru_attention_w_dim=4, gru_attention_w_l2=0.1,
                       dnn_hidden_units=(200, 80), dnn_activation='dice', dnn_use_bn=False,
                       l2_reg_embedding=1e-6, l2_reg_dnn=0, dnn_dropout=0, seed=1024, task='binary'):
    """
    :param constant_feature_columns: An iterable containing all constant features
    :param behavior_feature_columns: An iterable containing all timeseries features
    :param behavior_sparse_indicator: list,to indicate  sequence sparse field
    :param gru_type: str
    :param gru_hidden_size: int
    :param gru_attention_w_dim: int
    :param gru_attention_w_l2: float
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    features = build_input_features(constant_feature_columns + behavior_feature_columns)

    features['seq_length'] = features.get('seq_length', Input((1,), name='seq_length', dtype='int32'))
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
                                             l2_reg_embedding, seed, prefix="", seq_mask_zero=True)

    history_dense_value_list = get_dense_input(features, varlen_dense_feature_columns)
    history_dense = tf.keras.layers.Lambda(lambda x: tf.keras.backend.stack(x, axis=-1))(history_dense_value_list)
    history_dense = behavior_dense_normalizer(history_dense)
    history_dense = Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(history_dense)
    if len(varlen_sparse_feature_columns) > 0:
        history_emb_list = embedding_lookup(embedding_dict, features, varlen_sparse_feature_columns,
                                            return_feat_list=history_fc_names, to_list=True)
        history_emb = concat_func(history_emb_list, mask=True)
        history_input = Concatenate()([history_emb, history_dense])
    else:
        history_input = history_dense
    hist = get_sequence_pooling(history_input, user_behavior_length,
                                embedding_size=gru_hidden_size,
                                gru_type=gru_type,
                                att_weight_normalization=True,
                                attention_factor=gru_attention_w_dim,
                                l2_reg_w=gru_attention_w_l2)
    if len(constant_sparse_feature_columns) > 0:
        dnn_input_emb_list = embedding_lookup(embedding_dict, features, constant_sparse_feature_columns,
                                              mask_feat_list=behavior_sparse_indicator, to_list=True)
        deep_input_emb = concat_func(dnn_input_emb_list)
        deep_input_emb = Concatenate()([deep_input_emb, hist])
    else:
        deep_input_emb = hist
    deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)

    deep_dense_value_list = get_dense_input(features, constant_dense_feature_columns)

    if len(deep_dense_value_list) > 0:
        dense_dnn_input = Flatten()(concat_func(deep_dense_value_list))
        dense_dnn_input = constant_dense_normalizer(dense_dnn_input)
        dnn_input = concat_func([deep_input_emb, dense_dnn_input])
    else:
        dnn_input = deep_input_emb

    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)

    final_logit = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)

    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    return model
