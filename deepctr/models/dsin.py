# coding: utf-8
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Feng Y, Lv F, Shen W, et al. Deep Session Interest Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1905.06482, 2019.(https://arxiv.org/abs/1905.06482)

"""

from collections import OrderedDict

from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import (Concatenate, Dense, Embedding,
                                            Flatten, Input)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

from ..inputs import (build_input_features,
                      get_embedding_vec_list, get_inputs_list,SparseFeat,VarLenSparseFeat,DenseFeat,embedding_lookup,get_dense_input,combined_dnn_input)
from ..layers.core import DNN, PredictionLayer
from ..layers.sequence import (AttentionSequencePoolingLayer, BiasEncoding,
                               BiLSTM, Transformer)
from ..layers.utils import NoMask, concat_fun


def DSIN(dnn_feature_columns, sess_feature_list, embedding_size=8, sess_max_count=5, sess_len_max=10, bias_encoding=False,
         att_embedding_size=1, att_head_num=8, dnn_hidden_units=(200, 80), dnn_activation='sigmoid', dnn_dropout=0,
         dnn_use_bn=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, init_std=0.0001, seed=1024, task='binary',
         ):
    """Instantiates the Deep Session Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.    :param sess_feature_list: list,to indicate session feature sparse field (**now only support sparse feature**),must be a subset of ``feature_dim_dict["sparse"]``
    :param embedding_size: positive integer,sparse feature embedding_size.
    :param sess_max_count: positive int, to indicate the max number of sessions
    :param sess_len_max: positive int, to indicate the max length of each session
    :param bias_encoding: bool. Whether use bias encoding or postional encoding
    :param att_embedding_size: positive int, the embedding size of each attention head
    :param att_head_num: positive int, the number of attention head
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """
    #check_feature_config_dict(dnn_feature_columns)

    if (att_embedding_size * att_head_num != len(sess_feature_list) * embedding_size):
        raise ValueError(
            "len(session_feature_lsit) * embedding_size must equal to att_embedding_size * att_head_num ,got %d * %d != %d *%d" % (
            len(sess_feature_list), embedding_size, att_embedding_size, att_head_num))

    # sparse_input, dense_input, user_behavior_input_dict, _, user_sess_length = get_input(
    #     dnn_feature_columns, sess_feature_list, sess_max_count, sess_len_max)

    # def get_input(feature_dim_dict, seq_feature_list, sess_max_count, seq_max_len):
    #     sparse_input, dense_input = build_input_features(feature_dim_dict)
    #     user_behavior_input = {}
    #     for idx in range(sess_max_count):
    #         sess_input = OrderedDict()
    #         for i, feat in enumerate(seq_feature_list):
    #             sess_input[feat] = Input(
    #                 shape=(seq_max_len,), name='seq_' + str(idx) + str(i) + '-' + feat)
    #
    #         user_behavior_input["sess_" + str(idx)] = sess_input
    #
    #     user_behavior_length = {"sess_" + str(idx): Input(shape=(1,), name='seq_length' + str(idx)) for idx in
    #                             range(sess_max_count)}
    #     user_sess_length = Input(shape=(1,), name='sess_length')
    #
    #     return sparse_input, dense_input, user_behavior_input, user_behavior_length, user_sess_length


    features = build_input_features(dnn_feature_columns)

    sparse_feature_columns = list(filter(lambda x:isinstance(x,SparseFeat),dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []


    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "sess" + x, sess_feature_list))
    #user_behavior_input_dict = {"sess_"+str(i):{} for i in range(sess_max_count)}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            continue
            #history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)


    inputs_list = list(features.values())


    user_behavior_input_dict = {}
    for idx in range(sess_max_count):
        sess_input = OrderedDict()
        for i, feat in enumerate(sess_feature_list):
            sess_input[feat] = features["sess_"+str(idx)+"_"+feat]
                #Input(shape=(seq_max_len,), name='seq_' + str(idx) + str(i) + '-' + feat)

        user_behavior_input_dict["sess_" + str(idx)] = sess_input


    #user_behavior_length = {"sess_" + str(idx): Input(shape=(1,), name='seq_length' + str(idx)) for idx in
    #                            range(sess_max_count)}
    user_sess_length = Input(shape=(1,), name='sess_length')



    embedding_dict = {feat.embedding_name: Embedding(feat.dimension, embedding_size,
                                                  embeddings_initializer=RandomNormal(
                                                      mean=0.0, stddev=init_std, seed=seed),
                                                  embeddings_regularizer=l2(
                                                      l2_reg_embedding),
                                                  name='sparse_emb_' +
                                                       str(i) + '-' + feat.name,
                                                  mask_zero=(feat.name in sess_feature_list)) for i, feat in
                             enumerate(sparse_feature_columns)}



    query_emb_list = embedding_lookup(embedding_dict,features,sparse_feature_columns,sess_feature_list,sess_feature_list)#query是单独的
    keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns, history_fc_names, history_fc_names)
    dnn_input_emb_list = embedding_lookup(embedding_dict,features,sparse_feature_columns,mask_feat_list=sess_feature_list)
    dense_value_list = get_dense_input(features, dense_feature_columns)




    #query_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, dnn_feature_columns["sparse"],
    #                                        sess_feature_list, sess_feature_list)

    query_emb = concat_fun(query_emb_list)

    #dnn_input_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, dnn_feature_columns["sparse"],
    #                                             mask_feat_list=sess_feature_list)
    dnn_input_emb = concat_fun(dnn_input_emb_list)
    dnn_input_emb = Flatten()(NoMask()(dnn_input_emb))

    tr_input = sess_interest_division(embedding_dict, user_behavior_input_dict, sparse_feature_columns,
                                      sess_feature_list, sess_max_count, bias_encoding=bias_encoding)

    Self_Attention = Transformer(att_embedding_size, att_head_num, dropout_rate=0, use_layer_norm=False,
                                 use_positional_encoding=(not bias_encoding), seed=seed, supports_masking=True,
                                 blinding=True)
    sess_fea = sess_interest_extractor(
        tr_input, sess_max_count, Self_Attention)

    interest_attention_layer = AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True,
                                                             supports_masking=False)(
        [query_emb, sess_fea, user_sess_length])

    lstm_outputs = BiLSTM(len(sess_feature_list) * embedding_size,
                          layers=2, res_layers=0, dropout_rate=0.2, )(sess_fea)
    lstm_attention_layer = AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True)(
        [query_emb, lstm_outputs, user_sess_length])

    dnn_input_emb = Concatenate()(
        [dnn_input_emb, Flatten()(interest_attention_layer), Flatten()(lstm_attention_layer)])
    # if len(dense_input) > 0:
    #     deep_input_emb = Concatenate()(
    #         [deep_input_emb] + list(dense_input.values()))
    dnn_input_emb = combined_dnn_input([dnn_input_emb],dense_value_list)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, dnn_use_bn, seed)(dnn_input_emb)
    output = Dense(1, use_bias=False, activation=None)(output)
    output = PredictionLayer(task)(output)

    sess_input_list = []
    # sess_input_length_list = []
    for i in range(sess_max_count):
        sess_name = "sess_" + str(i)
        sess_input_list.extend(get_inputs_list(
            [user_behavior_input_dict[sess_name]]))
        # sess_input_length_list.append(user_behavior_length_dict[sess_name])

    # model_input_list = get_inputs_list([sparse_input, dense_input]) + sess_input_list + [
    #     user_sess_length]
    #

    model = Model(inputs=inputs_list+[user_sess_length], outputs=output)

    return model


def sess_interest_division(sparse_embedding_dict, user_behavior_input_dict, sparse_fg_list, sess_feture_list,
                           sess_max_count,
                           bias_encoding=True):
    tr_input = []
    for i in range(sess_max_count):
        sess_name = "sess_" + str(i)
        keys_emb_list = get_embedding_vec_list(sparse_embedding_dict, user_behavior_input_dict[sess_name],
                                               sparse_fg_list, sess_feture_list, sess_feture_list)
        # [sparse_embedding_dict[feat](user_behavior_input_dict[sess_name][feat]) for feat in
        #             sess_feture_list]
        keys_emb = concat_fun(keys_emb_list)
        tr_input.append(keys_emb)
    if bias_encoding:
        tr_input = BiasEncoding(sess_max_count)(tr_input)
    return tr_input


def sess_interest_extractor(tr_input, sess_max_count, TR):
    tr_out = []
    for i in range(sess_max_count):
        tr_out.append(TR(
            [tr_input[i], tr_input[i]]))
    sess_fea = concat_fun(tr_out, axis=1)
    return sess_fea