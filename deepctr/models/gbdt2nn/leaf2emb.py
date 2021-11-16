import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.python.keras.initializers import Constant, glorot_normal, Zeros
from tensorflow.python.keras.regularizers import l2

from deepctr.feature_column import build_input_features
from deepctr.inputs import get_dense_input
from deepctr.layers.utils import concat_func


class Leaf2Embedding(Layer):
    def __init__(self, n_split, maxleaf, embsize, use_bn=False,
                 l2_reg=0.0, seed=1024, *args, **kwargs):
        self.n_split = n_split
        self.maxleaf = maxleaf
        self.embsize = embsize
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.seed = seed
        super(Leaf2Embedding, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.max_ntree_per_split = tf.cast(input_shape[-1] / self.n_split, dtype=tf.int32)
        self.embed_w = self.add_weight(name='embed_w',
                                       shape=(self.n_split, self.max_ntree_per_split * self.maxleaf, self.embsize),
                                       initializer=glorot_normal(seed=self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True)
        super(Leaf2Embedding, self).build(input_shape)

    def call(self, inputs, training=True):
        batch_size = tf.shape(inputs)[0]
        inputs = tf.cast(tf.reshape(inputs, [-1]), dtype=tf.int32)
        one_hot = tf.one_hot(inputs, depth=self.maxleaf, axis=-1)
        one_hot = tf.reshape(one_hot, [batch_size, self.n_split, -1])
        one_hot = tf.transpose(one_hot, (1, 0, 2))

        leaf_emb = tf.matmul(one_hot, self.embed_w)
        leaf_emb = tf.transpose(leaf_emb, (1, 0, 2))
        leaf_emb = tf.keras.layers.Reshape((-1,), name='embedded_leaf')(leaf_emb)
        if self.use_bn:
            leaf_emb = tf.keras.layers.BatchNormalization()(leaf_emb)
        return leaf_emb

    def get_config(self):
        config = {'n_split': self.n_split,
                  'maxleaf': self.maxleaf,
                  'embsize': self.embsize,
                  'l2_reg': self.l2_reg,
                  'use_bn': self.use_bn,
                  'seed': self.seed
                  }
        base_config = super(Leaf2Embedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_split * self.embsize


class Embedding2Score(Layer):
    def __init__(self, n_split, out_bias_initializer,
                 l2_reg=0.0, seed=1024, *args, **kwargs):
        self.n_split = n_split
        self.l2_reg = l2_reg
        self.seed = seed
        self.out_bias_initializer = out_bias_initializer
        super(Embedding2Score, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.embsize = tf.cast(input_shape[-1] / self.n_split, dtype=tf.int32)
        self.out_w = self.add_weight(name='out_w',
                                     shape=(self.n_split, self.embsize, 1),
                                     initializer=glorot_normal(seed=self.seed),
                                     regularizer=l2(self.l2_reg),
                                     trainable=True)
        self.out_b = self.add_weight(name='out_b',
                                     shape=(self.n_split, 1, 1),
                                     initializer=self.out_bias_initializer,
                                     trainable=True)
        super(Embedding2Score, self).build(input_shape)

    def call(self, inputs, training=True):
        out = tf.keras.layers.Reshape([self.n_split, -1])(inputs)
        out = tf.transpose(out, [1, 0, 2])
        out = tf.matmul(out, self.out_w)
        out += self.out_b
        out = tf.transpose(out, [1, 0, 2])
        out = tf.keras.layers.Reshape((-1,), name='slip_scores')(out)
        return out

    def get_config(self):
        config = {'n_split': self.n_split,
                  'l2_reg': self.l2_reg,
                  'seed': self.seed,
                  'out_bias_initializer': self.out_bias_initializer
                  }
        base_config = super(Embedding2Score, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_split


# class EmbeddingLeafModel(Model):
#     def __init__(self, n_split, maxleaf, embsize, task='regression', out_bias=None,
#                  use_bn=False, l2_reg=0.0, dropout_rate=0.0, seed=1024, *args, **kwargs):
#         super(EmbeddingLeafModel, self).__init__(*args, **kwargs)
#         self.task = task
#         self.n_split = n_split
#         self.maxleaf = maxleaf
#         self.embsize = embsize
#         if out_bias is not None:
#             self.out_bias_initializer = Constant(out_bias, verify_shape=True)
#         else:
#             self.out_bias_initializer = Zeros()
#         self.use_bn = use_bn
#         self.l2_reg = l2_reg
#         self.dropout_rate = dropout_rate
#         self.seed = seed
#         self.embedding_layer = Leaf2Embedding(n_split=self.n_split,
#                                               maxleaf=self.maxleaf,
#                                               embsize=self.embsize,
#                                               l2_reg=self.l2_reg,
#                                               seed=self.seed)
#         self.mapping_to_score_layer = Embedding2Score(n_split=self.n_split,
#                                                       out_bias_initializer=self.out_bias_initializer,
#                                                       l2_reg=self.l2_reg,
#                                                       seed=self.seed)
#         if self.use_bn:
#             self.bn_layer = tf.keras.layers.BatchNormalization()
#         self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed)
#         self.activation_layer = tf.keras.layers.Activation('sigmoid') if self.task != 'regression' else None
#
#     def call(self, inputs, training=None, mask=None):
#         leaf_emb = self.embedding_layer(inputs)
#         if self.use_bn:
#             leaf_emb = self.bn_layer(leaf_emb)
#         leaf_emb = self.dropout_layer(leaf_emb)
#         slip_scores = self.mapping_to_score_layer(leaf_emb)
#         final_score = tf.reduce_sum(slip_scores, axis=-1)
#         if self.task != 'regression':
#             final_score = self.activation_layer(final_score)
#         return slip_scores, final_score


def EmbeddingLeafModel(dnn_feature_columns, n_split, maxleaf, embsize,
                       task='regression', out_bias=None, use_bn=False, l2_reg=0.0, dropout_rate=0.0, seed=1024):
    # prepare intermediate layers
    if out_bias is not None:
        out_bias_initializer = Constant(out_bias, verify_shape=True)
    else:
        out_bias_initializer = Zeros()
    embedding_layer = Leaf2Embedding(n_split=n_split,
                                     maxleaf=maxleaf,
                                     embsize=embsize,
                                     use_bn=use_bn,
                                     l2_reg=l2_reg,
                                     seed=seed,
                                     name='leaf2emb')
    mapping_to_score_layer = Embedding2Score(n_split=n_split,
                                             out_bias_initializer=out_bias_initializer,
                                             l2_reg=l2_reg,
                                             seed=seed,
                                             name='emb2score')

    # prepare inputs
    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    dense_value_list = get_dense_input(features, dnn_feature_columns)
    dnn_input = concat_func(dense_value_list)

    # embedding leaf-nodes
    leaf_emb = embedding_layer(dnn_input)
    leaf_emb = tf.keras.layers.Dropout(dropout_rate, seed=seed)(leaf_emb)

    # mapping embeddings to scores
    slip_scores = mapping_to_score_layer(leaf_emb)
    final_score = tf.reduce_sum(slip_scores, axis=-1, name='final_score')
    if task != 'regression':
        final_score = tf.keras.layers.Activation('sigmoid', name='final_score')(final_score)

    model = Model(inputs=inputs_list, outputs=(slip_scores, final_score))
    return model

