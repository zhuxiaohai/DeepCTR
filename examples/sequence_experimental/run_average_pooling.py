import numpy as np
import pandas as pd
import tensorflow as tf
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names, build_input_features
from deepctr.models.sequence.attentional_pooling import AttentionalPooling
from deepctr.inputs import get_dense_input


def get_xy_fd(hash_flag=False):
    constant_feature_columns = [SparseFeat('user', 5, embedding_dim=10, use_hash=hash_flag),
                       SparseFeat('gender', 2, embedding_dim=4, use_hash=hash_flag),
                       # SparseFeat('item_id', 3 + 1, embedding_dim=8, use_hash=hash_flag),
                       # SparseFeat('cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag),
                       DenseFeat('pay_score', 3)]

    behavior_feature_columns = [
        VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=5 + 1, embedding_dim=8, embedding_name='item_id'),
                         maxlen=4, length_name="seq_length"),
        VarLenSparseFeat(SparseFeat('hist_cate_id', 5 + 1, embedding_dim=4, embedding_name='cate_id'),
                         maxlen=4, length_name="seq_length"),
        DenseFeat('hist_dense1', 4),
        DenseFeat('hist_dense2', 4)]

    behavior_sparse_indicator = ["item_id", "cate_id"]
    uid = np.array([0, 1, 2, 3, 4])
    ugender = np.array([0, 0, 1, 1, 0])
    # iid = np.array([1, 2, 3])  # 0 is mask value
    # cate_id = np.array([1, 2, 2])  # 0 is mask value
    score = np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.3], [0.3, 0.2, 0.3],
                      [0.4, 0.2, 0.3], [0.5, 0.2, 0.3]])

    hist_iid = np.array([[1, 2, 3, 0], [2, 2, 3, 0], [3, 2, 0, 0],
                         [4, 5, 0, 0], [5, 1, 2, 0]])
    hist_cate_id = np.array([[1, 2, 2, 0], [2, 2, 2, 0], [3, 2, 0, 0],
                             [4, 2, 0, 0], [5, 2, 2, 0]])
    dense1 = np.array([[0.5, 0.1, 0.2, 0], [0.7, 0.6, 0.3, 0], [0.3, 0.2, 0, 0],
                       [0.1, 0.1, 0, 0], [0.2, 0.1, 0.2, 0]])
    dense2 = np.array([[0.2, 0.2, 0.2, 0], [0.5, 0.1, 0.1, 0], [0.1, 0.2, 0, 0],
                       [0.4, 0.2, 0, 0], [0.3, 0.1, 0.1, 0]])

    behavior_length = np.array([3, 3, 2, 2, 3])

    feature_dict = {'user': uid,
                    'gender': ugender,
                    # 'item_id': iid, 'cate_id': cate_id,
                    'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id,
                    'pay_score': score, "seq_length": behavior_length,
                    'hist_dense1': dense1, 'hist_dense2': dense2}

    x = {name: feature_dict[name] for name in get_feature_names(
        constant_feature_columns + behavior_feature_columns)}
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    return x, y, constant_feature_columns, behavior_feature_columns, behavior_sparse_indicator


def make_list(features):
    new_feats = {}
    for name, value in features.items():
        if name in ['y']:
            continue
        elif (name.find('hist') < 0) and (name != 'pay_score'):
            new_feats[name] = value
        else:
            ini = tf.ones_like(value, dtype=tf.int32)
            end = tf.strings.length(value) - 2
            value = tf.strings.substr(value, ini, end)
            value = tf.strings.split(value, ',').to_tensor()
            if name in  ['hist_dense1', 'hist_dense2', 'pay_score']:
                value = tf.strings.to_number(value)
            else:
                value = tf.strings.to_number(value, out_type=tf.int32)
            new_feats[name] = value
    return new_feats, features['y']


def stack_constant_dense(features, label):
    new = {feature.name: features[feature.name] for feature in constant_dense_feature_columns}
    return tf.concat(list(new.values()), -1)


def stack_sequence_dense(features, label):
    new = {feature.name: features[feature.name] for feature in varlen_dense_feature_columns}
    return tf.stack(list(new.values()), -1)


if __name__ == '__main__':
    x, y, constant_feature_columns, behavior_feature_columns, behavior_sparse_indicator = get_xy_fd()
    # df = pd.DataFrame()
    # for name, value in x.items():
    #     print(name)
    #     df[name] = value.tolist()
    # df['y'] = y
    # df.to_csv('../data/sequence/toy_sequence.csv', index=None)

    csv_train_ds = tf.data.experimental.make_csv_dataset(
        '../data/sequence/toy_sequence.csv',
        batch_size=2,
        shuffle_seed=2,
        shuffle=True,
        ignore_errors=True,)
    csv_train_adapt_ds = tf.data.experimental.make_csv_dataset(
        '../data/sequence/toy_sequence.csv',
        batch_size=2,
        shuffle=False,
        ignore_errors=True,
        num_epochs=1,)

    csv_val_ds = tf.data.experimental.make_csv_dataset(
        '../data/sequence/toy_sequence.csv',
        batch_size=2,
        shuffle=False,
        shuffle_seed=2,
        num_epochs=1,
        ignore_errors=True,)

    csv_train_ds_mapped = csv_train_ds.map(make_list)
    csv_train_adapt_ds_mapped = csv_train_adapt_ds.map(make_list)
    csv_val_ds_mapped = csv_val_ds.map(make_list)

    features = build_input_features(constant_feature_columns + behavior_feature_columns)
    constant_dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), constant_feature_columns) if constant_feature_columns else [])
    varlen_dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), behavior_feature_columns) if behavior_feature_columns else [])
    history_dense_value_list = get_dense_input(features, varlen_dense_feature_columns)
    constant_dense_value_list = get_dense_input(features, constant_dense_feature_columns)

    sequence_normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
    sequence_normalizer.adapt(csv_train_adapt_ds_mapped.map(stack_sequence_dense))
    constant_normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
    constant_normalizer.adapt(csv_train_adapt_ds_mapped.map(stack_constant_dense))

    model = AttentionalPooling(constant_feature_columns, behavior_feature_columns, behavior_sparse_indicator,
                               sequence_dense_normalizer=sequence_normalizer,
                               constant_dense_normalizer=constant_normalizer,
                               dnn_hidden_units=[4, 4], dnn_dropout=0.6)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
    model.compile(optimizer, "binary_crossentropy")
    history = model.fit(csv_train_ds_mapped,
                        epochs=5, steps_per_epoch=3,
                        validation_data=csv_val_ds_mapped)
    print(model.predict(csv_val_ds_mapped))