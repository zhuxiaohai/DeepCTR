import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns, get_linear_logit, DEFAULT_GROUP_NAME
from deepctr.layers.core import ModifiedPredictionLayer, DNN
from deepctr.layers.interaction import FM
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input


def GBDT_FM(linear_feature_columns, dnn_feature_columns, leaf_embed_model,
            dnn_hidden_units=(32, 16), dnn_use_bn=False, l2_reg_dnn=0, dnn_dropout=0, dnn_activation='relu',
            num_outdim=1, l2_reg_linear=1e-5, l2_reg_embedding=1e-5, seed=1024):
    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear', l2_reg=l2_reg_linear)

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)

    fm_logit = add_func([FM()(concat_func(v, axis=1)) for k, v in group_embedding_dict.items() if k in DEFAULT_GROUP_NAME])

    dnn_input = combined_dnn_input([], dense_value_list)

    dnn_output = leaf_embed_model(dnn_input)

    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                     seed=seed)(dnn_output)

    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)

    final_logit = add_func([linear_logit, fm_logit, dnn_logit])

    output = ModifiedPredictionLayer(multiclass_num=num_outdim)(final_logit)

    model = tf.keras.Model(inputs=inputs_list, outputs=output)

    return model


def GBDT_Resnet(linear_feature_columns, dnn_feature_columns, leaf_embed_model,
            dnn_hidden_units=(32, 16), dnn_use_bn=False, l2_reg_dnn=0, dnn_dropout=0, dnn_activation='relu',
            num_outdim=1, l2_reg_embedding=1e-5, seed=1024):
    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features,
                                                                        dnn_feature_columns,
                                                                        l2_reg_embedding,
                                                                        seed)

    dnn_input = combined_dnn_input([], dense_value_list)

    dnn_output = leaf_embed_model(dnn_input)

    dnn_output = combined_dnn_input(group_embedding_dict, [dnn_output])

    dnn_output_fnn = DNN(dnn_hidden_units[:-1], dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                         seed=seed)(dnn_output)

    dnn_output_fnn = tf.keras.layers.Dense(
        dnn_hidden_units[-1], use_bias=True, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output_fnn)

    dnn_output_fnn = tf.keras.layers.BatchNormalization()(dnn_output_fnn)

    dnn_output = tf.keras.layers.Dense(
        dnn_hidden_units[-1], use_bias=True, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)

    dnn_output = tf.keras.layers.Add()([dnn_output, dnn_output_fnn])

    final_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)

    output = ModifiedPredictionLayer(multiclass_num=num_outdim)(final_logit)

    model = tf.keras.Model(inputs=inputs_list, outputs=output)

    return model