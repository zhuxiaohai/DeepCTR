import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.metrics import AUC, Mean

from deepctr.models.gbdt2nn.gbdt_predict_leaf import gbdt_predict
from deepctr.models.gbdt2nn.leaf2emb import EmbeddingLeafModel, Embedding2Score, Leaf2Embedding
from deepctr.feature_column import DenseFeat, get_feature_names, SparseFeat
from deepctr.callbacks import ModifiedExponentialDecay
from deepctr.layers import custom_objects
from deepctr.layers.utils import NoMask
from deepctr.models.gbdt2nn.basenet import GBDT_FM

custom_objects['NoMask'] = NoMask
custom_objects['Mean'] = Mean
custom_objects['AUC'] = AUC
custom_objects['ModifiedExponentialDecay'] = ModifiedExponentialDecay
custom_objects['Embedding2Score'] = Embedding2Score
custom_objects['Leaf2Embedding'] = Leaf2Embedding
custom_objects['EmbeddingLeafModel'] = EmbeddingLeafModel
train_cate = pd.read_csv('../data/risk_offline/risk_offline_cate/train/train_cate.csv')
test_cate = pd.read_csv('../data/risk_offline/risk_offline_cate/test/test_cate.csv')


def load_data(data):
    data_dir = '../data/'
    trn_x = np.load(data_dir+"/%s/train_features.npy"%data).astype(np.float32)
    trn_y = np.load(data_dir+"/%s/train_labels.npy"%data).astype(np.float32)
    vld_x = np.load(data_dir+"/%s/test_features.npy"%data).astype(np.float32)
    vld_y = np.load(data_dir+"/%s/test_labels.npy"%data).astype(np.float32)

    vld_x1 = np.load(data_dir+"/%s/test_3oot_features.npy"%data).astype(np.float32)
    vld_y1= np.load(data_dir+"/%s/test_3oot_labels.npy"%data).astype(np.float32)
    vld_x2 = np.load(data_dir+"/%s/test_4oot_features.npy"%data).astype(np.float32)
    vld_y2 = np.load(data_dir+"/%s/test_4oot_labels.npy"%data).astype(np.float32)
    vld_x3 = np.load(data_dir+"/%s/test_5oot_features.npy"%data).astype(np.float32)
    vld_y3 = np.load(data_dir+"/%s/test_5oot_labels.npy"%data).astype(np.float32)

    mean = np.mean(trn_x, axis=0)
    std = np.std(trn_x, axis=0)
    trn_x = (trn_x - mean) / (std + 1e-5)
    vld_x = (vld_x - mean) / (std + 1e-5)

    vld_x1 = (vld_x1 - mean) / (std + 1e-5)
    vld_x2 = (vld_x2 - mean) / (std + 1e-5)
    vld_x3 = (vld_x3 - mean) / (std + 1e-5)

    return trn_x, trn_y, vld_x, vld_y, vld_x1, vld_y1, vld_x2, vld_y2, vld_x3, vld_y3


def load_lightgbm(trained_gbdt_model):
    import pickle
    path = '../data/risk_offline' + '/%s' % trained_gbdt_model
    pkl_file = open(path, 'rb')
    gbm = pickle.load(pkl_file)
    pkl_file.close()
    return gbm


args = {'nslices': 1, 'maxleaf': 2, 'embsize': 32, 'feat_per_group': 200, 'group_method': 'Equal'}

train_x, train_y, test_x, test_y, oot1_x, oot1_y, oot2_x, oot2_y, oot3_x, oot3_y = load_data('risk_offline')
gbm = load_lightgbm('lightgbm_model2.pickle')
leaf_preds, test_leaf_preds, tree_outputs, test_tree_outputs, \
group_average, used_features, n_models, max_ntree_per_split, min_len_features = \
    gbdt_predict(train_x, test_x, gbm, args)
model_input = {'gbm_leaf_predictions': leaf_preds}
test_model_input = {'gbm_leaf_predictions': test_leaf_preds}

emb_model = EmbeddingLeafModel([DenseFeat('gbm_leaf_predictions', leaf_preds.shape[1])],
                               n_models,
                               args['maxleaf'] + 1,
                               args['embsize'],
                               'binary',
                               group_average)
opt = tf.keras.optimizers.Adam(learning_rate=ModifiedExponentialDecay(0.001, max_iter_num=100))
tree_outputs = np.asarray(tree_outputs).reshape((n_models, leaf_preds.shape[0])).transpose((1, 0))
test_tree_outputs = np.asarray(test_tree_outputs).reshape((n_models, test_leaf_preds.shape[0])).transpose((1, 0))
emb_model.compile(loss=['mse', None], optimizer=opt, metrics=[['mse'], ['AUC']])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_final_score_auc',
                                             patience=5,
                                             mode='max',
                                             restore_best_weights=True)
history = emb_model.fit(model_input, [tree_outputs, train_y],
                          batch_size=256,
                          validation_data=(test_model_input, [test_tree_outputs, test_y]),
                          epochs=100,
                          callbacks=[callback])

model_new = tf.keras.Model(inputs=emb_model.input, outputs=emb_model.get_layer('leaf2emb').output)

train_cate = pd.read_csv('../data/risk_offline/risk_offline_cate/train/train_cate.csv')
test_cate = pd.read_csv('../data/risk_offline/risk_offline_cate/test/test_cate.csv')
sparse_features = ['ali_rain_score', 'bj_jc_m36_consume_cnt', 'td_zhixin_score', 'hds_36m_purchase_steady',
            'hds_36m_total_purchase_cnt', 'hds_36m_month_max_purchase_money_excp_doub11_12',
            'hds_36m_doub11_12_total_purchase_money', 'ab_local_ratio', 'ab_mobile_cnt', 'cust_id_area',
            'cust_work_city', 'immediate_relation_cnt', 'relation_contact_cnt', 'study_app_cnt',
            'ab_local_cnt', 'ab_prov_cnt', 'credit_repayment_score_bj_2', 'td_xyf_dq_score'] +\
           ['hds_phone_rich_rank', 'hds_mobile_rich', 'hds_recent_consumme_active_rank', 'idcard_district_grade',
            'idcard_rural_flag', 'selffill_degree', 'selffill_is_have_creditcard', 'selffill_marital_status',
            'hds_mobile_reli_rank_Ma', 'hds_mobile_reli_rank_Mb', 'hds_mobile_reli_rank_M0', 'is_ios', 'is_male']
train_cate[sparse_features] = train_cate[sparse_features].replace([-1], 0)
test_cate[sparse_features] = test_cate[sparse_features].replace([-1], 0)
cat_feature_columns = [SparseFeat(feat, vocabulary_size=train_cate[feat].max() + 1,
                                  embedding_dim=4) for i, feat in enumerate(sparse_features)]
num_feature_columns = [DenseFeat('gbm_leaf_predictions', leaf_preds.shape[1])]
dnn_feature_columns = cat_feature_columns + num_feature_columns
linear_feature_columns = cat_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
train_cate_input = {name: train_cate[name] if name != 'gbm_leaf_predictions' else leaf_preds for name in feature_names }
test_cate_input = {name: test_cate[test_cate['set'] == '2test'][name] if name != 'gbm_leaf_predictions' else test_leaf_preds
                   for name in feature_names}

# def get_optimizer(**param):
#     return keras.optimizers.Adam(learning_rate=ModifiedExponentialDecay(**param))
#
# def build_model_fn(embedding_dim, **params):
#     cat_feature_columns = [SparseFeat(feat, vocabulary_size=train_cate[feat].max() + 1,
#                                       embedding_dim=embedding_dim) for i, feat in enumerate(sparse_features)]
#     num_feature_columns = [DenseFeat('gbm_leaf_predictions', leaf_preds.shape[1])]
#     dnn_feature_columns = cat_feature_columns + num_feature_columns
#     linear_feature_columns = cat_feature_columns
#     return GBDT_FM(linear_feature_columns, dnn_feature_columns, model_new, **params)
#
# build_model_param_dict = {
#                           'embedding_dim': ('int', {'low': 4, 'high': 8, 'step': 2}),
#                           'dnn_hidden_units': (('int', {'low': 1, 'high': 3}), ('int', {'low': 8, 'high': 64, 'step': 8})),
#                           'dnn_use_bn': ('categorical', {'choices': [True, False]}),
#                           'l2_reg_embedding': ('float', {'low': 1e-5, 'high': 1e-1, 'log': True}),
#                           'l2_reg_dnn': ('float', {'low': 1e-5, 'high': 1e-1, 'log': True})
#                           }
# compile_param_dict = {'loss': keras.losses.binary_crossentropy,
#                       'optimizer': {'initial_learning_rate': ('float', {'low': 1e-5, 'high': 1e-1, 'log': True}),
#                                     'max_iter_num': 100*int(train_cate.shape[0] / 32)},
#                       'metrics': AUC(name='task1_AUC')
#                       }
#
# from deepctr.models.tuner.optuna_search_keras import OptunaSearchKeras
# op = OptunaSearchKeras(build_model_param_dict=build_model_param_dict,
#                        compile_param_dict=compile_param_dict,
#                        build_model_fn=build_model_fn,
#                        build_optimizer_instance=get_optimizer,
#                        eval_metric="task1_AUC",
#                        coef_train_val_disparity=0,
#                        optimization_direction='maximize',
#                        early_stop_rounds=10,
#                        optuna_verbosity=1,
#                        n_trials=100, n_startup_trials=10)
# op.search(train_cate_input,
#           train_y,
#           batch_size=32,
#           epochs=100,
#           validation_data=(test_cate_input, test_y),
#           verbose=0)
# train_param = op.get_params()
# print(train_param)
# print(op.study.best_trial.params)
# model2 = op.best_model
# print(model2.evaluate(test_cate_input, test_y, batch_size=32))

model2 = GBDT_FM(linear_feature_columns, dnn_feature_columns, model_new,
                 num_outdim=1, dnn_hidden_units=(32, 16), dnn_use_bn=False,
                 l2_reg_embedding=1e-5, l2_reg_dnn=1e-5, dnn_dropout=0, dnn_activation='relu', seed=1024)
optimizer = keras.optimizers.Adam(learning_rate=ModifiedExponentialDecay(0.001, max_iter_num=100))
model2.compile(optimizer=optimizer,
               loss=keras.losses.binary_crossentropy,
               metrics=AUC(name='task1_AUC'))
history = model2.fit(train_cate_input,
                     train_y,
                     batch_size=32,
                     epochs=100,
                     validation_data=(test_cate_input, test_y),
                     callbacks=[
                            tf.keras.callbacks.EarlyStopping(monitor='val_task1_AUC',
                                                             patience=10,
                                                             mode='max',
                                                             restore_best_weights=True)
                        ]
                     )

pred_y = gbm.predict(test_x)
auc_score = roc_auc_score(test_y, pred_y)
fpr, tpr, _ = roc_curve(test_y, pred_y)
ks = np.max(np.abs(tpr - fpr))
print(auc_score, ks)
pred_y = model2.predict(test_cate_input)
auc_score = roc_auc_score(test_y, pred_y)
fpr, tpr, _ = roc_curve(test_y, pred_y)
ks = np.max(np.abs(tpr - fpr))
print(auc_score, ks)


leaf_preds, test_leaf_preds, tree_outputs, test_tree_outputs, \
group_average, used_features, n_models, max_ntree_per_split, min_len_features = \
    gbdt_predict(train_x, oot1_x, gbm, args)
test_cate_input = {name: test_cate[test_cate['set'] == '3oot'][name] if name != 'gbm_leaf_predictions' else test_leaf_preds
                   for name in feature_names}
pred_y = gbm.predict(oot1_x)
auc_score = roc_auc_score(oot1_y, pred_y)
fpr, tpr, _ = roc_curve(oot1_y, pred_y)
ks = np.max(np.abs(tpr - fpr))
print(auc_score, ks)
pred_y = model2.predict(test_cate_input)
auc_score = roc_auc_score(oot1_y, pred_y)
fpr, tpr, _ = roc_curve(oot1_y, pred_y)
ks = np.max(np.abs(tpr - fpr))
print(auc_score, ks)


leaf_preds, test_leaf_preds, tree_outputs, test_tree_outputs, \
group_average, used_features, n_models, max_ntree_per_split, min_len_features = \
    gbdt_predict(train_x, oot2_x, gbm, args)
test_cate_input = {name: test_cate[test_cate['set'] == '4oot'][name] if name != 'gbm_leaf_predictions' else test_leaf_preds
                   for name in feature_names}
pred_y = gbm.predict(oot2_x)
auc_score = roc_auc_score(oot2_y, pred_y)
fpr, tpr, _ = roc_curve(oot2_y, pred_y)
ks = np.max(np.abs(tpr - fpr))
print(auc_score, ks)
pred_y = model2.predict(test_cate_input)
auc_score = roc_auc_score(oot2_y, pred_y)
fpr, tpr, _ = roc_curve(oot2_y, pred_y)
ks = np.max(np.abs(tpr - fpr))
print(auc_score, ks)


leaf_preds, test_leaf_preds, tree_outputs, test_tree_outputs, \
group_average, used_features, n_models, max_ntree_per_split, min_len_features = \
    gbdt_predict(train_x, oot3_x, gbm, args)
test_cate_input = {name: test_cate[test_cate['set'] == '5oot'][name] if name != 'gbm_leaf_predictions' else test_leaf_preds
                   for name in feature_names}
pred_y = gbm.predict(oot3_x)
auc_score = roc_auc_score(oot3_y, pred_y)
fpr, tpr, _ = roc_curve(oot3_y, pred_y)
ks = np.max(np.abs(tpr - fpr))
print(auc_score, ks)
pred_y = model2.predict(test_cate_input)
auc_score = roc_auc_score(oot3_y, pred_y)
fpr, tpr, _ = roc_curve(oot3_y, pred_y)
ks = np.max(np.abs(tpr - fpr))
print(auc_score, ks)