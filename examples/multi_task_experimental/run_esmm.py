#%%
import os
import re
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve

from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.metrics import AUC
from tensorflow.python.keras.models import load_model, Model
from tensorflow.keras.metrics import Mean
from tensorflow import keras
import tensorflow as tf

from deepctr.layers import custom_objects
from deepctr.layers.utils import NoMask
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models.multitask.esmm import ESMM
from deepctr.models.multitask.call_backs import MyEarlyStopping
from deepctr.models.multitask.multitaskbase import MultiTaskModelBase

custom_objects['NoMask'] = NoMask
custom_objects['MultiTaskModelBase'] = MultiTaskModelBase
custom_objects['Mean'] = Mean
custom_objects['AUC'] = AUC


if __name__ == "__main__":
    # configure
    task_name = 'preloan_istrans_overdue'
    run_name = 'uncertainty_weight_esmm2'
    checkpoint_dir = '.\\' + task_name + '\ckt\\' + run_name
    tensorboard_dir = '.\\' + task_name + '\log_dir\\' + run_name
    summary_dir = '.\\' + task_name + '\metrics\\' + run_name
    tasks = [('istrans', 'binary'), ('conditional_fpd4', 'binary'), ('fpd4', 'binary')]
    loss_fns = {'istrans': keras.losses.BinaryCrossentropy(),
                'fpd4': keras.losses.BinaryCrossentropy()}
    metrics_logger = {'istrans': AUC(name='istrans_auc'),
                      'fpd4': AUC(name='fpd4_auc')}
    loss_weights = {'istrans': 1, 'fpd4': 6}
    uncertainty = True
    batch_size = 256
    mode = 'train'

    # read data
    data = pd.read_csv('../data/train_for_multi2.csv')
    col_x = ['td_i_cnt_partner_all_imbank_365d',
             'duotou_br_als_m3_id_pdl_allnum',
             'marketing_channel_pred_1',
             'td_i_length_first_all_consumerfinance_365d',
             'duotou_br_als_m12_cell_nbank_allnum',
             'tx_m12_id_platnum',
             'duotou_br_als_m12_id_caon_allnum',
             'duotou_br_alf_apirisk_all_sum',
             'model_key_pred_1',
             'duotou_bh_rl_creditlimitsum',
             'dxm_dt_score',
             'credit_score_ronghuixf',
             'regist_channel_pred_1',
             'cs_mf_score_dt',
             'credit_score_sh',
             'duotou_br_als_m3_id_nbank_allnum',
             'br_frg_list_level',
             'td_3m_idcard_lending_cnt',
             'duotou_br_alf_apirisk_all_mean',
             'cs_hc_phone_score',
             'tx_m6_cell_allnum',
             'duotou_br_als_m12_id_pdl_allnum',
             'dxm_qzf',
             'ali_rain_score',
             'td_zhixin_score',
             'ab_local_cnt',
             'ab_mobile_cnt',
             'cust_id_area',
             'cust_gender',
             'immediate_relation_cnt',
             'hds_36m_month_max_purchase_money_excp_doub11_12',
             'ab_local_ratio',
             'hds_36m_purchase_steady',
             'cust_work_city',
             'relation_contact_cnt',
             'td_xyf_dq_score',
             'selffill_is_have_creditcard',
             'credit_repayment_score_bj_2',
             'bj_jc_m36_consume_cnt',
             'selffill_degree',
             'selffill_marital_status',
             'operation_sys',
             'hds_mobile_rich',
             'ab_prov_cnt',
             'hds_36m_total_purchase_cnt',
             'idcard_district_grade',
             'study_app_cnt',
             'hds_recent_consumme_active_rank',
             'hds_mobile_reli_rank',
             'hds_36m_doub11_12_total_purchase_money',
             'idcard_rural_flag']
    sparse_features = ['marketing_channel_pred_1',
                         'model_key_pred_1',
                         'regist_channel_pred_1',
                         'cust_id_area',
                         'cust_gender',
                         'cust_work_city',
                         'selffill_is_have_creditcard',
                         'selffill_marital_status',
                         'operation_sys',
                         'hds_mobile_reli_rank',
                         'idcard_rural_flag']
    data['conditional_fpd4'] = data['fpd4']
    data['conditional_fpd4_weight'] = 1.0
    data['conditional_fpd4_mask'] = 1.0
    data['fpd4_weight'] = 1.0
    data['fpd4_mask'] = 1.0
    data.loc[data['fpd4'].isnull(), 'conditional_fpd4_weight'] = 1.0
    data.loc[data['fpd4'].isnull(), 'fpd4_weight'] = 1.0
    data.loc[data['fpd4'].isnull(), 'conditional_fpd4_mask'] = 0
    data.loc[data['fpd4'].isnull(), 'fpd4_mask'] = 0
    data['istrans_weight'] = 1.0
    data['istrans_mask'] = 1.0
    data['conditional_fpd4'] = data['conditional_fpd4'].fillna(0)
    data['fpd4'] = data['fpd4'].fillna(0)
    data[col_x] = data[col_x].fillna(-1)

    # define input tensors
    dense_features = []
    for col in col_x:
        if col not in sparse_features:
            dense_features.append(col)

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].unique().shape[0], embedding_dim=8)
                              for i, feat in enumerate(sparse_features)] + \
                             [DenseFeat(feat, 1, ) for feat in dense_features]
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)

    # generate input data for model
    train = data[data['set'] == '1train']
    test = data[data['set'] == '2test']
    oot = data[data['set'] == '3oot']
    model_input = {name: train[name] for name in feature_names}
    model_batch_input = {name: train[name].iloc[:1] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    oot_model_input = {name: oot[name] for name in feature_names}

    # train or predict
    if mode == 'train':
        model = ESMM(dnn_feature_columns, tasks=tasks)
        optimizers = keras.optimizers.Adam()
        model.compile(optimizer=optimizers,
                      loss=loss_fns,
                      metrics=metrics_logger,
                      loss_weights=loss_weights,
                      uncertainly=uncertainty)
        try:
            checkpoints = [checkpoint_dir + '\\' + name for name in os.listdir(checkpoint_dir)]
            latest_checkpoint = max(checkpoints).split('.index')[0]
            model.train_on_batch(model_batch_input,
                                 {'istrans': train[['istrans']].iloc[:1], 'fpd4': train[['fpd4']].iloc[:1]})
            model.load_weights(latest_checkpoint)
            _, last_epoch, last_lr = latest_checkpoint.split('-')
            print('Restoring from ', latest_checkpoint)
            last_epoch = int(last_epoch)
            last_lr = float(last_lr)
        except:
            print('Creating a new model')
            last_epoch = 0
            last_lr = 0.001
        history = model.fit(model_input,
                            {'istrans': train[['istrans']],
                             'fpd4': train[['fpd4']]},
                            sample_weight={'istrans': train[['istrans_weight']], 'fpd4': train[['fpd4_weight']]},
                            batch_size=256,
                            epochs=100,
                            initial_epoch=last_epoch,
                            verbose=2,
                            validation_data=(test_model_input,
                                             {'istrans': test[['istrans']], 'fpd4': test[['fpd4']]},
                                             {'istrans': test[['istrans_weight']], 'fpd4': test[['fpd4_weight']]}),
                            callbacks=[
                                      MyEarlyStopping('val_fpd4_auc',
                                                      patience=10,
                                                      savepath=checkpoint_dir,
                                                      coef_of_balance=0.4,
                                                      direction='maximize'),
                                      TensorBoard(log_dir=tensorboard_dir)
                            ]
                            )
    else:
        best_metric = -1
        best_model = None
        for i in os.listdir(checkpoint_dir):
            if i.find('best_model') >= 0:
                metric = float(re.match('.*auc(.*).h5', i)[1])
                if metric > best_metric:
                    best_metric = metric
                    best_model = i
        print('loading ', checkpoint_dir + '\\' + best_model)
        model = load_model(checkpoint_dir + '\\' + best_model, custom_objects=custom_objects)
        intermediate_layer = model.get_layer(tasks[1][0])
        intermediate_model = Model(model.input, outputs={tasks[1][0]: intermediate_layer.output})
        file_writer = tf.summary.create_file_writer(summary_dir)
        print('final_result')
        for name in ['istrans', 'conditional_fpd4', 'fpd4']:
            if name != 'conditional_fpd4':
                predictions = model.predict(model_input)
            else:
                predictions = intermediate_model.predict(model_input)
            auc_score = roc_auc_score(train[name].values, predictions[name][:, 0], sample_weight=train[name+'_mask'])
            fpr, tpr, _ = roc_curve(train[name].values, predictions[name][:, 0], sample_weight=train[name+'_mask'])
            ks = np.max(np.abs(tpr - fpr))
            print(' {}: train_auc {:4f} train_ks {:4f}'.format(name, auc_score, ks))
            with file_writer.as_default():
                tf.summary.scalar(name+'_ks', ks, step=1)
                tf.summary.scalar(name+'_auc', auc_score, step=1)

        for name in ['istrans', 'conditional_fpd4', 'fpd4']:
            if name != 'conditional_fpd4':
                predictions = model.predict(test_model_input)
            else:
                predictions = intermediate_model.predict(test_model_input)
            auc_score = roc_auc_score(test[name].values, predictions[name][:, 0], sample_weight=test[name+'_mask'])
            fpr, tpr, _ = roc_curve(test[name].values, predictions[name][:, 0], sample_weight=test[name+'_mask'])
            ks = np.max(np.abs(tpr - fpr))
            print(' {}: test_auc {:4f} test_ks {:4f}'.format(name, auc_score, ks))
            with file_writer.as_default():
                tf.summary.scalar(name+'_ks', ks, step=2)
                tf.summary.scalar(name+'_auc', auc_score, step=2)

        for name in ['istrans', 'conditional_fpd4', 'fpd4']:
            if name != 'conditional_fpd4':
                predictions = model.predict(oot_model_input)
            else:
                predictions = intermediate_model.predict(oot_model_input)
            auc_score = roc_auc_score(oot[name].values, predictions[name][:, 0], sample_weight=oot[name+'_mask'])
            fpr, tpr, _ = roc_curve(oot[name].values, predictions[name][:, 0], sample_weight=oot[name+'_mask'])
            ks = np.max(np.abs(tpr - fpr))
            print(' {}: oot_auc {:4f} oot_ks {:4f}'.format(name, auc_score, ks))
            with file_writer.as_default():
                tf.summary.scalar(name+'_ks', ks, step=3)
                tf.summary.scalar(name+'_auc', auc_score, step=3)
