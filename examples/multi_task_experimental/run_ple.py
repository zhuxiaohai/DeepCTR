#%%
import platform
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve

from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.metrics import AUC
from tensorflow.python.keras.models import load_model
from tensorflow.keras.metrics import Mean
from tensorflow import keras
import tensorflow as tf

from deepctr.layers import custom_objects
from deepctr.layers.utils import NoMask
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models.multitask.ple import PLE, CGC
from deepctr.models.multitask.call_backs import MyEarlyStopping, MyRecorder
from deepctr.models.multitask.multitaskbase import MultiTaskModelBase
from deepctr.models.multitask.utils import calc_lift, cal_psi_score

custom_objects['NoMask'] = NoMask
custom_objects['CGC'] = CGC
custom_objects['MultiTaskModelBase'] = MultiTaskModelBase
custom_objects['Mean'] = Mean
custom_objects['AUC'] = AUC


if __name__ == "__main__":
    # configure
    task_name = 'preloan_istrans_overdue2'
    run_name = '1_1_weight_ple_fpd4_nomask'
    if platform.system() == 'Windows':
        joint_symbol = '\\'
    else:
        joint_symbol = '/'
    checkpoint_dir = joint_symbol.join([task_name, 'ckt', run_name])
    tensorboard_dir = joint_symbol.join([task_name, 'log_dir', run_name])
    summary_dir = joint_symbol.join([task_name, 'metrics', run_name])
    trend_dir = joint_symbol.join([task_name, 'trend', run_name])
    if not os.path.exists(trend_dir):
        os.makedirs(trend_dir)
    tasks = {'istrans': 'binary', 'fpd4': 'binary'}
    loss_fns = {'istrans': keras.losses.binary_crossentropy,
                'fpd4': keras.losses.binary_crossentropy}
    metrics_logger = {'istrans': AUC,
                      'fpd4': AUC}
    loss_weights = {'istrans': 1, 'fpd4': 1}
    config = {'layer1': {'hidden_unit': 8, 'tasks': {'share': 2, 'istrans': 2, 'fpd4': 2}},
              'layer2': {'hidden_unit': 8, 'tasks': {'share': 2, 'istrans': 2, 'fpd4': 2}},
              }
    uncertainty = False
    batch_size = 256
    mode = 'train'

    # read data
    data = pd.read_csv('data/train_for_multi.csv')
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
    data['fpd4_weight'] = 1.0
    data['fpd4_mask'] = 1.0
    data.loc[data['fpd4'].isnull(), 'fpd4_weight'] = 1.0
    data.loc[data['fpd4'].isnull(), 'fpd4_mask'] = 0
    data['istrans_weight'] = 1.0
    data['istrans_mask'] = 1.0
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
    test_model_input = {name: test[name] for name in feature_names}
    oot_model_input = {name: oot[name] for name in feature_names}
    model_batch_input = {name: train[name].iloc[:1] for name in feature_names}
    callback_data = (test_model_input,
                     {'istrans': test[['istrans']],
                      'fpd4': test[['fpd4']]},
                     {'istrans': test['istrans_weight'],
                      'fpd4': test['fpd4_weight']})
    callback_data = list(tf.data.Dataset.from_tensor_slices(callback_data).shuffle(test['istrans'].shape[0]).batch(
        batch_size).take(1))[0]

    # train or predict
    if mode == 'train':
        model = PLE(dnn_feature_columns,
                    tasks,
                    config)
        last_shared_weights = [weight for weight in model.get_layer('cgc_layer2').trainable_weights
                               if weight.name.find('share') >= 0] + \
                              [weight for weight in model.get_layer('cgc_layer1').trainable_weights
                               if weight.name.find('share') >= 0]
        gradnorm_config = {'alpha': 0.1, 'last_shared_weights': last_shared_weights}
        optimizers = keras.optimizers.Adam()
        model.compile(optimizers=optimizers,
                      loss_fns=loss_fns,
                      metrics_logger=metrics_logger,
                      loss_weights=loss_weights,
                      uncertainly=uncertainty,
                      # gradnorm_config=gradnorm_config,
                      )

        try:
            checkpoints = [joint_symbol.join([checkpoint_dir, name]) for name in os.listdir(checkpoint_dir)]
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
                            {'istrans': train[['istrans']], 'fpd4': train[['fpd4']]},
                            sample_weight={'istrans': train['istrans_weight'], 'fpd4': train['fpd4_weight']},
                            batch_size=256,
                            epochs=3,
                            initial_epoch=last_epoch,
                            verbose=2,
                            validation_data=(test_model_input,
                                             {'istrans': test[['istrans']], 'fpd4': test[['fpd4']]},
                                             {'istrans': test['istrans_weight'], 'fpd4': test['fpd4_weight']}),
                            callbacks=[
                                      MyEarlyStopping('val_fpd4_AUC',
                                                      patience=10,
                                                      savepath=checkpoint_dir,
                                                      coef_of_balance=0.4,
                                                      direction='maximize'),
                                      TensorBoard(log_dir=tensorboard_dir),
                                      MyRecorder(log_dir=tensorboard_dir,
                                                 data=callback_data)
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
        print('loading ', joint_symbol.join([checkpoint_dir, best_model]))
        model = load_model(joint_symbol.join([checkpoint_dir, best_model]), custom_objects=custom_objects)
        file_writer = tf.summary.create_file_writer(summary_dir)
        print('final_result')
        for name in tasks.keys():
            fig = plt.figure(figsize=(8, 10))
            fig.suptitle(run_name + '_' + name)
            for index, set_name in enumerate(['1train', '2test', '3oot']):
                set_data = data[data['set'] == set_name]
                predictions = model.predict({name: set_data[name] for name in feature_names})
                auc_score = roc_auc_score(set_data[name].values, predictions[name][:, 0], sample_weight=set_data[name+'_mask'])
                fpr, tpr, _ = roc_curve(set_data[name].values, predictions[name][:, 0], sample_weight=set_data[name+'_mask'])
                ks = np.max(np.abs(tpr - fpr))
                pred = predictions[name][:, 0]
                target = set_data[name].values
                weight = set_data[name+'_mask'].values
                pred = pred[weight != 0]
                target = target[weight != 0]
                if set_name != '1train':
                    psi = cal_psi_score(pred, expected)
                    title = '{} ks_{:.2f} auc_{:.2f} psi_{:.2f}'.format(set_name, ks, auc_score, psi)
                else:
                    expected = pred
                    title = '{} ks_{:.2f} auc_{:.2f}'.format(set_name, ks, auc_score)
                df = pd.DataFrame({'pred': pred, 'target': target})
                ax = fig.add_subplot(3, 1, index+1)
                _ = calc_lift(df, 'pred', 'target', ax=ax, groupnum=10, title_name=title)
                print(' {}: {} auc {:4f} ks {:4f}'.format(name, set_name, auc_score, ks))
                with file_writer.as_default():
                    tf.summary.scalar(name+'_ks', ks, step=index+1)
                    tf.summary.scalar(name+'_auc', auc_score, step=index+1)
            fig.savefig(joint_symbol.join([trend_dir, name]))