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
import kerastuner as kt

from deepctr.layers import custom_objects
from deepctr.layers.utils import NoMask
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models.multitask_modified.single_task import SimpleDNN
from deepctr.callbacks import MyEarlyStopping
from deepctr.models.multitask_modified.multitaskbase import MultiTaskModelBase
from deepctr.metrics import calc_lift, cal_psi_score, calc_cum

custom_objects['NoMask'] = NoMask
custom_objects['MultiTaskModelBase'] = MultiTaskModelBase
custom_objects['Mean'] = Mean
custom_objects['AUC'] = AUC


def build_model(hp):
    num_layers = 2
    dnn_hidden_units = []
    for i in range(num_layers):
        dnn_hidden_units.append(hp.Int('dnn_hidden_unit_' + str(i), 32, 128, step=32))
    dnn_use_bn = hp.Boolean('dnn_use_bn', default=False)
    l2_reg_embedding = hp.Float('l2_reg_embedding', 1e-5, 0.5, sampling='log')
    l2_reg_dnn = hp.Float('l2_reg_dnn', 1e-5, 0.5, sampling='log')
    dnn_dropout = hp.Float('dnn_dropout', 0.0, 0.5)
    model = SimpleDNN(dnn_feature_columns,
                      tasks,
                      dnn_hidden_units=dnn_hidden_units,
                      dnn_use_bn=dnn_use_bn,
                      l2_reg_embedding=l2_reg_embedding,
                      l2_reg_dnn=l2_reg_dnn,
                      dnn_dropout=dnn_dropout
                      )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', 1e-4, 0.01, sampling='log')),
                  loss=loss_fns,
                  metrics=metrics_logger,
                  uncertainly=uncertainty,
                  gradnorm_config=gradnorm_config)
    return model


if __name__ == "__main__":
    # configure
    project_name = 'preloan_istrans_overdue2'
    single_name = 'istrans'
    run_name = 'single2_expertfeatures_{}'.format(single_name)
    if platform.system() == 'Windows':
        joint_symbol = '\\'
    else:
        joint_symbol = '/'
    checkpoint_dir = joint_symbol.join([project_name, 'ckt', run_name])
    tensorboard_dir = joint_symbol.join([project_name, 'log_dir', run_name])
    summary_dir = joint_symbol.join([project_name, 'metrics', run_name])
    trend_dir = joint_symbol.join([project_name, 'trend', run_name])
    tasks = {single_name: 'binary'}
    loss_fns = {single_name: keras.losses.binary_crossentropy}
    metrics_logger = {single_name: AUC(name=single_name + '_AUC')}
    uncertainty = False
    gradnorm = False
    gradnorm_config = None
    batch_size = 256
    mode = 'train'

    # read data
    data = pd.read_csv('../data/train_for_multi2.csv')
    # fpd4
    # col_x = ['ali_rain_score',
    #          'td_zhixin_score',
    #          'ab_local_cnt',
    #          'ab_mobile_cnt',
    #          'cust_id_area',
    #          'cust_gender',
    #          'immediate_relation_cnt',
    #          'hds_36m_month_max_purchase_money_excp_doub11_12',
    #          'ab_local_ratio',
    #          'hds_36m_purchase_steady',
    #          'cust_work_city',
    #          'relation_contact_cnt',
    #          'td_xyf_dq_score',
    #          'selffill_is_have_creditcard',
    #          'credit_repayment_score_bj_2',
    #          'bj_jc_m36_consume_cnt',
    #          'selffill_degree',
    #          'selffill_marital_status',
    #          'operation_sys',
    #          'hds_mobile_rich',
    #          'ab_prov_cnt',
    #          'hds_36m_total_purchase_cnt',
    #          'idcard_district_grade',
    #          'study_app_cnt',
    #          'hds_recent_consumme_active_rank',
    #          'hds_mobile_reli_rank',
    #          'hds_36m_doub11_12_total_purchase_money',
    #          'idcard_rural_flag']
    # sparse_features = [  'cust_id_area',
    #                      'cust_gender',
    #                      'cust_work_city',
    #                      'selffill_is_have_creditcard',
    #                      'selffill_marital_status',
    #                      'operation_sys',
    #                      'hds_mobile_reli_rank',
    #                      'idcard_rural_flag']
    # istrans
    sparse_features = ['marketing_channel_pred_1',
                       'model_key_pred_1',
                       'regist_channel_pred_1']
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
             'dxm_qzf']
    # 全量
    # sparse_features = ['marketing_channel_pred_1',
    #                      'model_key_pred_1',
    #                      'regist_channel_pred_1',
    #                      'cust_id_area',
    #                      'cust_gender',
    #                      'cust_work_city',
    #                      'selffill_is_have_creditcard',
    #                      'selffill_marital_status',
    #                      'operation_sys',
    #                      'hds_mobile_reli_rank',
    #                      'idcard_rural_flag']
    # col_x = ['td_i_cnt_partner_all_imbank_365d',
    #          'duotou_br_als_m3_id_pdl_allnum',
    #          'marketing_channel_pred_1',
    #          'td_i_length_first_all_consumerfinance_365d',
    #          'duotou_br_als_m12_cell_nbank_allnum',
    #          'tx_m12_id_platnum',
    #          'duotou_br_als_m12_id_caon_allnum',
    #          'duotou_br_alf_apirisk_all_sum',
    #          'model_key_pred_1',
    #          'duotou_bh_rl_creditlimitsum',
    #          'dxm_dt_score',
    #          'credit_score_ronghuixf',
    #          'regist_channel_pred_1',
    #          'cs_mf_score_dt',
    #          'credit_score_sh',
    #          'duotou_br_als_m3_id_nbank_allnum',
    #          'br_frg_list_level',
    #          'td_3m_idcard_lending_cnt',
    #          'duotou_br_alf_apirisk_all_mean',
    #          'cs_hc_phone_score',
    #          'tx_m6_cell_allnum',
    #          'duotou_br_als_m12_id_pdl_allnum',
    #          'dxm_qzf',
    #          'ali_rain_score',
    #          'td_zhixin_score',
    #          'ab_local_cnt',
    #          'ab_mobile_cnt',
    #          'cust_id_area',
    #          'cust_gender',
    #          'immediate_relation_cnt',
    #          'hds_36m_month_max_purchase_money_excp_doub11_12',
    #          'ab_local_ratio',
    #          'hds_36m_purchase_steady',
    #          'cust_work_city',
    #          'relation_contact_cnt',
    #          'td_xyf_dq_score',
    #          'selffill_is_have_creditcard',
    #          'credit_repayment_score_bj_2',
    #          'bj_jc_m36_consume_cnt',
    #          'selffill_degree',
    #          'selffill_marital_status',
    #          'operation_sys',
    #          'hds_mobile_rich',
    #          'ab_prov_cnt',
    #          'hds_36m_total_purchase_cnt',
    #          'idcard_district_grade',
    #          'study_app_cnt',
    #          'hds_recent_consumme_active_rank',
    #          'hds_mobile_reli_rank',
    #          'hds_36m_doub11_12_total_purchase_money',
    #          'idcard_rural_flag']
    data['fpd4_weight'] = 1.0
    data['fpd4_mask'] = 1.0
    if run_name.find('nomask') >= 0:
        data.loc[data['fpd4'].isnull(), 'fpd4_weight'] = 1.0
    else:
        data.loc[data['fpd4'].isnull(), 'fpd4_weight'] = 0.0
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
        # fdp4
        # model = SimpleDNN(dnn_feature_columns,
        #                   tasks,
        #                   dnn_hidden_units=(96, 96),
        #                   dnn_use_bn=True,
        #                   l2_reg_embedding=0.0005835949243783095,
        #                   l2_reg_dnn=1e-05,
        #                   dnn_dropout=0.5)
        # last_lr = 0.01
        # istrans
        model = SimpleDNN(dnn_feature_columns,
                          tasks,
                          dnn_hidden_units=(96, 96),
                          dnn_use_bn=True,
                          l2_reg_embedding=0.49999999999999994,
                          l2_reg_dnn=1e-05,
                          dnn_dropout=0.158437822002694)
        last_lr = 0.001
        optimizers = keras.optimizers.Adam(learning_rate=last_lr)
        model.compile(optimizer=optimizers,
                      loss=loss_fns,
                      metrics=metrics_logger,
                      uncertainly=uncertainty,
                      gradnorm_config=gradnorm_config)
        # plot_model(model, to_file=joint_symbol.join([checkpoint_dir, 'model_viz.png']), show_shapes=True,
        #            show_layer_names=True)
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
        print('last epoch', last_epoch)
        print('last lr', last_lr)
        history = model.fit(model_input,
                            {task_name: train[[task_name]] for task_name in tasks.keys()},
                            sample_weight={task_name: train[task_name + '_weight'] for task_name in tasks.keys()},
                            batch_size=256,
                            epochs=100,
                            initial_epoch=last_epoch,
                            verbose=2,
                            validation_data=(test_model_input,
                                             {task_name: test[[task_name]] for task_name in tasks.keys()},
                                             {task_name: test[task_name + '_weight'] for task_name in tasks.keys()}),
                            callbacks=[
                                      MyEarlyStopping('val_{}_AUC'.format(list(tasks.keys())[0]),
                                                      patience=10,
                                                      savepath=checkpoint_dir,
                                                      coef_of_balance=0.4,
                                                      direction='maximize'),
                                      TensorBoard(log_dir=tensorboard_dir)
                            ]
                            )
    elif mode == 'tuning':
        tuner = kt.BayesianOptimization(
            build_model,
            objective=kt.Objective('val_{}_AUC'.format(list(tasks.keys())[0]), direction="max"),
            max_trials=100,
            num_initial_points=10)
        tuner.search(x=model_input,
                     y={task_name: train[[task_name]] for task_name in tasks.keys()},
                     sample_weight={task_name: train[task_name + '_weight'] for task_name in tasks.keys()},
                     batch_size=batch_size,
                     verbose=2,
                     epochs=100,
                     validation_data=(test_model_input,
                                      {task_name: test[[task_name]] for task_name in tasks.keys()},
                                      {task_name: test[task_name + '_weight'] for task_name in tasks.keys()}),
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_{}_AUC'.format(list(tasks.keys())[0]),
                                                              patience=10)]
                     )
        # even if you had not searched the best param, tuner object could find the best hyperparameters
        # as well as the best model from the tuning_dir you passed in
        best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
        print(best_hyperparameters.values)
        # model = tuner.get_best_models(1)[0]
    else:
        if not os.path.exists(trend_dir):
            os.makedirs(trend_dir)
        best_metric = -1
        best_model = None
        for i in os.listdir(checkpoint_dir):
            if i.find('best_model') >= 0:
                metric = float(re.match('.*AUC(.*).h5', i)[1])
                if metric > best_metric:
                    best_metric = metric
                    best_model = i
        print('loading ', joint_symbol.join([checkpoint_dir, best_model]))
        model = load_model(joint_symbol.join([checkpoint_dir, best_model]), custom_objects=custom_objects)
        file_writer = tf.summary.create_file_writer(summary_dir)
        print('final_result')
        for task_name in tasks.keys():
            fig = plt.figure(figsize=(8, 10))
            fig.suptitle(run_name + '_' + task_name)
            for index, set_name in enumerate(['1train', '2test', '3oot']):
                set_data = data[data['set'] == set_name]
                predictions = model.predict({name: set_data[name] for name in feature_names})
                auc_score = roc_auc_score(set_data[task_name].values, predictions[task_name][:, 0], sample_weight=set_data[task_name+'_mask'])
                fpr, tpr, _ = roc_curve(set_data[task_name].values, predictions[task_name][:, 0], sample_weight=set_data[task_name+'_mask'])
                ks = np.max(np.abs(tpr - fpr))
                pred = predictions[task_name][:, 0]
                target = set_data[task_name].values
                weight = set_data[task_name+'_mask'].values
                pred = pred[weight != 0]
                target = target[weight != 0]
                if set_name == '1train':
                    expected = pred
                    title = '{} ks_{:.2f} auc_{:.2f}'.format(set_name, ks, auc_score)
                elif set_name == '2test':
                    psi = cal_psi_score(pred, expected)
                    title = '{} ks_{:.2f} auc_{:.2f} psi_{:.2f}'.format(set_name, ks, auc_score, psi)
                else:
                    title = '{} ks_{:.2f} auc_{:.2f}'.format(set_name, ks, auc_score)
                df = pd.DataFrame({'pred': pred, 'target': target})
                ax = fig.add_subplot(3, 1, index+1)
                _ = calc_lift(df, 'pred', 'target', ax=ax, groupnum=10, title_name=title)
                print(' {}: {} auc {:4f} ks {:4f}'.format(task_name, set_name, auc_score, ks))
                with file_writer.as_default():
                    tf.summary.scalar(task_name+'_ks', ks, step=index+1)
                    tf.summary.scalar(task_name+'_auc', auc_score, step=index+1)
            fig.savefig(joint_symbol.join([trend_dir, task_name]))

        print('cross_validation')
        set_data = data[data['set'] == '3oot']
        predictions = model.predict({name: set_data[name] for name in feature_names})
        fig_cross = plt.figure(figsize=(8, 10))
        fig_cross.suptitle(run_name + '_cross')
        ax_cross = fig_cross.add_subplot(2, 1, 1)
        if single_name == 'fpd4':
            df = pd.DataFrame({'pred': predictions['fpd4'][:, 0], 'target': set_data['istrans'].values})
            _ = calc_cum(df, 'pred', 'target', ax=ax_cross, groupnum=10, title_name='istrans')
        elif single_name == 'istrans':
            pred = predictions['istrans'][:, 0]
            target = set_data['fpd4'].values
            weight = set_data['fpd4_mask'].values
            pred = pred[weight != 0]
            target = target[weight != 0]
            df = pd.DataFrame({'pred': pred, 'target': target})
            _ = calc_cum(df, 'pred', 'target', ax=ax_cross, groupnum=10, title_name='fpd4')
        fig_cross.savefig(joint_symbol.join([trend_dir, 'cross']))