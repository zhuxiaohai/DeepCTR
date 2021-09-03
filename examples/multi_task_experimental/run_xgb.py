#%%
import platform
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb

from tensorflow.python.keras.metrics import AUC
from tensorflow.keras.metrics import Mean
import tensorflow as tf

from deepctr.layers import custom_objects
from deepctr.layers.utils import NoMask
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models.multitask.multitaskbase import MultiTaskModelBase
from deepctr.models.multitask.utils import calc_lift, cal_psi_score, calc_cum
from deepctr.models.multitask.grid_search_xgb import OptunaSearchXGB

custom_objects['NoMask'] = NoMask
custom_objects['MultiTaskModelBase'] = MultiTaskModelBase
custom_objects['Mean'] = Mean
custom_objects['AUC'] = AUC


if __name__ == "__main__":
    # configure
    project_name = 'preloan_istrans_overdue2'
    single_name = 'istrans'
    run_name = 'xgb2_expertfeatures_{}_mask'.format(single_name)
    mode = 'train'
    if platform.system() == 'Windows':
        joint_symbol = '\\'
    else:
        joint_symbol = '/'
    checkpoint_dir = joint_symbol.join([project_name, 'ckt', run_name])
    tensorboard_dir = joint_symbol.join([project_name, 'log_dir', run_name])
    summary_dir = joint_symbol.join([project_name, 'metrics', run_name])
    trend_dir = joint_symbol.join([project_name, 'trend', run_name])
    tasks = {single_name: 'binary'}
    if run_name.find('uncertainty') >= 0:
        uncertainty = True
    else:
        uncertainty = False
    if run_name.find('gradnorm') >= 0:
        gradnorm = True
    else:
        gradnorm = False
    if run_name.find('bias') >= 0:
        add_bias = True
    else:
        add_bias = False
    batch_size = 256

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
    bias_features = {'istrans': ['pre_loan_flag']}
    data['fpd4_weight'] = 1.0
    data['fpd4_mask'] = 1.0
    if run_name.find('fpd4_nomask') >= 0:
        data.loc[data['fpd4'].isnull(), 'fpd4_weight'] = 1.0
    else:
        data.loc[data['fpd4'].isnull(), 'fpd4_weight'] = 0.0
    data.loc[data['fpd4'].isnull(), 'fpd4_mask'] = 0
    data['istrans_weight'] = 1.0
    data['istrans_mask'] = 1.0
    data.loc[data['pre_loan_flag'] == 1, 'istrans_mask'] = 0.0
    if run_name.find('istrans_mask') >= 0:
        data.loc[data['pre_loan_flag'] == 1, 'istrans_weight'] = 0.0
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

    if add_bias:
        bias_feature_columns_list = []
        for task_name, columns in bias_features.items():
            bias_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].unique().shape[0], embedding_dim=8)
                                    for i, feat in enumerate(columns)]
            bias_feature_columns_list += bias_feature_columns
        feature_names = get_feature_names(dnn_feature_columns + bias_feature_columns_list)
        bias_feature_names = get_feature_names(bias_feature_columns_list)
        for feat in bias_feature_names:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
    else:
        bias_feature_names = []
        feature_names = get_feature_names(dnn_feature_columns)

    # generate input data for model
    train = data[data['set'] == '1train']
    test = data[data['set'] == '2test']

    # train or predict
    if mode == 'train':
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        op = OptunaSearchXGB()
        tuning_param_dict = {'objective': 'binary:logistic',
                             'verbosity': 0,
                             'seed': 2,
                             'num_parallel_tree': ('int', {'low': 1, 'high': 4}),
                             'max_depth': ('int', {'low': 2, 'high': 6}),
                             'reg_lambda': ('int', {'low': 1, 'high': 20}),
                             'reg_alpha': ('int', {'low': 1, 'high': 20}),
                             'gamma': ('int', {'low': 0, 'high': 3}),
                             'min_child_weight': ('int', {'low': 1, 'high': 30}),
                             'base_score': ('discrete_uniform', {'low': 0.5, 'high': 0.9, 'q': 0.1}),
                             'colsample_bytree': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                             'colsample_bylevel': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                             'colsample_bynode': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                             'subsample': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                             'eta': ('discrete_uniform', {'low': 0.07, 'high': 1.2, 'q': 0.01}),
                             'rate_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                             'skip_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                             'tree_method': ('categorical', {'choices': ['auto', 'exact', 'approx', 'hist']}),
                             'booster': ('categorical', {'choices': ['gbtree', 'dart']}),
                             'sample_type': ('categorical', {'choices': ['uniform', 'weighted']}),
                             'normalize_type': ('categorical', {'choices': ['tree', 'forest']})}
        op.search(train[train[single_name + '_weight'] == 1][feature_names],
                  train[train[single_name + '_weight'] == 1][single_name],
                  tuning_param_dict, cv=1, coef_train_val_disparity=0.4, eval_metric=['roc_auc'],
                  eval_set=[(test[test[single_name + '_weight'] == 1][feature_names],
                             test[test[single_name + '_weight'] == 1][single_name])],
                  optuna_verbosity=1, early_stopping_rounds=30, n_warmup_steps=10)
        train_param = op.get_params()
        print(train_param)
        train_dmatrix = xgb.DMatrix(train[train[single_name + '_weight'] == 1][feature_names],
                                    train[train[single_name + '_weight'] == 1][single_name])
        model = xgb.train(train_param, train_dmatrix, num_boost_round=train_param['n_iterations'])
        model.save_model(joint_symbol.join([checkpoint_dir, 'xgb']))
    else:
        if not os.path.exists(trend_dir):
            os.makedirs(trend_dir)
        best_metric = -1
        best_model = None
        for i in os.listdir(checkpoint_dir):
            if i.find('xgb') >= 0:
                model = xgb.Booster()
                print('loading ', joint_symbol.join([checkpoint_dir, i]))
                model.load_model(joint_symbol.join([checkpoint_dir, i]))
                break
        file_writer = tf.summary.create_file_writer(summary_dir)
        print('final_result')
        for task_name in tasks.keys():
            fig = plt.figure(figsize=(8, 10))
            fig.suptitle(run_name + '_' + task_name)
            for index, set_name in enumerate(['1train', '2test', '3oot']):
                set_data = data[data['set'] == set_name]
                for name in bias_feature_names:
                    set_data[name] = [0] * set_data.shape[0]
                dmatrix = xgb.DMatrix(set_data[feature_names], set_data[task_name])
                predictions = model.predict(dmatrix)
                auc_score = roc_auc_score(set_data[task_name].values, predictions, sample_weight=set_data[task_name+'_mask'].values)
                fpr, tpr, _ = roc_curve(set_data[task_name].values, predictions, sample_weight=set_data[task_name+'_mask'].values)
                ks = np.max(np.abs(tpr - fpr))
                pred = predictions
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

        print('bias_study')
        for task_name in bias_features.keys():
            for bias_task in ['biased', 'all']:
                fig = plt.figure(figsize=(8, 10))
                fig.suptitle(run_name + '_' + task_name + '_' + bias_task)
                for index, set_name in enumerate(['1train', '2test', '3oot']):
                    set_data = data[data['set'] == set_name]
                    if bias_task == 'biased':
                        masks = -set_data[task_name+'_mask'].values
                        masks[masks == 0.0] = 1.0
                        masks[masks == -1.0] = 0.0
                    else:
                        masks = set_data[task_name+'_mask'].values
                        masks[masks != 1.0] = 1.0
                    for name in bias_feature_names:
                        set_data[name] = [0]*set_data.shape[0]
                    dmatrix = xgb.DMatrix(set_data[feature_names], set_data[task_name])
                    predictions = model.predict(dmatrix)
                    auc_score = roc_auc_score(set_data[task_name].values, predictions, sample_weight=masks)
                    fpr, tpr, _ = roc_curve(set_data[task_name].values, predictions, sample_weight=masks)
                    ks = np.max(np.abs(tpr - fpr))
                    pred = predictions
                    target = set_data[task_name].values
                    weight = masks
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
                    print(' {}_{}: {} auc {:4f} ks {:4f}'.format(task_name, bias_task, set_name, auc_score, ks))
                    with file_writer.as_default():
                        tf.summary.scalar(task_name+'_'+bias_task+'_ks', ks, step=index+1)
                        tf.summary.scalar(task_name+'_'+bias_task+'_auc', auc_score, step=index+1)
                    if (bias_task == 'biased') & (set_name == '3oot'):
                        pred = predictions
                        weight = masks
                        fig2, ax2 = plt.subplots()
                        _, _, _ = ax2.hist(pred[weight == 1.0], bins=50, density=True, alpha=1)
                        _, _, _ = ax2.hist(pred[weight == 0.0], bins=50, density=True, alpha=0.2)
                        title = 'biased_1: {:.2f} biased_0: {:.2f} gap: {:.2f}'.format(
                            pred[weight == 1.0].mean(),
                            pred[weight == 0.0].mean(),
                            abs(pred[weight == 1.0].mean() - pred[weight == 0.0].mean()))
                        ax2.legend(['biased_1', 'biased_0'])
                        fig2.suptitle(run_name + '_' + task_name + '_' + set_name + '\n' + title)
                        fig2.savefig(joint_symbol.join([trend_dir, task_name+'_'+set_name+'_hist']))
                fig.savefig(joint_symbol.join([trend_dir, task_name+'_'+bias_task]))

        print('cross_validation')
        set_data = data[data['set'] == '3oot']
        for name in bias_feature_names:
            set_data[name] = [0] * set_data.shape[0]
        dmatrix = xgb.DMatrix(set_data[feature_names])
        predictions = model.predict(dmatrix)
        fig_cross = plt.figure(figsize=(8, 10))
        fig_cross.suptitle(run_name + '_cross')
        ax_cross = fig_cross.add_subplot(2, 1, 1)
        if single_name == 'fpd4':
            df = pd.DataFrame({'pred': predictions, 'target': set_data['istrans'].values})
            _ = calc_cum(df, 'pred', 'target', ax=ax_cross, groupnum=10, title_name='istrans')
        elif single_name == 'istrans':
            pred = predictions
            target = set_data['fpd4'].values
            weight = set_data['fpd4_mask'].values
            pred = pred[weight != 0]
            target = target[weight != 0]
            df = pd.DataFrame({'pred': pred, 'target': target})
            _ = calc_cum(df, 'pred', 'target', ax=ax_cross, groupnum=10, title_name='fpd4')
        fig_cross.savefig(joint_symbol.join([trend_dir, 'cross']))