# %%
import platform
import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score, roc_curve

from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.metrics import AUC
from tensorflow.python.keras.models import load_model, save_model
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.metrics import Mean
from tensorflow import keras
import tensorflow as tf
import kerastuner as kt

from deepctr.layers import custom_objects
from deepctr.layers.utils import NoMask
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models.multitask.mmoe import MMOE, MMOELayer, MMOE_BIAS
from deepctr.models.multitask.call_backs import MyEarlyStopping, MyRecorder, ModifiedExponentialDecay
from deepctr.models.multitask.multitaskbase import MultiTaskModelBase
from deepctr.models.multitask.utils import calc_lift, cal_psi_score, calc_cum
import numpy as np
custom_objects['NoMask'] = NoMask
custom_objects['MMOELayer'] = MMOELayer
custom_objects['MultiTaskModelBase'] = MultiTaskModelBase
custom_objects['Mean'] = Mean
custom_objects['AUC'] = AUC


#
# def map_score(x):
#     if isinstance(x, float):
#         return np.nan
#     if x == 'DFLT_VAL' or x == '-2' or x == '库无记录':
#         return np.nan
#     if x == '0':
#         return '0'
#     temp = x.split(',')[0][1:]
#     if temp == '0':
#         return '1'
#     elif temp == '-999':
#         return '0'
#     else:
#         return temp
#
# data = pd.read_csv('test.csv')
# backup = data.copy()
# # data = pd.DataFrame(data=data, index = [0])
# col_x = ['tz_m12_platform_infos_max_all_overdue_repay_plat_cnt_2', 'cs_hc_phone_score',
#          'upa_max_consume_amt_6m', 'ab_local_ratio', 'ab_mobile_cnt', 'td_i_length_first_all_consumerfinance_365d',
#          'yysc_mobile_in_net_period', 'cs_hnsk_xef', 'duotou_br_als_m3_id_pdl_allnum', 'operation_sys',
#          'credit_repayment_score_bj_2',
#          'tz_evesums_m24_verif_sum', 'hds_mobile_reli_rank', 'selffill_is_have_creditcard', 'bwjk_xyf',
#          'duotou_bes_m1_overdue_money',
#          'credit_score_ronghuixf', 'duotou_br_als_m12_id_pdl_allnum', 'td_zhixin_score',
#          'duotou_br_als_m12_id_caon_allnum', 'ab_prov_cnt',
#          'dxm_dt_score', 'td_3m_idcard_lending_cnt', 'td_xyf_dq_i_ratio_cnt_grp_max_partner_loan_all',
#          'immediate_relation_cnt', 'bj_jc_m36_consume_cnt',
#          'duotou_br_als_m3_id_nbank_allnum', 'study_app_cnt', 'selffill_marital_status', 'tx_m6_cell_allnum',
#          'cs_mf_score_dt', 'cust_work_city',
#          'ali_rain_score', 'selffill_degree', 'pre_loan_flag', 'cust_gender', 'upa_failed_deal_cnt_6m',
#          'td_i_cnt_partner_all_imbank_365d', 'td_xyf_dq_score',
#          'duotou_br_alf_apirisk_all_sum', 'hds_36m_month_max_purchase_money_excp_doub11_12', 'credit_score_sh',
#          'wy_credit_score_credit_apply',
#          'duotou_br_als_m12_cell_nbank_allnum', 'tx_m12_id_platnum', 'umeng_score', 'relation_contact_cnt',
#          'br_frg_list_level',
#          'mg_callb_contacts_number_statistic_cnt_to_applied', 'area_risk_level', 'duotou_bes_m3_repay_times',
#          'ab_local_cnt']
#
# data[col_x] = data[col_x].replace([-99, -1, np.nan, '-1', '-99', '-1111', '-999', -999], np.nan)
#
# data['wy_credit_score_credit_apply'] = data['wy_credit_score_credit_apply'].astype(float)
# data['upa_failed_deal_cnt_6m'] = data['upa_failed_deal_cnt_6m'].astype(float)
# data['upa_max_consume_amt_6m'] = data['upa_max_consume_amt_6m'].astype(float)
# data['cust_gender'] = data['cust_gender'].map({'男': 0.0, '女': 1.0}).astype(float)
# data['umeng_score'] = data['umeng_score'].replace([-98], np.nan).astype(float)
# data['td_xyf_dq_i_ratio_cnt_grp_max_partner_loan_all'] = data['td_xyf_dq_i_ratio_cnt_grp_max_partner_loan_all'].map(
#     map_score).astype(float)
# data['yysc_mobile_in_net_period'] = data['yysc_mobile_in_net_period'].map(map_score).astype(float)
# data['selffill_is_have_creditcard'] = data['selffill_is_have_creditcard'].map({'N': 0.0, 'Y': 1.0, '0': np.nan}).astype(float)
# data['operation_sys'] = data['operation_sys'].replace(
#     {'IOS': 'ios', 'iPhone OS': 'ios', 'iOS': 'ios', 'Android': 'android'})
# data['operation_sys'] = data['operation_sys'].map({'ios': 0.0, 'android': 1.0}).astype(float)
# data['ab_prov_cnt'] = data['ab_prov_cnt'].astype(float)
# data['hds_mobile_reli_rank'] = data['hds_mobile_reli_rank'].map({'M0': np.nan, 'Ma': 0.0, 'Mb': 1.0}).astype(float)
# data['td_i_cnt_partner_all_imbank_365d'] = data['td_i_cnt_partner_all_imbank_365d'].map(map_score).astype(float)
# data['td_i_length_first_all_consumerfinance_365d'] = data['td_i_length_first_all_consumerfinance_365d'].map(
#     map_score).astype(float)
# data['tx_m6_cell_allnum'] = data['tx_m6_cell_allnum'].astype(float)
#
#
# quantile_transformer = QuantileTransformer(random_state=0)
# quantile_transformer.fit(data[(data['set'] == '1train') | (data['set'] == '2test')][col_x])
# data[col_x] = quantile_transformer.transform(data[col_x])
# data[col_x] = data[col_x].fillna(-1)
# best_metric = -1
# best_model = None
# project_name = 'preloan_istrans_overdue2'
# run_name = 'uncertainty_weight_fpd4_mask_mob3_k11_mask6'
# mode = 'test'
# if platform.system() == 'Windows':
#     joint_symbol = '\\'
# else:
#     joint_symbol = '/'
# checkpoint_dir = joint_symbol.join([project_name, 'ckt', run_name])
# tensorboard_dir = joint_symbol.join([project_name, 'log_dir', run_name])
# summary_dir = joint_symbol.join([project_name, 'metrics', run_name])
# trend_dir = joint_symbol.join([project_name, 'trend', run_name])
# for i in os.listdir(checkpoint_dir):
#     if i.find('best_model') >= 0:
#         metric = float(re.match('.*AUC(.*).h5', i)[1])
#         if metric > best_metric:
#             best_metric = metric
#             best_model = i
# print('loading ', joint_symbol.join([checkpoint_dir, best_model]))
# model = load_model(joint_symbol.join([checkpoint_dir, best_model]), custom_objects=custom_objects)
# n_bins=20
# weight_dict = {}
# for round_num, ft in enumerate(col_x):
#     weight_dict[ft] = np.array([list(range(1, n_bins + 2))] * data.shape[0])
#     bin_ary = np.array([i / n_bins for i in range(1, n_bins + 1)])
#     bin_ary = np.append(bin_ary, -1)
#     cent_hat = np.abs(np.expand_dims(np.expand_dims(data[ft], -1), -1) - np.expand_dims(bin_ary, -1))
#     weight_dict[ft + '_weight'] = 1.0 / (cent_hat + 1e-7)
# predictions = model.predict(weight_dict)
# print(backup.shape)
# print(predictions['fpd4'].shape)
# backup['fpd4_pred2'] = predictions['fpd4'].flatten()
# backup['istrans_pred2'] = predictions['istrans'].flatten()
# backup['mob3_k11_pred2'] = predictions['mob3_k11'].flatten()
#
# data = backup
# data[col_x] = data[col_x].replace([-99, -1, np.nan, '-1', '-99', '-1111', '-999', -999], np.nan)
#
# data['wy_credit_score_credit_apply'] = data['wy_credit_score_credit_apply'].astype(float)
# data['upa_failed_deal_cnt_6m'] = data['upa_failed_deal_cnt_6m'].astype(float)
# data['upa_max_consume_amt_6m'] = data['upa_max_consume_amt_6m'].astype(float)
# data['cust_gender'] = data['cust_gender'].map({'男': 0.0, '女': 1.0}).astype(float)
# data['umeng_score'] = data['umeng_score'].replace([-98], np.nan).astype(float)
# data['td_xyf_dq_i_ratio_cnt_grp_max_partner_loan_all'] = data['td_xyf_dq_i_ratio_cnt_grp_max_partner_loan_all'].map(
#     map_score).astype(float)
# data['yysc_mobile_in_net_period'] = data['yysc_mobile_in_net_period'].map(map_score).astype(float)
# data['selffill_is_have_creditcard'] = data['selffill_is_have_creditcard'].map({'N': 0.0, 'Y': 1.0, '0': np.nan}).astype(float)
# data['operation_sys'] = data['operation_sys'].replace(
#     {'IOS': 'ios', 'iPhone OS': 'ios', 'iOS': 'ios', 'Android': 'android'})
# data['operation_sys'] = data['operation_sys'].map({'ios': 0.0, 'android': 1.0}).astype(float)
# data['ab_prov_cnt'] = data['ab_prov_cnt'].astype(float)
# data['hds_mobile_reli_rank'] = data['hds_mobile_reli_rank'].map({'M0': np.nan, 'Ma': 0.0, 'Mb': 1.0}).astype(float)
# data['td_i_cnt_partner_all_imbank_365d'] = data['td_i_cnt_partner_all_imbank_365d'].map(map_score).astype(float)
# data['td_i_length_first_all_consumerfinance_365d'] = data['td_i_length_first_all_consumerfinance_365d'].map(
#     map_score).astype(float)
# data['tx_m6_cell_allnum'] = data['tx_m6_cell_allnum'].astype(float)
# with open('fill_na_dict.json', 'r') as f:
#     fill_na_dict = json.load(f)
# data[col_x] = data[col_x].fillna(fill_na_dict)
# import pickle
# with open('mms4.pkl', 'rb') as f:
#     mms = pickle.load(f)
# data[col_x] = mms.transform(data[col_x])
#
# model_input_dict = {name: data[name] for name in col_x}
# model = load_model('model_demo4')
# predictions = model.predict(model_input_dict)
# backup['fpd4_pred'] = predictions['fpd4'].flatten()
# backup['istrans_pred'] = predictions['istrans'].flatten()
# backup['mob3_k11_pred'] = predictions['mob3_k11'].flatten()
#
# backup.to_csv('evaluation_multitask2.csv', index=None)
#
#

def map_score(x):
    if isinstance(x, float):
        return np.nan
    if x == 'DFLT_VAL' or x == '-2' or x =='库无记录':
        return np.nan
    if x == '0':
        return '0'
    temp = x.split(',')[0][1:]
    if temp == '0':
        return '1'
    elif temp == '-999':
        return '0'
    else:
        return temp


if __name__ == "__main__":
    # configure
    project_name = 'preloan_istrans_overdue2'
    run_name = 'uncertainty_weight_fpd4_mask_mob3_k11_mask7'
    mode = 'test'
    if platform.system() == 'Windows':
        joint_symbol = '\\'
    else:
        joint_symbol = '/'
    checkpoint_dir = joint_symbol.join([project_name, 'ckt', run_name])
    tensorboard_dir = joint_symbol.join([project_name, 'log_dir', run_name])
    summary_dir = joint_symbol.join([project_name, 'metrics', run_name])
    trend_dir = joint_symbol.join([project_name, 'trend', run_name])
    if not os.path.exists(trend_dir):
        os.makedirs(trend_dir)
    tasks = {'istrans': 'binary', 'fpd4': 'binary', 'mob3_k11': 'binary'}
    loss_fns = {'istrans': keras.losses.binary_crossentropy,
                'fpd4': keras.losses.binary_crossentropy,
                'mob3_k11': keras.losses.binary_crossentropy}
    # loss_fns = {'istrans': keras.losses.BinaryCrossentropy(),
    #             'fpd4': keras.losses.BinaryCrossentropy()}
    metrics_logger = {'istrans': AUC,
                      'fpd4': AUC,
                      'mob3_k11': AUC}
    loss_weights = {'istrans': 1, 'fpd4': 6, 'mob3_k11': 6}
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
    data = pd.read_csv('data/test2.csv')
    col_x = ['tz_m12_platform_infos_max_all_overdue_repay_plat_cnt_2',
             'cs_hc_phone_score',
             'upa_max_consume_amt_6m',
             'ab_local_ratio',
             'ab_mobile_cnt',
             'td_i_length_first_all_consumerfinance_365d',
             'yysc_mobile_in_net_period',
             'cs_hnsk_xef',
             'duotou_br_als_m3_id_pdl_allnum',
             'operation_sys',
             'credit_repayment_score_bj_2',
             'tz_evesums_m24_verif_sum',
             'hds_mobile_reli_rank',
             'selffill_is_have_creditcard',
             'bwjk_xyf',
             'duotou_bes_m1_overdue_money',
             'credit_score_ronghuixf',
             'duotou_br_als_m12_id_pdl_allnum',
             'td_zhixin_score',
             'duotou_br_als_m12_id_caon_allnum',
             'ab_prov_cnt',
             'dxm_dt_score',
             'td_3m_idcard_lending_cnt',
             'td_xyf_dq_i_ratio_cnt_grp_max_partner_loan_all',
             'immediate_relation_cnt',
             'bj_jc_m36_consume_cnt',
             'duotou_br_als_m3_id_nbank_allnum',
             'study_app_cnt',
             'selffill_marital_status',
             'tx_m6_cell_allnum',
             'cs_mf_score_dt',
             'cust_work_city',
             'ali_rain_score',
             'selffill_degree',
             'pre_loan_flag',
             'cust_gender',
             'upa_failed_deal_cnt_6m',
             'td_i_cnt_partner_all_imbank_365d',
             'td_xyf_dq_score',
             'duotou_br_alf_apirisk_all_sum',
             'hds_36m_month_max_purchase_money_excp_doub11_12',
             'credit_score_sh',
             'wy_credit_score_credit_apply',
             'duotou_br_als_m12_cell_nbank_allnum',
             'tx_m12_id_platnum',
             'umeng_score',
             'relation_contact_cnt',
             'br_frg_list_level',
             'mg_callb_contacts_number_statistic_cnt_to_applied',
             'area_risk_level',
             'duotou_bes_m3_repay_times',
             'ab_local_cnt']
    bias_features = {'istrans': ['pre_loan_flag']}
    bias_dropout_dict = {'istrans': 0.1, 'fpd4': 0.1}

    data['fpd4_weight'] = 1.0
    data['fpd4_mask'] = 1.0
    if run_name.find('fpd4_nomask') >= 0:
        data.loc[(data['if_t4'] == 0), 'fpd4_weight'] = 1.0
    else:
        data.loc[(data['if_t4'] == 0), 'fpd4_weight'] = 0.0
    data.loc[(data['if_t4'] == 0), 'fpd4_mask'] = 0

    data['mob3_k11_weight'] = 1.0
    data['mob3_k11_mask'] = 1.0
    if run_name.find('mob3_k11_nomask') >= 0:
        data.loc[(data['if_mob3_t11'] == 0), 'mob3_k11_weight'] = 1.0
    else:
        data.loc[(data['if_mob3_t11'] == 0), 'mob3_k11_weight'] = 0.0
    data.loc[(data['if_mob3_t11'] == 0), 'mob3_k11_mask'] = 0

    data['istrans_weight'] = 1.0
    data['istrans_mask'] = 1.0
    data.loc[data['pre_loan_flag'] == 1, 'istrans_mask'] = 0.0
    if run_name.find('istrans_mask') >= 0:
        data.loc[data['pre_loan_flag'] == 1, 'istrans_weight'] = 0.0

    data['fpd4'] = data['fpd4'].fillna(0)
    data.loc[data['fpd4'] == -1, 'fpd4'] = 0
    data['mob3_k11'] = data['mob3_k11'].fillna(0)
    data.loc[data['mob3_k11'] == -1, 'mob3_k11'] = 0

    data[col_x] = data[col_x].replace([-99, -1, np.nan, '-1', '-99', '-1111', '-999', -999], np.nan)
    data['wy_credit_score_credit_apply'] = data['wy_credit_score_credit_apply'].astype(float)
    data['upa_failed_deal_cnt_6m'] = data['upa_failed_deal_cnt_6m'].astype(float)
    data['upa_max_consume_amt_6m'] = data['upa_max_consume_amt_6m'].astype(float)
    data['cust_gender'] = data['cust_gender'].map({'男': 0.0, '女': 1.0})
    data['umeng_score'] = data['umeng_score'].replace([-98], np.nan)
    data['td_xyf_dq_i_ratio_cnt_grp_max_partner_loan_all'] = data['td_xyf_dq_i_ratio_cnt_grp_max_partner_loan_all'].map(
        map_score).astype(float)
    data['yysc_mobile_in_net_period'] = data['yysc_mobile_in_net_period'].map(map_score).astype(float)
    data['selffill_is_have_creditcard'] = data['selffill_is_have_creditcard'].map({'N': 0.0, 'Y': 1.0, '0': np.nan})
    data['operation_sys'] = data['operation_sys'].replace(
        {'IOS': 'ios', 'iPhone OS': 'ios', 'iOS': 'ios', 'Android': 'android'})
    data['operation_sys'] = data['operation_sys'].map({'ios': 0.0, 'android': 1.0})
    data['ab_prov_cnt'] = data['ab_prov_cnt'].astype(float)
    data['hds_mobile_reli_rank'] = data['hds_mobile_reli_rank'].map({'M0': np.nan, 'Ma': 0.0, 'Mb': 1.0})
    data['td_i_cnt_partner_all_imbank_365d'] = data['td_i_cnt_partner_all_imbank_365d'].map(map_score).astype(float)
    data['td_i_length_first_all_consumerfinance_365d'] = data['td_i_length_first_all_consumerfinance_365d'].map(
        map_score).astype(float)
    data['tx_m6_cell_allnum'] = data['tx_m6_cell_allnum'].astype(float)

    # quantile_transformer = QuantileTransformer(random_state=0)
    # quantile_transformer.fit(data[(data['set'] == '1train') | (data['set'] == '2test')][col_x])
    import pickle
    with open('quantile_transformer.pkl', 'rb') as f:
        quantile_transformer = pickle.load(f)
    data[col_x] = quantile_transformer.transform(data[col_x])
    data[col_x] = data[col_x].fillna(-1)

    n_bins = 20
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
    for task_name in ['mob3_k11']:
        for index, set_name in enumerate(['1train', '2test', '3oot']):
            set_data = data[(data['set'] == set_name)]
            print(set_data[set_data[task_name + '_mask'] == 1].shape)
            weight_dict = {}
            for round_num, ft in enumerate(col_x):
                weight_dict[ft] = np.array([list(range(1, n_bins+1))] * set_data.shape[0])
                bin_ary = np.array([i / n_bins for i in range(1, n_bins)])
                bin_ary = np.append(bin_ary, -1)
                cent_hat = np.abs(np.expand_dims(np.expand_dims(set_data[ft], -1), -1) - np.expand_dims(bin_ary, -1))
                weight_dict[ft + '_weight'] = 1.0 / (cent_hat + 1e-7)
            predictions = model.predict(weight_dict)
            auc_score = roc_auc_score(set_data[task_name].values, predictions[task_name][:, 0],
                                      sample_weight=set_data[task_name + '_mask'].values)
            fpr, tpr, _ = roc_curve(set_data[task_name].values, predictions[task_name][:, 0],
                                    sample_weight=set_data[task_name + '_mask'].values)
            ks = np.max(np.abs(tpr - fpr))
            print(' {}: {} auc {:4f} ks {:4f}'.format(task_name, set_name, auc_score, ks))
