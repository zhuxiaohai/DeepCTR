#%%
import platform
import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score, roc_curve

from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.metrics import AUC
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.metrics import Mean
from tensorflow import keras
import tensorflow as tf
import kerastuner as kt

from deepctr.layers import custom_objects
from deepctr.layers.utils import NoMask
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from deepctr.models.multitask.mmoe import MMOE, MMOELayer, MMOE_BIAS
from deepctr.models.multitask.call_backs import MyEarlyStopping, MyRecorder, ModifiedExponentialDecay
from deepctr.models.multitask.multitaskbase import MultiTaskModelBase
from deepctr.models.multitask.utils import calc_lift, cal_psi_score, calc_cum

custom_objects['NoMask'] = NoMask
custom_objects['MMOELayer'] = MMOELayer
custom_objects['MultiTaskModelBase'] = MultiTaskModelBase
custom_objects['Mean'] = Mean
custom_objects['AUC'] = AUC


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


def build_model(hp):
    # num_layers = hp.Int('num_bottom_layers', 1, 2, step=1)
    num_layers = 2
    bottom_shared_units = []
    for i in range(num_layers):
        bottom_shared_units.append(hp.Int('dnn_hidden_unit_' + str(i), 32, 128, step=32))
    bottom_shared_use_bn = hp.Boolean('bottom_shared_use_bn', default=False)
    num_experts = hp.Int('num_experts', 4, 12, step=2)
    expert_dim = hp.Int('expert_dim', 4, 12, step=4)
    l2_reg_embedding = hp.Float('l2_reg_embedding', 1e-5, 0.5, sampling='log')
    l2_reg_dnn = hp.Float('l2_reg_dnn', 1e-5, 0.5, sampling='log')
    model = MMOE_BIAS(dnn_feature_columns,
                      tasks,
                      bias_feature_columns_dict,
                      bias_dropout_dict,
                      num_experts=num_experts,
                      expert_dim=expert_dim,
                      bottom_shared_units=bottom_shared_units,
                      bottom_shared_use_bn=bottom_shared_use_bn,
                      l2_reg_embedding=l2_reg_embedding,
                      l2_reg_dnn=l2_reg_dnn,
                      )
    if gradnorm:
        last_shared_weights = [weight for weight in model.get_layer('mmoe_layer').trainable_weights
                               if weight.name.find('expert') >= 0]
        gradnorm_config = {'alpha': hp.Float('l2_reg_dnn', 0.1, 1.5, step=0.1),
                           'last_shared_weights': last_shared_weights}
    else:
        gradnorm_config = None
    model.compile(optimizers=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', 1e-4, 0.01, sampling='log')),
                  loss_fns=loss_fns,
                  metrics_logger=metrics_logger,
                  uncertainly=uncertainty,
                  gradnorm_config=gradnorm_config)
    return model


if __name__ == "__main__":
    # configure
    project_name = 'preloan_istrans_overdue2'
    run_name = 'uncertainty_weight_fpd4_mask_mob3_k11_mask7'
    mode = 'train'
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
    data = pd.read_csv('../data/test2.csv')
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
        pass
    else:
        data.loc[(data['if_t4'] != 1), 'fpd4_weight'] = 0.0
    data.loc[(data['if_t4'] != 1), 'fpd4_mask'] = 0
    
    data['mob3_k11_weight'] = 1.0
    data['mob3_k11_mask'] = 1.0
    if run_name.find('mob3_k11_nomask') >= 0:
        pass
    else:
        data.loc[(data['if_mob3_t11'] != 1), 'mob3_k11_weight'] = 0.0
    data.loc[(data['if_mob3_t11'] != 1), 'mob3_k11_mask'] = 0
    
    data['istrans_weight'] = 1.0
    data['istrans_mask'] = 1.0
    data.loc[data['pre_loan_flag'] != 0, 'istrans_mask'] = 0.0
    if run_name.find('istrans_mask') >= 0:
        data.loc[data['pre_loan_flag'] != 0, 'istrans_weight'] = 0.0
        
    data['fpd4'] = data['fpd4'].fillna(0)
    data['mob3_k11'] = data['mob3_k11'].fillna(0)

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

    quantile_transformer = QuantileTransformer(random_state=0)
    quantile_transformer.fit(data[(data['set'] == '1train') | (data['set'] == '2test')][col_x])
    data[col_x] = quantile_transformer.transform(data[col_x])
    data[col_x] = data[col_x].fillna(-1)
    import pickle
    with open('../data/quantile_transformer.pkl', 'wb') as f:
        pickle.dump(quantile_transformer, f)

    # import toad
    # combiner = toad.transform.Combiner()
    # combiner.fit(data[(data['set'] == '1train') | (data['set'] == '2test')][col_x + ['fpd4']],
    #              y='fpd4',
    #              method='quantile',
    #              n_bins=20,
    #              empty_separate=True,
    #              )
    # bins = combiner.export()
    # df_bin = combiner.transform(data[col_x])

    n_bins = 20
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat(col, vocabulary_size=n_bins+1, embedding_dim=4),
                                               maxlen=n_bins, combiner='mean', weight_name=col+'_weight')
                              for col in col_x]

    dnn_feature_columns = varlen_feature_columns

    if add_bias:
        bias_feature_columns_dict = {}
        bias_feature_columns_list = []
        for task_name, columns in bias_features.items():
            bias_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].unique().shape[0], embedding_dim=8)
                                    for i, feat in enumerate(columns)]
            bias_feature_columns_list += bias_feature_columns
            bias_feature_columns_dict[task_name] = bias_feature_columns
        feature_names = get_feature_names(dnn_feature_columns + bias_feature_columns_list)
        bias_feature_names = get_feature_names(bias_feature_columns_list)
        for feat in bias_feature_names:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
    else:
        bias_feature_columns_dict = None
        bias_dropout_dict = None
        bias_feature_names = []
        feature_names = get_feature_names(dnn_feature_columns)

    # generate input data for model
    train = data[data['set'] == '1train']
    test = data[data['set'] == '2test']
    weight_dict = {}
    for round_num, ft in enumerate(col_x):
        weight_dict[ft] = np.array([list(range(1, n_bins+1))]*train.shape[0])
        bin_ary = np.array([i / n_bins for i in range(1, n_bins)])
        bin_ary = np.append(bin_ary, -1)
        cent_hat = np.abs(np.expand_dims(np.expand_dims(train[ft], -1), -1) - np.expand_dims(bin_ary, -1))
        weight_dict[ft+'_weight'] = 1.0 / (cent_hat + 1e-7)
    model_input = weight_dict

    weight_dict = {}
    for round_num, ft in enumerate(col_x):
        weight_dict[ft] = np.array([list(range(1, n_bins+1))]*test.shape[0])
        bin_ary = np.array([i / n_bins for i in range(1, n_bins)])
        bin_ary = np.append(bin_ary, -1)
        cent_hat = np.abs(np.expand_dims(np.expand_dims(test[ft], -1), -1) - np.expand_dims(bin_ary, -1))
        weight_dict[ft+'_weight'] = 1.0 / (cent_hat + 1e-7)
    test_model_input = weight_dict

    weight_dict = {}
    for round_num, ft in enumerate(col_x):
        weight_dict[ft] = np.array([list(range(1, n_bins+1))]*train[ft].iloc[:1].shape[0])
        bin_ary = np.array([i / n_bins for i in range(1, n_bins)])
        bin_ary = np.append(bin_ary, -1)
        cent_hat = np.abs(np.expand_dims(np.expand_dims(train[ft].iloc[:1], -1), -1) - np.expand_dims(bin_ary, -1))
        weight_dict[ft+'_weight'] = 1.0 / (cent_hat + 1e-7)
    model_batch_input = weight_dict

    callback_data = (test_model_input,
                     {task_name: test[[task_name]] for task_name in tasks.keys()},
                     {task_name: test[task_name+'_weight'] for task_name in tasks.keys()})
    callback_data = list(tf.data.Dataset.from_tensor_slices(callback_data).shuffle(test['istrans'].shape[0]).batch(
        batch_size).take(1))[0]

    # train_dataset = tf.data.Dataset.from_tensor_slices((model_input,
    #                                                     {'istrans': train['istrans'].values,
    #                                                      'fpd4': train['fpd4'].values}))
    # train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # val_dataset = tf.data.Dataset.from_tensor_slices((test_model_input,
    #                                                   {'istrans': test['istrans'].values,
    #                                                    'fpd4': test['fpd4'].values}))
    # val_dataset = val_dataset.batch(batch_size)

    # train or predict
    if mode == 'train':
        model = MMOE_BIAS(dnn_feature_columns,
                          tasks,
                          bias_feature_columns_dict,
                          bias_dropout_dict,
                          num_experts=4,
                          expert_dim=8,
                          bottom_shared_units=(96, 96),
                          bottom_shared_use_bn=False,
                          l2_reg_embedding=2.7618e-05,
                          l2_reg_dnn=1.1261e-05
                          )
        # plot_model(aa, to_file=joint_symbol.join([checkpoint_dir, 'model_viz.png']), show_shapes=True,
        #            show_layer_names=True)
        if gradnorm:
            last_shared_weights = [weight for weight in model.get_layer('mmoe_layer').trainable_weights
                                   if weight.name.find('expert') >= 0]
            gradnorm_config = {'alpha': 0.3, 'last_shared_weights': last_shared_weights}
        else:
            gradnorm_config = None
        # last_lr = 0.003
        last_lr = 0.001
        optimizers = keras.optimizers.Adam(learning_rate=ModifiedExponentialDecay(last_lr))
        # optimizers = keras.optimizers.Adam(learning_rate=last_lr)
        model.compile(optimizers=optimizers,
                      loss_fns=loss_fns,
                      metrics_logger=metrics_logger,
                      loss_weights=loss_weights,
                      uncertainly=uncertainty,
                      gradnorm_config=gradnorm_config)
        try:
            checkpoints = [joint_symbol.join([checkpoint_dir, name]) for name in os.listdir(checkpoint_dir)]
            latest_checkpoint = max(checkpoints).split('.index')[0]
            model.train_on_batch(model_batch_input,
                                 {task_name: train[[task_name]].iloc[:1] for task_name in tasks.keys()})
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
                            sample_weight={task_name: train[task_name+'_weight'] for task_name in tasks.keys()},
                            batch_size=256,
                            epochs=100,
                            initial_epoch=last_epoch,
                            verbose=2,
                            validation_data=(test_model_input,
                                             {task_name: test[[task_name]] for task_name in tasks.keys()},
                                             {task_name: test[task_name+'_weight'] for task_name in tasks.keys()}),
                            callbacks=[
                                # ReduceLROnPlateau(monitor='val_fpd4_AUC', factor=0.7, mode='max',
                                #                   patience=2, verbose=1, min_delta=0.01),
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