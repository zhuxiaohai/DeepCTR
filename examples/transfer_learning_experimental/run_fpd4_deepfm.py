import os
import re
import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.metrics import AUC, Mean
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import TensorBoard
from deepctr.feature_column import DenseFeat, SparseFeat, get_feature_names
from deepctr.models.transferlearning.domain_adaptation import DomainAdaptation
from deepctr.models.transferlearning.transferloss import DomainAdversarialLoss, MMDLoss, LMMDLoss
from deepctr.models.multitask_modified.multitaskbase import MultiTaskModelBase
from deepctr.models.transferlearning.basenet import DeepFM
from deepctr.callbacks import MyEarlyStopping, ModifiedExponentialDecay, MyRecorder
from deepctr.layers import custom_objects
from deepctr.layers.utils import NoMask
from deepctr.metrics import calc_lift
from deepctr.models.transferlearning.utils import plot_tsne_source_target, proxy_a_distance

custom_objects['NoMask'] = NoMask
custom_objects['MultiTaskModelBase'] = MultiTaskModelBase
custom_objects['Mean'] = Mean
custom_objects['AUC'] = AUC
custom_objects['ModifiedExponentialDecay'] = ModifiedExponentialDecay


project_name = 'k3dq'
run_name = 'fpd4_mask_deepfm_dzdq'
mode = 'test'
joint_symbol = '/'
checkpoint_dir = joint_symbol.join([project_name, 'ckt', run_name])
tensorboard_dir = joint_symbol.join([project_name, 'log_dir', run_name])
summary_dir = joint_symbol.join([project_name, 'metrics', run_name])
trend_dir = joint_symbol.join([project_name, 'trend', run_name])

if run_name.find('_lmmd') >= 0:
    loss_fns = {'fpd4': keras.losses.categorical_crossentropy}
    metrics_logger = {'fpd4': AUC(label_weights=[0, 1], name='fpd4_AUC')}
    tasks = {'fpd4': 'multiclass'}
else:
    loss_fns = {'fpd4': keras.losses.binary_crossentropy}
    metrics_logger = {'fpd4': AUC(name='fpd4_AUC')}
    tasks = {'fpd4': 'binary'}
batch_size = 256
epochs = 100

dz_ts = pd.read_csv('../data/transferlearning/data_K3_dz_samples.csv')
dq_ts = pd.read_csv('../data/transferlearning/data_dq_0716_updated.csv')

dz_ts['fpd4_weight'] = 1.0
dz_ts['fpd4_mask'] = 1.0
if run_name.find('fpd4_nomask') >= 0:
    pass
else:
    dz_ts.loc[(dz_ts['if_fpd_t4'] != 1), 'fpd4_weight'] = 0.0
dz_ts.loc[(dz_ts['if_fpd_t4'] != 1), 'fpd4_mask'] = 0

dq_ts['fpd4_weight'] = 1.0
dq_ts['fpd4_mask'] = 1.0
if run_name.find('fpd4_nomask') >= 0:
    pass
else:
    dq_ts.loc[(dq_ts['if_fpd_t4'] != 1), 'fpd4_weight'] = 0.0
dq_ts.loc[(dq_ts['if_fpd_t4'] != 1), 'fpd4_mask'] = 0

dz_all = dz_ts[(dz_ts.shouxin_date>='2019-09-01') & (dz_ts.shouxin_date<'2020-05-01')][
        ['cust_no','shouxin_date','fpd4', 'fpd4_weight', 'fpd4_mask'] +
        ['ali_rain_score','bj_jc_m36_consume_cnt','td_zhixin_score','hds_36m_purchase_steady','hds_36m_total_purchase_cnt','hds_36m_month_max_purchase_money_excp_doub11_12','hds_36m_doub11_12_total_purchase_money','hds_phone_rich_rank','hds_mobile_rich','hds_mobile_reli_rank','hds_recent_consumme_active_rank'] +
        ['ab_local_ratio','ab_mobile_cnt','app_type','cust_gender','cust_id_area','cust_work_city','idcard_district_grade','idcard_rural_flag','immediate_relation_cnt','operation_sys','relation_contact_cnt','study_app_cnt','selffill_degree','selffill_is_have_creditcard','selffill_marital_status','ab_local_cnt','ab_prov_cnt']]

dz_all['hds_mobile_reli_rank_Ma'] = dz_all.hds_mobile_reli_rank.apply(lambda x : 1 if x=='Ma' else 0)
dz_all['hds_mobile_reli_rank_Mb'] = dz_all.hds_mobile_reli_rank.apply(lambda x : 1 if x=='Mb' else 0)
dz_all['hds_mobile_reli_rank_M0'] = dz_all.hds_mobile_reli_rank.apply(lambda x : 1 if x=='M0' else 0)
dz_all['hds_recent_consumme_active_rank'] = dz_all.hds_recent_consumme_active_rank.apply(lambda x : float(str(x).replace('}','')))
dz_all['is_ios'] = dz_all['operation_sys'].apply(lambda x : 1 if x in ['ios','IOS','iPhone OS'] else(0 if x in ['android','Android'] else -99))
dz_all['is_male'] = dz_all['cust_gender'].apply(lambda x : 1 if x=='M' else(0 if x=='F' else -99))
dz_all['selffill_is_have_creditcard'] = dz_all['selffill_is_have_creditcard'].apply(lambda x : 1 if x == 'Y' else(0 if x == 'N' else -99))
dz_all['credit_repayment_score_bj_2'] = -99
dz_all['td_xyf_dq_score'] = -99
# dz_all.fillna(-99, inplace=True)
dz_all = dz_all.replace([-99], np.nan)

source = dz_all

candidate_features = ['cust_no','shouxin_date','fpd4', 'fpd4_weight', 'fpd4_mask'] +\
        ['ali_rain_score','bj_jc_m36_consume_cnt','credit_repayment_score_bj_2',
         'td_xyf_dq_score','td_zhixin_score','hds_36m_purchase_steady','hds_36m_total_purchase_cnt',
         'hds_36m_month_max_purchase_money_excp_doub11_12','hds_36m_doub11_12_total_purchase_money',
         'hds_phone_rich_rank','hds_mobile_rich','hds_mobile_reli_rank','hds_recent_consumme_active_rank'] +\
        ['ab_local_ratio','ab_mobile_cnt','app_type','cust_gender','cust_id_area','cust_work_city',
         'idcard_district_grade','idcard_rural_flag','immediate_relation_cnt','operation_sys','relation_contact_cnt',
         'study_app_cnt','selffill_degree','selffill_is_have_creditcard','selffill_marital_status','ab_local_cnt','ab_prov_cnt']

if mode == 'test':
    dq_oot = pd.concat([pd.read_csv('../data/transferlearning/dqts_201908_202107_1.csv'),
                        pd.read_csv('../data/transferlearning/dqts_201908_202107_2.csv'),
                        pd.read_csv('../data/transferlearning/dqts_201908_202107_3.csv')],
                       axis=0)
    dq_oot = dq_oot[(dq_oot.shouxin_date > '2020-06-09') &
                    (dq_oot.shouxin_date < '2020-08-01') &
                    (dq_oot.istrans == 1) &
                    (dq_oot.if_t4 == 1)]
    dq_oot['fpd4_weight'] = 1.0
    dq_oot['fpd4_mask'] = 1.0
    if run_name.find('fpd4_nomask') >= 0:
        pass
    else:
        dq_oot.loc[(dq_oot['if_t4'] != 1), 'fpd4_weight'] = 0.0
    dq_oot.loc[(dq_oot['if_t4'] != 1), 'fpd4_mask'] = 0
    dq_all = pd.concat([dq_ts[(dq_ts.shouxin_date >= '2019-07-23')][candidate_features], dq_oot[candidate_features]], axis=0)
else:
    dq_all = dq_ts[(dq_ts.shouxin_date >= '2019-07-23')][candidate_features]

dq_all['hds_mobile_reli_rank_Ma'] = dq_all.hds_mobile_reli_rank.apply(lambda x : 1 if x=='Ma' else 0)
dq_all['hds_mobile_reli_rank_Mb'] = dq_all.hds_mobile_reli_rank.apply(lambda x : 1 if x=='Mb' else 0)
dq_all['hds_mobile_reli_rank_M0'] = dq_all.hds_mobile_reli_rank.apply(lambda x : 1 if x=='M0' else 0)
dq_all['hds_recent_consumme_active_rank'] = dq_all.hds_recent_consumme_active_rank.apply(lambda x : float(str(x).replace('}','')))
dq_all['is_ios'] = dq_all['operation_sys'].apply(lambda x : 1 if x in ['ios','IOS','iPhone OS'] else(0 if x in ['android','Android'] else -99))
dq_all['is_male'] = dq_all['cust_gender'].apply(lambda x : 1 if x=='M' else(0 if x=='F' else -99))
dq_all['selffill_is_have_creditcard'] = dq_all['selffill_is_have_creditcard'].apply(lambda x : 1 if x == 'Y' else(0 if x == 'N' else -99))
dq_all['td_xyf_dq_score'] = dq_all['td_xyf_dq_score'].apply(lambda x : -99 if x==-99 else (x*1000))
# dq_all.fillna(-99, inplace=True)
dq_all = dq_all.replace([-99], np.nan)

target = dq_all

col_x = ['ali_rain_score','bj_jc_m36_consume_cnt','td_zhixin_score','hds_36m_purchase_steady','hds_36m_total_purchase_cnt',
         'hds_36m_month_max_purchase_money_excp_doub11_12','hds_36m_doub11_12_total_purchase_money','hds_phone_rich_rank',
         'hds_mobile_rich','hds_recent_consumme_active_rank','ab_local_ratio','ab_mobile_cnt','cust_id_area',
         'cust_work_city','idcard_district_grade','idcard_rural_flag','immediate_relation_cnt','relation_contact_cnt',
         'study_app_cnt','selffill_degree','selffill_is_have_creditcard','selffill_marital_status','ab_local_cnt',
         'ab_prov_cnt','hds_mobile_reli_rank_Ma','hds_mobile_reli_rank_Mb','hds_mobile_reli_rank_M0','is_ios','is_male',
         'credit_repayment_score_bj_2','td_xyf_dq_score']

early_features = [col for col in col_x if target[target.shouxin_date < '2020-05-01'][col].unique().shape[0] < 5]
bin_features = [col for col in col_x if col not in early_features]
import toad
combiner = toad.transform.Combiner()
combiner.fit(target[target.shouxin_date < '2020-05-01'][bin_features + ['fpd4']],
             y='fpd4',
             method='quantile',
             n_bins=5,
             empty_separate=True,
             )
sparse_features = []
dense_features = []
for feature in col_x:
    if feature in early_features:
        source[feature + '_bin'] = source[feature].astype(str)
        target[feature + '_bin'] = target[feature].astype(str)
        sparse_features.append(feature + '_bin')
    else:
        bins = combiner.export()[feature]
        if len(bins) <= 3:
            source[feature + '_bin'] = combiner.transform(source[feature]).astype(str)
            target[feature + '_bin'] = combiner.transform(target[feature]).astype(str)
            sparse_features.append(feature + '_bin')
        else:
            source[feature + '_bin'] = source[feature].fillna(-1)
            target[feature + '_bin'] = target[feature].fillna(-1)
            dense_features.append(feature + '_bin')

from sklearn.preprocessing import LabelEncoder
temp = pd.concat([source[sparse_features], target[target.shouxin_date < '2020-05-01'][sparse_features]], axis=0)
for feat in sparse_features:
    lbe = LabelEncoder()
    lbe.fit(temp[feat])
    source[feat] = lbe.transform(source[feat])
    target[feat] = lbe.transform(target[feat])

mms = MinMaxScaler(feature_range=(0, 1))
source[dense_features] = mms.fit_transform(source[dense_features])
mms = MinMaxScaler(feature_range=(0, 1))
mms.fit(target[target.shouxin_date < '2020-05-01'][dense_features])
target[dense_features] = mms.transform(target[dense_features])

fixlen_feature_columns = [SparseFeat(feat,
                                     vocabulary_size=temp[feat].unique().shape[0],
                                     embedding_dim=8)
                          for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                        for feat in dense_features]
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

def concat_all(source, target):
    source_x, source_y, source_weight = source
    target_x, target_y, target_weight = target
    combined_x = {name: tf.concat([source_x[name], target_x[name]], 0) for name in source_x.keys()}
    combined_y = {name: tf.concat([source_y[name], target_y[name]], 0) for name in source_y.keys()}
    combined_weight = {name: tf.concat([source_weight[name], target_weight[name]], 0) for name in source_weight.keys()}
    return (source, target, (combined_x, combined_y, combined_weight))


def concat(source, target):
    source_x, source_y, source_weight = source
    target_x, target_y, target_weight = target
    combined_x = {name: tf.concat([source_x[name], target_x[name]], 0) for name in source_x.keys()}
    combined_y = {name: tf.concat([source_y[name], target_y[name]], 0) for name in source_y.keys()}
    combined_weight = {name: tf.concat([source_weight[name], target_weight[name]], 0) for name in source_weight.keys()}
    return (combined_x, combined_y, combined_weight)


source.loc[:, 'set'] = '1train'
target.loc[(target.shouxin_date < '2020-05-01'), 'set'] = '2test'
target.loc[(target.shouxin_date >= '2020-05-01') & (target.shouxin_date < '2020-06-01'), 'set'] = '3oot'
target.loc[(target.shouxin_date >= '2020-06-01') & (target.shouxin_date < '2020-07-01'), 'set'] = '4oot'
target.loc[(target.shouxin_date >= '2020-07-01') & (target.shouxin_date <= '2020-08-01'), 'set'] = '5oot'
data = pd.concat([source, target], axis=0).reset_index()

source_x = {name: source[name] for name in feature_names}
source_weight = {task_name: source[task_name+'_weight'] for task_name in tasks.keys()}
if run_name.find('_lmmd') >= 0:
    enc = OneHotEncoder(handle_unknown='ignore')
    source_y = {task_name: enc.fit_transform(source[[task_name]]).toarray().astype(np.float32) for task_name in tasks.keys()}
else:
    source_y = {task_name: source[[task_name]] for task_name in tasks.keys()}

target_x = {name: target[target.shouxin_date < '2020-05-01'][name] for name in feature_names}
target_weight = {task_name: target[target.shouxin_date < '2020-05-01'][task_name+'_weight'] for task_name in tasks.keys()}
if run_name.find('_lmmd') >= 0:
    enc = OneHotEncoder(handle_unknown='ignore')
    target_y = {task_name: enc.fit_transform(target[target.shouxin_date < '2020-05-01'][[task_name]]).toarray().astype(np.float32)
                for task_name in tasks.keys()}
else:
    target_y = {task_name: target[target.shouxin_date < '2020-05-01'][[task_name]] for task_name in tasks.keys()}

source_batch_num = math.ceil(len(source) / batch_size)
target_batch_num = math.ceil(len(target[target.shouxin_date < '2020-05-01']) / batch_size)
if run_name.find('_da') >= 0:
    train_iter_num = max(source_batch_num, target_batch_num)
else:
    if run_name.find('_dzdq') >= 0:
        train_iter_num = max(source_batch_num, target_batch_num)
    else:
        train_iter_num = source_batch_num
max_iter_num = epochs * train_iter_num

source_dataset = tf.data.Dataset.from_tensor_slices((source_x, source_y, source_weight))
target_dataset = tf.data.Dataset.from_tensor_slices((target_x, target_y, target_weight))
source_dataset = source_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True, seed=0).repeat().batch(batch_size)
target_dataset = target_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True, seed=0).repeat().batch(batch_size)

if run_name.find('_da') >= 0:
    train_dataset = tf.data.Dataset.zip((source_dataset.take(train_iter_num), target_dataset.take(train_iter_num)))
    if run_name.find('dzdq') >= 0:
        train_dataset = train_dataset.map(lambda x1, y1: concat_all(x1, y1))
else:
    if run_name.find('dzdq') >= 0:
        train_dataset = tf.data.Dataset.zip((source_dataset.take(train_iter_num), target_dataset.take(train_iter_num))
                                            ).map(lambda x1, y1: concat(x1, y1))
    else:
        train_dataset = source_dataset.take(train_iter_num)

if run_name.find('_da') >= 0:
    if run_name.find('_lmmd') >= 0:
        enc = OneHotEncoder(handle_unknown='ignore')
        val_dataset = tf.data.Dataset.zip(
            (source_dataset,
             tf.data.Dataset.from_tensor_slices(
                 ({name: target[target['set'] == '3oot'][name] for name in feature_names},
                  {task_name: enc.fit_transform(target[target['set'] == '3oot'][[task_name]]).toarray().astype(np.float32)
                   for task_name in tasks.keys()},
                  {task_name: target[target['set'] == '3oot'][task_name + '_weight'] for task_name in tasks.keys()})
             ).batch(batch_size)
            )
        )
    else:
        val_dataset = tf.data.Dataset.zip(
            (source_dataset,
             tf.data.Dataset.from_tensor_slices(
                 ({name: target[target['set'] == '3oot'][name] for name in feature_names},
                  {task_name: target[target['set'] == '3oot'][[task_name]] for task_name in tasks.keys()},
                  {task_name: target[target['set'] == '3oot'][task_name + '_weight'] for task_name in tasks.keys()})
             ).batch(batch_size)
            )
        )
else:
    if run_name.find('_lmmd') >= 0:
        enc = OneHotEncoder(handle_unknown='ignore')
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {name: target[target['set'] == '3oot'][name] for name in feature_names},
                {task_name: enc.fit_transform(target[target['set'] == '3oot'][[task_name]]).toarray().astype(np.float32)
                 for task_name in tasks.keys()},
                {task_name: target[target['set'] == '3oot'][task_name + '_weight'] for task_name in tasks.keys()}
            )
        )
    else:
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {name: target[target['set'] == '3oot'][name] for name in feature_names},
                {task_name: target[target['set'] == '3oot'][[task_name]] for task_name in tasks.keys()},
                {task_name: target[target['set'] == '3oot'][task_name+'_weight'] for task_name in tasks.keys()}
            )
        )
    val_dataset = val_dataset.batch(batch_size)

if mode == 'train':
    if run_name.find('_lmmd') >= 0:
        output_dim = 2
    else:
        output_dim = 1
    model = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(96, 96),
                   l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                   dnn_activation='relu', dnn_use_bn=False, tasks=tasks)

    last_lr = 0.001
    optimizer = keras.optimizers.Adam(learning_rate=ModifiedExponentialDecay(last_lr, max_iter_num=max_iter_num))
    model.compile(optimizer=optimizer,
                  loss=loss_fns,
                  metrics=metrics_logger,
                  loss_weights=None,
                  uncertainly=None,
                  gradnorm_config=None)

    if run_name.find('_da') >= 0:
        feature_extractor = keras.Model(inputs=model.input, outputs=model.get_layer('bottom_shared_dnn').output)
        if run_name.find('_lmmd') >= 0:
            da_loss = LMMDLoss(2, max_iter_num=max_iter_num)
        elif run_name.find('_mmd') >= 0:
            da_loss = MMDLoss()
        else:
            da_loss = DomainAdversarialLoss(max_iter_grl=max_iter_num, dnn_units=[24, 8])
        dann = DomainAdaptation(feature_extractor, model)
        dann.compile(da_loss=da_loss,
                     optimizer_da_loss=tf.keras.optimizers.Adam(
                         learning_rate=ModifiedExponentialDecay(0.001, max_iter_num=max_iter_num)))
        dann.fit(train_dataset,
                 validation_data=val_dataset,
                 epochs=epochs,
                 callbacks=[MyEarlyStopping('val_fpd4_AUC',
                                            patience=10,
                                            savepath=checkpoint_dir,
                                            coef_of_balance=0,
                                            direction='maximize'),
                            TensorBoard(log_dir=tensorboard_dir),
                            MyRecorder(tensorboard_dir, None,
                                       gradient_freq=0, experts_freq=0, lr_freq=1)

                 ]
                 )
        set_data = data[data['set'] == '3oot']
        predictions = dann.predict({name: set_data[name] for name in feature_names})
        if run_name.find('_lmmd') >= 0:
            pred = predictions['fpd4'][:, 1]
        else:
            pred = predictions['fpd4'][:, 0]
        score = roc_auc_score(set_data['fpd4'].values, pred, sample_weight=set_data['fpd4_mask'].values)
        print(score)

        predictions = model.predict({name: set_data[name] for name in feature_names})
        if run_name.find('_lmmd') >= 0:
            pred = predictions['fpd4'][:, 1]
        else:
            pred = predictions['fpd4'][:, 0]
        score = roc_auc_score(set_data['fpd4'].values, pred, sample_weight=set_data['fpd4_mask'].values)
        print(score)
    else:
        history = model.fit(train_dataset,
                            epochs=epochs,
                            validation_data=val_dataset,
                            callbacks=[
                                MyEarlyStopping('val_fpd4_AUC',
                                                patience=10,
                                                savepath=checkpoint_dir,
                                                coef_of_balance=0,
                                                direction='maximize'),
                                TensorBoard(log_dir=tensorboard_dir),
                                MyRecorder(tensorboard_dir, None,
                                           gradient_freq=0, experts_freq=0, lr_freq=1)
                            ]
                            )
        set_data = data[data['set'] == '3oot']
        predictions = model.predict({name: set_data[name] for name in feature_names})
        if run_name.find('_lmmd') >= 0:
            pred = predictions['fpd4'][:, 1]
        else:
            pred = predictions['fpd4'][:, 0]
        score = roc_auc_score(set_data['fpd4'].values, pred, sample_weight=set_data['fpd4_mask'].values)
        print(score)
else:
    if not os.path.exists(trend_dir):
        os.makedirs(trend_dir)
    best_metric = -1
    best_model = None
    for i in os.listdir(checkpoint_dir):
        if i.find('best_') >= 0:
            metric = float(re.match('.*AUC(.*).h5', i)[1])
            if metric > best_metric:
                best_metric = metric
                best_model = i
    print('loading ', joint_symbol.join([checkpoint_dir, best_model]))
    model = load_model(joint_symbol.join([checkpoint_dir, best_model]), custom_objects=custom_objects)

    set_data = data[data['set'] == '3oot']
    predictions = model.predict({name: set_data[name] for name in feature_names})
    if run_name.find('_lmmd') >= 0:
        pred = predictions['fpd4'][:, 1]
    else:
        pred = predictions['fpd4'][:, 0]
    score = roc_auc_score(set_data['fpd4'].values, pred, sample_weight=set_data['fpd4_mask'].values)
    print(score)


    F = keras.Model(inputs=model.input, outputs=model.get_layer('bottom_shared_dnn').output)
    file_writer = tf.summary.create_file_writer(summary_dir)
    for task_name in tasks.keys():
        fig = plt.figure(figsize=(8, 20))
        fig.suptitle(run_name + '_' + task_name)
        for index, set_name in enumerate(['1train', '2test', '3oot', '4oot', '5oot']):
            set_data = data[data['set'] == set_name]
            predictions = model.predict({name: set_data[name] for name in feature_names})
            if run_name.find('_lmmd') >= 0:
                pred = predictions[task_name][:, 1]
            else:
                pred = predictions[task_name][:, 0]
            auc_score = roc_auc_score(set_data[task_name].values, pred,
                                      sample_weight=set_data[task_name + '_mask'].values)
            fpr, tpr, _ = roc_curve(set_data[task_name].values, pred,
                                    sample_weight=set_data[task_name + '_mask'].values)
            ks = np.max(np.abs(tpr - fpr))
            target = set_data[task_name].values
            weight = set_data[task_name + '_mask'].values
            pred = pred[weight != 0]
            target = target[weight != 0]
            title = '{} ks_{:.2f} auc_{:.2f}'.format(set_name, ks, auc_score)
            df = pd.DataFrame({'pred': pred, 'target': target})
            ax = fig.add_subplot(5, 1, index + 1)
            _ = calc_lift(df, 'pred', 'target', ax=ax, groupnum=10, title_name=title)
            print(' {}: {} auc {:4f} ks {:4f}'.format(task_name, set_name, auc_score, ks))
            with file_writer.as_default():
                tf.summary.scalar(task_name+'_ks', ks, step=index+1)
                tf.summary.scalar(task_name+'_auc', auc_score, step=index+1)
        fig.savefig(joint_symbol.join([trend_dir, task_name]))

        fig, ax = plt.subplots()
        # a distantance
        source_data = data[data['set'] == '1train']
        print('source shape ', source_data.shape)
        _, source_data = train_test_split(source_data, test_size=min(2000, source_data.shape[0]-1),
                                          random_state=42, shuffle=source_data[task_name])
        set_name = '2test'
        set_data = data[data['set'] == set_name]
        print(set_name, 'shape ', set_data.shape)
        _, set_data = train_test_split(set_data, test_size=min(2000, set_data.shape[0]-1),
                                       random_state=42, shuffle=set_data[task_name])
        for i in range(2):
            print('source label {} shape:'.format(i), source_data[source_data[task_name] == i].shape)
            print('{} label {} shape'.format(set_name, i), set_data[set_data[task_name] == i].shape)
            source_x = F.predict({name: source_data[source_data[task_name] == i][name] for name in feature_names})
            target_x = F.predict({name: set_data[set_data[task_name] == i][name] for name in feature_names})
            a_score = proxy_a_distance(source_x, target_x)
            print(set_name, 'a_score_{}'.format(i), a_score)
            with file_writer.as_default():
                tf.summary.scalar(task_name+'_a_score_{}'.format(i), a_score, step=index + 1)
        source_x = F.predict({name: source_data[name] for name in feature_names})
        target_x = F.predict({name: set_data[name] for name in feature_names})
        a_score = proxy_a_distance(source_x, target_x)
        print(set_name, 'a_score', a_score)
        with file_writer.as_default():
            tf.summary.scalar(task_name+'_a_score', a_score, step=index + 1)
        _ = plot_tsne_source_target(source_x, source_data[task_name].values, target_x, set_data[task_name].values, ax,
                                    name='{}_feature_map'.format(task_name))
        path_name = joint_symbol.join([trend_dir, '{}_feature_map.jpg'.format(task_name)])
        fig.savefig(path_name)




