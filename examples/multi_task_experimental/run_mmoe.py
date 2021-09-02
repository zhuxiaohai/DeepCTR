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
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.metrics import Mean
from tensorflow import keras
import tensorflow as tf
import kerastuner as kt

from deepctr.layers import custom_objects
from deepctr.layers.utils import NoMask
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models.multitask.mmoe import MMOE, MMOELayer, MMOE_BIAS
from deepctr.models.multitask.call_backs import MyEarlyStopping, MyRecorder
from deepctr.models.multitask.multitaskbase import MultiTaskModelBase
from deepctr.models.multitask.utils import calc_lift, cal_psi_score, calc_cum

custom_objects['NoMask'] = NoMask
custom_objects['MMOELayer'] = MMOELayer
custom_objects['MultiTaskModelBase'] = MultiTaskModelBase
custom_objects['Mean'] = Mean
custom_objects['AUC'] = AUC


def build_model(hp):
    # num_layers = hp.Int('num_bottom_layers', 1, 2, step=1)
    num_layers = 1
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
    run_name = 'uncertainty_weight_fpd4_mask_istrans_mask'
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
    tasks = {'istrans': 'binary', 'fpd4': 'binary'}
    loss_fns = {'istrans': keras.losses.binary_crossentropy,
                'fpd4': keras.losses.binary_crossentropy}
    # loss_fns = {'istrans': keras.losses.BinaryCrossentropy(),
    #             'fpd4': keras.losses.BinaryCrossentropy()}
    metrics_logger = {'istrans': AUC,
                      'fpd4': AUC}
    loss_weights = {'istrans': 1, 'fpd4': 6}
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
    bias_features = {'istrans': ['pre_loan_flag']}
    bias_dropout_dict = {'istrans': 0.1, 'fpd4': 0.1}
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
    model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    model_batch_input = {name: train[name].iloc[:1] for name in feature_names}
    callback_data = (test_model_input,
                     {'istrans': test[['istrans']],
                      'fpd4': test[['fpd4']]},
                     {'istrans': test['istrans_weight'],
                      'fpd4': test['fpd4_weight']})
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
        # model = MMOE_BIAS(dnn_feature_columns,
        #                   tasks,
        #                   bias_feature_columns_dict,
        #                   bias_dropout_dict,
        #                   num_experts=12,
        #                   expert_dim=4,
        #                   bottom_shared_units=(128,),
        #                   bottom_shared_use_bn=False,
        #                   l2_reg_embedding=1e-05,
        #                   l2_reg_dnn=1e-05
        #                   )
        model = MMOE_BIAS(dnn_feature_columns,
                          tasks,
                          bias_feature_columns_dict,
                          bias_dropout_dict,
                          num_experts=6,
                          expert_dim=12,
                          bottom_shared_units=(64,),
                          bottom_shared_use_bn=True,
                          l2_reg_embedding=0.0021129,
                          l2_reg_dnn=0.070863
                          )
        # plot_model(aa, to_file=joint_symbol.join([checkpoint_dir, 'model_viz.png']), show_shapes=True,
        #            show_layer_names=True)
        if gradnorm:
            last_shared_weights = [weight for weight in model.get_layer('mmoe_layer').trainable_weights
                                   if weight.name.find('expert') >= 0]
            gradnorm_config = {'alpha': 1, 'last_shared_weights': last_shared_weights}
        else:
            gradnorm_config = None
        # last_lr = 0.003
        last_lr = 0.0018228
        optimizers = keras.optimizers.Adam(learning_rate=last_lr)
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
                            {'istrans': train[['istrans']], 'fpd4': train[['fpd4']]},
                            sample_weight={'istrans': train['istrans_weight'], 'fpd4': train['fpd4_weight']},
                            batch_size=256,
                            epochs=100,
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
    elif mode == 'tuning':
        tuner = kt.BayesianOptimization(
            build_model,
            objective=kt.Objective('val_fpd4_AUC', direction="max"),
            max_trials=50,
            num_initial_points=10)
        tuner.search(x=model_input,
                     y={'istrans': train[['istrans']], 'fpd4': train[['fpd4']]},
                     sample_weight={'istrans': train['istrans_weight'], 'fpd4': train['fpd4_weight']},
                     batch_size=batch_size,
                     verbose=2,
                     epochs=100,
                     validation_data=(test_model_input,
                                      {'istrans': test[['istrans']], 'fpd4': test[['fpd4']]},
                                      {'istrans': test['istrans_weight'], 'fpd4': test['fpd4_weight']}),
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_fpd4_AUC',
                                                              patience=10)]
                     )
        # even if you had not searched the best param, tuner object could find the best hyperparameters
        # as well as the best model from the tuning_dir you passed in
        best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
        print(best_hyperparameters.values)
        # model = tuner.get_best_models(1)[0]
    else:
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
        outputs = {layer.name: layer.gates_output
                   for layer in model.layers
                   if (layer.name.find('cgc_layer') >= 0) or (layer.name.find('mmoe') >= 0)}
        intermediate_model = keras.Model(model.input, outputs=outputs)
        file_writer = tf.summary.create_file_writer(summary_dir)
        print('final_result')
        for task_name in tasks.keys():
            fig = plt.figure(figsize=(8, 10))
            fig.suptitle(run_name + '_' + task_name)
            for index, set_name in enumerate(['1train', '2test', '3oot']):
                set_data = data[data['set'] == set_name]
                predictions = model.predict({name: set_data[name] if name not in bias_feature_names
                    else pd.Series([0]*set_data.shape[0], index=set_data.index) for name in feature_names})
                auc_score = roc_auc_score(set_data[task_name].values, predictions[task_name][:, 0], sample_weight=set_data[task_name+'_mask'].values)
                fpr, tpr, _ = roc_curve(set_data[task_name].values, predictions[task_name][:, 0], sample_weight=set_data[task_name+'_mask'].values)
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

        print('bias_study')
        for task_name in bias_features.keys():
            if add_bias:
                bias_weights_series = pd.Series()
                for variable in [variable for variable in model.trainable_variables if variable.name.find(task_name+'_bias_') >= 0]:
                    bias_weights = variable.numpy()[:, 0].tolist()
                    bias_names = [variable.name.replace('sparse_emb_', '').
                                      replace(task_name+'_bias_', '').
                                      replace('embeddings:0', '') + str(i) for i in range(len(bias_weights))]
                    bias_weights_series = bias_weights_series.append(pd.Series(bias_weights, index=bias_names))
                fig, ax = plt.subplots()
                bias_weights_series.plot(ax=ax, kind='bar')
                ax.set_xticks(range(bias_weights_series.shape[0]))
                ax.set_xticklabels(labels=bias_weights_series.index, rotation=-20, horizontalalignment='left')
                fig.savefig(joint_symbol.join([trend_dir, task_name + '_bias_weights']), bbox_inches='tight')

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
                    predictions = model.predict({name: set_data[name] if name not in bias_feature_names
                    else pd.Series([0]*set_data.shape[0], index=set_data.index) for name in feature_names})
                    auc_score = roc_auc_score(set_data[task_name].values, predictions[task_name][:, 0], sample_weight=masks)
                    fpr, tpr, _ = roc_curve(set_data[task_name].values, predictions[task_name][:, 0], sample_weight=masks)
                    ks = np.max(np.abs(tpr - fpr))
                    pred = predictions[task_name][:, 0]
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
                        pred = predictions[task_name][:, 0]
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

        print('plot experts')
        set_name = '3oot'
        set_data = data[data['set'] == set_name]
        intermediate_results = intermediate_model.predict({name: set_data[name] if name not in bias_feature_names
                    else pd.Series([0]*set_data.shape[0], index=set_data.index) for name in feature_names})
        fig2 = plt.figure(figsize=(8, 10))
        fig2.suptitle(run_name + '_experts')
        for layer_index, (layer_name, layer_outputs) in enumerate(intermediate_results.items()):
            ax = fig2.add_subplot(len(intermediate_results), 1, layer_index + 1)
            ax.set_title(layer_name)
            gate_output_series = pd.Series()
            for gate_name, gate_output in layer_outputs.items():
                for i in range(gate_output.shape[-1]):
                    expert_name = gate_name + '_expert_' + str(i)
                    gate_output_series = gate_output_series.append(pd.Series({expert_name: gate_output[:, i].mean()}))
            gate_output_series.plot(kind='bar', ax=ax)
            ax.set_xticklabels(labels=gate_output_series.index, rotation=-20, horizontalalignment='left')
            ax.grid()
        fig2.savefig(joint_symbol.join([trend_dir, 'experts']))

        print('cross_validation')
        set_data = data[data['set'] == '3oot']
        predictions = model.predict({name: set_data[name] if name not in bias_feature_names
                    else pd.Series([0]*set_data.shape[0], index=set_data.index) for name in feature_names})
        fig_cross = plt.figure(figsize=(8, 10))
        fig_cross.suptitle(run_name + '_cross')
        ax_cross = fig_cross.add_subplot(2, 1, 1)
        df = pd.DataFrame({'pred': predictions['fpd4'][:, 0], 'target': set_data['istrans'].values})
        _ = calc_cum(df, 'pred', 'target', ax=ax_cross, groupnum=10, title_name='istrans')
        pred = predictions['istrans'][:, 0]
        target = set_data['fpd4'].values
        weight = set_data['fpd4_mask'].values
        pred = pred[weight != 0]
        target = target[weight != 0]
        ax_cross = fig_cross.add_subplot(2, 1, 2)
        df = pd.DataFrame({'pred': pred, 'target': target})
        _ = calc_cum(df, 'pred', 'target', ax=ax_cross, groupnum=10, title_name='fpd4')
        fig_cross.savefig(joint_symbol.join([trend_dir, 'cross']))
