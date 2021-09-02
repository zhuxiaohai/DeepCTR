import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import ESMM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.layers import custom_objects
from deepctr.layers.utils import NoMask
from tensorflow.python.keras.callbacks import Callback, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import save_model, load_model, Model
from tensorflow.python.keras.metrics import AUC
from tensorflow.python.keras import backend as K
custom_objects['NoMask'] = NoMask


class MyEarlyStopping(Callback):
    def __init__(self, monitor, patience=0, savepath=None, coef_of_balance=0.2, direction='maximize'):
        super(MyEarlyStopping, self).__init__()
        self.patience = patience
        self.savepath = savepath
        self.best_weights = None
        self.coef_of_balance = coef_of_balance
        self.monitor = monitor
        self.direction = direction

    def _compare_op(self, x, y):
        if self.direction == 'maximize':
            return x > y
        else:
            return x < y

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        if self.direction == 'maximize':
            self.best = -np.inf
        else:
            self.best = np.inf
        self.best_epoch = -1
        self.best_monitor = None

    def on_epoch_end(self, epoch, logs=None):
        if self.direction == 'maximize':
            current_metric = logs[self.monitor] - \
                             self.coef_of_balance * (abs(logs[self.monitor] - logs[self.monitor[4:]]))
        else:
            current_metric = logs[self.monitor] + \
                             self.coef_of_balance * (abs(logs[self.monitor] - logs[self.monitor[4:]]))

        if self._compare_op(current_metric, self.best):
            self.best = current_metric
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.best_monitor = logs[self.monitor]
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.save_weights(self.savepath + '-{:02d}-{:.5f}'.format(self.stopped_epoch + 1,
                                                                                K.get_value(self.model.optimizer.lr)))
                self.model.stop_training = True
                print('\n')
                print("Restoring model weights from the end of the best epoch: %05d" % (self.best_epoch + 1))
                self.model.set_weights(self.best_weights)
                save_model(self.model, 'best_model_epoch{}_{}{:.4f}.h5'.format(self.best_epoch + 1,
                                                                               self.monitor,
                                                                               self.best_monitor))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


if __name__ == "__main__":
    mode = 'train'
    data = pd.read_csv('../data/train_for_multi.csv')
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
    data['fpd4'] = data['fpd4'].fillna(0)
    data[col_x] = data[col_x].fillna(-1)
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

    train = data[data['set'] == '1train']
    test = data[data['set'] == '2test']
    oot = data[data['set'] == '3oot']
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    oot_model_input = {name: oot[name] for name in feature_names}
    train_y = [train['istrans'].values, train['fpd4'].values]
    test_y = [test['istrans'].values, test['fpd4'].values]
    oot_y = [oot['istrans'].values, oot['fpd4'].values]

    if mode == 'train':
        model = ESMM(dnn_feature_columns)

        checkpoint_dir = '.\ckt\model'
        tensorboard_dir = '.\log_dir'
        try:
            dir_name = checkpoint_dir[:checkpoint_dir.find('\model')]
            checkpoints = [dir_name + '\\' + name for name in os.listdir(checkpoint_dir[:checkpoint_dir.find('\model')])]
            latest_checkpoint = max(checkpoints).split('.index')[0]
            model.load_weights(latest_checkpoint)
            _, last_epoch, last_lr = latest_checkpoint.split('-')
            print('Restoring from ', latest_checkpoint)
            last_epoch = int(last_epoch)
            last_lr = float(last_lr)
        except:
            print('Creating a new model')
            last_epoch = 0
            last_lr = 0.001
        model.compile(optimizer=Adam(lr=last_lr), loss="binary_crossentropy", metrics=[AUC()])
        reduce_lr = ReduceLROnPlateau(monitor='val_lambda_auc',
                                      factor=0.8,
                                      patience=5,
                                      min_lr=0.0001,
                                      mode='max',
                                      verbose=1)
        tensorboard = TensorBoard(log_dir=tensorboard_dir)
        earlystop = MyEarlyStopping('val_lambda_auc', patience=10, savepath=checkpoint_dir)
        history = model.fit(train_model_input,
                            train_y,
                            batch_size=256,
                            verbose=1,
                            epochs=100,
                            initial_epoch=last_epoch,
                            validation_data=(test_model_input, test_y),
                            callbacks=[reduce_lr, earlystop, tensorboard])
    else:
        best_metric = -1
        best_model = None
        for i in os.listdir('../../deepctr/models/multitask/'):
            if i.find('best_model') >= 0:
                metric = float(re.match('.*auc(.*).h5', i)[1])
                if metric > best_metric:
                    best_metric = metric
                    best_model = i
        print('loading ', best_model)
        model = load_model(best_model, custom_objects=custom_objects)
        cvr_layer = model.get_layer('cvr')
        cvr_model = Model(model.input, outputs=cvr_layer.output)
        print('final_result')
        train_predictions = model.predict(train_model_input)
        for index, (name, predictions) in enumerate(zip(['ctr', 'ctcvr'], train_predictions)):
            train_auc_score = roc_auc_score(train_y[index], predictions)
            fpr, tpr, _ = roc_curve(train_y[index], predictions)
            train_ks = np.max(np.abs(tpr - fpr))
            print(' {}: train_auc {:4f} train_ks {:4f}'.format(name, train_auc_score, train_ks))
        predictions = cvr_model.predict(train_model_input)
        train_auc_score = roc_auc_score(train_y[1][train_y[0] != 0], predictions[train_y[0] != 0])
        fpr, tpr, _ = roc_curve(train_y[1][train_y[0] != 0], predictions[train_y[0] != 0])
        train_ks = np.max(np.abs(tpr - fpr))
        print(' {}: train_auc {:4f} train_ks {:4f}'.format('cvr', train_auc_score, train_ks))

        test_predictions = model.predict(test_model_input)
        for index, (name, predictions) in enumerate(zip(['ctr', 'ctcvr'], test_predictions)):
            test_auc_score = roc_auc_score(test_y[index], predictions)
            fpr, tpr, _ = roc_curve(test_y[index], predictions)
            test_ks = np.max(np.abs(tpr - fpr))
            print(' {}: test_auc {:4f} test_ks {:4f}'.format(name, test_auc_score, test_ks))
        predictions = cvr_model.predict(test_model_input)
        test_auc_score = roc_auc_score(test_y[1][test_y[0] != 0], predictions[test_y[0] != 0])
        fpr, tpr, _ = roc_curve(test_y[1][test_y[0] != 0], predictions[test_y[0] != 0])
        test_ks = np.max(np.abs(tpr - fpr))
        print(' {}: test_auc {:4f} test_ks {:4f}'.format('cvr', test_auc_score, test_ks))

        oot_predictions = model.predict(oot_model_input)
        for index, (name, predictions) in enumerate(zip(['ctr', 'ctcvr'], oot_predictions)):
            oot_auc_score = roc_auc_score(oot_y[index], predictions)
            fpr, tpr, _ = roc_curve(oot_y[index], predictions)
            oot_ks = np.max(np.abs(tpr - fpr))
            print(' {}: oot_auc {:4f} oot_ks {:4f}'.format(name, oot_auc_score, oot_ks))
        predictions = cvr_model.predict(oot_model_input)
        oot_auc_score = roc_auc_score(oot_y[1][oot_y[0] != 0], predictions[oot_y[0] != 0])
        fpr, tpr, _ = roc_curve(oot_y[1][oot_y[0] != 0], predictions[oot_y[0] != 0])
        oot_ks = np.max(np.abs(tpr - fpr))
        print(' {}: oot_auc {:4f} oot_ks {:4f}'.format('cvr', oot_auc_score, oot_ks))