import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import save_model
from tensorflow.python.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from deepctr.models.sequence.attentional_pooling import TimeSeries, RnnAttentionalLayer
from deepctr.layers import custom_objects
custom_objects['RnnAttentionalLayer'] = RnnAttentionalLayer


def get_xy_fd(hash_flag=False):
    constant_feature_columns = [SparseFeat('user', 3, embedding_dim=10, use_hash=hash_flag),
                       SparseFeat('gender', 2, embedding_dim=4, use_hash=hash_flag),
                       SparseFeat('item_id', 3 + 1, embedding_dim=8, use_hash=hash_flag),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag),
                       DenseFeat('pay_score', 3)]

    behavior_feature_columns = [
        VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                         maxlen=4, length_name="seq_length"),
        VarLenSparseFeat(SparseFeat('hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'), maxlen=4,
                         length_name="seq_length"),
        DenseFeat('df1', 4),
        DenseFeat('df2', 4)]

    behavior_sparse_indicator = ["item_id", "cate_id"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    cate_id = np.array([1, 2, 2])  # 0 is mask value
    score = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 2, 2, 0], [1, 2, 2, 0], [1, 2, 0, 0]])
    df1 = np.array([[0.5, 0.1, 0.2, 0], [0.7, 0.6, 0.3, 0], [0.3, 0.2, 0, 0]])
    df2 = np.array([[0.2, 0.2, 0.2, 0], [0.5, 0.1, 0.1, 0], [0.1, 0.2, 0, 0]])

    behavior_length = np.array([3, 3, 2])

    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                    'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id,
                    'pay_score': score, "seq_length": behavior_length,
                    'df1': df1, 'df2': df2}

    x = {name: feature_dict[name] for name in get_feature_names(
        constant_feature_columns + behavior_feature_columns)}
    y = np.array([1, 0, 1])
    return x, y, constant_feature_columns, behavior_feature_columns, behavior_sparse_indicator


# def auc(y_true, y_pred):
#     return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
#

class TrainAUC(Callback):
    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(train_model_input)
        auc_score = roc_auc_score(train_y, predictions)
        fpr, tpr, _ = roc_curve(train_y, predictions)
        ks = np.max(np.abs(tpr - fpr))
        print(' train_auc {:4f} train_ks {:4f}'.format(auc_score, ks))
        predictions = self.model.predict(oot1_model_input)
        auc_score = roc_auc_score(oot1_y, predictions)
        fpr, tpr, _ = roc_curve(oot1_y, predictions)
        ks = np.max(np.abs(tpr - fpr))
        print(' oot1_auc {:4f} oot1_ks {:4f}'.format(auc_score, ks))
        predictions = self.model.predict(oot2_model_input)
        auc_score = roc_auc_score(oot2_y, predictions)
        fpr, tpr, _ = roc_curve(oot2_y, predictions)
        ks = np.max(np.abs(tpr - fpr))
        print(' oot2_auc {:4f} oot2_ks {:4f}'.format(auc_score, ks))


class EarlyStoppingAtMinKS(Callback):
    def __init__(self, patience=0, savepath=None):
        super(EarlyStoppingAtMinKS, self).__init__()
        self.patience = patience
        self.savepath = savepath
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -1

    def on_epoch_end(self, epoch, logs=None):
        train_predictions = self.model.predict(train_model_input)
        train_auc_score = roc_auc_score(train_y, train_predictions)
        fpr, tpr, _ = roc_curve(train_y, train_predictions)
        train_ks = np.max(np.abs(tpr - fpr))
        print(' train_auc {:4f} train_ks {:4f}'.format(train_auc_score, train_ks))

        test_predictions = self.model.predict(test_model_input)
        test_auc_score = roc_auc_score(test_y, test_predictions)
        fpr, tpr, _ = roc_curve(test_y, test_predictions)
        test_ks = np.max(np.abs(tpr - fpr))
        print(' test_auc {:4f} test_ks {:4f}'.format(test_auc_score, test_ks))

        oot1_predictions = self.model.predict(oot1_model_input)
        oot1_auc_score = roc_auc_score(oot1_y, oot1_predictions)
        fpr, tpr, _ = roc_curve(oot1_y, oot1_predictions)
        oot1_ks = np.max(np.abs(tpr - fpr))
        print(' oot1_auc {:4f} oot1_ks {:4f}'.format(oot1_auc_score, oot1_ks))

        oot2_predictions = self.model.predict(oot2_model_input)
        oot2_auc_score = roc_auc_score(oot2_y, oot2_predictions)
        fpr, tpr, _ = roc_curve(oot2_y, oot2_predictions)
        oot2_ks = np.max(np.abs(tpr - fpr))
        print(' oot2_auc {:4f} oot2_ks {:4f}'.format(oot2_auc_score, oot2_ks))

        if train_ks > test_ks:
            if oot2_ks > self.best:
                self.best = oot2_ks
                self.wait = 0
                self.best_weights = self.model.get_weights()
                self.model.save_weights(self.savepath)
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print("Restoring model weights from the end of the best epoch.")
                    self.model.set_weights(self.best_weights)
                    save_model(self.model, 'final.h5')

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


# if __name__ == "__main__":
#     if tf.__version__ >= '2.0.0':
#         tf.compat.v1.disable_eager_execution()
#     x, y, constant_feature_columns, behavior_feature_columns, behavior_sparse_indicator = get_xy_fd()
#
#     # 4.Define Model,train,predict and evaluate
#     model = TimeSeries(constant_feature_columns, behavior_feature_columns, behavior_sparse_indicator,
#                        dnn_hidden_units=[4, 4], dnn_dropout=0.6)
#
#     model.compile('adam', 'binary_crossentropy',
#                   metrics=['binary_crossentropy'])
#     history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)


if __name__ == "__main__":
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    from sklearn.preprocessing import StandardScaler

    behavior_dense_features = ['max_overdue_day_calc_current',
                               'use_rate',
                               'repayment_rate']
    behavior_dense_list = []
    for feat in behavior_dense_features:
        for j in range(0, 180, 30):
            behavior_dense_list.append(feat + '_at_p' + str(j))

    constant_dense_features = \
        ['fpd_max_overdue_day_calc_his_recent90',
         'fpd_max_overdue_day_calc_his_recent180',
         'm0_loan_cnt_his_recent180',
         'm0_loan_cnt_his_recent90',
         'fpd_cnt_his_recent180',
         'fpd_cnt_his_recent30',
         'fpd_cnt_his_recent90',
         'm0_loan_cnt_his_recent30',
         'm0_periods_cnt_his_recent30',
         'platform_score',
         'im_cnt',
         'agent_job_level_code_mapped',
         'team_beyond_a3_num',
         'commercial_7del_cnt',
         'cros_accp_cnt',
         'shop_ershou_performance_m3',
         'ershou_housedel_m3',
         'ershou_showing_morning_m1',
         'ershou_showing_morning_m2',
         'team_agent_num_min',
         'commercial_15showing_cnt',
         'im_m1',
         'shop_cnt',
         'team_agent_num_max',
         'shop_on_job_cnt',
         'view_house_showing_cnt',
         'team_person_num',
         'team_agent_num_avg',
         'cust_hold_cnt_m1']
    label = 'mob2_dpd10_<4'
    columns = behavior_dense_list + constant_dense_features
    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")
    oot1_df = pd.read_csv('./oot1.csv')
    oot2_df = pd.read_csv('./oot2.csv')

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    oot1_df = oot1_df.fillna(0)
    oot2_df = oot2_df.fillna(0)

    scaler = StandardScaler()
    # 读取Model
    # scaler = joblib.load('scaler.pkl')
    # train = scaler.transform(train_df[columns])
    train = scaler.fit_transform(train_df[columns])
    joblib.dump(scaler, 'scaler.pkl')
    test = scaler.transform(test_df[columns])
    oot1 = scaler.transform(oot1_df[columns])
    oot2 = scaler.transform(oot2_df[columns])

    train = pd.DataFrame(train, columns=columns)
    train[label] = train_df[label].values
    test = pd.DataFrame(test, columns=columns)
    test[label] = test_df[label].values
    oot1 = pd.DataFrame(oot1, columns=columns)
    oot1[label] = oot1_df[label].values
    oot2 = pd.DataFrame(oot2, columns=columns)
    oot2[label] = oot2_df[label].values

    constant_sparse_features = []
    for feat in constant_sparse_features:
        lbe = LabelEncoder()
        train[feat] = lbe.fit_transform(train[feat])
        test[feat] = lbe.transform(test[feat])
        oot1[feat] = lbe.transform(oot1[feat])
        oot2[feat] = lbe.transform(oot2[feat])

    constant_feature_columns = [SparseFeat(feat, train[feat].nunique(), embedding_dim=8)
                                for feat in constant_sparse_features] + \
                               [DenseFeat(feat, 1) for feat in constant_dense_features]

    behavior_feature_columns = [DenseFeat(feat, 6) for feat in behavior_dense_features]

    # 3.generate input data for model
    train_model_input = {}
    test_model_input = {}
    oot1_model_input = {}
    oot2_model_input = {}
    final_model_input = {}
    for feat in constant_dense_features:
        train_model_input[feat] = train[feat].values
        test_model_input[feat] = test[feat].values
        oot1_model_input[feat] = oot1[feat].values
        oot2_model_input[feat] = oot2[feat].values

    for feat in behavior_dense_features:
        behavior_list = [feat + '_at_p' + str(i) for i in range(150, -30, -30)]
        train_model_input[feat] = train[behavior_list].values
        test_model_input[feat] = test[behavior_list].values
        oot1_model_input[feat] = oot1[behavior_list].values
        oot2_model_input[feat] = oot2[behavior_list].values

    train_model_input['seq_length'] = np.array([6] * train.shape[0])
    test_model_input['seq_length'] = np.array([6] * test.shape[0])
    oot1_model_input['seq_length'] = np.array([6] * oot1.shape[0])
    oot2_model_input['seq_length'] = np.array([6] * oot2.shape[0])

    train_y = train[label].values
    test_y = test[label].values
    oot1_y = oot1[label].values
    oot2_y = oot2[label].values

    # 4.Define Model,train,predict and evaluate
    model = TimeSeries(constant_feature_columns, behavior_feature_columns, [],
                       dnn_hidden_units=[4, 4], dnn_dropout=0.6)

    checkpoint_dir = './ckt/model'
    try:
        model.load_weights(checkpoint_dir)
        print('Restoring from', checkpoint_dir)
    except:
        print('Creating a new model')
    model.compile("adam", "binary_crossentropy")
    # checkpoint = ModelCheckpoint(filepath=checkpoint_dir,
    #                              monitor='val_auc',
    #                              save_weights_only=True,
    #                              save_best_only=True,
    #                              mode='max')
    # earlystop = EarlyStopping(monitor='val_auc',
    #                           patience=5,
    #                           mode='max',
    #                           restore_best_weights=True)
    history = model.fit(train_model_input, train_y,
                        verbose=1, epochs=100, callbacks=[EarlyStoppingAtMinKS(10, checkpoint_dir)])
    # def proba_to_score(prob, pdo=60, rate=2, base_odds=35, base_score=750):
    #     factor = pdo / np.log(rate)
    #     offset = base_score - factor * np.log(base_odds)
    #     return factor * (np.log(1 - prob) - np.log(prob)) + offset
    #
    # model = load_model('final.h5', custom_objects)
    # predictions = model.predict(train_model_input)
    # auc_score = roc_auc_score(train_y, predictions)
    # fpr, tpr, _ = roc_curve(train_y, predictions)
    # ks = np.max(np.abs(tpr - fpr))
    # print(' train_auc {:4f} train_ks {:4f}'.format(auc_score, ks))
    # predictions = model.predict(oot1_model_input)
    # auc_score = roc_auc_score(oot1_y, predictions)
    # fpr, tpr, _ = roc_curve(oot1_y, predictions)
    # ks = np.max(np.abs(tpr - fpr))
    # print(' oot1_auc {:4f} oot1_ks {:4f}'.format(auc_score, ks))
    # predictions = model.predict(oot2_model_input)
    # auc_score = roc_auc_score(oot2_y, predictions)
    # fpr, tpr, _ = roc_curve(oot2_y, predictions)
    # ks = np.max(np.abs(tpr - fpr))
    # print(' oot2_auc {:4f} oot2_ks {:4f}'.format(auc_score, ks))