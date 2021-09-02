import os
import math
import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.metrics import AUC, Mean
from tensorflow.python.keras.models import load_model

from deepctr.models.transferlearning.domain_adaptation import DomainAdaptation
from deepctr.models.transferlearning.transferloss import DomainAdversarialLoss, LMMDLoss, MMDLoss
from deepctr.models.transferlearning.utils import plot_tsne_source_target, proxy_a_distance
from deepctr.models.multitask.multitaskbase import MultiTaskModelBase
from deepctr.models.multitask.call_backs import MyEarlyStopping, ModifiedExponentialDecay
from deepctr.layers import custom_objects
from deepctr.layers.utils import NoMask


custom_objects['NoMask'] = NoMask
custom_objects['MultiTaskModelBase'] = MultiTaskModelBase
custom_objects['Mean'] = Mean
custom_objects['AUC'] = AUC
custom_objects['ModifiedExponentialDecay'] = ModifiedExponentialDecay

project_name = 'k3dq'
run_name = 'toy_lmmd_da'
mode = 'test'
joint_symbol = '/'
checkpoint_dir = joint_symbol.join([project_name, 'ckt', run_name])
tensorboard_dir = joint_symbol.join([project_name, 'log_dir', run_name])
summary_dir = joint_symbol.join([project_name, 'metrics', run_name])
trend_dir = joint_symbol.join([project_name, 'trend', run_name])
epochs = 1
batch_size = 8

source_x, source_y = make_blobs(300, centers=[[0, 0], [0, 1]], cluster_std=0.2, random_state=0)
target_x, target_y = make_blobs(300, centers=[[1, -1], [1, 0]], cluster_std=0.2, random_state=0)
source_y2 = source_y[:, np.newaxis].astype(int)
target_y2 = target_y[:, np.newaxis].astype(int)

if run_name.find('_lmmd') >= 0:
    enc = OneHotEncoder(handle_unknown='ignore')
    source_y2 = enc.fit_transform(source_y2).toarray().astype(np.float32)
    target_y2 = enc.fit_transform(target_y2).toarray().astype(np.float32)

train_source_batch_num = math.ceil(len(source_x) / batch_size)
train_target_batch_num = math.ceil(len(target_x) / batch_size)
train_iter_num = max(train_source_batch_num, train_target_batch_num)
train_iter_num = 10000
max_iter_num = epochs * train_iter_num

source_dataset = tf.data.Dataset.from_tensor_slices((source_x, source_y2))
target_dataset = tf.data.Dataset.from_tensor_slices((target_x, target_y2))
source_dataset = source_dataset.shuffle(buffer_size=300, reshuffle_each_iteration=True, seed=0).repeat().batch(batch_size).take(train_iter_num)
target_dataset = target_dataset.shuffle(buffer_size=300, reshuffle_each_iteration=True, seed=0).repeat().batch(batch_size).take(train_iter_num)
x = tf.data.Dataset.zip((source_dataset, target_dataset))

if run_name.find('_da') >= 0:
    val_dataset = tf.data.Dataset.from_tensor_slices(((source_x, source_y2), (target_x, target_y2)))
else:
    val_dataset = tf.data.Dataset.from_tensor_slices((target_x, target_y2))
val_dataset = val_dataset.batch(2 * batch_size)

source_df = pd.DataFrame(source_x, columns=['x', 'y'])
source_df.loc[:, 'label'] = source_y
source_df.loc[:, 'set'] = '1source'
target_df = pd.DataFrame(target_x, columns=['x', 'y'])
target_df.loc[:, 'label'] = target_y
target_df.loc[:, 'set'] = '2target'
data = pd.concat([source_df, target_df], axis=0)

if mode == 'train':
    if run_name.find('_lmmd') >= 0:
        model = tf.keras.Sequential([tf.keras.layers.Input(2),
                                     tf.keras.layers.Dense(15, name='bottom_shared_dnn'),
                                     tf.keras.layers.Activation('relu'),
                                     tf.keras.layers.Dense(2),
                                     tf.keras.layers.Activation('softmax')])
    else:
        model = tf.keras.Sequential([tf.keras.layers.Input(2),
                                     tf.keras.layers.Dense(15, name='bottom_shared_dnn'),
                                     tf.keras.layers.Activation('relu'),
                                     tf.keras.layers.Dense(1),
                                     tf.keras.layers.Activation('sigmoid')])

    last_lr = 0.001
    optimizer = keras.optimizers.Adam(learning_rate=ModifiedExponentialDecay(last_lr, max_iter_num=max_iter_num))
    if run_name.find('_lmmd') >= 0:
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=tf.keras.metrics.categorical_accuracy)
    else:
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=tf.keras.metrics.binary_accuracy)

    if run_name.find('_da') >= 0:
        feature_extractor = keras.Model(inputs=model.input, outputs=model.get_layer('bottom_shared_dnn').output)
        if run_name.find('_lmmd') >= 0:
            da_loss = LMMDLoss(2, max_iter_num=max_iter_num)
        elif run_name.find('_mmd') >= 0:
            da_loss = MMDLoss()
        else:
            da_loss = DomainAdversarialLoss(max_iter_grl=max_iter_num)
        dann = DomainAdaptation(feature_extractor, model, da_loss)
        dann.compile(
                     da_loss=da_loss,
                     optimizer_da_loss=tf.keras.optimizers.Adam(learning_rate=ModifiedExponentialDecay(last_lr, max_iter_num=max_iter_num)))
        if run_name.find('_lmmd') >= 0:
            metric = 'val_categorical_accuracy'
        else:
            metric = 'val_binary_accuracy'
        dann.fit(x,
                 validation_data=val_dataset,
                 epochs=epochs,
                 callbacks=[MyEarlyStopping(metric,
                                            patience=10,
                                            savepath=checkpoint_dir,
                                            coef_of_balance=0,
                                            direction='maximize')
                            ]
                 )
        pred = model.predict(source_x)
        score = accuracy_score(source_y, (pred[:, 0] > 0.5).astype(int))
        print(score)
    else:
        if run_name.find('_lmmd') >= 0:
            metric = 'val_categorical_accuracy'
        else:
            metric = 'val_binary_accuracy'
        history = model.fit(source_dataset,
                            epochs=epochs,
                            validation_data=val_dataset,
                            callbacks=[MyEarlyStopping(metric,
                                                       patience=10,
                                                       savepath=checkpoint_dir,
                                                       coef_of_balance=0,
                                                       direction='maximize')
                                         ]
                            )
        pred = model.predict(source_x)
        score = accuracy_score(source_y, (pred[:, 0] > 0.5).astype(int))
        print(score)
    pred = model.predict(source_x)
    score = accuracy_score(source_y, (pred[:, 0] > 0.5).astype(int))
    print(score)

else:
    if not os.path.exists(trend_dir):
        os.makedirs(trend_dir)
    best_metric = -1
    best_model = None
    for i in os.listdir(checkpoint_dir):
        if i.find('best_') >= 0:
            best_model = i
            break
    print('loading ', joint_symbol.join([checkpoint_dir, best_model]))
    model = load_model(joint_symbol.join([checkpoint_dir, best_model]), custom_objects=custom_objects)
    pred = model.predict(source_x)
    score = accuracy_score(source_y, (pred[:, 0] > 0.5).astype(int))
    print(score)
    file_writer = tf.summary.create_file_writer(summary_dir)

    fig = plt.figure(figsize=(8, 10))
    fig.suptitle(run_name)
    for index, set_name in enumerate(['1source', '2target']):
        set_data = data[data['set'] == set_name]
        predictions = model.predict(set_data[['x', 'y']].values)
        if run_name.find('_lmmd') >= 0:
            y_pred = np.argmax(predictions, 1)
        else:
            y_pred = (predictions[:, 0] > 0.5).astype(int)
        score = accuracy_score(set_data['label'].values, y_pred)
        print(set_name, 'accuracy', score)
        with file_writer.as_default():
            tf.summary.scalar('accuracy', score, step=index+1)

    F = keras.Model(inputs=model.input, outputs=model.get_layer('bottom_shared_dnn').output)
    source_data = data[data['set'] == '1source']
    print('source shape ', source_data.shape)
    for index, set_name in enumerate(['2target']):
        set_data = data[data['set'] == set_name]
        print(set_name, 'shape ', set_data.shape)
        source_x = F.predict(source_data[['x', 'y']].values)
        target_x = F.predict(set_data[['x', 'y']].values)
        a_score = proxy_a_distance(source_x, target_x)
        print(set_name, 'a_score', a_score)
        with file_writer.as_default():
            tf.summary.scalar('a_score', a_score, step=index+1)
        for i in range(2):
            print(set_name, i, ' shape ', source_data[source_data['label'] == i].shape)
            source_x = F.predict(source_data[source_data['label'] == i][['x', 'y']].values)
            target_x = F.predict(set_data[set_data['label'] == i][['x', 'y']].values)
            a_score = proxy_a_distance(source_x, target_x)
            print(set_name, 'a_score_{}'.format(i), a_score)
            with file_writer.as_default():
                tf.summary.scalar('a_score_{}'.format(i), a_score, step=index + 1)

        emb_s = F.predict(source_data[['x', 'y']].values)
        emb_t = F.predict(set_data[['x', 'y']].values)
        plot_tsne_source_target(emb_s, source_data['label'].values, emb_t, set_data['label'].values,
                                path_name=joint_symbol.join([trend_dir, set_name]))