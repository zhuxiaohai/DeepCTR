# -*- coding:utf-8 -*-
"""
Author:
    Xiaohai Zhu(zhuxiaohai_sast@163.com)

Reference:
    [1] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, Andrew Rabinovich.
    GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.
    Proceedings of the 35 th International Conference on Machine Learning.
"""

import platform

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.metrics import MeanSquaredError
from tensorflow.python.keras.callbacks import TensorBoard

from deepctr.layers.core import DNN
from deepctr.layers.utils import combined_dnn_input
from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.models.multitask.multitaskbase import MultiTaskModelBase
from deepctr.models.multitask.call_backs import MyRecorder
from utils import get_multitask_test_data


def toy_gradnorm_model(num_tasks, feature_columns, output_dim, seed=1024):
    features = build_input_features(feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, feature_columns,
                                                                         l2_reg=0.0, seed=seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_output = DNN([100, 100], name='bottom_shared_dnn', seed=seed)(dnn_input)

    outputs = {}
    for i in range(num_tasks):
        logit = tf.keras.layers.Dense(output_dim, use_bias=True, activation=None)(dnn_output)
        outputs['task_' + str(i)] = logit

    model = MultiTaskModelBase(inputs=inputs_list, outputs=outputs)

    return model


if __name__ == '__main__':
    # configure
    project_name = 'test_gradnorm'
    run_name = 'gradnorm_1_100'
    if platform.system() == 'Windows':
        joint_symbol = '\\'
    else:
        joint_symbol = '/'
    checkpoint_dir = joint_symbol.join([project_name, 'ckt', run_name])
    tensorboard_dir = joint_symbol.join([project_name, 'log_dir', run_name])
    summary_dir = joint_symbol.join([project_name, 'metrics', run_name])
    sample_size = 10000
    output_dim = 1
    num_tasks = 2
    gradnorm = True
    uncertainty = True
    loss_fns = {'task_0': keras.losses.mean_squared_error,
                'task_1': keras.losses.mean_squared_error}
    metrics_logger = {'task_0': MeanSquaredError,
                      'task_1': MeanSquaredError}
    loss_weights = {'task_0': 1, 'task_1': 1}

    model_input, y_train, model_test_input, y_test, feature_columns = get_multitask_test_data(sample_size,
                                                                                              outout_dim=output_dim)
    model = toy_gradnorm_model(num_tasks, feature_columns, output_dim)
    if gradnorm:
        last_shared_weights = [weight for weight in model.get_layer('bottom_shared_dnn').trainable_weights
                               if weight.name.find('kernel1') >= 0]
        gradnorm_config = {'alpha': 0.2, 'last_shared_weights': last_shared_weights}
    else:
        gradnorm_config = None
    last_lr = 0.001
    optimizers = keras.optimizers.Adam(learning_rate=last_lr)
    model.compile(optimizers=optimizers,
                  loss_fns=loss_fns,
                  metrics_logger=metrics_logger,
                  loss_weights=loss_weights,
                  uncertainly=uncertainty,
                  gradnorm_config=gradnorm_config)
    last_epoch = 0
    print('last epoch', last_epoch)
    print('last lr', last_lr)
    history = model.fit(model_input,
                        y_train,
                        batch_size=256,
                        epochs=300,
                        initial_epoch=last_epoch,
                        verbose=2,
                        validation_data=(model_test_input, y_test),
                        callbacks=[
                            TensorBoard(log_dir=tensorboard_dir),
                            MyRecorder(log_dir=tensorboard_dir,
                                       data=(model_test_input, y_test))
                        ]
                        )




