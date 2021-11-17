import numpy as np
import tensorflow as tf

from deepctr.feature_column import DenseFeat

from sklearn.model_selection import train_test_split


def get_multitask_test_data(sample_size=10000, sigmas=[1, 100], dense_feature_num=250, outout_dim=100, num_tasks=2):
    feature_columns = []
    model_input = {}
    model_test_input = {}
    B = np.random.normal(scale=10, size=(dense_feature_num, outout_dim)).astype(np.float32)
    sigmas = np.array(sigmas).astype(np.float32)
    epsilons = np.random.normal(scale=3.5, size=(num_tasks, dense_feature_num, outout_dim)).astype(np.float32)
    x = np.random.uniform(-1, 1, size=(sample_size, dense_feature_num)).astype(np.float32)
    x = x / np.linalg.norm(x)
    X_train, X_test = train_test_split(x, test_size=0.3, random_state=42)

    for i in range(dense_feature_num):
        feature_columns.append(
            DenseFeat(
                'dense_feature_' + str(i),
                1,
                dtype=tf.float32,
            )
        )
        model_input['dense_feature_' + str(i)] = X_train[:, i]
        model_test_input['dense_feature_' + str(i)] = X_test[:, i]

    y_train = {}
    y_test = {}
    for i in range(num_tasks):
        # eq (3) on the paper:
        # each target is $\sigma_i \tanh((B + \epsilon_i)) \mathbf{x}) $
        y_train['task_' + str(i)] = sigmas[i] * np.tanh(X_train.dot(B + epsilons[i]))
        y_test['task_' + str(i)] = sigmas[i] * np.tanh(X_test.dot(B + epsilons[i]))

    return model_input, y_train, model_test_input, y_test, feature_columns
