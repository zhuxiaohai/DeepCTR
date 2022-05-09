import numpy as np
import matplotlib.pyplot as plt

import optuna
from optuna.integration import TFKerasPruningCallback

import tensorflow as tf

from deepctr.callbacks import EarlyStopping


def train_val_score(train_score, val_score, w):
    output_scores = val_score - abs(train_score - val_score) * w
    return output_scores


class Objective(object):
    def __init__(self, build_model_fn, build_optimizer_instance,
                 metric, coef_train_val_disparity, early_stop_rounds,
                 build_model_param_dict, compile_param_dict,
                 optimization_direction, args, kwargs):
        """ objective will be called once in a trial
        dtrain: xgbdmatrix for train_val
        cv: int, number of stratified folds
        folds:  of list of k tuples (in ,out) for cv, where 'in' is a list of indices into dtrain
                    for training in the ith cv and 'out' is a list of indices into dtrain
                    for validation in the ith cv
                when folds is defined, cv will not be used
        metric: str, the metric to optimize
        optimization_direction: str, 'maximize' or 'minimize'
        feval: callable, custom metric fuction for monitor
        coef_train_val_disparity: float, coefficient for train_val balance
        tuning_param_dict: dict
        max_boosting_rounds: int, max number of trees in a trial of a set of parameters
        early_stop: int, number of rounds(trees) for early stopping
        random_state: int
        """
        self.coef_train_val_disparity = coef_train_val_disparity
        self.build_model_fn = build_model_fn
        self.build_optimizer_instance = build_optimizer_instance
        self.metric = metric
        self.early_stop_rounds = early_stop_rounds
        self.optimization_direction = optimization_direction
        self.build_model_param_dict = build_model_param_dict
        self.compile_param_dict = compile_param_dict
        self.fit_args = args
        self.fit_kwargs = kwargs
        self.best_model = None
        if self.optimization_direction == 'maximize':
            self.history_best = -np.inf
        else:
            self.history_best = np.inf

    def _compare_op(self, x, y):
        if self.optimization_direction == 'maximize':
            return x > y
        else:
            return x < y

    def __call__(self, trial):
        tf.keras.backend.clear_session()
        trial_param_dict = {}
        for key, param in self.build_model_param_dict.items():
            # dynamic
            if isinstance(param, tuple):
                # hidden units
                if not isinstance(param[0], str):
                    n_layers_param = param[0]
                    if isinstance(n_layers_param, tuple):
                        suggest_type = n_layers_param[0]
                        suggest_param = n_layers_param[1]
                        num_layers = eval('trial.suggest_' + suggest_type)(
                            'num_layers_*_{}'.format(key), **suggest_param)
                    else:
                        num_layers = n_layers_param
                    units_param = param[1]
                    hidden_units = []
                    for i in range(num_layers):
                        if isinstance(units_param, tuple):
                            suggest_type = units_param[0]
                            suggest_param = units_param[1]
                            hidden_units.append(eval('trial.suggest_' + suggest_type)(
                                'num_units_layer*{}_*_{}'.format(i, key), **suggest_param))
                        else:
                            hidden_units.append(units_param)
                    trial_param_dict[key] = hidden_units
                # the others
                else:
                    suggest_type = param[0]
                    suggest_param = param[1]
                    trial_param_dict[key] = eval('trial.suggest_' + suggest_type)(key, **suggest_param)
            # static
            else:
                trial_param_dict[key] = param
        model = self.build_model_fn(**trial_param_dict)

        trial_compile_dict = {}
        for key, param in self.compile_param_dict.items():
            if key == 'optimizer':
                trial_optimizer_dict = {}
                for optimizer_key, optimizer_param in param.items():
                    if isinstance(optimizer_param, tuple):
                        suggest_type = optimizer_param[0]
                        suggest_param = optimizer_param[1]
                        trial_optimizer_dict[optimizer_key] = eval('trial.suggest_' + suggest_type)(optimizer_key, **suggest_param)
                    else:
                        trial_optimizer_dict[optimizer_key] = optimizer_param
                trial_compile_dict[key] = self.build_optimizer_instance(**trial_optimizer_dict)
            else:
                trial_compile_dict[key] = param
        model.compile(**trial_compile_dict)

        early_stopping = EarlyStopping('val_'+self.metric,
                                         patience=self.early_stop_rounds,
                                         coef_of_balance=self.coef_train_val_disparity,
                                         persistence=False,
                                         direction=self.optimization_direction)
        pruning = TFKerasPruningCallback(trial, 'val_'+self.metric)
        history = model.fit(callbacks=[pruning, early_stopping], *self.fit_args, **self.fit_kwargs)
        cvresult = history.history
        best_epoch = early_stopping.best_epoch
        train_score = cvresult[self.metric][best_epoch]
        val_score = cvresult['val_' + self.metric][best_epoch]
        best_score = train_val_score(train_score, val_score, self.coef_train_val_disparity)
        if self._compare_op(best_score, self.history_best):
            self.history_best = best_score
            self.best_model = model
        trial.set_user_attr("n_epochs", len(cvresult[self.metric]))
        trial.set_user_attr("best_epochs", best_epoch + 1)
        trial.set_user_attr("train_score", train_score)
        trial.set_user_attr("val_score", val_score)
        return best_score


class OptunaSearchKeras(object):
    def __init__(self, build_model_param_dict, compile_param_dict,
               build_model_fn, build_optimizer_instance,
               eval_metric, coef_train_val_disparity=0.4, early_stop_rounds=10, optimization_direction='maximize',
               n_startup_trials=20, n_warmup_steps=20, interval_steps=1, pruning_percentile=75,
               maximum_time=60*10, n_trials=100, random_state=2, optuna_verbosity=1):
        self.build_model_param_dict = build_model_param_dict
        self.compile_param_dict = compile_param_dict
        self.build_model_fn = build_model_fn
        self.build_optimizer_instance = build_optimizer_instance
        self.eval_metric = eval_metric
        self.coef_train_val_disparity = coef_train_val_disparity
        self.early_stop_rounds = early_stop_rounds
        self.optimization_direction = optimization_direction
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.interval_steps = interval_steps
        self.pruning_percentile = pruning_percentile
        self.maximum_time = maximum_time
        self.n_trials = n_trials
        self.random_state = random_state
        self.optuna_verbosity = optuna_verbosity

        self.best_model = None
        self.study = None
        self.model_static_param = {}
        self.model_dynamic_param = {}
        self.compile_static_param = {}
        self.compile_dynamic_param = {}

    def get_params(self):
        """
        how to use the best params returned by this function:
        train_param = instance.get_params()
        model = xgb.train(train_param, train_dmatrix, num_boost_round=train_param['n_iterations'])
        test_probability_1d_array = model.predict(test_dmatrix)
        """
        if self.study:
            best_trial = self.study.best_trial
            best_param = best_trial.params
            collected_best_param = {}
            for key, value in best_param.items():
                if key.find('_*_') >= 0:
                    second_key, real_key = key.split('_*_')
                    if second_key.find('_layer*') >= 0:
                        name, layer_no = second_key.split('_layer*')
                        layer_no = int(layer_no)
                        if collected_best_param.get(real_key, None) is not None:
                            collected_best_param[real_key].append((layer_no, value))
                        else:
                            collected_best_param[real_key] = [(layer_no, value)]
                else:
                    collected_best_param[key] = value

            model_param = {}
            compile_param = {}
            for key, value in collected_best_param.items():
                if isinstance(value, list):
                    value.sort(key=lambda x: x[0])
                    value = [i[1] for i in value]
                if key in self.model_dynamic_param:
                    model_param[key] = value
                else:
                    compile_param[key] = value
            model_param.update(self.model_static_param)
            return model_param, compile_param, best_trial.user_attrs
        else:
            return None

    def plot_optimization(self):
        if self.study:
            return optuna.visualization.plot_optimization_history(self.study)

    def plot_score(self):
        if self.study:
            trial_df = self.study.trials_dataframe()
            _, ax1 = plt.subplots()
            ax1.plot(trial_df.index,
                     trial_df.user_attrs_train_score,
                     label='train')
            ax1.plot(trial_df.index,
                     trial_df.user_attrs_val_score,
                     label='val')
            plt.legend()
            plt.show()

    def plot_importance(self, names=None):
        if self.study:
            return optuna.visualization.plot_param_importances(self.study, params=names)

    def search(self, *args, **kwargs):
        for key, param in self.build_model_param_dict.items():
            if not isinstance(param, tuple):
                self.model_static_param[key] = param
            else:
                self.model_dynamic_param[key] = param
        for key, param in self.compile_param_dict.items():
            if not isinstance(param, dict):
                self.compile_static_param[key] = param
            else:
                self.compile_dynamic_param[key] = param

        objective = Objective(build_model_fn=self.build_model_fn,
                              build_optimizer_instance=self.build_optimizer_instance,
                              metric=self.eval_metric,
                              coef_train_val_disparity=self.coef_train_val_disparity,
                              early_stop_rounds=self.early_stop_rounds,
                              build_model_param_dict=self.build_model_param_dict,
                              compile_param_dict=self.compile_param_dict,
                              optimization_direction=self.optimization_direction,
                              args=args,
                              kwargs=kwargs)
        if self.optuna_verbosity == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        # prune a step(a boosting round) if it's worse than the bottom (1 - percentile) in history
        pruner = optuna.pruners.PercentilePruner(percentile=self.pruning_percentile,
                                                 n_warmup_steps=self.n_warmup_steps,
                                                 interval_steps=self.interval_steps,
                                                 n_startup_trials=self.n_startup_trials)
        study = optuna.create_study(direction=self.optimization_direction, sampler=sampler, pruner=pruner)
        study.optimize(objective, timeout=self.maximum_time, n_trials=self.n_trials, n_jobs=1)
        self.study = study
        self.best_model = objective.best_model
        print("Number of finished trials: ", len(study.trials))


if __name__ == '__main__':
    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.models import Sequential

    N_TRAIN_EXAMPLES = 3000
    N_VALID_EXAMPLES = 1000
    CLASSES = 10
    BATCHSIZE = 128


    class BuildModel(tf.keras.Model):
        def __init__(self, hidden_units, dropout, activation):
            super(BuildModel, self).__init__()
            print('new')
            n_layers = len(hidden_units)
            self.blocks = []
            self.dropout = []
            for i in range(n_layers):
                num_hidden = hidden_units[i]
                self.blocks.append(Dense(num_hidden,
                                         activation=activation,
                                         kernel_initializer=tf.initializers.glorot_normal(seed=10)))
                self.dropout.append(Dropout(rate=dropout))
            self.output_layer = Dense(CLASSES,
                                      activation="softmax",
                                      kernel_initializer=tf.initializers.glorot_normal(seed=10))

        def call(self, inputs, training=None, mask=None):
            out = inputs
            for layer, dropout in zip(self.blocks, self.dropout):
                out = layer(out)
                out = dropout(out, training=training)
            out = self.output_layer(out)
            return out

        def compute_output_shape(self, input_shape):
            shape = tf.TensorShape(input_shape).as_list()
            return shape[0], CLASSES

    def build_model(hidden_units, dropout, activation):
        n_layers = len(hidden_units)
        model = Sequential()
        for i in range(n_layers):
            num_hidden = hidden_units[i]
            model.add(Dense(num_hidden, activation=activation))
            model.add(Dropout(rate=dropout))
        model.add(Dense(CLASSES, activation="softmax"))
        return model

    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)[:N_TRAIN_EXAMPLES].astype("float32") / 255
    x_valid = x_valid.reshape(10000, 784)[:N_VALID_EXAMPLES].astype("float32") / 255

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train[:N_TRAIN_EXAMPLES], CLASSES)
    y_valid = keras.utils.to_categorical(y_valid[:N_VALID_EXAMPLES], CLASSES)

    build_model_param_dict = {'hidden_units': (('int', {'low': 1, 'high': 3}), ('int', {'low': 2, 'high': 6})),
                              'dropout': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                              'activation': 'relu',
                              }
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = tf.keras.metrics.CategoricalAccuracy()
    compile_param_dict = {'loss': loss,
                          'optimizer': {'lr': ('float', {'low': 1e-5, 'high': 1e-1, 'log': True})},
                          'metrics': metrics,
                          'run_eagerly': True
                          }
    op = OptunaSearchKeras(build_model_param_dict=build_model_param_dict,
                           compile_param_dict=compile_param_dict,
                           build_model_fn=BuildModel,
                           build_optimizer_instance=keras.optimizers.RMSprop,
                           eval_metric="categorical_accuracy",
                           coef_train_val_disparity=0.4,
                           optimization_direction='maximize',
                           early_stop_rounds=10,
                           optuna_verbosity=1,
                           n_startup_trials=1,
                           n_trials=10)

    op.search(x_train,
              y_train,
              validation_data=(x_valid, y_valid),
              batch_size=BATCHSIZE,
              verbose=1,
              epochs=100)

    train_param = op.get_params()
    print(train_param)
    print(op.study.best_trial.params)
    model = build_model(**train_param[0])
    model.compile(**op.compile_static_param, optimizer=keras.optimizers.RMSprop(**train_param[1]))
    model.fit(x_train,
              y_train,
              validation_data=(x_valid, y_valid),
              batch_size=BATCHSIZE,
              verbose=0,
              epochs=train_param[2]['best_epochs'])
    test_pred = model.predict(x_valid)
    op.plot_optimization().show()
    op.plot_importance().show()
    op.plot_score()


