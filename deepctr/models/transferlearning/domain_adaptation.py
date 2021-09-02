"""
Author:
    zhuxiaohai_sast@163.com
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine import data_adapter
from types import MethodType

class DomainAdaptation(Model):
    def __init__(self,
                 feature_extractor,
                 main_model,
                 *args, **kwargs):
        super(DomainAdaptation, self).__init__(*args, **kwargs)
        self.feature_extractor = feature_extractor
        self.main_model = main_model
        self.build(self.main_model.input_shape)

    def build(self, input_shape):
        super(DomainAdaptation, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        return self.main_model(inputs, training=training)

    def compile(self,
                da_loss,
                optimizer_da_loss,
                **kwargs):
        super(DomainAdaptation, self).compile(**kwargs)

        # main model
        # self.loss_fn_main_model = loss_fn_main_model
        # self.metrics_main_model = metrics_main_model
        self.optimizer = self.main_model.optimizer
        # self.loss_logger_main_model = Mean("main_loss", dtype=tf.float32)

        # domain adaptation
        self.da_loss = da_loss
        self.optimizer_da_loss = optimizer_da_loss

    def train_step(self, data):
        source, target = data[0], data[1]
        batch_source_data, batch_source_label, batch_source_weight = data_adapter.unpack_x_y_sample_weight(source)
        batch_target_data, batch_target_label, batch_target_weight = data_adapter.unpack_x_y_sample_weight(target)
        if len(data) > 2:
            batch_main_model_data, batch_main_model_label, batch_main_model_weight = \
                data_adapter.unpack_x_y_sample_weight(data[2])
        else:
            batch_main_model_data, batch_main_model_label, batch_main_model_weight = \
                batch_source_data, batch_source_label, batch_source_weight

        with tf.GradientTape(persistent=True) as tape:
            # main_model
            batch_main_model_pred = self.main_model(batch_main_model_data, training=True)
            try:
                loss_main = self.main_model._compiled_loss(batch_main_model_label, batch_main_model_pred, batch_main_model_weight,
                                                           regularization_losses=self.main_model.losses)
            except:
                loss_main = self.main_model.compiled_loss(batch_main_model_label, batch_main_model_pred, batch_main_model_weight,
                                                          regularization_losses=self.main_model.losses)

            # map to the space for alignment
            kwargs = {}
            if self.da_loss.conditional_capability:
                kwargs['source_label'] = batch_source_label
                kwargs['target_logits'] = self.main_model(batch_target_data, training=True)
            batch_source_data = self.feature_extractor(batch_source_data, training=True)
            batch_target_data = self.feature_extractor(batch_target_data, training=True)
            # compute alignment loss of transfer learning
            loss_da = self.da_loss(batch_source_data, batch_target_data, training=True, **kwargs)

            loss = loss_main + loss_da + sum(self.da_loss.losses)

        vars = self.main_model.trainable_variables
        grads = tape.gradient(loss, vars)
        self.optimizer.apply_gradients(zip(grads, vars))

        if len(self.da_loss.trainable_variables) > 0:
            vars = self.da_loss.trainable_variables
            grads = tape.gradient(loss, vars)
            self.optimizer_da_loss.apply_gradients(zip(grads, vars))

        try:
            self.main_model.compiled_metrics.update_state(batch_main_model_label, batch_main_model_pred, batch_main_model_weight)
        except:
            pass

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        source, target = data[0], data[1]
        if source is not None:
            batch_source_data, batch_source_label, batch_source_weight = data_adapter.unpack_x_y_sample_weight(source)
        batch_target_data, batch_target_label, batch_target_weight = data_adapter.unpack_x_y_sample_weight(target)
        if len(data) > 2:
            batch_main_model_data, batch_main_model_label, batch_main_model_weight = \
                data_adapter.unpack_x_y_sample_weight(data[2])
        else:
            batch_main_model_data, batch_main_model_label, batch_main_model_weight = \
                batch_target_data, batch_target_label, batch_target_weight

        # main_model forward
        batch_main_model_pred = self.main_model(batch_main_model_data, training=False)
        try:
            self.main_model._compiled_loss(batch_main_model_label, batch_main_model_pred, batch_main_model_weight,
                                          regularization_losses=self.main_model.losses)
        except:
            self.main_model.compiled_loss(batch_main_model_label, batch_main_model_pred, batch_main_model_weight,
                                           regularization_losses=self.main_model.losses)

        # da_model forward
        kwargs = {}
        if self.da_loss.conditional_capability:
            kwargs['source_label'] = batch_source_label
            kwargs['target_logits'] = self.main_model(batch_target_data, training=True)
        if source is not None:
            batch_source_data = self.feature_extractor(batch_source_data, training=False)
        else:
            batch_source_data = None
        batch_target_data = self.feature_extractor(batch_target_data, training=False)
        self.da_loss(batch_source_data, batch_target_data, training=False, **kwargs)

        try:
            self.main_model.compiled_metrics.update_state(batch_main_model_label, batch_main_model_pred, batch_main_model_weight)
        except:
            pass

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        return self(x, training=False)

    @property
    def metrics(self):
        return self.main_model.metrics + self.da_loss.metrics