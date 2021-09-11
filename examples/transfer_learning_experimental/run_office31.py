import tensorflow as tf
from tensorflow.python.keras import layers
from deepctr.call_backs import ModifiedExponentialDecay
from deepctr.models.transferlearning.domain_adaptation import DomainAdaptation
from deepctr.models.transferlearning.transferloss import DomainAdversarialLoss
from deepctr.models.multitask.composite_optimizer import CompositeOptimizer

IMAGE_SIZE = (256, 256)
CROP_SIZE = 224
BATCH_SIZE = 2
AUTO = 2
NUM_CLASSES = 31
last_lr = 0.01
train_iter_num = 3
epochs = 4
max_iter_num = epochs * train_iter_num

source_dir = "../data/office31/amazon"
target_dir = "../data/office31/webcam"

train_aug = tf.keras.Sequential(
    [
        layers.RandomCrop(CROP_SIZE, CROP_SIZE, seed=1024),
        layers.RandomFlip("horizontal"),
        layers.Rescaling(1./255),
        layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
    ]
)

test_aug = tf.keras.Sequential(
    [
        layers.Resizing(CROP_SIZE, CROP_SIZE),
        layers.Rescaling(1./255),
        layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
    ]
)

source_ds = tf.keras.preprocessing.image_dataset_from_directory(
    source_dir,
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

source_ds_train = (
    source_ds
    .shuffle(BATCH_SIZE * 100)
    .repeat()
    .map(lambda x, y: (train_aug(x, training=True), y))
)

target_ds = tf.keras.preprocessing.image_dataset_from_directory(
    target_dir,
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

target_ds_train = (
    target_ds
    .shuffle(BATCH_SIZE * 100)
    .repeat()
    .map(lambda x, y: (train_aug(x, training=True), y))
)


target_ds_val = (
    target_ds
    .map(lambda x, y: (None, (test_aug(x, training=False), y)))
)

train_ds = tf.data.Dataset.zip((source_ds_train.take(train_iter_num), target_ds_train.take(train_iter_num)))

raw_model = tf.keras.applications.ResNet50(
                                        include_top=True,
                                        weights="imagenet",
                                        classes=1000,
                                       )
raw_model_base = tf.keras.Model(inputs=raw_model.input, outputs=raw_model.get_layer('avg_pool').output)
feature_extractor = tf.keras.Sequential(
    [
    raw_model_base,
    layers.Dense(256),
    layers.Activation('relu')
    ])
out = feature_extractor(raw_model.input)
classifier = layers.Dense(NUM_CLASSES)
logit = classifier(out)
model = tf.keras.Model(inputs=raw_model.input, outputs=logit)

optimizer1 = tf.keras.optimizers.SGD(momentum=0.9, decay=0.001,
                                    learning_rate=ModifiedExponentialDecay(0.1*last_lr, max_iter_num=max_iter_num))
optimizer2 = tf.keras.optimizers.SGD(momentum=0.9, decay=0.001,
                                    learning_rate=ModifiedExponentialDecay(last_lr, max_iter_num=max_iter_num))
composite_optimizer = CompositeOptimizer([
    (optimizer1, lambda: feature_extractor.trainable_variables),
    (optimizer2, lambda: classifier.trainable_variables)])

model.compile(optimizer=composite_optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.SparseCategoricalAccuracy())

da_loss = DomainAdversarialLoss(max_iter_grl=max_iter_num, dnn_units=[256, 256], use_bn=True)
dann = DomainAdaptation(feature_extractor, model)
dann.compile(da_loss=da_loss,
             optimizer_da_loss=tf.keras.optimizers.SGD(momentum=0.9, decay=0.001,
                                    learning_rate=ModifiedExponentialDecay(last_lr, max_iter_num=max_iter_num)))
dann.fit(train_ds,
         validation_data=target_ds_val,
         epochs=epochs,
         )

