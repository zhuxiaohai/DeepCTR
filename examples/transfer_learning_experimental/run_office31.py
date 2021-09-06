import tensorflow as tf
from tensorflow.python.keras import layers

IMAGE_SIZE = (256, 256)
CROP_SIZE = 224
BATCH_SIZE = 256
AUTO = 2

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
    .batch(BATCH_SIZE)
    .map(lambda x, y: (train_aug(x, training=True), y), num_parallel_calls=AUTO)
    .prefetch(AUTO)
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
    .batch(BATCH_SIZE)
    .map(lambda x, y: (train_aug(x, training=True), y), num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

model = tf.keras.applications.ResNet50(
                                        include_top=True,
                                        weights="imagenet",
                                        input_tensor=None,
                                        input_shape=None,
                                        pooling=None,
                                        classes=1000,
                                       )

