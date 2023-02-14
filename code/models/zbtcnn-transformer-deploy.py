import os
import sys
import random
import time
from collections import deque
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Activation, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import optimizers
from tensorflow.keras import utils
import keras_tuner as kt
import joblib
from pushbullet import Pushbullet

# CD to top level git directory
if ".git" not in os.listdir("."):
    os.chdir("../../")

# Check GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.config.list_physical_devices()


# %%
import gzip
compressed_bstr = b"\x1f\x8b\x08\x00\xdf\xd7\x7e\x62\x02\xff\xa5\x55\x4b\x8f\xdb\x20\x10\x3e\x67\x7f\x05\xe5\x04\x92\x13\xed\xa3\x2f\x45\xbb\xab\x5e\xaa\xaa\x97\x6a\x2f\x3d\x59\x16\x22\x06\xa7\x53\x61\xec\x00\xde\xcd\xb6\xea\x7f\xef\x80\x13\x39\x91\x9d\x55\xba\xbd\xd8\x30\x7c\xf3\xcd\x83\x99\x01\xea\xb6\x71\x81\xf8\x6e\xd5\xba\xa6\xd4\xde\x13\xe9\x89\x6f\x2f\x4a\x23\x71\xfd\xe5\xe1\xfb\xf2\x62\xa6\x74\x45\x84\x00\x0b\x41\x08\xe6\xb5\xa9\x38\x0a\x67\x71\xb1\x10\x42\xc9\x20\x85\x20\x77\xe4\x5b\x63\xf5\x1e\xbb\xd6\xa1\x97\x0f\x70\xa7\x43\xe7\x2c\x52\x2f\x5c\x67\x19\xa3\xf6\x11\x14\xc8\xb9\xaf\x81\x66\x84\xce\x37\xe9\xbb\xa5\x3c\x23\x3e\xa8\xa6\x0b\x77\x88\x7c\xf8\xfa\xf0\x99\x2f\xfa\xfd\x42\xe9\xb2\x51\x9a\xd1\x2e\x54\xf3\x8f\x14\xc5\xad\x81\x60\xc0\x6a\xcf\xf8\xe0\x63\x14\x08\x0c\xa5\xd5\x2e\x3c\xef\xec\x67\x44\x85\xe7\x56\x67\xa4\xb5\xb2\xd6\xc9\x1b\xa8\xc8\x91\xff\x51\x36\x8b\x4b\x0c\xe4\xe8\x00\xe5\xda\x78\x3d\x71\x3e\xc4\xc8\x87\xf0\x72\x96\x4c\x31\xe8\xdd\x63\xf4\x9e\xf2\xfc\xaa\xd8\xef\x6e\x71\x77\xb9\xdf\xc5\x25\xcf\xce\x43\xce\xaf\x0a\x4e\xaa\xc6\x11\x20\x60\x49\x72\x04\x43\x48\xf1\x44\x01\x14\x08\xbe\x98\x7d\xda\x07\xde\xe7\x23\x9e\x8e\x2e\x20\x7f\xd1\xe0\x60\x63\x1c\x65\xb4\x48\x6f\xd1\x84\xea\xca\x20\x22\xf9\x3d\x3d\x6d\xbc\xd6\xf5\x60\xfb\xa5\xcc\x85\x26\x48\x93\x91\xce\x6b\x95\x91\xca\x69\x8d\xc0\x9c\x81\x0d\x2c\x82\x72\x28\xce\x4e\xe5\xbf\xe0\x8f\xf2\xe9\xa4\x5d\x6b\x66\xb4\x4d\x26\xf9\x2e\xd2\x6a\x25\x30\x88\xc6\x3d\x8b\xce\xcb\xf5\x2e\xd8\xde\x06\x5e\x07\x41\xe5\x97\x31\xd7\x67\x60\x6e\x8a\xa2\xbf\x1a\x0c\x3a\x15\x0e\xc5\x3e\x60\xcd\xea\xa7\x2e\x43\x86\xbd\xf0\x9b\xa6\x74\x41\xd0\xb5\x10\x74\x49\x8c\xac\x57\x4a\x12\x9f\xc1\x12\x93\x34\x4a\x1d\xcf\x73\x9a\x84\xb1\x9b\xa2\x38\xfe\xe3\x01\x2d\x16\x60\x95\xde\xb2\x0d\x2f\x62\x74\xe0\xc1\xfa\x20\x6d\xa9\xd9\x26\x76\x9c\xe3\x24\x96\x39\x99\xa2\xdc\xf4\x55\xb1\x89\x6e\x63\x7a\xe3\x6d\x23\xab\x10\x4e\xb7\xee\xc8\xa9\xe5\x94\xfa\x62\x0f\x64\xfc\x0f\x3f\x6c\x15\xfc\x8d\x8a\x26\x25\x68\x3c\x33\xfa\xb2\x19\x37\x37\x16\x09\x7a\x72\xbb\x6e\x3b\xd1\x05\x30\x78\xe1\x23\x46\x4c\x5c\xfb\x2a\xc2\xa8\x38\x45\xd8\x36\x4f\xda\x9d\xcf\x58\x99\x46\x26\xce\xa4\x27\x94\x93\x4f\x53\xac\xb5\xdc\xfe\x1f\xb1\x81\x1a\xc2\x14\x73\xd9\x29\x29\x1e\xb5\xf3\xd0\xd8\xf3\xd9\xb1\x22\x22\xf7\xa1\xf2\x7d\x6a\xa0\x11\xbf\x72\x80\x80\x57\x5b\x38\x56\x3f\x61\x43\x1a\x73\xfa\xdd\x99\x1c\x2a\x32\x04\xe7\xe3\x20\xd9\xa6\xda\xdd\xa6\x96\x83\x5d\x7a\x63\x03\xd8\x26\x90\x2d\x3e\x2d\xd2\x05\xff\x04\xe1\x07\xc3\x82\xa6\x9c\x48\xab\x10\xfc\xe6\x8e\x50\xb4\x49\x8b\x43\xa6\xb4\xc8\xe3\x5c\x02\x7e\x30\x27\xe9\xdb\x9b\x0f\xef\xdf\x5d\x5f\x5e\xd1\xa1\x93\x15\x94\x81\xfd\x82\x96\x25\x9d\x8c\xe4\x83\x8f\x51\x02\xab\x2e\x68\x74\x54\xf6\x23\x48\x46\x9a\x84\x2c\x38\x3f\xf9\xb0\x1e\xb6\x0d\xd6\x27\xca\xf1\x71\x66\xfc\x2f\x17\xd9\x71\x54\xc0\x07\x00\x00"
exec(gzip.decompress(compressed_bstr))

print(gpu.name)


# %%
# Data parameters
SEQ_LEN = 48 #hours
FUTURE_PERIOD_PREDICT = 1 #hours

# Hyperparameter Optimizer parameters
MAX_TRIALS = 100 # 10t x 20e ~ 4h

# Model parameters
EPOCHS = 20
BATCH_SIZE = 16

NAME = "TRANSFORMER-01"


# %%
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


# %%
def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="linear")(x)
    return keras.Model(inputs, outputs)


# %%
def build_model_hp(hp):
    kwargs = {
        "input_shape": train_x.shape[1:],
        "head_size": hp.Int("head_size", min_value=32, max_value=1024, step=32),
        "num_heads": hp.Int("num_heads", min_value=2, max_value=10, step=1),
        "ff_dim": 4,
        "num_transformer_blocks": hp.Int("num_transformer_blocks", min_value=2, max_value=10, step=1),
        "mlp_units": [hp.Int(f"mlp_units_{i}", min_value=32, max_value=512, step=32) for i in range(hp.Int("mlp_count", min_value=1, max_value=4, step=1))],
        "mlp_dropout": hp.Float("mlp_dropout", min_value=0.1, max_value=0.7, step=0.1),
        "dropout": hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1),
    }

    model = build_model(**kwargs)

    model.compile(
        loss="mape",
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate", values=[.001, .0001, .00001, .000001])),
        metrics=["mse"],
    )

    return model


# %%
## Load Data

train_x = joblib.load("data/pickle/train_x.dump")
train_y = joblib.load("data/pickle/train_y.dump")
validation_x = joblib.load("data/pickle/validation_x.dump")
validation_y = joblib.load("data/pickle/validation_y.dump")


# %%
# Initialize Tuner

tuner = kt.BayesianOptimization(
                    max_trials=MAX_TRIALS,
                    hypermodel=build_model_hp,
                    objective="val_loss",
                    overwrite=False,
                    directory=f"tuners/{NAME}",
                    project_name=f"{NAME}",)

tuner.search_space_summary(extended=True)

callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]


# %%
with tf.device("/device:GPU:0"):
    tuner.search(train_x, train_y, 
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                validation_data=(validation_x, validation_y))


# %%

best_model = tuner.get_best_models(num_models=1)[0]
tf.keras.models.save_model(best_model, f"models/{NAME}.model")

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
with open("models/{NAME}.json") as outfd:
    json.dump(best_hp.values, outfd)


# %%

pb = Pushbullet("o.nyntgspLep97yl0oPDbp0nAbMIDUGiO5")
push = pb.push_note(f"{os.uname()[1]} @ {time.asctime()}", "ZBTCNNT Completed")


