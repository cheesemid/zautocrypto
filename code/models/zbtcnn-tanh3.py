#!/usr/bin/env python3


# ## ZBTCNN TANH 3
# Predicts price increase  
# Uses technicals aswell from csv_formatter.ipynb
# 
# Creates model using Bayesian Hyperparameter Optimization

import os
import sys
import random
import time
from collections import deque
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Activation, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import optimizers
from tensorflow.keras import utils
import keras_tuner as kt
import json
import joblib
import os

# Check GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.config.list_physical_devices()



# Data parameters
SEQ_LEN = 48 #hours
FUTURE_PERIOD_PREDICT = 1 #hours

# Hyperparameter Optimizer parameters
MAX_TRIALS = 40 # 10t x 20e ~ 4h

# Model parameters
EPOCHS = 20
BATCH_SIZE = 16
# NAME = f"TANH3-e{EPOCHS}-b{BATCH_SIZE}-s{SEQ_LEN}-fpp{FUTURE_PERIOD_PREDICT}-{int(time.time())}"
NAME = "TANH3-e20-b16-s48-fpp1-1645639574"



## Import data
# DATA MUST BE FORMATTED USING CSV_FORMATTER.IPYNB

csv_file = "data/formatted/BTCUSDT-1h-data.csv"

data = pd.read_csv(csv_file, skiprows=[0], names=["timestamp", "open", "high", "low", "close", "volume", "rsi", "ema"])

data.set_index("timestamp", inplace=True)

# data.head()



## Add min max bounds to data

price_max = 80000.0
volume_max = 60000.0
rsi_max = 100.0

price_min = 2000.0
volume_min = 0.0
rsi_min = 0.0

max_df = pd.DataFrame()

max_df["timestamp"] = []

for col in data.columns:
    max_df[col] = []

max_df = max_df.append({"timestamp": str(int(time.time())),
                "open": price_max,
                "high": price_max,
                "low": price_max,
                "close": price_max,
                "volume": volume_max,
                "rsi": rsi_max,
                "ema": price_max,
                "target": price_max}, ignore_index=True)

max_df = max_df.append({"timestamp": str(int(time.time())),
                "open": price_min,
                "high": price_min,
                "low": price_min,
                "close": price_min,
                "volume": volume_min,
                "rsi": rsi_min,
                "ema": price_min,
                "target": price_min}, ignore_index=True)

max_df.set_index("timestamp", inplace=True)

max_df.head()



## Formatting data and Scaler Initialization

# def classify(current, future):
#     return float((future - current) / current)


data["target"] = data["close"].shift(-FUTURE_PERIOD_PREDICT)

# # Cut off NaNs
# # data = data[:-FUTURE_PERIOD_PREDICT]
data.dropna(inplace=True)

# data["target"] = list(map(classify, data["close"], data["future"]))
# # data[["close", "future", "target"]].tail()
# data = data.drop("future", 1)

# Fit scalers
price_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
volume_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
rsi_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

data = data.append(max_df)

price_scaler.fit(np.array(data["close"]).reshape(-1, 1))
volume_scaler.fit(np.array(data["volume"]).reshape(-1, 1))
rsi_scaler.fit(np.array(data["rsi"]).reshape(-1, 1))

#Dump scalers
try:
    os.mkdir(f"scalers/{NAME}")
except:
    pass

joblib.dump(price_scaler, f"scalers/{NAME}/price_scaler")
joblib.dump(volume_scaler, f"scalers/{NAME}/volume_scaler")
joblib.dump(rsi_scaler, f"scalers/{NAME}/rsi_scaler")

# Remove min max boundary values
data = data[:-2]


# Split dataset # Dont split here; do split after shuffle
# last_5_pct = int(len(data) * .95)

# train_data = data[:last_5_pct]
# validation_data = data[last_5_pct:]

# print(f"{len(train_data)} :: {len(validation_data)}")

# data.head()



## Helper
## Ratios of buy to sell targets

## See how balanced the input data is

# close_target_list = list(zip(data["close"], data["target"]))

# sell_counter = len([x for x in close_target_list if x[0] > x[1]])
# buy_counter = len([x for x in close_target_list if x[0] < x[1]])

# pct_sell = sell_counter / len(data)
# pct_buy = buy_counter / len(data)

# print(f"{pct_sell} :: {pct_buy}")



## Preprocess Data

def preprocess_df_p1(df_p):

    df = pd.DataFrame()
    for col in df_p.columns:
        df[col] = df_p[col]

    for col in df.columns:
        scaler = None
        if col in ["open", "high", "low", "close", "ema", "target"]:
            scaler = price_scaler
        elif col == "volume":
            scaler = volume_scaler
        elif col == "rsi":
            scaler = rsi_scaler
        else:
            raise Exception("Column not recognized and scaler cannot be determined")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df[col] = scaler.transform(np.array(df[col]).reshape(-1, 1))

        
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    sequential_data = []
    prev_periods = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_periods.append([n for n in i[:-1]])
        if len(prev_periods) == SEQ_LEN:
            sequential_data.append([np.array(prev_periods), i[-1]])

    return sequential_data
    # random.shuffle(sequential_data)

def preprocess_df_p2(seq_data):

    # Balance buys and sells
    buys = []
    sells = []

    for seq, target in seq_data:

        if target < seq[-1][3]: #compares to close column
            sells.append([seq, target])
        elif target > seq[-1][3]:
            buys.append([seq, target])

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    local_seq_data = buys + sells

    random.shuffle(local_seq_data)

    X = [d[0] for d in local_seq_data]
    Y = [d[1] for d in local_seq_data]

    return np.array(X), np.array(Y)
    

# train_x, train_y = preprocess_df(train_data)
# validation_x, validation_y = preprocess_df(validation_data)

# Preprocess and split data here
seq_data_full = preprocess_df_p1(data)

last_5_pct = int(len(seq_data_full) * .95)

seq_data_train = seq_data_full[:last_5_pct]
seq_data_val = seq_data_full[last_5_pct:]

train_x, train_y = preprocess_df_p2(seq_data_train)
validation_x, validation_y = preprocess_df_p2(seq_data_val)



## Dataset metrics

# close_target_list = list(zip([x[-1][3] for x in list(train_x)], list(train_y)))

# train_sell_counter = len([x for x in close_target_list if x[0] > x[1]])
# train_buy_counter = len([x for x in close_target_list if x[0] < x[1]])

# close_target_list = list(zip([x[-1][3] for x in list(validation_x)], list(validation_y)))

# val_sell_counter = len([x for x in close_target_list if x[0] > x[1]])
# val_buy_counter = len([x for x in close_target_list if x[0] < x[1]])

# print(f"Train : Validation == {len(train_x)} : {len(validation_x)}")
# print(f"Train\t\tBuys : Sells == {train_buy_counter} : {train_sell_counter}")
# print(f"Validation\tBuys : Sells == {val_buy_counter} : {val_sell_counter}")



## Make model

def build_model(hp):
    model = Sequential()

    learning_rate = hp.Choice("learning_rate", values=[.01, .001, .0001])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=1e-6)

    layer_count = hp.Int("lstm_count", min_value=1, max_value=8, step=1)

    for i in range(layer_count):
        neurons = hp.Int(f"lstm_neurons_{i}", min_value=32, max_value=1024, step=32)
        dropout_rate = hp.Float(f"dropout_rate_{i}", 0, 0.4, step=0.1)

        model.add(LSTM(neurons, input_shape=(train_x.shape[1:]), activation="tanh", return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())

    neurons = hp.Int(f"lstm_neurons_last", min_value=32, max_value=1024, step=32)
    dropout_rate = hp.Float(f"dropout_rate_last_lstm", 0, 0.4, step=0.1)

    model.add(LSTM(neurons, input_shape=(train_x.shape[1:]), activation="tanh"))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    neurons = hp.Int(f"dense_neurons", min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float(f"dropout_rate_last_lstm", 0, 0.4, step=0.1)

    model.add(Dense(neurons, activation="relu"))
    model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(1, activation="tanh"))

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer, metrics=["mse", "mean_absolute_percentage_error"])

    return model

tb = TensorBoard(log_dir=f"logs/{NAME}")

    # filepath = NAME + "-e{epoch:02d}-vmse{val_loss:.5f}-" + str(int(time.time()))
    # checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor="mse", verbose=1, save_best_only=True, mode="max"))



# Initialize Tuner

tuner = kt.BayesianOptimization(
                    max_trials=MAX_TRIALS,
                    hypermodel=build_model,
                    objective="val_loss",
                    overwrite=False,
                    directory=f"tuners/{NAME}",
                    project_name=f"{NAME}",)

tuner.search_space_summary(extended=True)


os.system("clear")

with tf.device("/device:GPU:0"):
    tuner.search(train_x, train_y, 
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[tb],
                validation_data=(validation_x, validation_y))



from pushbullet import Pushbullet
pb = Pushbullet("o.nyntgspLep97yl0oPDbp0nAbMIDUGiO5")
push = pb.push_note(f"{time.asctime()}", "ML Training Done")



best_model = tuner.get_best_models(num_models=1)[0]
tf.keras.models.save_model(best_model, f"models/{NAME}.model")



best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

with open(f"temp/{NAME}.hyperparameters.json", "w") as outfd:
    json.dump(best_hp.values, outfd)

print(f"\n\nSaved best model to: " + f"models/{NAME}.model" + "\nSaved best hyperparameters to: " + f"temp/{NAME}.hyperparameters.json")


