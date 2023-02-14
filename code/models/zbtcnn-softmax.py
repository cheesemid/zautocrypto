#!/usr/bin/env python3

# BTC PRICE PREDICTION NN
# 122620 zed

# pylint: disable=import-error
import os
import sys
import random
import time
import logging
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

# ohhh boy here we go

SEQ_LEN = 24 #hours
FUTURE_PERIOD_PREDICT = 3 #hours

# Change time format on csv read
do_time_fix = False

# Model parameters
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"e{EPOCHS}-b{BATCH_SIZE}-s{SEQ_LEN}-fpp{FUTURE_PERIOD_PREDICT}-{int(time.time())}"

csv_file = "output.csv"


model = None
tb = None
checkpoint = None
data = None

logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)


def read_data():
    global data

    logger.info("Reading data")
    data = pd.read_csv(csv_file, skiprows=[0], names=["timestamp","open","high","low","close","volume","close_time","quote_av","trades","tb_base_av","tb_quote_av","ignore"])

    if do_time_fix:
        for i in range(len(data["timestamp"])):
            t = data["timestamp"][i].split(".")[0]


            # 2017-08-17 04:00:00.000
            tstruct = time.strptime(t, "%Y-%m-%d %H:%M:%S")
            epoch_sec = int(time.mktime(tstruct))

            data["timestamp"][i] = epoch_sec
        
        data.to_csv(f"{csv_file.split('.')[0]}-formatted.csv")

    data.set_index("timestamp", inplace=True)


def classify(current, future):
    if float(future) > float(current):
        return 1
    return 0

def format_data():
    global data

    logger.info("Formatting Data")

    data = data.drop(["trades", "quote_av", "tb_base_av", "tb_quote_av", "ignore", "close_time"], 1)


    data["future"] = data["close"].shift(-FUTURE_PERIOD_PREDICT)

    # Cut off NaNs
    # data = data[:-FUTURE_PERIOD_PREDICT]
    data.dropna(inplace=True)

    data["target"] = list(map(classify, data["close"], data["future"]))
    # data[["close", "future", "target"]].tail()
    data = data.drop("future", 1)

    # Split dataset
    last_5_pct = int(len(data) * .95)

    train_data = data[:last_5_pct]
    validation_data = data[last_5_pct:]

    # print(f"{len(train_data)} :: {len(validation_data)}")

    return train_data, validation_data



def preprocess_df(df):

    logger.info("Preprocessing Data")

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            df[col] = preprocessing.StandardScaler().fit_transform(df[col].values.reshape(-1,1))

    df.dropna(inplace=True)

    sequential_data = []
    prev_periods = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_periods.append([n for n in i[:-1]])
        if len(prev_periods) == SEQ_LEN:
            sequential_data.append([np.array(prev_periods), i[-1]])

    # random.shuffle(sequential_data)

    # Balance buys and sells
    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells

    random.shuffle(sequential_data)

    X = [d[0] for d in sequential_data]
    Y = [d[1] for d in sequential_data]

    return np.array(X), np.array(Y)

def create_model(train_x):
    global model
    global tb
    global checkpoint

    logger.info("Creating model")

    model = Sequential()
    optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)


    model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation="tanh", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation="tanh", return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation="tanh"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation="softmax"))



    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    tb = TensorBoard(log_dir=f"logs/{NAME}")

    filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}-" + str(int(time.time()))
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max"))


def fit_model(train_x, train_y, validation_x, validation_y):
    global model

    logger.info("Fitting model")

    with tf.device("/device:GPU:0"):
        history = model.fit(x=train_x, y=train_y,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=(validation_x, validation_y),
                            callbacks=[tb, checkpoint]
                            )
        return history

def do_train():
    # Check GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    read_data()

    train_data, validation_data = format_data()
    train_x, train_y = preprocess_df(train_data)
    validation_x, validation_y = preprocess_df(validation_data)

    create_model(train_x)
    history = fit_model(train_x, train_y, validation_x, validation_y)

def do_multi_train():
    global SEQ_LEN
    global FUTURE_PERIOD_PREDICT
    global EPOCHS
    global NAME

    logger.info("Doing Multi-train")

    # fmt: SEQ_LEN, FUTURE_PERIOD_PREDICT, EPOCHS
    attributes = [
        [72, 1, 20],
        [48, 1, 20],
        [24, 1, 20],
        [12, 1, 20]
    ]

    read_data()

    train_data, validation_data = format_data()

    

    for a in attributes:
        SEQ_LEN, FUTURE_PERIOD_PREDICT, EPOCHS = a
        NAME = f"e{EPOCHS}-b{BATCH_SIZE}-s{SEQ_LEN}-fpp{FUTURE_PERIOD_PREDICT}-{int(time.time())}"

        logger.info(f"Running with attributes: {a}")
        logger.info(f"Name: {NAME}")

        train_x, train_y = preprocess_df(train_data)
        validation_x, validation_y = preprocess_df(validation_data)
        create_model(train_x)
        history = fit_model(train_x, train_y, validation_x, validation_y)


def main():
    # do_train()
    do_multi_train()
    

if __name__ == "__main__":
    main()



