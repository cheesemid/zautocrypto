#!/usr/bin/env pytho3

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
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras import utils

from pushbullet import Pushbullet


SEQ_LEN = 48 #hours
FUTURE_PERIOD_PREDICT = 1 #hours

# Model parameters
EPOCHS = 20
BATCH_SIZE = 64
NAME = f"softmax2-e{EPOCHS}-b{BATCH_SIZE}-s{SEQ_LEN}-fpp{FUTURE_PERIOD_PREDICT}-{int(time.time())}"

model = None
tb = None
checkpoint = None
data = None
train_data = None
validation_data = None

train_x = None
train_y = None
validation_x = None
validation_y = None

def load_data():
    global data
    ## Import data
    # DATA MUST BE FORMATTED USING CSV_FORMATTER.IPYNB

    csv_file = "data/formatted/BTCUSDT-1h-data.csv"

    data = pd.read_csv(csv_file, skiprows=[0], names=["timestamp", "open", "high", "low", "close", "volume", "rsi", "ema"])

    data.set_index("timestamp", inplace=True)

    data.head()



def format_data():
    global data
    global train_data
    global validation_data
    ## Formatting data

    def classify(current, future):
        if float(future) > float(current):
            return 1
        return 0



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

    print(f"{len(train_data)} :: {len(validation_data)}")

    data.head()


## Preprocess Data

def preprocess_df(df):
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
    





## Make model
def create_model():
    global model
    global tb
    global checkpoint

    model = Sequential()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)


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

    # saved_model = tf.train.load_checkpoint("")
    # model.load_weights(saved_model)

    tb = TensorBoard(log_dir=f"logs/{NAME}")

    filepath = NAME + "-e{epoch:02d}-vacc{val_accuracy:.3f}-" + str(int(time.time()))
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max"))



## Fit model
def train():
    global model
    with tf.device("/device:GPU:0"):
        history = model.fit(x=train_x, y=train_y,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=(validation_x, validation_y),
                            callbacks=[tb, checkpoint]
                            )



# from pushbullet import Pushbullet
# pb = Pushbullet("o.nyntgspLep97yl0oPDbp0nAbMIDUGiO5")
# push = pb.push_note(f"{time.asctime()}", "ML Training Done")

def do_train(local_epochs=None, local_batch_size=None, local_seq_len=None, local_future_period_predict=None):
    global EPOCHS
    global BATCH_SIZE
    global SEQ_LEN
    global FUTURE_PERIOD_PREDICT
    global NAME

    global train_x
    global train_y
    global validation_x
    global validation_y

    if local_epochs:
        EPOCHS = local_epochs

    if local_batch_size:
        BATCH_SIZE = local_batch_size

    if local_seq_len:
        SEQ_LEN = local_seq_len

    if local_future_period_predict:
        FUTURE_PERIOD_PREDICT = local_future_period_predict

    NAME = f"softmax2-e{EPOCHS}-b{BATCH_SIZE}-s{SEQ_LEN}-fpp{FUTURE_PERIOD_PREDICT}-{int(time.time())}"

    load_data()
    format_data()
    train_x, train_y = preprocess_df(train_data)
    validation_x, validation_y = preprocess_df(validation_data)

    create_model()

    train()


if __name__ == "__main__":
    multi_train = True

    if multi_train:

        ## fmt: SEQ_LEN, FUTURE_PERIOD_PREDICT, EPOCHS, BATCH_SIZE
        attributes = [
            [24, 1, 20, 64],
            [24, 2, 20, 64],
            [24, 3, 20, 64],
            [24, 4, 20, 64],

            [12, 1, 20, 64],
            [12, 2, 20, 64],
            [12, 3, 20, 64],
            [12, 4, 20, 64],


            [24, 1, 20, 32],
            [24, 2, 20, 32],
            [24, 3, 20, 32],
            [24, 4, 20, 32],

            [12, 1, 20, 32],
            [12, 2, 20, 32],
            [12, 3, 20, 32],
            [12, 4, 20, 32],
        ]


        for i in range(len(attributes)):
            print(f"RUNNING MODEL {i} of {len(attributes)}")
            a = attributes[i]
            do_train(local_epochs=a[2], local_batch_size=a[3], local_seq_len=a[0], local_future_period_predict=a[1])

    

    pb = Pushbullet("o.nyntgspLep97yl0oPDbp0nAbMIDUGiO5")
    push = pb.push_note(f"{time.asctime()}", "ML Training Done")