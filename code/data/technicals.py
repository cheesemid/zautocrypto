#!/usr/bin/env python3

# Functions for technical indicators

import numpy as np

def rsi_start(period, data):
    # Returns RSI(data[period])

    sum_gains = 0
    sum_losses = 0

    last = data[0]

    for i in data[1:period]:
        if i > last:
            sum_gains += i - last
        elif i < last:
            sum_losses += last - i
        last = i

    avg_gain = sum_gains / period
    avg_loss = sum_losses / period

    rsi_ret = 100.0 - (100.0/(1+(avg_gain/avg_loss)))

    return (rsi_ret, avg_gain, avg_loss)


def rsi(period, last_avg_gain, last_avg_loss, next_delta):
    if next_delta > 0:
        avg_gain = (last_avg_gain * (period - 1)) + next_delta
        avg_loss = (last_avg_loss * (period - 1))
    else:
        avg_gain = (last_avg_gain * (period - 1))
        avg_loss = (last_avg_loss * (period - 1)) - next_delta
        
    rsi_ret = 100.0 - (100.0 / (1 + (avg_gain/avg_loss)))

    return (rsi_ret, avg_gain, avg_loss)

def ema():
    pass