#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import talib

"""
    ===Ta-Lib Install===
    Links:
        https://visualstudio.microsoft.com/visual-cpp-build-tools/
        https://github.com/afnhsn/TA-Lib_x64
        https://github.com/mrjbq7/ta-lib/issues/127#issuecomment-280222942

    1. Install Visual C++ Build Tools
    2. DL afnhsn/TA-Lib_x64/TA-Lib_x64.zip (disregard exe)
    3. Move the "TA-Lib_x64.zip/TA-Lib_x64/ta-lib" folder to C:\
    4. Open Developer Command Prompt for VS[2019]
    5. CD to C:\ta-lib\c\make\cdr\win32\msvc
    6. Build the Library with: nmake
    7. pip install ta-lib

"""


plt.rcParams["figure.figsize"] = (20,20)

# nn inputs
# [fastema-slowema,rsi,price vs bbands]

def loadcsv(filename):
    df = pd.read_csv(filename)
    return df

def ema(df, column="close", period=100):
    ema = df[column].ewm(period).mean()
    #emadata.plot()
    return ema

def rsi(df, column="close", period=30):
    # pylint: disable=no-member
    rsi = talib.RSI(df[column], timeperiod=period)
    return rsi

def bbands(df, column="close", period=20):
    # pylint: disable=no-member
    # https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands
    upperBB, middleBB, lowerBB = talib.BBANDS(df[column], timeperiod=period, nbdevup=2, nbdevdn=2, matype=0)
    return [upperBB,middleBB,lowerBB]

if __name__ == "__main__":
    df = loadcsv("BTCUSDT-1h-data.csv")
    datapoints = 500
    df["close"].head(datapoints).plot()
    ##print(ema(df))
    ##print(df)
    ema(df).head(datapoints).plot()
    ema(df, period=30).head(datapoints).plot()
    #rsi(df).head(200).plot()
    for i in bbands(df):
        i.head(datapoints).plot()
    
    #print(talib.get_functions())
    #print(talib.get_function_groups())