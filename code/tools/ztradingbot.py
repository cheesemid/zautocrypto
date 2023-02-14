#!/usr/bin/env python3


# absolute hot garbage
# please do not use unless you dont care at all about money

from time import sleep
from TradingView_SignalScraper import *
from binance.client import Client
import bkeys
import math

dodebug = True
client = Client(bkeys.api_key,bkeys.secret_key)

class wallet():
    # pylint: disable=no-self-argument
    # pylint: disable=not-callable
    dodebug = True
    btcusd = 0

    def __init__(self, usd, btc = 0):
        self.usd = usd
        self.btc = btc

    def debugger(self,msg):
        if self.dodebug:
            print(msg)

    def contents(func):
        def inner(self,deltabtc,deltausd):
            func(self,deltabtc,deltausd)
            print("USD: " + str(self.usd) + " : " + "BTC: " + str(self.btc) + " :: " + "Total USD: " + str(self.getusdportfoliovalue()))
        return inner

    def getusdportfoliovalue(self):
        return self.usd + (getbtcprice()*self.btc)

    def getpositions(self):
        return [self.usd,self.btc]

    @contents
    def move(self,deltausd,deltabtc):
        self.usd = self.usd + deltausd
        self.btc = self.btc + deltabtc

def debugger(msg):
        if dodebug:
            print(msg)

def getbtcprice():
    for ticker in client.get_all_tickers():
        if "BTCUSDT" in ticker.values():
            return float(ticker["price"])

def decide(signals):
    decs = []
    for signal in signals:
        if signal >= .5:
            decs.append(1)
        elif signal < .5 and signal > 0:
            decs.append(signal*2)
        elif signal <= 0:
            decs.append(0)
    return decs

def makemove(w,decision):
    [dec5,dec30,dec60,dec240] = decision
    totalval = w.getusdportfoliovalue()
    valperinterval = math.floor(totalval/4)

    end5 = valperinterval * dec5
    end30 = valperinterval * dec30
    end60 = valperinterval * dec60
    end240 = valperinterval * dec240

    usdshouldbeinbtc = end5 + end30 + end60 + end240
    positions = w.getpositions()

    btcprice = getbtcprice()
    deltausd = positions[1]*btcprice - usdshouldbeinbtc
    deltabtc = -(deltausd/btcprice)

    debugger([deltausd,deltabtc])

    debugger([dec5,dec30,dec60,dec240])
    if abs(deltausd) >= w.getusdportfoliovalue()/20:
        w.move(deltausd,deltabtc)
        debugger("Did move")
    return 0

def trader(w):
    print("Starting BTC price: " + str(getbtcprice()))
    try:
        while True:
            signal5 = get_signal("BTCUSDT",5)
            signal30 = get_signal("BTCUSDT",60)
            signal60 = get_signal("BTCUSDT",60)
            signal240 = get_signal("BTCUSDT",240)

            signallist = [signal5,signal30,signal60,signal240]
            debugger(signallist)
            
            decision = decide(signallist)
            makemove(w,decision)
            sleep(30)
    except KeyboardInterrupt:
        print("Ending Portfolio Value: " + str(w.getusdportfoliovalue()))
        print("Ending BTC Price: " + str(getbtcprice()))
        quit()

trader(wallet(1000))