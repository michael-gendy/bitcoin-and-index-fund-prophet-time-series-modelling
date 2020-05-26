# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:48:26 2020

@author: Michael
"""

#import libraries 
import cryptowatch as cw
#import time
import pandas as pd
#import json
#import pytz
#import json
from datetime import datetime, timedelta
#from requests import Request, Session
#from requests.exceptions import ConnectionError, Timeout, TooManyRedirects 
import matplotlib.pyplot as plt
import prophet

print("All libraries imported successfully")

#making api call - crypto watch 

cw.api_key = 'QSIYN3W065D5BWUA7AFR'

bitcoin = cw.markets.list("kraken")

# #checking for successful connection 
# if '200' in str(bitcoin._http_response):
#     print("Successful Connection")
# else: 
#     print("Unsuccessful: {}".format(bitcoin._http_response))

# #creating http response variable, transform to a dict and inspect keys
# bitcoin_response = bitcoin._http_response 
# bitcoin_dict = bitcoin_response.json()

# for key in bitcoin_dict.keys():
#     print(key)

# #checking allowance 
# print(bitcoin_dict["allowance"])

# #checking the result
# bitcoin_result_df = pd.DataFrame(bitcoin_dict["result"])
# bitcoin_result_df.head()

# #testing getting daily open and close 
# candles=cw.markets.get("KRAKEN:BTCUSD", ohlc=True, periods=["1d"])

# #prints all attributes of object/response
# print(dir(candles))

# close_timestamp, daily_open, daily_close = (
#                     candles.of_1d[-1][0],
#                     candles.of_1d[-1][1],
#                     candles.of_1d[-1][4]
# )

# print(datetime.utcfromtimestamp(close_timestamp) - timedelta(days=1))
# print(daily_open)
# print(daily_close)

# #getting all price opens 
# df = pd.DataFrame(candles.of_1d)
# df.head()

# #checking which dates we get
# for index, date_utc in enumerate(df[0]):
#     print(datetime.utcfromtimestamp(date_utc))

#create a function to get price action of any market, exchange and period combination 
def get_price_action(exchange, market, period):
    candles=cw.markets.get("{}:{}".format(exchange, market)
                           , ohlc=True
                           , periods=["{}".format(period)])
    df = pd.DataFrame(getattr(candles, 'of_{}'.format(period)))
    df.columns=(["date","open", "high", "low", "close", "volume", "volume in {}".format(market[-3:])])
    df["date"] = pd.to_datetime(df["date"], unit='s')
    df.set_index("date", inplace=True)
    return df

print("Function defined sucessfully")

df_btc_usd_kraken_1h = get_price_action(exchange="KRAKEN"
                                    ,market = "BTCUSD"
                                    ,period = '1h')

df_btc_usd_kraken_1d = get_price_action(exchange="KRAKEN"
                                    ,market = "BTCUSD"
                                    ,period = '1d')

df_btc_usd_kraken_1w = get_price_action(exchange="KRAKEN"
                                    ,market = "BTCUSD"
                                    ,period = '1w')
#set BTC price/time axis
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_ylabel('BTC Price/$', color='blue')
ax1.set_xlabel('Date')
ax1.plot(df_btc_usd_kraken_1w.index
        ,df_btc_usd_kraken_1w["close"]
        ,color = 'blue')

#add second axis for volume
ax2 = ax1.twinx()

ax2.set_ylabel('BTC Volume/BTC', color='red')
ax2.plot(df_btc_usd_kraken_1w.index
         ,df_btc_usd_kraken_1w["volume"]
         ,color= 'red')
fig.tight_layout()

plt.show()






