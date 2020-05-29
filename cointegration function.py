# -*- coding: utf-8 -*-
"""
Created on Tue May 19 13:20:35 2020

@author: Michael
"""
"""
The purpose of this code is to call price action for a cryto pairing, 
and find how it's correlation with another time series varies over a specified timeframe

"""
import cryptowatch as cw
#import time
import pandas as pd
#import json
#import pytz
#import json
from datetime import datetime, timedelta
from scipy.stats import pearsonr
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt

cw.api_key = 'QSIYN3W065D5BWUA7AFR'

def get_price_action(exchange, market, period):
    candles=cw.markets.get("{}:{}".format(exchange, market)
                           , ohlc=True
                           , periods=["{}".format(period)])
    df = pd.DataFrame(getattr(candles, 'of_{}'.format(period)))
    df.columns=(["date","open", "high", "low", "close", "volume", "volume in {}".format(market[-3:])])
    df["date"] = pd.to_datetime(df["date"], unit='s')
    df.set_index("date", inplace=True)
    return df

#calling BTC:USD and ETH:USD hourly price action in KRAKEN exchange 
df_eth_usd_kraken_1h = get_price_action(exchange="KRAKEN"
                                    ,market = "ETHUSD"
                                    ,period = "1h"
                                    )
df_btc_usd_kraken_1h = get_price_action(exchange="KRAKEN"
                                    ,market = "BTCUSD"
                                    ,period = "1h")

#define function that takes two time series dataframes, marges them, splits into specified timeframe and finds correlation coefficient during that time 
def cointegration_function(df1, var1, df2, var2, timeframe):
    #slice the dataframes, ensure both indexes are named same and joining to one dataframe 
    df1_slice = df1.loc[:, ['{}'.format(var1)]]
    df2_slice = df2.loc[:, ['{}'.format(var2)]]
    df1_slice.index.name = 'date'
    df2_slice.index.name = 'date'
    df_comb = pd.merge(df1_slice, df2_slice, on='date')
    #add week/month/day numbers 
    df_comb['day'] = df_comb.index.strftime('%x')
    df_comb['week'] = df_comb.index.strftime('%Y-%W')
    df_comb['month'] = df_comb.index.strftime('%Y-%m')
    
    #create empty dict for results 
    results = {}
    #split into timeframe, and calc correlation coefficient for all data points in that timeframe
    #if statement error-handles for variables that are named the same
    if var1 == var2:
        for tf in df_comb['{}'.format(timeframe)].unique():
            timeframe_data = df_comb[df_comb['{}'.format(timeframe)] == tf]
            corr = pearsonr(timeframe_data['{}'.format(var1 + '_x')], timeframe_data['{}'.format(var2 + '_y')])
            results[tf] = [corr[0]]
    else: 
  #split into timeframe, and calc correlation coefficient for all data points in that 
        for tf in df_comb['{}'.format(timeframe)].unique():
            timeframe_data = df_comb[df_comb['{}'.format(timeframe)] == tf]
            corr = pearsonr(timeframe_data['{}'.format(var1)], timeframe_data['{}'.format(var2)])
            results[tf] = [corr[0]]
    result_df = pd.DataFrame.from_dict(results, orient='index', columns=['pearsons_coeff'])
    result_df.index.name = 'timeframe'
    
    return result_df
"""
#finding the correlation of BTC and ETH 
btc_eth_weekly_corr = cointegration_function(df1 = df_eth_usd_kraken_1h
                                        ,var1 = 'close'
                                        ,df2 = df_btc_usd_kraken_1h
                                        ,var2 = 'close'
                                        ,timeframe='week')

#bar plot of correlation coefficients per timeframe of two time series'
sns.set(style="whitegrid")
_ = sns.barplot(x = btc_eth_weekly_corr.index, y= btc_eth_weekly_corr['pearsons_coeff'])
plt.show()
"""
######################################################################################
####################EXAMPLE MAPPING VS STOCKS ########################################
######################################################################################
tickers = {'NASDAQ' : '^IXIC'
           ,'S&P' : '^GSPC'
           ,'DOW' : '^DJI'}

#pulling in nasdaq data
nasdaq = yf.Ticker(tickers['NASDAQ'])
nasdaq_df = pd.DataFrame(nasdaq.history(period='max'))

#pulling in BTC data 
#can only get daily close data for NASDAQ, therefore need to match with daily BTC data
btc_usd_coinbase_1d = get_price_action(exchange='COINBASE'
                                       ,market = 'BTCUSD'
                                       ,period='1d')

nasdaq_btc_monthly_corr = cointegration_function(nasdaq_df
                                                 ,'Close'
                                                 ,btc_usd_coinbase_1d
                                                 ,'close'
                                                 ,timeframe='month')

#inspect output 
nasdaq_btc_monthly_corr.head()


#plot monthly correlation coefficients 
#try and merge on BTC_USD monthly to plot both on the same axis 
fig1, ax1 = plt.subplots(figsize=(6,6))

sns.barplot(x = nasdaq_btc_monthly_corr.index
                ,y = nasdaq_btc_monthly_corr['pearsons_coeff']
                ,ax =ax1)
ax1.set_xlabel("Month")
ax1.set_ylabel("Pearson's Correlation Coefficient")

#only take every 12th x label due to overlap 
for i, t in enumerate(ax1.get_xticklabels()):
    if (i % 12) != 0:
        t.set_visible(False)

#compare against price graphs 
fig2, ax2 = plt.subplots(figsize=(6,6))

sns.lineplot(x = btc_usd_coinbase_1d.index
            ,y = btc_usd_coinbase_1d['close']
            ,ax=ax2)

sns.lineplot(x = nasdaq_df.index
           ,y= nasdaq_df['Close']
           ,ax=ax2)

ax2.set_xlim([datetime(2015, 1, 1), datetime(2020, 6, 1)])
ax1.set_xlabel("Date")
ax1.set_ylabel("Price/USD")


plt.show()
