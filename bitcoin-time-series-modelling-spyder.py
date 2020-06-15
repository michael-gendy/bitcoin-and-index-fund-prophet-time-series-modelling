# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:48:26 2020

@author: Michael
"""

#import libraries 
import cryptowatch as cw
import pandas as pd 
import matplotlib.pyplot as plt
import fbprophet
from fbprophet.plot import add_changepoints_to_plot
import seaborn as sns
from scipy.stats import pearsonr
import yfinance as yf

print("All libraries imported successfully")

#making api call - crypto watch 

cw.api_key = 'QSIYN3W065D5BWUA7AFR'

bitcoin = cw.markets.list("kraken")

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

#pull weekly bitcoin price actoin 
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

####################################################################################
######################ADDITIVE TIME SERIES MODELLING################################
##############################USING PROPHET#########################################
####################################################################################


#call daily BTCUSD price action
btc = get_price_action(exchange="KRAKEN"
                      ,market = "BTCUSD"
                      ,period = '1d')

#replace date and close data names for prophet
btc.reset_index(inplace=True)
btc.rename(columns={'close':'y', 'date': 'ds'}, inplace=True)

#create the model and fit the data 
#changepoint_prior_scale is used to control how sensitive the trend is to change
#higher = more sensitive  lower = less sensitive 
btc_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15
                                ,changepoint_range=0.9 # also setting checkpoitns for 90% (up from 80%) of range
                                ,n_changepoints=30 #specified amount won't necessarily appear in add_changepoints_to_plot, but will always appear once you inspect the .changepoints attribute
                                )
btc_prophet.fit(btc)

#create a dataframe with new dates 
btc_forecast = btc_prophet.make_future_dataframe(periods = 365
                                               ,freq = 'D')

#make predictions in the new dataframe
btc_forecast = btc_prophet.predict(btc_forecast)

fig = btc_prophet.plot(btc_forecast
                ,xlabel = 'Date'
                ,ylabel = 'BTC/USD')

#adding the changepoints from the model
_ = add_changepoints_to_plot(fig.gca()#this gets axis from figure
                            ,btc_prophet
                            ,btc_forecast)

fig

###################################################################################
##############PLOTTING PRICE CHANGEPOINTS OVER OTHER METRICS#######################
###################################################################################
trends = pd.read_csv("C:/Users/Michael/Documents/GitHub/bitcoin-time-series-modeling-/bitcoin_google_trends.csv")
trends['Month'] = pd.to_datetime(trends['Month'])

btc_daily_changepoints = pd.DataFrame([pd.to_datetime(date) for date in btc_prophet.changepoints])
btc_daily_changepoints
fig, ax = plt.subplots(figsize=(12, 6))

sns.lineplot(trends['Month']
           ,trends['worldwide_relative_bitcoin_interest']
           ,ax=ax)
plt.ylabel('Worldwide Relative Bitcoin Index')
plt.xlabel('Date')

ax.vlines(btc_daily_changepoints[0]
          ,ymin = -5
          ,ymax = 100
          ,colors = 'r'
          ,linewidth = 0.5, linestyles = 'dashed', label='Prophet Change Points')

ax.set(ylim=(0, 100))
ax.legend()
plt.show()

###################################################################################
##############VISUALISING OVERALL TRENDS - WEEKLY/MONTHLY/YEARLY ##################
###################################################################################

btc_prophet.plot_components(btc_forecast)

###################################################################################
##############USING MORE GRANULAR DATA OVER A SHORTER TIMEFRAME ###################
###################################################################################



#call daily BTCUSD price action
btc_1h = get_price_action(exchange="KRAKEN"
                      ,market = "BTCUSD"
                      ,period = '1h')

#replace date and close data names for prophet
btc_1h.reset_index(inplace=True)
btc_1h.rename(columns={'close':'y', 'date': 'ds'}, inplace=True)

#create the model and fit the data 
#changepoint_prior_scale is used to control how sensitive the trend is to change
#higher = more sensitive  lower = less sensitive 
btc_1h_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15
                                ,changepoint_range=0.9 # also setting checkpoitns for 90% (up from 80%) of range
                                ,n_changepoints=30 #specified amount won't necessarily appear in add_changepoints_to_plot, but will always appear once you inspect the .changepoints attribute
                                )
btc_1h_prophet.fit(btc_1h)

btc_1h_forecast = btc_1h_prophet.make_future_dataframe(periods = 365
                                               ,freq = 'h')

btc_1h_forecast = btc_1h_prophet.predict(btc_1h_forecast)

fig = btc_1h_prophet.plot(btc_1h_forecast
                ,xlabel = 'Date'
                ,ylabel = 'BTC/USD')

#adding the changepoints from the model
_ = add_changepoints_to_plot(fig.gca()#this gets axis from figure
                            ,btc_1h_prophet
                            ,btc_1h_forecast)

btc_1h_prophet.plot_components(btc_1h_forecast)

###################################################################################
##############CREATE A FUNCTON TO TEST DIFFERENT HYPERPARAMETERS ##################
##########################ALSO ADDING HALVING DATES TO MODEL#######################

#this function takes a specified dataframe, splits into training and test dataframes, 
#and compares different hyperparameters by plotting each model and visualising the result

#call daily BTCUSD price action
btc = get_price_action(exchange="KRAKEN"
                      ,market = "BTCUSD"
                      ,period = '1d')

#replace date and close data names for prophet
btc.reset_index(inplace=True)
btc.rename(columns={'close':'y', 'date': 'ds'}, inplace=True)

#adding halving dates to model
halvings = pd.DataFrame({
  'holiday': 'halving',
  'ds': pd.to_datetime(['2012-11-28', '2016-07-09', '2020-05-11']),
  'lower_window': 0,
  'upper_window': 90,
})


def find_optimal_prophet_model(df
                              ,cpps
                              ,special_events
                              ,periods
                              ,freq
                              ,test_size):
    #split off last 10% for validation 
    df_train= df.sort_values('ds', ascending=True).head(int(len(btc)*((1-test_size)/100)))
    df_test = df.sort_values('ds', ascending=True).tail(int(len(btc)*(test_size/100)))

    #for each hyperparameter
    for param in cpps:
        print("Starting model for cpps:{}".format(param))
        #create the model
        model = fbprophet.Prophet(changepoint_prior_scale=param
                                        ,changepoint_range=0.9
                                        ,n_changepoints=30 
                                        ,holidays=special_events)
        #fit the model
        model.fit(df_train)
        
        #create forcast by extrapolating training data
        future_df_train = model.make_future_dataframe(periods = periods
                                                       ,freq = '{}'.format(freq))
        
        #fit the model on the extrapolated training data 
        forecast_train = model.predict(future_df_train)
        
        #plot the model with predicted value
        model.plot(forecast_train, uncertainty=True)
        
        #plot actual dataset to compare 
        plt.plot(df_test['ds']
                ,df_test['y']
                ,label = 'actual'
                ,color = 'r')

        plt.title('Validation data v. forecast {}'.format(param))
        plt.ylim(df['y'].min(), df['y'].max())
        plt.legend();
        
    
cpps=(0.15,0.2#, 0.25,0.30,0.35,0.4,0.45, 0.50
      )
find_optimal_prophet_model(btc, cpps, halvings, 365, 'D', 20)

#clear that prophet does not handle price action well for extremely volatile assets such as BTC
#retesting with less volatile data - index funds


tickers = {'NASDAQ' : '^IXIC'
           ,'S&P' : '^GSPC'
           ,'DOW' : '^DJI'}

#pulling in nasdaq data
nasdaq = yf.Ticker(tickers['NASDAQ'])
nasdaq_df = pd.DataFrame(nasdaq.history(period='max'))
#set in prophet format 
nasdaq_df.reset_index(inplace=True)
nasdaq_df.rename(columns={'Date' : 'ds', 'Close' : 'y'}, inplace=True)
nasdaq_df = nasdaq_df[nasdaq_df['ds'] >= '2015-01-01']

#add special events 
events = pd.DataFrame({
  'holiday': 'halving',
  'ds': pd.to_datetime([]),
  'lower_window': 0,
  'upper_window': 90,
})

cpps=(0.1,0.3, 0.5,0.7,0.9)
find_optimal_prophet_model(nasdaq_df, cpps, events, 365, 'D', 10)



############################################################################
###############EVLALUATING CORRELATION BETWEEN Y_TEST AND Y_PRED############
############################################################################

#since the model performed poorly on Bitcoin data, lets choose a less (but still somewhat) volatile market - indexes

#set tickers
tickers = {'NASDAQ' : '^IXIC'
           ,'S&P' : '^GSPC'
           ,'DOW' : '^DJI'}

#pulling in nasdaq data
index_ticket = yf.Ticker(tickers['NASDAQ']) #choose index here
index_df = pd.DataFrame(index_ticket.history(period='max'))

#reduce timeframe and change format
index_df = index_df[index_df.index >= '2015-01-01']
index_df.reset_index(inplace=True)
index_df.rename(columns={'Date' : 'ds', 'Close' : 'y'}, inplace=True)


#Evaulate prophet model vs test data using rolling correlation function

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

#create the model and fit the data 
#changepoint_prior_scale is used to control how sensitive the trend is to change
#higher = more sensitive  lower = less sensitive 
index_prophet = fbprophet.Prophet(changepoint_prior_scale=0.05
                                ,changepoint_range=0.9 # also setting checkpoitns for 90% (up from 80%) of range
                                ,n_changepoints=30 #specified amount won't necessarily appear in add_changepoints_to_plot, but will always appear once you inspect the .changepoints attribute
                                )
index_prophet.fit(index_df)

#create a dataframe with new dates 
index_forecast = index_prophet.make_future_dataframe(periods = 365
                                               ,freq = 'D')

#make predictions in the new dataframe
index_forecast = index_prophet.predict(index_forecast)

#obtain prophet predictions 
index_predictions = index_forecast.loc[:, ('ds', 'yhat')] #yhat is prophets prediction
index_predictions.rename(columns={'ds' : 'date'}, inplace=True)
index_predictions.set_index('date', inplace=True)

index_predictions.tail()

#recall price action before format changes
index_ticket = yf.Ticker(tickers['NASDAQ'])
index_df = pd.DataFrame(index_ticket.history(period='max'))
index_df = index_df[index_df.index >= '2015-01-01']


#find correlations

prophet_preditions_vs_actual_correlation = cointegration_function(index_predictions
                                                 ,'yhat'
                                                 ,index_df
                                                 ,'Close'
                                                 ,timeframe='month')

#plot monthly correlation coefficients 
fig1, ax1 = plt.subplots(figsize=(12,6))

sns.barplot(x = prophet_preditions_vs_actual_correlation.index
                ,y = prophet_preditions_vs_actual_correlation['pearsons_coeff']
                ,ax =ax1)
ax1.set_xlabel("Month")
ax1.set_ylabel("Pearson's Correlation Coefficient")

#only take every 12th x label due to overlap 
for i, t in enumerate(ax1.get_xticklabels()):
    if (i % 12) != 0:
        t.set_visible(False)

plt.show()

#compare with model visuslisation 
fig = index_prophet.plot(index_forecast
                ,xlabel = 'Date'
                ,ylabel = 'BTC/USD')

fig
