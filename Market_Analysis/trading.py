import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import quandl
def quandl_data(dataset,col_name,collapse="daily", key="digit_your_key"):
    """
    import quandl dataset: close,open,high,low,volume
    """
    data = quandl.get(dataset,collapse=collapse, api_key= key)
    n_assets = len(dataset)
    close = data[list(map(lambda x,y,z: x + y + z,dataset,[" - "]*n_assets,["Settle"]*n_assets))]
    open_= data[list(map(lambda x,y,z: x + y + z,dataset,[" - "]*n_assets,["Open"]*n_assets))]
    high= data[list(map(lambda x,y,z: x + y + z,dataset,[" - "]*n_assets,["High"]*n_assets))]
    low= data[list(map(lambda x,y,z: x + y + z,dataset,[" - "]*n_assets,["Low"]*n_assets))]
    open_close_max_min = pd.concat([open_,close,high,low],axis=1)
    open_close_max_min = open_close_max_min.replace({0:np.nan})
    open_close_max_min.dropna(inplace=True)
    close = open_close_max_min[list(map(lambda x,y,z: x + y + z,dataset,[" - "]*n_assets,["Settle"]*n_assets))]
    close.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["close"]*n_assets))]
    open_= open_close_max_min[list(map(lambda x,y,z: x + y + z,dataset,[" - "]*n_assets,["Open"]*n_assets))]
    open_.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["open"]*n_assets))]
    high= open_close_max_min[list(map(lambda x,y,z: x + y + z,dataset,[" - "]*n_assets,["High"]*n_assets))]
    high.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["high"]*n_assets))]
    low= open_close_max_min[list(map(lambda x,y,z: x + y + z,dataset,[" - "]*n_assets,["Low"]*n_assets))]
    low.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["low"]*n_assets))]
    return close,open_,high,low

def mav(stochastic,p=3):
    """
    Calculation of modified average: 
    MAVt = MAVt-1 + (Pt - MAVt-1)/n
    n default 3 periods (p=3)
    First MAV is = Simple moving average 
    """
    stochastic.dropna(inplace=True)
    n_days = stochastic.shape[0] -p
    mav = []
    mav1 = stochastic.rolling(p).mean()
    mav1.dropna(inplace=True)
    mav.append(mav1.values[0])
    for n in range(n_days):
        mav.append(mav[-1]+((stochastic.iloc[n+p]-mav[-1])/p))
    return pd.DataFrame(mav,stochastic.index[p-1:],columns=stochastic.columns)

def preferred_stochastic(close,high,low,col_name,n=8,p=3):
    """
    FAST STOCHASTIC
    %K = 100[(C-Ln)/(Hn-Ln)]
    %D = 3 period Modified Moving Average of %K
    
    PREFFERED(SLOW) STOCHASTIC
    %K_slow = %D from the FAST STOCHASTIC
    %D_slow = 3 period Modified Moving Average of %K_slow
    
    INPUTS
    n = 8 periods
    3 periods of smoothing fast line
    3 periods of smoothing fast line 
    """
    n_assets = len(col_name)
    k_fast = 100*((close.values - low.rolling(n).min())/(high.rolling(n).max().values-low.rolling(n).min()))
    k_fast.dropna(inplace=True)
    k_fast.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["k_fast"]*n_assets))]
    k_slow = mav(k_fast,p)
    k_slow.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["k_slow"]*n_assets))]
    d_slow = mav(k_slow)
    d_slow.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["d_slow"]*n_assets))]
    return k_slow,d_slow

def plot_pref_stochastic(k_slow,d_slow,l=25,h=25):
    """
    Plot pref_stochastic
    """
    n_assets=k_slow.shape[1]
    plt.figure(figsize=(l,h))
    for n in range(n_assets):
        plt.subplot(math.ceil(n_assets/2),2,n+1)
        plt.plot(k_slow[k_slow.columns[n]], label='%k_fast')
        plt.plot(d_slow[d_slow.columns[n]], label='%d_slow')
        plt.axhline(y=25, color='r', linestyle='-')
        plt.axhline(y=75, color='r', linestyle='-')
        plt.title(k_slow.columns[n])
        plt.legend()

def macd_dema(prices):
    """
    MACD è data dalla differenza tra due volte una EMA sul valore del titolo e una doppia EMA 
    (cioè una EMA sulla EMA del titolo, indicata con l'acronimo EMA2)
    """
    n_days = prices.shape[0] -1
    ema = [prices.iloc[0]]
    ema1 = [prices.iloc[0]]
    ema2 = [prices.iloc[0]]
    signal_line = []
    for n in range(n_days):
        ema.append(ema[-1]+0.213*(prices.iloc[n+1]-ema[-1]))
        ema1.append(ema1[-1]+0.213*(ema[-1]-ema1[-1]))
        ema2.append(ema2[-1]+0.108*(ema1[-1]-ema2[-1]))
    ema1=pd.DataFrame(ema1)
    ema2=pd.DataFrame(ema2)
    macd = ema1 -ema2
    macd.index = prices.index
    macd.columns=prices.columns
    for n in range(n_days):
        signal_line.append(macd.iloc[n] + 0.199*(macd.iloc[n+1]-macd.iloc[n])) 
    signal_line = pd.DataFrame(signal_line)
    signal_line.index = macd.index[1:]
    signal_line.columns = macd.columns
    diff = macd-signal_line
    return macd, signal_line, diff

def plot_macd(macd,signal,diff,l=25,h=25):
    """
    Plot macd
    """
    n_assets=macd.shape[1]
    plt.figure(figsize=(l,h))
    for n in range(n_assets):
        plt.subplot(math.ceil(n_assets/2),2,n+1)
        plt.plot(macd[macd.columns[n]], label='MACD')
        plt.plot(signal[signal.columns[n]], label='Signal')
        plt.bar(diff.index,diff[diff.columns[n]], label='Diff')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title(macd.columns[n])
        plt.legend()

def RSI(prices,p=13):
    """
    RSI = 100 - (100/(1+RS)) dove RS = (Media variazioni positive)/(Media variazioni negative)
    """
    diff = prices - prices.shift(1)
    diff_positive=diff*0
    diff_negative=diff*0
    diff_positive[diff>0] = diff[diff>0]
    diff_negative[diff<0] = -diff[diff<0]
    diff_positive_average = diff_positive.rolling(p).mean()
    diff_negative_average = diff_negative.rolling(p).mean()
    rs= diff_positive_average/diff_negative_average
    rsi = 100 - (100/(1+rs))
    return rsi

def TRI(prices,p1=13,p2=3,p3=3):
    """
    TRI = RSI(13) + SIMPLE MOV 3 days(RSI(3))
    """
    tri = RSI(prices,p1) + RSI(prices,p2).rolling(p3).mean()
    return tri

def plot_TRI(TRI,l=25,h=25):
    """
    Plot TRI
    """
    n_assets=TRI.shape[1]
    plt.figure(figsize=(l,h))
    for n in range(n_assets):
        plt.subplot(math.ceil(n_assets/2),2,n+1)
        plt.plot(TRI[TRI.columns[n]], label='TRI')
        plt.title(TRI.columns[n])
        plt.legend()

def detrended_oscillator(prices,freq=7):
    detrended = prices - prices.rolling(freq).mean()
    detrended.dropna(inplace=True)
    return detrended

def plot_detrended_oscillator(oscillator,period,l=25,h=25):
    """
    Plot detrended oscillator
    Le rette signal rappresentano la soglia per determinare overbought e oversold
    Il dataframe period è diverso dal dataframe quando nei mesi precedenti c'e stata un'alta volatilità che ha
    alterato i valori dell'overbought e oversold quindi non la includiamo per il calcolo delle due linee overbought
    oversold ma scegliamo un perodo precedente o successivo
    signal_up= quantile(0.985)*0.7
    signal_down= quantile(0.015)*0.7
    """
    n_assets=oscillator.shape[1]
    plt.figure(figsize=(l,h))
    for n in range(n_assets):
        signal_up=period[period.columns[n]].quantile(0.985)*0.7
        signal_down=period[period.columns[n]].quantile(0.015)*0.7
        plt.subplot(math.ceil(n_assets/2),2,n+1)
        plt.plot(oscillator[oscillator.columns[n]])
        plt.axhline(y=0, color='r', linestyle='-')
        plt.axhline(y=signal_up, color='k', linestyle='--')
        plt.axhline(y=signal_down, color='k', linestyle='--')
        plt.title(oscillator.columns[n])
        
def VHF(close,high,low,n=7):
    """
    To calculate the Vertical Horizontal Filter:
    Select the number of periods (n). 
    This should be based on the length of the cycle that you are analyzing. 
    The most popular is 28 days (for intermediate cycles).
    Determine the highest closing price (HCP) in n periods.
    Determine the lowest closing price (LCP) in n periods.
    Calculate the range of closing prices in n periods:
               HCP - LCP
    Next, calculate the movement in closing price for each period:
               Closing price [today] - Closing price [yesterday]
    Add up all price movements for n periods, disregarding whether they are up or down:
               Sum of absolute values of ( Close [today] - Close [yesterday] ) for n periods
    Final formula:
               VHF = (HCP - LCP) / (Sum of absolute values for n periods)
               
    TRADING SIGNALS
    Vertical Horizontal Filter does not, itself, generate trading signals, 
    but determines whether signals are taken from trend or momentum indicators.

    Rising values indicate a trend.
    Falling values indicate a ranging market.
    High values precede the end of a trend.
    Low values precede a trend start.
    """
    highest = high.rolling(n).max()
    lowests = low.rolling(n).min()
    max_low = highest - lowests.values
    closings = close - close.shift(1)
    abs_closings = np.absolute(closings)
    sum_closings = abs_closings.rolling(n).sum()
    vhf = max_low/sum_closings.values
    return vhf

def plot_VHF(VHF,l=25,h=25):
    """
    Plot detrended oscillator
    """
    n_assets=VHF.shape[1]
    plt.figure(figsize=(l,h))
    for n in range(n_assets):
        plt.subplot(math.ceil(n_assets/2),2,n+1)
        plt.plot(VHF[VHF.columns[n]])
        plt.title(VHF.columns[n])
        
def DMA(close,window=25,shift=5):
    dis_ma = close.rolling(window=window).mean().shift(shift)
    return dis_ma
        
def buy_scan(close_monthly,close_daily,k_slow_daily,d_slow_daily,column):
    """
    Return the probabilities to open a buy position
    """
    k_slow_daily.columns = column
    d_slow_daily.columns = column
    close_monthly.columns = column
    close_daily.columns = column
    monthly_DMA_25x5 = close_monthly.iloc[-1]>DMA(close_monthly).iloc[-1]
    macd_monthly= macd_dema(close_monthly)[2].iloc[-1]>0
    daily_DMA_25x5 = close_daily.iloc[-1]>DMA(close_daily,25,5).iloc[-1]
    macd_daily = macd_dema(close_daily)[2].iloc[-1]>0
    daily_DMA_3x3 = close_daily.iloc[-1]<DMA(close_daily,3,3).iloc[-1]
    stoch_daily = k_slow_daily.iloc[-1] < d_slow_daily.iloc[-1]
    prob = pd.concat([monthly_DMA_25x5,macd_monthly,daily_DMA_25x5,macd_daily,daily_DMA_3x3,stoch_daily],axis=1).transpose()
    prob.reset_index(drop=True,inplace=True)
    prob.insert(0,"indicators",["monthly_DMA_25x5","macd_monthly","daily_DMA_25x5","macd_daily","daily_DMA_3x3","stoch_daily"])
    prob.set_index("indicators",inplace=True)
    prob_=prob*1
    prob_=(prob_.sum()/6)*100
    return prob,prob_
        
def sell_scan(close_monthly,close_daily,k_slow_daily,d_slow_daily,column):
    """
    Return the probabilities to open a buy position
    """
    k_slow_daily.columns = column
    d_slow_daily.columns = column
    close_monthly.columns = column
    close_daily.columns = column
    monthly_DMA_25x5 = close_monthly.iloc[-1]<DMA(close_monthly).iloc[-1]
    macd_monthly= macd_dema(close_monthly)[2].iloc[-1]<0
    daily_DMA_25x5 = close_daily.iloc[-1]<DMA(close_daily,25,5).iloc[-1]
    macd_daily = macd_dema(close_daily)[2].iloc[-1]<0
    daily_DMA_3x3 = close_daily.iloc[-1]>DMA(close_daily,3,3).iloc[-1]
    stoch_daily = k_slow_daily.iloc[-1] > d_slow_daily.iloc[-1]
    prob = pd.concat([monthly_DMA_25x5,macd_monthly,daily_DMA_25x5,macd_daily,daily_DMA_3x3,stoch_daily],axis=1).transpose()
    prob.reset_index(drop=True,inplace=True)
    prob.insert(0,"indicators",["monthly_DMA_25x5","macd_monthly","daily_DMA_25x5","macd_daily","daily_DMA_3x3","stoch_daily"])
    prob.set_index("indicators",inplace=True)
    prob_=prob*1
    prob_=(prob_.sum()/6)*100
    return prob,prob_

def plot_detrended_oscillator2(oscillator,signal_oscillator,p=5,l=25,h=25):
    """
    Plot detrended oscillator
    Le rette signal rappresentano la soglia per determinare overbought e oversold
    Il dataframe period è diverso dal dataframe quando nei mesi precedenti c'e stata un'alta volatilità che ha
    alterato i valori dell'overbought e oversold quindi non la includiamo per il calcolo delle due linee overbought
    oversold ma scegliamo un perodo precedente o successivo
    signal_up= quantile(0.985)*0.7
    signal_down= quantile(0.015)*0.7
    p si riferisce al periodo da considerare per calcolare il quantile ad esempio 5 per Ipercomprato/ipervenduto di breve termine quindi segnaleranno delle correzioni modeste mentre 20 per correzioni piu significative
    """
    n_assets=oscillator.shape[1]
    vol= oscillator.rolling(2).std()
    plt.figure(figsize=(l,h))
    for n in range(n_assets):
        #signal_up=oscillator[oscillator.columns[n]].rolling(5).quantile(0.985)*0.7 + vol[vol.columns[n]]
        #signal_down=oscillator[oscillator.columns[n]].rolling(5).quantile(0.015)*0.7 - vol[vol.columns[n]]
        signal_up=signal_oscillator[signal_oscillator.columns[n]].rolling(p).quantile(0.985)*0.7
        signal_down=signal_oscillator[signal_oscillator.columns[n]].rolling(p).quantile(0.015)*0.7 
        plt.subplot(math.ceil(n_assets/2),2,n+1)
        plt.plot(oscillator[oscillator.columns[n]])
        plt.axhline(y=0, color='k', linestyle='-')
        plt.plot(signal_up[oscillator.index[0]:oscillator.index[-1]], color='r', linestyle='--')
        plt.plot(signal_down[oscillator.index[0]:oscillator.index[-1]], color='g', linestyle='--')
        plt.title(oscillator.columns[n])
