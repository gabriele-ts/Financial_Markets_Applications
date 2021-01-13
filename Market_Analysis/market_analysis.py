import quandl
def quandl_dataset(dataset, collapse="daily", key="eY51mj69BzDt2VXszsHd"):
    """
    Import a dataset from Quandl
    """
    data = quandl.get(dataset,collapse=collapse, api_key= key)
    data.columns = map(str.lower, data.columns)
    prices= data["settle"]
    high= data["high"]
    low= data["low"]
    volume= data["volume"]
    open_ = data["open"]
    return data,prices,high,low,volume,open_

import numpy as np
import pandas as pd
def quandl_returns(dataset,col_name,collapse="daily",key="eY51mj69BzDt2VXszsHd"):
    """
    Returns calculation from Quandl dataset
    """
    n_assets=len(col_name)
    prices = []
    for n in range(n_assets):
        prices.append(quandl_dataset(dataset[n],collapse=collapse)[1])
    prices=pd.concat(prices,axis=1)
    prices.dropna(inplace=True)
    prices.columns = [col_name]
    ret = np.log(prices) -np.log(prices.shift(1))
    ret = pd.DataFrame(ret)
    ret.columns = [col_name]
    ret.dropna(inplace=True)
    return ret, prices

def annualized_return(returns, freq="Daily"):
    if freq == "Daily":
        ann_ret = returns.mean()*252
    elif freq == "Weekly":
        ann_ret = returns.mean()*52
    elif freq == "Monthly":
        ann_ret = returns.mean()*12
    else:
        print("choose freq= Daily, Weekly or Monthly")
    return ann_ret

def check_ann_log_ret(prices,returns,freq="Daily"):
    """
    coumpounding of the log returns for the total period to verify if 
    compounding the first price for the total period return the correct
    final price
    """
    if freq == "Daily":
        n = returns.shape[0]/252
        last = prices.iloc[0]*np.exp(annualized_return(returns,freq)*n)
    elif freq == "Weekly":
        n = returns.shape[0]/52
        last = prices.iloc[0]*np.exp(annualized_return(returns,freq)*n)
    elif freq == "Monthly":
        n = returns.shape[0]/12
        last = prices.iloc[0]*np.exp(annualized_return(returns,freq)*n)
    else:
        print("choose freq= Daily, Weekly or Monthly")
    return last 

def annualized_volatility(returns, freq="Daily"):
    """
    Calculazion of annualized volatility and return a Dataframe
    """
    period_volatility = returns.std()
    if freq=="Daily":
        ann_vol = period_volatility*np.sqrt(252)
    elif freq=="Weekly":
        ann_vol = period_volatility*np.sqrt(52)
    elif freq=="Monthly":
        ann_vol = period_volatility*np.sqrt(12)
    else:
        print("choose freq= Daily, Weekly or Monthly")
    return ann_vol

def sharpe_ratio(returns,risk_free,freq="Daily"):
    """
    Sharpe Ratio calculation
    """
    sharpe_r = (annualized_return(returns,freq=freq)-risk_free)/annualized_volatility(returns,freq=freq)
    return sharpe_r

def drawdown(returns):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index= 1000*np.exp(returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    drawdowns.columns = ["Drawdowns"]
    return drawdowns

def annualized_semideviation(returns, freq="Daily"):
    """
    Returns the semideviation aka negative semideviation of r
    """
    is_negative = returns < 0
    semideviation = returns[is_negative].std(ddof=0)
    if freq=="Daily":
        ann_sem = semideviation*np.sqrt(252)
    elif freq=="Weekly":
        ann_sem = semideviation*np.sqrt(52)
    elif freq=="Monthly":
        ann_sem = semideviation*np.sqrt(12)
    else:
        print("choose freq= Daily, Weekly or Monthly")
    return ann_sem

def var_historic(returns,level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    var = -np.percentile(returns,level)
    return var

def c_var(returns,level=5):
    """
    Computes the Conditional VaR
    """
    is_beyond = returns <= -var_historic(returns,level=level)
    return -returns[is_beyond].mean()

from scipy.stats import norm
import scipy
def var_gaussian(returns, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = scipy.stats.skew(returns)
        k = scipy.stats.kurtosis(returns)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(returns.mean() + z*returns.std(ddof=0))

import scipy.stats
def is_normal(returns, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(returns)
    return p_value > level

def summary_stats(returns,freq = "Daily",risk_free=0.03):
    ann_ret = returns.aggregate(annualized_return, freq=freq)
    ann_vol = returns.aggregate(annualized_volatility, freq=freq)
    ann_semdev= returns.aggregate(annualized_semideviation, freq=freq)
    ann_sharpe = returns.aggregate(sharpe_ratio,risk_free=risk_free,freq=freq)
    dd = returns.aggregate(lambda returns: drawdown(returns).min())
    skew = returns.aggregate(lambda returns: scipy.stats.skew(returns))
    kurt = returns.aggregate(lambda returns: scipy.stats.kurtosis(returns))
    norm = returns.aggregate(is_normal)
    var5 = returns.aggregate(var_gaussian, modified=True)
    cvar5 = returns.aggregate(c_var)
    
    return pd.DataFrame( {"Annualized Return":ann_ret,
                          "Annualized Volatility":ann_vol,
                          "Annualized Semideviation": ann_semdev,
                          "Annualized Sharpe Ratio":ann_sharpe,
                          "Max Drawdown":dd,
                          "Skewness":skew,
                          "Excess Kurtosis":kurt,
                          "Normality Test":norm,
                          "Cornish-Fisher VaR (5%)":var5,
                          "Historic CVaR (5%)":cvar5
                         })

def dot_com_bubble(data):
    return data["2000-03-13":"2001-09-30"]

def sept_11_attacks(data):
    return data["2001-09-10":"2001-10-5"]

def fin_crisis2007_09(data):
    return data["2007-10-11":"2009-03-15"]

def lehman_crash(data):
    return data["2008-09-15":"2009-03-15"]

def coronavirus(data):
    return data["2020-02-24":]

def moving_average(prices,windows=200):
    return prices.rolling(windows).mean()

import matplotlib.pyplot as plt
import math
def plot_prices(prices,tot_prices, windows=200):
    n_assets=prices.shape[1]
    for n in range(n_assets):
        plt.figure(figsize=(15,15))
        plt.subplot(math.ceil(n_assets/2),2,n+1)
        plt.plot(prices[prices.columns[n]])
        plt.plot(moving_average(tot_prices[tot_prices.columns[n]])[prices.index[0]:prices.index[-1]])
        plt.title(prices.columns[n])
        
import seaborn as sns
def correlation(returns):
    corr = returns.corr()
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr, annot=True, linewidths=.5,vmin=-1, vmax=1, ax=ax)
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    
def rolling_volatility(returns, windows=20, plot=True):
    n_assets=returns.shape[1]
    if plot:
        plt.figure(figsize=(25,10))
        for n in range(n_assets):
            plt.subplot(math.ceil(n_assets/2),2,n+1) 
            plt.plot(returns[returns.columns[n]].rolling(windows).std(ddof=0)*np.sqrt(252))
            plt.title(returns.columns[n])
    return returns.rolling(windows).std(ddof=0)*np.sqrt(252)

def mom(prices):
    return (prices[-1] / prices[0]) *100


def momentum(prices,window=5,plot=True):
    moment = prices.rolling(window=window).apply(mom, raw=True)
    n_assets=prices.shape[1]
    if plot: 
        plt.figure(figsize=(25,10))
        for n in range(n_assets):
            plt.subplot(math.ceil(n_assets/2),2,n+1)
            plt.plot(prices[prices.columns[n]].rolling(window=window).apply(mom, raw=True))
            plt.title(prices.columns[n])
    return moment

def displaced_ma(prices,tot_prices,window=25,shift=5,plot=True,l=25,h=25):
    dis_ma = tot_prices.rolling(window=window).mean().shift(shift)
    n_assets=prices.shape[1]
    if plot:
        plt.figure(figsize=(l,h))
        for n in range(n_assets):
            plt.subplot(math.ceil(n_assets/2),2,n+1)
            plt.plot(prices[prices.columns[n]],label='Price')
            plt.plot(tot_prices[tot_prices.columns[n]].rolling(window=window).mean().shift(shift)[prices.index[0]:prices.index[-1]],label='DMA')
            plt.title(prices.columns[n])
            plt.legend()
    return dis_ma

def open_close_max_min(dataset,col_name,collapse="daily", key="eY51mj69BzDt2VXszsHd"):
    n_assets = len(col_name)
    prices = []
    high = []
    low = []
    open_ = []
    for n in range(n_assets):
        prices.append(quandl_dataset(dataset[n],collapse=collapse)[1])
        high.append(quandl_dataset(dataset[n],collapse=collapse)[2])
        low.append(quandl_dataset(dataset[n],collapse=collapse)[3])
        open_.append(quandl_dataset(dataset[n],collapse=collapse)[5])
    prices=pd.concat(prices,axis=1)
    high=pd.concat(high,axis=1)
    low=pd.concat(low,axis=1)
    open_=pd.concat(open_,axis=1)
    open_prices_max_min = pd.concat([prices,high,low,open_],axis=1)
    open_prices_max_min.dropna(inplace=True)
    close = open_prices_max_min[["settle"]]
    close.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["close"]*n_assets))]
    high = open_prices_max_min[["high"]]
    high.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["high"]*n_assets))]
    low = open_prices_max_min[["low"]]
    low.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["low"]*n_assets))]
    open_ = open_prices_max_min[["open"]]
    open_.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["open"]*n_assets))]
    return close, high, low, open_

def preferred_stochastic(dataset,col_name,collapse="daily",n=8,p=3):
    """
    In dataset insert the functon close_max_min that will return close, high and low of a dataset.
    
    """
    n_assets = len(col_name)
    close = open_close_max_min(dataset,col_name,collapse=collapse)[0]
    high = open_close_max_min(dataset,col_name,collapse=collapse)[1]
    low = open_close_max_min(dataset,col_name,collapse=collapse)[2]
    k_fast = 100*((close.values - low.rolling(n).min())/(high.rolling(n).max().values-low.rolling(n).min()))
    k_fast.dropna(inplace=True)
    k_fast.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["k_fast"]*n_assets))]
    k_slow = mav(k_fast,p)
    k_slow.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["k_slow"]*n_assets))]
    d_slow = mav(k_slow)
    d_slow.columns = [list(map(lambda x,y,z: x + y + z,col_name,["_"]*n_assets,["d_slow"]*n_assets))]
    return k_fast,k_slow,d_slow

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
        
def mav(stochastic,p=3):
    """
    Calculation of modified average
    """
    n_days = stochastic.shape[0] -p
    mav = []
    mav1 = stochastic.rolling(p).mean()
    mav1.dropna(inplace=True)
    mav.append(mav1.values[0])
    for n in range(n_days):
        mav.append(mav[-1]+((stochastic.iloc[n+p]-mav[-1])/p))
    return pd.DataFrame(mav,stochastic.index[p-1:],columns=stochastic.columns)

def volume(dataset,col_name, key="eY51mj69BzDt2VXszsHd"):
    n_assets = len(col_name)
    volume = []
    for n in range(n_assets):
        volume.append(quandl_dataset(dataset[n])[4])
    volume=pd.concat(volume,axis=1)
    volume.dropna(inplace=True)
    volume.columns = [col_name]
    return volume

def open_interest(dataset,col_name, key="eY51mj69BzDt2VXszsHd"):
    n_assets = len(col_name)
    open_interest = []
    for n in range(n_assets):
        open_interest.append(quandl_dataset(dataset[n])[5])
    open_interest=pd.concat(open_interest,axis=1)
    open_interest.dropna(inplace=True)
    open_interest.columns = [col_name]
    return open_interest

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
    return macd, signal_line

def plot_macd(macd,signal,l=25,h=25):
    """
    Plot macd
    """
    n_assets=macd.shape[1]
    plt.figure(figsize=(l,h))
    for n in range(n_assets):
        plt.subplot(math.ceil(n_assets/2),2,n+1)
        plt.plot(macd[macd.columns[n]], label='MACD')
        plt.plot(signal[signal.columns[n]], label='Signal')
        plt.title(macd.columns[n])
        plt.legend()
        
def detrended_oscillator(prices,freq=7):
    n_days = prices.shape[0] -(freq-1)
    mov_av = prices.rolling(freq).mean()
    detrended = []
    for n in range (n_days):
        detrended.append(prices.iloc[n+(freq-1)] - mov_av.iloc[n+(freq-1)])
    detrended = pd.DataFrame(detrended)
    return detrended

def plot_detrended_oscillator(oscillator,l=25,h=25):
    """
    Plot detrended oscillator
    """
    n_assets=oscillator.shape[1]
    plt.figure(figsize=(l,h))
    for n in range(n_assets):
        plt.subplot(math.ceil(n_assets/2),2,n+1)
        plt.plot(oscillator[oscillator.columns[n]])
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title(oscillator.columns[n])
        

def plot_detrended_oscillator(oscillator,l=25,h=25):
    """
    Plot detrended oscillator
    """
    n_assets=oscillator.shape[1]
    plt.figure(figsize=(l,h))
    for n in range(n_assets):
        plt.subplot(math.ceil(n_assets/2),2,n+1)
        plt.plot(oscillator[oscillator.columns[n]])
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title(oscillator.columns[n])
        
"""
a= detrend[["Schatz"]].copy()
n= pd.DataFrame()
n= n.append(a.loc[a.idxmax()])
a.drop(a.idxmax(),inplace=True)
n= n.append(a.loc[a.idxmax()])
a.drop(a.idxmax(),inplace=True)
n= n.append(a.loc[a.idxmax()])

"""

def roney_ratio(dataset,col_name,collapse="daily",n=30):
    n_assets = len(col_name)
    close = open_close_max_min(dataset,col_name,collapse=collapse)[0]
    high = open_close_max_min(dataset,col_name,collapse=collapse)[1]
    low = open_close_max_min(dataset,col_name,collapse=collapse)[2]
    open_=open_close_max_min(dataset,col_name,collapse=collapse)[3]
    net_change= close.values-open_
    max_min_range = high.values - low
    average_range=max_min_range.rolling(n).mean()
    roney_ratio = net_change.values/average_range
    roney_ratio.dropna(inplace=True)
    return roney_ratio