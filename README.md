 #  A Whale off the Port(folio)

 In this assignment, you'll get to use what you've learned this week to evaluate the performance among various algorithmic, hedge, and mutual fund portfolios and compare them against the S&P 500.


```python
import pandas as pd
import numpy as np
import os
import Risk_Kit2 as rk
import datetime as dt
from scipy.optimize import minimize
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
%load_ext autoreload
%autoreload 2
%matplotlib inline
```

## I used several functions in my analysis: Some of these functions were taken from 2 Coursera courses that I completed: 
## 1. [Introduction to Portfolio Construction and Analysis with Python](https://www.coursera.org/learn/introduction-portfolio-construction-python)
## 2. [Advanced Portfolio Construction and Analysis with Python](https://www.coursera.org/learn/advancedportfolio-construction-python)
## Some of the functions that were taken from these courses were modified to fit the requirements of this assignment. 

# Data Cleaning

In this section, you will need to read the CSV files into DataFrames and perform any necessary data cleaning steps. After cleaning, combine all DataFrames into a single DataFrame.

Files:
1. whale_returns.csv
2. algo_returns.csv
3. sp500_history.csv

## Whale Returns
Read the Whale Portfolio daily returns and clean the data

### Because I've included these .csv files in my repository, I've commented out the paths I used to import the files while working from my computer


```python
# Reading whale returns
# common_path=Path('Homework_2')
# whale_returns_csv=os.path.join(common_path,'whale_returns.csv')
# algo_returns_csv=os.path.join(common_path,'algo_returns.csv')
# sp500_history_csv=os.path.join(common_path,'sp500_history.csv')

whale_returns=pd.read_csv('whale_returns.csv', index_col='Date', parse_dates=True, infer_datetime_format=True)

# whale_returns=pd.read_csv(whale_returns_csv, index_col='Date', parse_dates=True, infer_datetime_format=True)
whale_returns.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <th>PAULSON &amp; CO.INC.</th>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <th>BERKSHIRE HATHAWAY INC</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-03-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2015-03-03</td>
      <td>-0.001266</td>
      <td>-0.004981</td>
      <td>-0.000496</td>
      <td>-0.006569</td>
    </tr>
    <tr>
      <td>2015-03-04</td>
      <td>0.002230</td>
      <td>0.003241</td>
      <td>-0.002534</td>
      <td>0.004213</td>
    </tr>
    <tr>
      <td>2015-03-05</td>
      <td>0.004016</td>
      <td>0.004076</td>
      <td>0.002355</td>
      <td>0.006726</td>
    </tr>
    <tr>
      <td>2015-03-06</td>
      <td>-0.007905</td>
      <td>-0.003574</td>
      <td>-0.008481</td>
      <td>-0.013098</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Count nulls
whale_returns.isnull().sum()
```




    SOROS FUND MANAGEMENT LLC      0
    PAULSON & CO.INC.              0
    TIGER GLOBAL MANAGEMENT LLC    0
    BERKSHIRE HATHAWAY INC         0
    dtype: int64




```python
# Drop nulls
whale_returns.dropna(inplace=True)
whale_returns.isna().sum()
```




    SOROS FUND MANAGEMENT LLC      0
    PAULSON & CO.INC.              0
    TIGER GLOBAL MANAGEMENT LLC    0
    BERKSHIRE HATHAWAY INC         0
    dtype: int64



## Algorithmic Daily Returns

Read the algorithmic daily returns and clean the data


```python
# Reading algorithmic returns
algo_returns=pd.read_csv('algo_returns.csv', index_col='Date', parse_dates=True, infer_datetime_format=True)
algo_returns.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algo 1</th>
      <th>Algo 2</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2014-05-28</td>
      <td>0.001745</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2014-05-29</td>
      <td>0.003978</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2014-05-30</td>
      <td>0.004464</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2014-06-02</td>
      <td>0.005692</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2014-06-03</td>
      <td>0.005292</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Count nulls
algo_returns.isna().sum()
```




    Algo 1    0
    Algo 2    6
    dtype: int64




```python
# Drop nulls
algo_returns.dropna(inplace=True)
algo_returns.isna().sum()
```




    Algo 1    0
    Algo 2    0
    dtype: int64



## S&P 500 Returns

Read the S&P500 Historic Closing Prices and create a new daily returns DataFrame from the data. 


```python
# Reading S&P 500 Closing Prices
sp500_returns=pd.read_csv('sp500_history.csv', index_col='Date', parse_dates=True, infer_datetime_format=True)
sp500_returns.sort_index(inplace=True)
sp500_returns.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2012-10-01</td>
      <td>$1444.49</td>
    </tr>
    <tr>
      <td>2012-10-02</td>
      <td>$1445.75</td>
    </tr>
    <tr>
      <td>2012-10-03</td>
      <td>$1450.99</td>
    </tr>
    <tr>
      <td>2012-10-04</td>
      <td>$1461.40</td>
    </tr>
    <tr>
      <td>2012-10-05</td>
      <td>$1460.93</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check Data Types
sp500_returns.dtypes
```




    Close    object
    dtype: object




```python
# Fix Data Types
sp500_returns['Close']=sp500_returns['Close'].str.replace('$', '')
sp500_returns['Close']=sp500_returns['Close'].astype('float')
sp500_returns.dtypes
```




    Close    float64
    dtype: object




```python
# Calculate Daily Returns
sp500_daily_rets=sp500_returns.pct_change()
```


```python
# Drop nulls
sp500_daily_rets.dropna(inplace=True)
sp500_daily_rets.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2012-10-02</td>
      <td>0.000872</td>
    </tr>
    <tr>
      <td>2012-10-03</td>
      <td>0.003624</td>
    </tr>
    <tr>
      <td>2012-10-04</td>
      <td>0.007174</td>
    </tr>
    <tr>
      <td>2012-10-05</td>
      <td>-0.000322</td>
    </tr>
    <tr>
      <td>2012-10-08</td>
      <td>-0.003457</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Rename Columns
sp500_daily_rets.columns=(['S&P 500'])
```

## Combine Whale, Algorithmic, and S&P 500 Returns


```python
# Concatenate all DataFrames into a single DataFrame
combined_df=pd.concat([whale_returns, algo_returns, sp500_daily_rets], axis=1, join='inner')
combined_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <th>PAULSON &amp; CO.INC.</th>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <th>BERKSHIRE HATHAWAY INC</th>
      <th>Algo 1</th>
      <th>Algo 2</th>
      <th>S&amp;P 500</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-03-03</td>
      <td>-0.001266</td>
      <td>-0.004981</td>
      <td>-0.000496</td>
      <td>-0.006569</td>
      <td>-0.001942</td>
      <td>-0.000949</td>
      <td>-0.004539</td>
    </tr>
    <tr>
      <td>2015-03-04</td>
      <td>0.002230</td>
      <td>0.003241</td>
      <td>-0.002534</td>
      <td>0.004213</td>
      <td>-0.008589</td>
      <td>0.002416</td>
      <td>-0.004389</td>
    </tr>
    <tr>
      <td>2015-03-05</td>
      <td>0.004016</td>
      <td>0.004076</td>
      <td>0.002355</td>
      <td>0.006726</td>
      <td>-0.000955</td>
      <td>0.004323</td>
      <td>0.001196</td>
    </tr>
    <tr>
      <td>2015-03-06</td>
      <td>-0.007905</td>
      <td>-0.003574</td>
      <td>-0.008481</td>
      <td>-0.013098</td>
      <td>-0.004957</td>
      <td>-0.011460</td>
      <td>-0.014174</td>
    </tr>
    <tr>
      <td>2015-03-09</td>
      <td>0.000582</td>
      <td>0.004225</td>
      <td>0.005843</td>
      <td>-0.001652</td>
      <td>-0.005447</td>
      <td>0.001303</td>
      <td>0.003944</td>
    </tr>
  </tbody>
</table>
</div>



---

# Portfolio Analysis

In this section, you will calculate and visualize performance and risk metrics for the portfolios.

## Performance

Calculate and Plot the daily returns and cumulative returns. Does any portfolio outperform the S&P 500? 


```python
# Plot daily returns
combined_df.plot(figsize=(18,12))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x20c96335f48>




![png](output_25_1.png)


## Algo 1 and Berkshire Hathaway both had better total returns than the S&P 500


```python
# Plot cumulative returns
(1+combined_df).cumprod().plot(figsize=(18,12))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x20c97d92888>




![png](output_27_1.png)


---

## Risk

Determine the _risk_ of each portfolio:

1. Create a box plot for each portfolio. 
2. Calculate the standard deviation for all portfolios
4. Determine which portfolios are riskier than the S&P 500
5. Calculate the Annualized Standard Deviation


```python
# Box plot to visually show risk
combined_df.boxplot(figsize=(18,12))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b48784088>




![png](output_30_1.png)


### The box plot shows that Tiger and Berkshire Hathaway have more dispersion of returns than the S&P 500, while Paulson and Algos 1 and 2 have lower dispersion of returns
* ### Algo 1 has about equal upside dispersion as the S&P 500, but its downside returns cluster closer together at less negative values.


```python
# Daily Standard Deviations
# Calculate the standard deviation for each portfolio. 
# Which portfolios are riskier than the S&P 500?
combined_df.std()
```




    SOROS FUND MANAGEMENT LLC      0.007895
    PAULSON & CO.INC.              0.007023
    TIGER GLOBAL MANAGEMENT LLC    0.010894
    BERKSHIRE HATHAWAY INC         0.012919
    Algo 1                         0.007620
    Algo 2                         0.008342
    S&P 500                        0.008554
    dtype: float64



### Tiger Global and Berkshire Hathaway both had higher risk than the S&P 500 in terms of volatility


```python
# Determine which portfolios are riskier than the S&P 500
combined_df.std()>combined_df.std()['S&P 500']
```




    SOROS FUND MANAGEMENT LLC      False
    PAULSON & CO.INC.              False
    TIGER GLOBAL MANAGEMENT LLC     True
    BERKSHIRE HATHAWAY INC          True
    Algo 1                         False
    Algo 2                         False
    S&P 500                        False
    dtype: bool



### Annualized Standard Deviation


```python
combined_df.std()*np.sqrt(252)
```




    SOROS FUND MANAGEMENT LLC      0.125335
    PAULSON & CO.INC.              0.111488
    TIGER GLOBAL MANAGEMENT LLC    0.172936
    BERKSHIRE HATHAWAY INC         0.205077
    Algo 1                         0.120967
    Algo 2                         0.132430
    S&P 500                        0.135786
    dtype: float64



## Rolling Statistics

Risk changes over time. Analyze the rolling statistics for Risk and Beta. 

1. Plot the rolling standard deviation of the various portfolios along with the rolling standard deviation of the S&P 500 (consider a 21 day window). Does the risk increase for each of the portfolios at the same time risk increases in the S&P?
2. Construct a correlation table for the algorithmic, whale, and S&P 500 returns. Which returns most closely mimic the S&P?
2. Choose one portfolio and plot a rolling beta between that portfolio's returns and S&P 500 returns. Does the portfolio seem sensitive to movements in the S&P 500?


```python
# Calculate and plot the rolling standard deviation for
# the S&P 500 and whale portfolios using a 21 trading day window
combined_df.rolling(21).std().plot(figsize=(18,12), title='Rolling Std Dev Method 1')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b48c221c8>




![png](output_38_1.png)


### It looks like volatility for most of the portfolios rises as the volatility of the S&P 500 rises, however, Tiger Global's volatility seems to spike upwards at random times. It has risen both when the S&P's volatility is rising but also when the S&P's volatility is falling

---

### Method 2: Here I construct my own rolling windows of size 21. I then return a list of the returns from the for each window.
* ## I used the following function


```python
def get_rolled_returns(rets, window_length):
    n_periods=rets.shape[0]
    start=n_periods-window_length # total number of windows needed
    
    windows=[(x, x+window_length) for x in range(start)] # Tuples of all the rolling windows
    rolled_returns=[rets.iloc[win[0]:win[1]] for win in windows] # Returns from start to end of each window. 
    # Returns list of DataFrames
    return rolled_returns
```


```python
rolled_rets=get_rolled_returns(combined_df, 21)
# Std Dev for each window. I then shift the index up by 21 periods, so that the first Std Dev calculated using periods
# 0-21 will be the Std Dev for period 21
rolled_std=pd.DataFrame(data=[r.std() for r in rolled_rets], index=combined_df.index[21:], columns=combined_df.columns)
rolled_std.plot(figsize=(18,12), title='Rolling Std Dev Method 2')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b48d98b88>




![png](output_43_1.png)



```python
# Construct a correlation table
combined_df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <th>PAULSON &amp; CO.INC.</th>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <th>BERKSHIRE HATHAWAY INC</th>
      <th>Algo 1</th>
      <th>Algo 2</th>
      <th>S&amp;P 500</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SOROS FUND MANAGEMENT LLC</td>
      <td>1.000000</td>
      <td>0.699914</td>
      <td>0.561243</td>
      <td>0.754360</td>
      <td>0.321211</td>
      <td>0.826873</td>
      <td>0.837864</td>
    </tr>
    <tr>
      <td>PAULSON &amp; CO.INC.</td>
      <td>0.699914</td>
      <td>1.000000</td>
      <td>0.434479</td>
      <td>0.545623</td>
      <td>0.268840</td>
      <td>0.678152</td>
      <td>0.669732</td>
    </tr>
    <tr>
      <td>TIGER GLOBAL MANAGEMENT LLC</td>
      <td>0.561243</td>
      <td>0.434479</td>
      <td>1.000000</td>
      <td>0.424423</td>
      <td>0.164387</td>
      <td>0.507414</td>
      <td>0.623946</td>
    </tr>
    <tr>
      <td>BERKSHIRE HATHAWAY INC</td>
      <td>0.754360</td>
      <td>0.545623</td>
      <td>0.424423</td>
      <td>1.000000</td>
      <td>0.292033</td>
      <td>0.688082</td>
      <td>0.751371</td>
    </tr>
    <tr>
      <td>Algo 1</td>
      <td>0.321211</td>
      <td>0.268840</td>
      <td>0.164387</td>
      <td>0.292033</td>
      <td>1.000000</td>
      <td>0.288243</td>
      <td>0.279494</td>
    </tr>
    <tr>
      <td>Algo 2</td>
      <td>0.826873</td>
      <td>0.678152</td>
      <td>0.507414</td>
      <td>0.688082</td>
      <td>0.288243</td>
      <td>1.000000</td>
      <td>0.858764</td>
    </tr>
    <tr>
      <td>S&amp;P 500</td>
      <td>0.837864</td>
      <td>0.669732</td>
      <td>0.623946</td>
      <td>0.751371</td>
      <td>0.279494</td>
      <td>0.858764</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
combined_df.corr()[combined_df.corr().iloc[:,-1]<1].idxmax()
```




    SOROS FUND MANAGEMENT LLC        SOROS FUND MANAGEMENT LLC
    PAULSON & CO.INC.                       PAULSON & CO.INC. 
    TIGER GLOBAL MANAGEMENT LLC    TIGER GLOBAL MANAGEMENT LLC
    BERKSHIRE HATHAWAY INC              BERKSHIRE HATHAWAY INC
    Algo 1                                              Algo 1
    Algo 2                                              Algo 2
    S&P 500                                             Algo 2
    dtype: object



### We can see that Algo 2 is the most highly correlated with the S&P 500

### Rolling Beta for Tiger Global


```python
# Calculate Beta for a single portfolio compared to the total market (S&P 500)
# (Your graph may differ, dependent upon which portfolio you are comparing)
rolling_cov=combined_df['TIGER GLOBAL MANAGEMENT LLC'].rolling(21).cov(combined_df['S&P 500'])
rolling_market_var=combined_df['S&P 500'].rolling(21).var()
rolling_beta=rolling_cov/rolling_market_var
rolling_beta.plot(figsize=(18,12))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b48f2fa48>




![png](output_48_1.png)


### Rolling Beta of Tiger Global found by constructing windows for each 21 day return period. Results look pretty close to Method 1.
* ### I used the following functions


```python
def get_beta_from_windows(all_rets, rets, window_length): 
    """
    I use the rolling returns that I previously calculated (rolled_rets). I pass each of these 21 period DataFrames into
    the fin_Beta function, and a beta for each 21 day period is returned. Then I put all of these Betas into a DataFrame
    and again shift the index up by 21 periods, so that the first Beta calculated applies to the 21st period.
    """
    rolling_betas=[find_Beta(r) for r in rets]
    return pd.DataFrame(data=rolling_betas, index=all_rets.index[window_length:], columns=all_rets.columns)

def find_Beta(rets):
    corr_mat=rets.corr()
    location=rets.columns.get_loc('S&P 500')
    vols=rets.std()
    covmat=(corr_mat.mul(vols, axis=0)).mul(vols.T, axis=1)
    cov_market=pd.DataFrame(covmat.iloc[location,:])
    market_var=cov_market.iloc[location]
    beta_mat=cov_market/market_var[0]
    return beta_mat['S&P 500']
```

### Tiger Global looks to be extremely sensitive to market movements. Its beta gets near 2 which I think is very high as it means that Tiger can move twice as much as the market at times. It seems to range between 2 and around .2-.3, but during a period in 2019 the beta dropped to -1 all of a sudden. Maybe the manager suddenly switched to negatively correlated assets as a defensive move? 


```python
rolled_betas=get_beta_from_windows(combined_df, rolled_rets, 21)
rolled_betas['TIGER GLOBAL MANAGEMENT LLC'].plot(figsize=(18,12), title='Rolling Beta Method 2')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b4daf23c8>




![png](output_52_1.png)


### Exponentially Weighted Average using the built-in 'ewm' function, using 21 as the halflife parameter. I also used the following formula to calculate alpha based on a halflife of 21 periods and graphed the result. The result was similar. This formula for alpha was found in the documentation of the ewm method

## $\alpha$ = 1 - $e^{ln(0.5) / halflife}$

### Challenge: Exponentially Weighted Average 

An alternative way to calculate a rolling window is to take the exponentially weighted moving average. This is like a moving window average, but it assigns greater importance to more recent observations. Try calculating the `ewm` with a 21 day half-life.


```python
# (OPTIONAL) YOUR CODE HERE
combined_df_ewm=combined_df.ewm(halflife=21).mean()
combined_df_ewm.plot(figsize=(18,12))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b49067f48>




![png](output_56_1.png)


### Method 2: Here I use the returns on the 20th trading day as my initial EMA (Exponentially Weighted Average). Then I use the following formula to calculate the next period's EMA: I use N=21
## $EMA_{t}$ =($Returns_t$ - $EMA_{t-1}$)*$\frac{2}{N+1}$+$EMA_{t-1}$
* ### I used the following function


```python
def get_my_ema(rets, window_length):
    """
    I append the returns from period 20 to my empty ema_list. I then treat this first period of returns as my initial
    EMA. Then, I use the next row of returns along with the returns previously appended to ema_list to calculate my new EMA.
    I repeat for the number of windows in the entire time period.
    """
    n_times=rets.shape[0]-window_length
    ema_list=[]
    ema_list.append(rets.iloc[window_length-1])
    k=2/(window_length+1)
    x=1
    for x in range(1,n_times):
        ema_list.append((rets.iloc[window_length-1+x]-ema_list[x-1])*k+ema_list[x-1])
    return pd.DataFrame(data=ema_list, index=rets.index[window_length:], columns=rets.columns)
```


```python
## Method 2
combined_df_ema=get_my_ema(combined_df, 21)
combined_df_ema.plot(figsize=(18,12), title='EMA using k=2/(N+1)')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b4db9da48>




![png](output_59_1.png)


## Sharpe Ratios
In reality, investment managers and thier institutional investors look at the ratio of return-to-risk, and not just returns alone. (After all, if you could invest in one of two portfolios, each offered the same 10% return, yet one offered lower risk, you'd take that one, right?)

Calculate and plot the annualized Sharpe ratios for all portfolios to determine which portfolio has the best performance


```python
# Annualized Sharpe Ratios
sharpes=(combined_df.mean()*252)/(combined_df.std()*np.sqrt(252))
sharpes
```




    SOROS FUND MANAGEMENT LLC      0.356417
    PAULSON & CO.INC.             -0.483570
    TIGER GLOBAL MANAGEMENT LLC   -0.121060
    BERKSHIRE HATHAWAY INC         0.621810
    Algo 1                         1.378648
    Algo 2                         0.501364
    S&P 500                        0.648267
    dtype: float64



### Sharpe Ratios show that Algo 1 outperforms every other portfolio. Algo 2 outperforms 3 managers, but has a lower Sharpe than Berkshire Hathaway and the S&P 500

### Sortino Ratios
* ### I used the following functions


```python
def sortino_ratio(r, riskfree_rate, periods_per_year):
    rf_per_period=(1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret=r-rf_per_period
    ann_ex_ret=annualize_rets(excess_ret, periods_per_year)
    # I only use negative returns to calculate the volatility, although when standard deviation is calculated, I think that
    # python still uses the total number of returns as the denominator 'N', both positive and negative,
    ann_neg_vol=annualize_vol(r[r<0], periods_per_year)
    return ann_ex_ret/ann_neg_vol

def annualize_vol(r, periods_per_year):
    return r.std()*(periods_per_year**.5)

def annualize_rets(r, periods_per_year):
    compounded_growth=(1+r).prod()
    n_periods=r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1
```


```python
sortino_ratio(combined_df, 0, 252)
```




    SOROS FUND MANAGEMENT LLC      0.390080
    PAULSON & CO.INC.             -0.708171
    TIGER GLOBAL MANAGEMENT LLC   -0.237255
    BERKSHIRE HATHAWAY INC         0.699785
    Algo 1                         2.427960
    Algo 2                         0.593063
    S&P 500                        0.746851
    dtype: float64



 ### plot() these sharpe ratios using a barplot. On the basis of this performance metric, do our algo strategies outperform both 'the market' and the whales?
 * ### Algo 1 outperforms everything else by far. Algo 2 is in the middle of the pack. Interestingly, Algo 1's lead over the other portfolios increases when we use the Sortino ratio and only consider the downside volatility. 


```python
# Visualize the sharpe ratios as a bar plot
sharpes.plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b4e1c5808>




![png](output_67_1.png)



```python
# Sortino Ratios
sortino_ratio(combined_df, 0, 252).plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b4e24a308>




![png](output_68_1.png)


---

# Portfolio Returns

In this section, you will build your own portfolio of stocks, calculate the returns, and compare the results to the Whale Portfolios and the S&P 500. 

1. Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.
2. Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock
3. Join your portfolio returns to the DataFrame that contains all of the portfolio returns
4. Re-run the performance and risk analysis with your portfolio to see how it compares to the others
5. Include correlation analysis to determine which stocks (if any) are correlated

## Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.


```python
my_stocks=pd.read_csv('3_5_stocks.csv')
my_stocks
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Close</th>
      <th>Date.1</th>
      <th>Close.1</th>
      <th>Date.2</th>
      <th>Close.2</th>
      <th>Date.3</th>
      <th>Close.3</th>
      <th>Date.4</th>
      <th>Close.4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3/3/2015 16:00:00</td>
      <td>102.34</td>
      <td>3/3/2015 16:00:00</td>
      <td>56.46</td>
      <td>3/3/2015 16:00:00</td>
      <td>87.62</td>
      <td>3/3/2015 16:00:00</td>
      <td>7.13</td>
      <td>3/3/2015 16:00:00</td>
      <td>81.58</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3/4/2015 16:00:00</td>
      <td>101.65</td>
      <td>3/4/2015 16:00:00</td>
      <td>55.79</td>
      <td>3/4/2015 16:00:00</td>
      <td>87.18</td>
      <td>3/4/2015 16:00:00</td>
      <td>6.88</td>
      <td>3/4/2015 16:00:00</td>
      <td>85.49</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3/5/2015 16:00:00</td>
      <td>102.52</td>
      <td>3/5/2015 16:00:00</td>
      <td>55.50</td>
      <td>3/5/2015 16:00:00</td>
      <td>86.74</td>
      <td>3/5/2015 16:00:00</td>
      <td>6.53</td>
      <td>3/5/2015 16:00:00</td>
      <td>86.10</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3/6/2015 16:00:00</td>
      <td>100.11</td>
      <td>3/6/2015 16:00:00</td>
      <td>53.37</td>
      <td>3/6/2015 16:00:00</td>
      <td>85.63</td>
      <td>3/6/2015 16:00:00</td>
      <td>6.42</td>
      <td>3/6/2015 16:00:00</td>
      <td>84.40</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3/9/2015 16:00:00</td>
      <td>100.66</td>
      <td>3/9/2015 16:00:00</td>
      <td>53.83</td>
      <td>3/9/2015 16:00:00</td>
      <td>85.16</td>
      <td>3/9/2015 16:00:00</td>
      <td>6.23</td>
      <td>3/9/2015 16:00:00</td>
      <td>82.53</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1038</td>
      <td>4/16/2019 16:00:00</td>
      <td>138.02</td>
      <td>4/16/2019 16:00:00</td>
      <td>56.39</td>
      <td>4/16/2019 16:00:00</td>
      <td>81.20</td>
      <td>4/16/2019 16:00:00</td>
      <td>13.57</td>
      <td>4/16/2019 16:00:00</td>
      <td>185.78</td>
    </tr>
    <tr>
      <td>1039</td>
      <td>4/17/2019 16:00:00</td>
      <td>138.52</td>
      <td>4/17/2019 16:00:00</td>
      <td>56.18</td>
      <td>4/17/2019 16:00:00</td>
      <td>81.43</td>
      <td>4/17/2019 16:00:00</td>
      <td>13.27</td>
      <td>4/17/2019 16:00:00</td>
      <td>187.55</td>
    </tr>
    <tr>
      <td>1040</td>
      <td>4/18/2019 16:00:00</td>
      <td>137.52</td>
      <td>4/18/2019 16:00:00</td>
      <td>54.37</td>
      <td>4/18/2019 16:00:00</td>
      <td>81.13</td>
      <td>4/18/2019 16:00:00</td>
      <td>13.32</td>
      <td>4/18/2019 16:00:00</td>
      <td>186.94</td>
    </tr>
    <tr>
      <td>1041</td>
      <td>4/22/2019 16:00:00</td>
      <td>137.83</td>
      <td>4/22/2019 16:00:00</td>
      <td>54.61</td>
      <td>4/22/2019 16:00:00</td>
      <td>82.90</td>
      <td>4/22/2019 16:00:00</td>
      <td>13.06</td>
      <td>4/22/2019 16:00:00</td>
      <td>185.38</td>
    </tr>
    <tr>
      <td>1042</td>
      <td>4/23/2019 16:00:00</td>
      <td>139.90</td>
      <td>4/23/2019 16:00:00</td>
      <td>54.82</td>
      <td>4/23/2019 16:00:00</td>
      <td>83.38</td>
      <td>4/23/2019 16:00:00</td>
      <td>13.17</td>
      <td>4/23/2019 16:00:00</td>
      <td>187.29</td>
    </tr>
  </tbody>
</table>
<p>1043 rows × 10 columns</p>
</div>



* # JNJ


```python
# Read the first stock
jnj=my_stocks.iloc[:,[0,1]]
jnj.set_index(['Date'], inplace=True)
jnj.index=pd.to_datetime(jnj.index)
jnj.insert(0, column='Symbol', value='JNJ')
jnj.index=jnj.index.normalize()
jnj.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-03-03</td>
      <td>JNJ</td>
      <td>102.34</td>
    </tr>
    <tr>
      <td>2015-03-04</td>
      <td>JNJ</td>
      <td>101.65</td>
    </tr>
    <tr>
      <td>2015-03-05</td>
      <td>JNJ</td>
      <td>102.52</td>
    </tr>
    <tr>
      <td>2015-03-06</td>
      <td>JNJ</td>
      <td>100.11</td>
    </tr>
    <tr>
      <td>2015-03-09</td>
      <td>JNJ</td>
      <td>100.66</td>
    </tr>
  </tbody>
</table>
</div>



* # MO


```python
# Read the second stock
mo=my_stocks.iloc[:,2:4]
mo.set_index(['Date.1'], inplace=True)
mo.index=pd.to_datetime(mo.index)
mo.insert(0, column='Symbol', value='MO')
mo.index=mo.index.normalize()
mo.rename(columns={'Close.1': 'Close'}, inplace=True)
mo.index.name='Date'
mo.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-03-03</td>
      <td>MO</td>
      <td>56.46</td>
    </tr>
    <tr>
      <td>2015-03-04</td>
      <td>MO</td>
      <td>55.79</td>
    </tr>
    <tr>
      <td>2015-03-05</td>
      <td>MO</td>
      <td>55.50</td>
    </tr>
    <tr>
      <td>2015-03-06</td>
      <td>MO</td>
      <td>53.37</td>
    </tr>
    <tr>
      <td>2015-03-09</td>
      <td>MO</td>
      <td>53.83</td>
    </tr>
  </tbody>
</table>
</div>



* # XOM


```python
# Read the third stock
xom=my_stocks.iloc[:,4:6]
xom.set_index(['Date.2'], inplace=True)
xom.index=pd.to_datetime(xom.index)
xom.insert(0, column='Symbol', value='XOM')
xom.index=xom.index.normalize()
xom.rename(columns={'Close.2': 'Close'}, inplace=True)
xom.index.name='Date'
xom.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-03-03</td>
      <td>XOM</td>
      <td>87.62</td>
    </tr>
    <tr>
      <td>2015-03-04</td>
      <td>XOM</td>
      <td>87.18</td>
    </tr>
    <tr>
      <td>2015-03-05</td>
      <td>XOM</td>
      <td>86.74</td>
    </tr>
    <tr>
      <td>2015-03-06</td>
      <td>XOM</td>
      <td>85.63</td>
    </tr>
    <tr>
      <td>2015-03-09</td>
      <td>XOM</td>
      <td>85.16</td>
    </tr>
  </tbody>
</table>
</div>



* # VALE


```python
vale=my_stocks.iloc[:,6:8]
vale.set_index(['Date.3'], inplace=True)
vale.index=pd.to_datetime(vale.index)
vale.insert(0, column='Symbol', value='VALE')
vale.index=vale.index.normalize()
vale.rename(columns={'Close.3': 'Close'}, inplace=True)
vale.index.name='Date'
vale.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-03-03</td>
      <td>VALE</td>
      <td>7.13</td>
    </tr>
    <tr>
      <td>2015-03-04</td>
      <td>VALE</td>
      <td>6.88</td>
    </tr>
    <tr>
      <td>2015-03-05</td>
      <td>VALE</td>
      <td>6.53</td>
    </tr>
    <tr>
      <td>2015-03-06</td>
      <td>VALE</td>
      <td>6.42</td>
    </tr>
    <tr>
      <td>2015-03-09</td>
      <td>VALE</td>
      <td>6.23</td>
    </tr>
  </tbody>
</table>
</div>



* # BABA


```python
baba=my_stocks.iloc[:,8:10]
baba.set_index(['Date.4'], inplace=True)
baba.index=pd.to_datetime(baba.index)
baba.insert(0, column='Symbol', value='BABA')
baba.index=baba.index.normalize()
baba.rename(columns={'Close.4': 'Close'}, inplace=True)
baba.index.name='Date'
baba.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-03-03</td>
      <td>BABA</td>
      <td>81.58</td>
    </tr>
    <tr>
      <td>2015-03-04</td>
      <td>BABA</td>
      <td>85.49</td>
    </tr>
    <tr>
      <td>2015-03-05</td>
      <td>BABA</td>
      <td>86.10</td>
    </tr>
    <tr>
      <td>2015-03-06</td>
      <td>BABA</td>
      <td>84.40</td>
    </tr>
    <tr>
      <td>2015-03-09</td>
      <td>BABA</td>
      <td>82.53</td>
    </tr>
  </tbody>
</table>
</div>



# Concatenate all stocks into a single DataFrame


```python
my_stocks=pd.concat([jnj,mo,xom,vale,baba], axis=1, join='inner')
my_stocks=my_stocks.drop(columns=my_stocks.columns[::2])
my_stocks.columns=(['JNJ', 'MO', 'XOM', 'VALE', 'BABA'])
my_stocks.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>JNJ</th>
      <th>MO</th>
      <th>XOM</th>
      <th>VALE</th>
      <th>BABA</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-03-03</td>
      <td>102.34</td>
      <td>56.46</td>
      <td>87.62</td>
      <td>7.13</td>
      <td>81.58</td>
    </tr>
    <tr>
      <td>2015-03-04</td>
      <td>101.65</td>
      <td>55.79</td>
      <td>87.18</td>
      <td>6.88</td>
      <td>85.49</td>
    </tr>
    <tr>
      <td>2015-03-05</td>
      <td>102.52</td>
      <td>55.50</td>
      <td>86.74</td>
      <td>6.53</td>
      <td>86.10</td>
    </tr>
    <tr>
      <td>2015-03-06</td>
      <td>100.11</td>
      <td>53.37</td>
      <td>85.63</td>
      <td>6.42</td>
      <td>84.40</td>
    </tr>
    <tr>
      <td>2015-03-09</td>
      <td>100.66</td>
      <td>53.83</td>
      <td>85.16</td>
      <td>6.23</td>
      <td>82.53</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Pivot so that each column of prices represents a unique symbol
my_stocks=my_stocks.rename_axis('Symbol', axis=1)
my_stocks.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbol</th>
      <th>JNJ</th>
      <th>MO</th>
      <th>XOM</th>
      <th>VALE</th>
      <th>BABA</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-03-03</td>
      <td>102.34</td>
      <td>56.46</td>
      <td>87.62</td>
      <td>7.13</td>
      <td>81.58</td>
    </tr>
    <tr>
      <td>2015-03-04</td>
      <td>101.65</td>
      <td>55.79</td>
      <td>87.18</td>
      <td>6.88</td>
      <td>85.49</td>
    </tr>
    <tr>
      <td>2015-03-05</td>
      <td>102.52</td>
      <td>55.50</td>
      <td>86.74</td>
      <td>6.53</td>
      <td>86.10</td>
    </tr>
    <tr>
      <td>2015-03-06</td>
      <td>100.11</td>
      <td>53.37</td>
      <td>85.63</td>
      <td>6.42</td>
      <td>84.40</td>
    </tr>
    <tr>
      <td>2015-03-09</td>
      <td>100.66</td>
      <td>53.83</td>
      <td>85.16</td>
      <td>6.23</td>
      <td>82.53</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop Nulls
my_stocks.dropna(inplace=True)
my_stocks.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbol</th>
      <th>JNJ</th>
      <th>MO</th>
      <th>XOM</th>
      <th>VALE</th>
      <th>BABA</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-03-03</td>
      <td>102.34</td>
      <td>56.46</td>
      <td>87.62</td>
      <td>7.13</td>
      <td>81.58</td>
    </tr>
    <tr>
      <td>2015-03-04</td>
      <td>101.65</td>
      <td>55.79</td>
      <td>87.18</td>
      <td>6.88</td>
      <td>85.49</td>
    </tr>
    <tr>
      <td>2015-03-05</td>
      <td>102.52</td>
      <td>55.50</td>
      <td>86.74</td>
      <td>6.53</td>
      <td>86.10</td>
    </tr>
    <tr>
      <td>2015-03-06</td>
      <td>100.11</td>
      <td>53.37</td>
      <td>85.63</td>
      <td>6.42</td>
      <td>84.40</td>
    </tr>
    <tr>
      <td>2015-03-09</td>
      <td>100.66</td>
      <td>53.83</td>
      <td>85.16</td>
      <td>6.23</td>
      <td>82.53</td>
    </tr>
  </tbody>
</table>
</div>



## Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock

### I construct 5 differently weighted portfolios: 
* ## 1: Equally weighted


```python
# Calculate weighted portfolio returns
equal_weights = pd.Series(1/5, index=my_stocks.columns)
my_stock_returns=my_stocks.pct_change()
my_EW_rets=(equal_weights*my_stock_returns).sum(axis=1)
my_EW_rets
```




    Date
    2015-03-03    0.000000
    2015-03-04   -0.002153
    2015-03-05   -0.009085
    2015-03-06   -0.022255
    2015-03-09   -0.008625
                    ...   
    2019-04-16    0.008958
    2019-04-17   -0.001970
    2019-04-18   -0.008521
    2019-04-22    0.000124
    2019-04-23    0.008676
    Length: 1043, dtype: float64



* ## 2: Global Minimum Variance Portfolio. Found by assuming that expected returns of each portfolio are equal to 1, and then using the 'minimize' function to find the minimum 'Negative Sharpe Ratio'. This will be the portfolio with the highest normal Sharpe ratio, and therefore the portfolio with the lowest variance
* ### I use the 'gmv' function, which appears under the 'Max Sharpe Portfolio' section. This function assumes each portfolio has expected returns of 1.0. It then uses these expected returns to call the 'msr' function, which will minimize the 'negative sharpe ratio', and therefore maximizing the normal Sharpe Ratio. So, the portfolio with the highest Sharpe Ratio is also the portfolio with the lowest std deviation


```python
# Construct GMV Portfolio
my_gmv_portfolio_weights=gmv(my_stock_returns.cov())
my_gmv_portfolio=my_stock_returns@my_gmv_portfolio_weights
my_gmv_portfolio
```




    Date
    2015-03-03         NaN
    2015-03-04   -0.003410
    2015-03-05    0.001780
    2015-03-06   -0.024417
    2015-03-09    0.001551
                    ...   
    2019-04-16    0.005115
    2019-04-17    0.002047
    2019-04-18   -0.012351
    2019-04-22    0.006605
    2019-04-23    0.009669
    Length: 1043, dtype: float64



* ## 3: Max Sharpe Ratio portfolio: I find Expected Returns using the 'predict' method of the LinearRegression class. I will then use these expected returns as an input to the 'msr' function, which will calculate weights that will maximize the Sharpe Ratio

### I use return premiums for the Fama-French factors as my explanatory variables: exp_var. This is monthly data, so I need  to convert my daily stock return data to monthly using the resample function with parameter '1M'. I confirm that this procedure is giving me correct total monthly returns by grouping the stock returns by year and month and then summing them. Then I make the monthly stock return index compatible by dropping the 'day' portion, so that it matches the Fama-French index. Then I proceed with the regression, using 50% of the data as a training set, and the remainder as an out of sample testing set


```python
def get_fff_m_returns():
    rets=pd.read_csv("F-F_Research_Data_Factors_m.csv",
                     header=0, index_col=0, na_values=-99.99)
    rets=rets/100
    rets.index=pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets
```


```python
fama_french=get_fff_m_returns()
mkt_excess=fama_french[['Mkt-RF']]           
exp_var=mkt_excess.copy()
exp_var['Constant']=1
exp_var['SMB']=fama_french['SMB']
exp_var['HML']=fama_french['HML']
exp_var['RF']=fama_french['RF']
```


```python
exp_var=exp_var.loc['2015-03':]
exp_var.drop(columns=['Constant', 'RF'], inplace=True)

my_stock_returns_monthly=my_stock_returns.copy()
my_stock_returns_monthly=my_stock_returns_monthly.resample('1M').sum()
my_stock_returns_monthly.index=my_stock_returns_monthly.index.strftime('%Y-%m')
my_stock_returns_monthly=my_stock_returns_monthly.loc[:'2018-12']

X=exp_var.values
Y=my_stock_returns_monthly.values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
my_regressor=LinearRegression()
my_regressor.fit(X_train, Y_train)
df_regression=pd.DataFrame(my_regressor.coef_, index=my_stock_returns_monthly.columns, columns=exp_var.columns)
```


```python
grouped_rets=my_stock_returns.groupby([my_stock_returns.index.year, my_stock_returns.index.month]).sum()
grouped_rets.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>JNJ</th>
      <th>MO</th>
      <th>XOM</th>
      <th>VALE</th>
      <th>BABA</th>
    </tr>
    <tr>
      <th>Date</th>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="top">2015</td>
      <td>3</td>
      <td>-0.015818</td>
      <td>-0.119438</td>
      <td>-0.029107</td>
      <td>-0.220872</td>
      <td>0.022459</td>
    </tr>
    <tr>
      <td>4</td>
      <td>-0.013471</td>
      <td>0.001868</td>
      <td>0.028297</td>
      <td>0.330905</td>
      <td>-0.021273</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.010078</td>
      <td>0.023965</td>
      <td>-0.024330</td>
      <td>-0.184355</td>
      <td>0.098747</td>
    </tr>
    <tr>
      <td>6</td>
      <td>-0.026509</td>
      <td>-0.044105</td>
      <td>-0.023323</td>
      <td>-0.057219</td>
      <td>-0.080481</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.028360</td>
      <td>0.107980</td>
      <td>-0.046711</td>
      <td>-0.097018</td>
      <td>-0.046912</td>
    </tr>
  </tbody>
</table>
</div>




```python
my_stock_returns_monthly.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbol</th>
      <th>JNJ</th>
      <th>MO</th>
      <th>XOM</th>
      <th>VALE</th>
      <th>BABA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-03</td>
      <td>-0.015818</td>
      <td>-0.119438</td>
      <td>-0.029107</td>
      <td>-0.220872</td>
      <td>0.022459</td>
    </tr>
    <tr>
      <td>2015-04</td>
      <td>-0.013471</td>
      <td>0.001868</td>
      <td>0.028297</td>
      <td>0.330905</td>
      <td>-0.021273</td>
    </tr>
    <tr>
      <td>2015-05</td>
      <td>0.010078</td>
      <td>0.023965</td>
      <td>-0.024330</td>
      <td>-0.184355</td>
      <td>0.098747</td>
    </tr>
    <tr>
      <td>2015-06</td>
      <td>-0.026509</td>
      <td>-0.044105</td>
      <td>-0.023323</td>
      <td>-0.057219</td>
      <td>-0.080481</td>
    </tr>
    <tr>
      <td>2015-07</td>
      <td>0.028360</td>
      <td>0.107980</td>
      <td>-0.046711</td>
      <td>-0.097018</td>
      <td>-0.046912</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_regression
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mkt-RF</th>
      <th>SMB</th>
      <th>HML</th>
    </tr>
    <tr>
      <th>Symbol</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>JNJ</td>
      <td>0.539400</td>
      <td>-0.518792</td>
      <td>-0.348539</td>
    </tr>
    <tr>
      <td>MO</td>
      <td>0.369444</td>
      <td>-1.395942</td>
      <td>0.361920</td>
    </tr>
    <tr>
      <td>XOM</td>
      <td>1.048292</td>
      <td>0.088621</td>
      <td>0.383012</td>
    </tr>
    <tr>
      <td>VALE</td>
      <td>1.518237</td>
      <td>-2.033449</td>
      <td>2.412702</td>
    </tr>
    <tr>
      <td>BABA</td>
      <td>2.670996</td>
      <td>-0.093738</td>
      <td>0.104801</td>
    </tr>
  </tbody>
</table>
</div>




```python
Y_predictions=my_regressor.predict(X_test)
```

## Because my 'test-size' parameter was 0.5, 'Y_predictions' has only half of the rows of the original monthly stock returns. This is why I use '[-23]' for the index parameter.


```python
df_predict=pd.DataFrame(data=Y_predictions, index=exp_var.index[-23:], columns=my_stock_returns_monthly.columns)
df_predict
my_expected_returns_monthly=df_predict.mean(axis=0)
my_expected_covariance=df_predict.cov()
my_expected_returns_monthly
```




    Symbol
    JNJ     0.005276
    MO     -0.017801
    XOM     0.001030
    VALE   -0.014211
    BABA    0.034176
    dtype: float64




```python
df_predict.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbol</th>
      <th>JNJ</th>
      <th>MO</th>
      <th>XOM</th>
      <th>VALE</th>
      <th>BABA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2017-02</td>
      <td>0.009694</td>
      <td>-0.027035</td>
      <td>-0.002524</td>
      <td>-0.053858</td>
      <td>0.036978</td>
    </tr>
    <tr>
      <td>2017-03</td>
      <td>0.026425</td>
      <td>-0.002650</td>
      <td>0.001465</td>
      <td>-0.026600</td>
      <td>0.060400</td>
    </tr>
    <tr>
      <td>2017-04</td>
      <td>0.019306</td>
      <td>0.011444</td>
      <td>0.009970</td>
      <td>0.040382</td>
      <td>0.061506</td>
    </tr>
    <tr>
      <td>2017-05</td>
      <td>0.045124</td>
      <td>0.034576</td>
      <td>-0.010626</td>
      <td>-0.001559</td>
      <td>0.051236</td>
    </tr>
    <tr>
      <td>2017-06</td>
      <td>-0.020031</td>
      <td>0.019192</td>
      <td>-0.062741</td>
      <td>0.021667</td>
      <td>-0.138226</td>
    </tr>
  </tbody>
</table>
</div>



### The predicted average returns for MO and VALE are negative. I don't think these are reliable values, since no stock would have negative expected returns. It's probably due to the small number of observations.

## I use the following functions to find weights that will maximize the Sharpe Ratio given the the expected monthly returns and their covariance. I use the 'minimize' function to minimize the 'negative sharpe ratio'. The weights that do this will also be maximizing the regular sharpe ratio.


```python
def portfolio_return(weights, returns):
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    return (weights.T @ covmat @ weights)**.5


def msr(riskfree_rate, er, cov):
    """
    Risk Free Rate + ER + Cov -----> Outputs an array of weights.
    """
    n=er.shape[0]
    initial_guess=np.repeat(1/n, n)
    #constraints
    bounds=((0,1), )*n
 
    weights_sum_to_1={
        'type': 'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio, given weights
        """
        rets=portfolio_return(weights, er)
        vol=portfolio_vol(weights, cov)
        return -(rets-riskfree_rate)/vol

    results=minimize(neg_sharpe_ratio, initial_guess,
                                   args=(riskfree_rate, er, cov,), method="SLSQP",
                                   options={'disp': False},
                                   constraints=(weights_sum_to_1),
                                   bounds=bounds
                                  )
    return results.x

def gmv(cov):
    """
    Returns the weight array of the Global Minimum Variance portfolio. Assumes expected returns are all 1.0
    """
    n=cov.shape[0]
    return msr(0, np.repeat(1,n), cov)

```

## I need to convert my predicted returns into daily returns if I want to be able to concatenate this portfolio to the others. So I take my predicted monthly returns and divide them by 21 to approximate daily returns. Then I 'resize' the dataframe so that it contains 1043 rows like the other portfolios
## To do this, I repeat the same sequence of returns 45 times (45*23=1035) and then I just add the first 8 rows to the end so that my total rows are 1043 which matches my daily stock data. I also change the datetimeIndex format to include days so that it will be compatible with the other portfolios


```python
my_predicted_max_sharpe_weights=msr(0, my_expected_returns_monthly, df_predict.cov())
my_predicted_max_sharpe=((df_predict/21)@my_predicted_max_sharpe_weights)
my_predicted_max_sharpe
my_predicted_max_sharpe.index=my_predicted_max_sharpe.index.strftime('%Y-%m-%d')
```


```python
my_predicted_max_sharpe=pd.concat([my_predicted_max_sharpe]*45)
my_predicted_max_sharpe=pd.concat([my_predicted_max_sharpe, my_predicted_max_sharpe.iloc[0:8]])
my_predicted_max_sharpe_weights
```




    array([4.86952244e-01, 0.00000000e+00, 0.00000000e+00, 2.02745806e-16,
           5.13047756e-01])




```python
my_predicted_max_sharpe.index=my_EW_rets.index
```

* ## 4: Next is an equal risk contribution portfolio. The stocks are weighted so that each stock contributes equally to the volatility of the portfolio. Again I use optimization: I minimize the mean-squared difference between calculated risk contribution and target (equal) risk contribution.
* ### I used the following functions


```python
def portfolio_vol(weights, covmat):
    return (weights.T @ covmat @ weights)**.5

def risk_contribution(w,cov):
    """
    Compute the contributions to risk of the constituents of a portfolio, 
    given a set of portfolio weights and a covariance matrix
    """
    total_portfolio_var = portfolio_vol(w,cov)**2
    # Marginal contribution of each constituent
    marginal_contrib = cov@w
    risk_contrib = np.multiply(marginal_contrib,w.T)/total_portfolio_var
    return risk_contrib



def target_risk_contributions(target_risk, cov):
    """
    Minimizes squared difference from target risk
    
    """
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def msd_risk(weights, target_risk, cov):
        """
        Returns the Mean Squared Difference in risk contributions
        between weights and target_risk
        """
        w_contribs = risk_contribution(weights, cov)
        return ((w_contribs-target_risk)**2).sum()
    
    weights = minimize(msd_risk, init_guess,
                       args=(target_risk, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def equal_risk_contributions(cov):
    """
    Calls target_risk_contributions with an equally weighted Series
    """
    n = cov.shape[0]
    return target_risk_contributions(target_risk=np.repeat(1/n,n), cov=cov)
```


```python
# Contruct portfolio where each stock contributes equally to risk. Equally weighted risk portfolio
eq_risk_cont_weights=equal_risk_contributions(my_stock_returns.cov())
eq_risk_returns=my_stock_returns@eq_risk_cont_weights
eq_risk_returns
```




    Date
    2015-03-03         NaN
    2015-03-04   -0.001520
    2015-03-05   -0.003407
    2015-03-06   -0.023728
    2015-03-09   -0.003596
                    ...   
    2019-04-16    0.006359
    2019-04-17    0.000253
    2019-04-18   -0.011204
    2019-04-22    0.003579
    2019-04-23    0.008798
    Length: 1043, dtype: float64



* ## 5: Finally, I construct a momentum based portfolio. The following function finds the best performer for a given day. The best performer is then overweighted the following day. So this is a trailing 24 hour momentum strategy. Weights can be customized: the 2nd parameter will be the weight on the following day for the top performing stock. The other four stocks are equally weighted, and their sum totals whatever is left after subtracting the top performer's weight from 1.0.
* ### I used the following function


```python
def ride_winner(df, weight_for_winner):
    symbol=''
    val=0
    location=-1
    n=len(df.columns)
    rows=df.shape[0]
    rest=1-weight_for_winner
    weight_list=None
    weighted_rets=pd.DataFrame().reindex_like(df)
    weighted_rets=df.copy()
    for x in range(1, rows-1):
        val=(df.iloc[x]).max()
        symbol=(df.iloc[x]).idxmax()
        location=df.columns.get_loc(symbol)
        weight_list=pd.Series(rest/(n-1), range(0,n))
        weight_list.at[location]=weight_for_winner
        
        weighted_rets.iloc[x+1]=df.iloc[x+1]*weight_list.values
    return weighted_rets.sum(axis=1)
```

### Then I define 2 different momentum portfolios. One assigns a weight of 50% to the previous day's top performer, and the other goes all-in by weighting the previous day's top performer by 100%


```python
momentum_weighted_rets=ride_winner(my_stock_returns, .5)
momentum_weighted_rets_100=ride_winner(my_stock_returns, 1)
```

## Join your custom portfolio returns to the DataFrame that contains all of the portfolio returns


```python
# Add your "Custom" portfolio to the larger dataframe of fund returns
all_returns=pd.concat([combined_df, my_EW_rets], axis=1, join='inner')
all_returns.rename(columns={0:'Equal Dollar Returns'}, inplace=True)
all_returns=pd.concat([all_returns,my_gmv_portfolio], axis=1, join='inner')
all_returns.rename(columns={0: 'GMV Portfolio'}, inplace=True)
all_returns.dropna(inplace=True)
all_returns=pd.concat([all_returns, my_predicted_max_sharpe], axis=1, join='inner')
all_returns.rename(columns={0:'Max Sharpe Portfolio'}, inplace=True)
all_returns=pd.concat([all_returns, eq_risk_returns], axis=1, join='inner')
all_returns.rename(columns={0:'Equal Risk Contribution'}, inplace=True)
all_returns=pd.concat([all_returns, momentum_weighted_rets], axis=1, join='inner')
all_returns.rename(columns={0:'Momentum'}, inplace=True)
all_returns=pd.concat([all_returns, momentum_weighted_rets_100], axis=1, join='inner')
all_returns.rename(columns={0:'Momentum 100%'}, inplace=True)
```


```python
# Only compare dates where return data exists for all the stocks (drop NaNs)
all_returns.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <th>PAULSON &amp; CO.INC.</th>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <th>BERKSHIRE HATHAWAY INC</th>
      <th>Algo 1</th>
      <th>Algo 2</th>
      <th>S&amp;P 500</th>
      <th>Equal Dollar Returns</th>
      <th>GMV Portfolio</th>
      <th>Max Sharpe Portfolio</th>
      <th>Equal Risk Contribution</th>
      <th>Momentum</th>
      <th>Momentum 100%</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-03-04</td>
      <td>0.002230</td>
      <td>0.003241</td>
      <td>-0.002534</td>
      <td>0.004213</td>
      <td>-0.008589</td>
      <td>0.002416</td>
      <td>-0.004389</td>
      <td>-0.002153</td>
      <td>-0.003410</td>
      <td>0.002088</td>
      <td>-0.001520</td>
      <td>-0.010765</td>
      <td>-0.010765</td>
    </tr>
    <tr>
      <td>2015-03-05</td>
      <td>0.004016</td>
      <td>0.004076</td>
      <td>0.002355</td>
      <td>0.006726</td>
      <td>-0.000955</td>
      <td>0.004323</td>
      <td>0.001196</td>
      <td>-0.009085</td>
      <td>0.001780</td>
      <td>0.001950</td>
      <td>-0.003407</td>
      <td>-0.003002</td>
      <td>0.007135</td>
    </tr>
    <tr>
      <td>2015-03-06</td>
      <td>-0.007905</td>
      <td>-0.003574</td>
      <td>-0.008481</td>
      <td>-0.013098</td>
      <td>-0.004957</td>
      <td>-0.011460</td>
      <td>-0.014174</td>
      <td>-0.022255</td>
      <td>-0.024417</td>
      <td>0.002298</td>
      <td>-0.023728</td>
      <td>-0.022724</td>
      <td>-0.023508</td>
    </tr>
    <tr>
      <td>2015-03-09</td>
      <td>0.000582</td>
      <td>0.004225</td>
      <td>0.005843</td>
      <td>-0.001652</td>
      <td>-0.005447</td>
      <td>0.001303</td>
      <td>0.003944</td>
      <td>-0.008625</td>
      <td>0.001551</td>
      <td>-0.003841</td>
      <td>-0.003596</td>
      <td>-0.007449</td>
      <td>-0.005489</td>
    </tr>
    <tr>
      <td>2015-03-10</td>
      <td>-0.010263</td>
      <td>-0.005341</td>
      <td>-0.012079</td>
      <td>-0.009739</td>
      <td>-0.001392</td>
      <td>-0.012155</td>
      <td>-0.016961</td>
      <td>-0.008523</td>
      <td>-0.009915</td>
      <td>0.002311</td>
      <td>-0.008871</td>
      <td>-0.009715</td>
      <td>-0.011704</td>
    </tr>
  </tbody>
</table>
</div>



## Re-run the performance and risk analysis with your portfolio to see how it compares to the others

### Annualized Standard Deviation: 
* ### GMV Portfolio has a low std dev which makes sense. 
* ### Equal Dollar returns having a higher std dev also makes sense. 
* ### The Max Sharpe Ratio Portfolio has an extremely low std dev. Its returns are extremely high as well. These outliers are probably due to the small sample size used for the regression. 
* ### The momentum portfolios have the highest std devs which makes sense since it's a performance chasing strategy.


```python
# Risk
all_returns.std()*np.sqrt(252)
```




    SOROS FUND MANAGEMENT LLC      0.125393
    PAULSON & CO.INC.              0.111517
    TIGER GLOBAL MANAGEMENT LLC    0.173019
    BERKSHIRE HATHAWAY INC         0.205146
    Algo 1                         0.121018
    Algo 2                         0.132492
    S&P 500                        0.135830
    Equal Dollar Returns           0.187949
    GMV Portfolio                  0.134142
    Max Sharpe Portfolio           0.034828
    Equal Risk Contribution        0.150123
    Momentum                       0.252997
    Momentum 100%                  0.423851
    dtype: float64



### Annualized Returns


```python
(((1+all_returns).prod()**(1/all_returns.shape[0]))**252-1)
```




    SOROS FUND MANAGEMENT LLC      0.037850
    PAULSON & CO.INC.             -0.057284
    TIGER GLOBAL MANAGEMENT LLC   -0.035380
    BERKSHIRE HATHAWAY INC         0.114173
    Algo 1                         0.173597
    Algo 2                         0.059599
    S&P 500                        0.083242
    Equal Dollar Returns           0.121365
    GMV Portfolio                  0.055160
    Max Sharpe Portfolio           0.271600
    Equal Risk Contribution        0.087778
    Momentum                       0.159226
    Momentum 100%                  0.189116
    dtype: float64




```python
# Rolling Std Dev
all_returns.rolling(21).std().plot(figsize=(18,12))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x20c9560a048>




![png](output_126_1.png)


### The  timeframe of stock returns that I was able to get (about 3 years) loooks like it was too short to be representative. The predicted returns of my max sharpe portfolio are too high, and its standard deviation is extremely low. Ideally, I would have had a larger sample size to work with. Because of the skewed nature of this portfolio, I am excluding it from the rest of the analysis.


```python
all_returns.drop(columns=['Max Sharpe Portfolio'], inplace=True)
```

## Rolling Beta for the GMV Portfolio. Method 1


```python
all_returns_rolling_cov=all_returns['GMV Portfolio'].rolling(21).cov(all_returns['S&P 500'])
rolling_market_var=all_returns['S&P 500'].rolling(21).var()
all_returns_rolling_beta=all_returns_rolling_cov/rolling_market_var
all_returns_rolling_beta.plot(figsize=(18,12))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b4ae3ecc8>




![png](output_130_1.png)


### The GMV portfolio beta seems to be around .5 on average. This makes sense that a portfolio with low variance would only move about half as much as the market does.

## Rolling Betas for all of the alternatively weighted portfolios. Method 2.


```python
my_rolled_rets=get_rolled_returns(all_returns[['Equal Dollar Returns', 'GMV Portfolio',
                                               'Equal Risk Contribution','Momentum', 'Momentum 100%','S&P 500']], 21)
my_rolled_betas=get_beta_from_windows(all_returns[['Equal Dollar Returns', 'GMV Portfolio',
                                               'Equal Risk Contribution', 'Momentum', 'Momentum 100%']],
                                      my_rolled_rets, 21)
```


```python
my_rolled_betas.plot(figsize=(18,12))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b4ae5b108>




![png](output_134_1.png)



```python
# Annualized Sharpe Ratios
my_sharpes=(all_returns.mean()*252)/(all_returns.std()*np.sqrt(252))
my_sharpes
```




    SOROS FUND MANAGEMENT LLC      0.359034
    PAULSON & CO.INC.             -0.473108
    TIGER GLOBAL MANAGEMENT LLC   -0.120425
    BERKSHIRE HATHAWAY INC         0.629941
    Algo 1                         1.383268
    Algo 2                         0.503342
    S&P 500                        0.656761
    Equal Dollar Returns           0.703421
    GMV Portfolio                  0.467524
    Equal Risk Contribution        0.635625
    Momentum                       0.710004
    Momentum 100%                  0.618466
    dtype: float64




```python
# Visualize the sharpe ratios as a bar plot
my_sharpes.plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b4ba17548>




![png](output_136_1.png)


## Sortino Ratios


```python
sortino_ratio(all_returns, 0, 252)
```




    SOROS FUND MANAGEMENT LLC      0.393632
    PAULSON & CO.INC.             -0.694342
    TIGER GLOBAL MANAGEMENT LLC   -0.236607
    BERKSHIRE HATHAWAY INC         0.710853
    Algo 1                         2.436731
    Algo 2                         0.595919
    S&P 500                        0.757693
    Equal Dollar Returns           0.994208
    GMV Portfolio                  0.533970
    Equal Risk Contribution        0.828455
    Momentum                       0.962488
    Momentum 100%                  0.647765
    dtype: float64




```python
sortino_ratio(all_returns, 0, 252).plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b4bad6a08>




![png](output_139_1.png)


## These Sortino ratios paint a clearer picture of Algo 1's dominance. When only downside risk is considered, several portfolios' ratio of excess return to risk goes up. This suggests that their Sharpe Ratios are being unfairly penalized for having upside volatility. Interestingly, Paulson and Tiger both have lower Sortino ratios. This may mean that they have disproportionate downside volatility or left tail risk.

## Total Wealth at end of period

### Algo 1 was the best portfolio to be invested in. It had the highest Sharpe ratio by far, and produced about as much wealth as the momentum portfolios, though with a lot less volatility


```python
(10000*(1+all_returns).cumprod()).plot(figsize=(18,12), title='Cumulative Wealth')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b4baa4308>




![png](output_143_1.png)


## Include correlation analysis to determine which stocks (if any) are correlated


```python
my_stock_returns.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbol</th>
      <th>JNJ</th>
      <th>MO</th>
      <th>XOM</th>
      <th>VALE</th>
      <th>BABA</th>
    </tr>
    <tr>
      <th>Symbol</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>JNJ</td>
      <td>1.000000</td>
      <td>0.358441</td>
      <td>0.427391</td>
      <td>0.204238</td>
      <td>0.208317</td>
    </tr>
    <tr>
      <td>MO</td>
      <td>0.358441</td>
      <td>1.000000</td>
      <td>0.255911</td>
      <td>0.096526</td>
      <td>0.116000</td>
    </tr>
    <tr>
      <td>XOM</td>
      <td>0.427391</td>
      <td>0.255911</td>
      <td>1.000000</td>
      <td>0.381230</td>
      <td>0.254624</td>
    </tr>
    <tr>
      <td>VALE</td>
      <td>0.204238</td>
      <td>0.096526</td>
      <td>0.381230</td>
      <td>1.000000</td>
      <td>0.273577</td>
    </tr>
    <tr>
      <td>BABA</td>
      <td>0.208317</td>
      <td>0.116000</td>
      <td>0.254624</td>
      <td>0.273577</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Interestingly, these 5 stocks are only moderately correlated. BABA has pretty low pairwise correlations with all other stocks. I suppose this makes sense, since these 5 stocks are all in different industries...although I would have expected a higher correlation between XOM and VALE, since they're both involved with natural resources.


```python
all_returns.corr().iloc[-6:, -6:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S&amp;P 500</th>
      <th>Equal Dollar Returns</th>
      <th>GMV Portfolio</th>
      <th>Equal Risk Contribution</th>
      <th>Momentum</th>
      <th>Momentum 100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>S&amp;P 500</td>
      <td>1.000000</td>
      <td>0.755508</td>
      <td>0.770717</td>
      <td>0.811437</td>
      <td>0.610252</td>
      <td>0.412715</td>
    </tr>
    <tr>
      <td>Equal Dollar Returns</td>
      <td>0.755508</td>
      <td>1.000000</td>
      <td>0.760458</td>
      <td>0.951012</td>
      <td>0.883686</td>
      <td>0.667428</td>
    </tr>
    <tr>
      <td>GMV Portfolio</td>
      <td>0.770717</td>
      <td>0.760458</td>
      <td>1.000000</td>
      <td>0.919201</td>
      <td>0.588059</td>
      <td>0.373795</td>
    </tr>
    <tr>
      <td>Equal Risk Contribution</td>
      <td>0.811437</td>
      <td>0.951012</td>
      <td>0.919201</td>
      <td>1.000000</td>
      <td>0.800646</td>
      <td>0.571468</td>
    </tr>
    <tr>
      <td>Momentum</td>
      <td>0.610252</td>
      <td>0.883686</td>
      <td>0.588059</td>
      <td>0.800646</td>
      <td>1.000000</td>
      <td>0.938277</td>
    </tr>
    <tr>
      <td>Momentum 100%</td>
      <td>0.412715</td>
      <td>0.667428</td>
      <td>0.373795</td>
      <td>0.571468</td>
      <td>0.938277</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### All of these portfolios are pretty highly correlated with each other. That makes intuitive sense since they are made up of the same 5 stocks.
* ### As a 'sanity check' it makes good sense that the GMV portfolio would have a lower correlation with the extreme momentum 100% portfolio. 
* ### The Equal Risk Contribution portfolio also has a low correlation to the Momentum 100% portfolio. This can be interpreted as saying that the Momentum 100% portfolio has far from a balanced risk exposure; in fact it's concentrated 100% in 1 stock.


```python
all_returns.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <th>PAULSON &amp; CO.INC.</th>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <th>BERKSHIRE HATHAWAY INC</th>
      <th>Algo 1</th>
      <th>Algo 2</th>
      <th>S&amp;P 500</th>
      <th>Equal Dollar Returns</th>
      <th>GMV Portfolio</th>
      <th>Equal Risk Contribution</th>
      <th>Momentum</th>
      <th>Momentum 100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SOROS FUND MANAGEMENT LLC</td>
      <td>1.000000</td>
      <td>0.699961</td>
      <td>0.561246</td>
      <td>0.754385</td>
      <td>0.321175</td>
      <td>0.826869</td>
      <td>0.837908</td>
      <td>0.652593</td>
      <td>0.585070</td>
      <td>0.667221</td>
      <td>0.529510</td>
      <td>0.360676</td>
    </tr>
    <tr>
      <td>PAULSON &amp; CO.INC.</td>
      <td>0.699961</td>
      <td>1.000000</td>
      <td>0.434551</td>
      <td>0.545465</td>
      <td>0.268692</td>
      <td>0.678214</td>
      <td>0.669612</td>
      <td>0.507858</td>
      <td>0.484871</td>
      <td>0.529521</td>
      <td>0.390066</td>
      <td>0.245803</td>
    </tr>
    <tr>
      <td>TIGER GLOBAL MANAGEMENT LLC</td>
      <td>0.561246</td>
      <td>0.434551</td>
      <td>1.000000</td>
      <td>0.424465</td>
      <td>0.164384</td>
      <td>0.507414</td>
      <td>0.624023</td>
      <td>0.434836</td>
      <td>0.511190</td>
      <td>0.499944</td>
      <td>0.336028</td>
      <td>0.213387</td>
    </tr>
    <tr>
      <td>BERKSHIRE HATHAWAY INC</td>
      <td>0.754385</td>
      <td>0.545465</td>
      <td>0.424465</td>
      <td>1.000000</td>
      <td>0.291912</td>
      <td>0.688112</td>
      <td>0.751297</td>
      <td>0.608966</td>
      <td>0.490328</td>
      <td>0.606944</td>
      <td>0.503816</td>
      <td>0.352034</td>
    </tr>
    <tr>
      <td>Algo 1</td>
      <td>0.321175</td>
      <td>0.268692</td>
      <td>0.164384</td>
      <td>0.291912</td>
      <td>1.000000</td>
      <td>0.288214</td>
      <td>0.279366</td>
      <td>0.254907</td>
      <td>0.168138</td>
      <td>0.231482</td>
      <td>0.197859</td>
      <td>0.125923</td>
    </tr>
    <tr>
      <td>Algo 2</td>
      <td>0.826869</td>
      <td>0.678214</td>
      <td>0.507414</td>
      <td>0.688112</td>
      <td>0.288214</td>
      <td>1.000000</td>
      <td>0.858828</td>
      <td>0.689425</td>
      <td>0.596409</td>
      <td>0.691693</td>
      <td>0.572155</td>
      <td>0.401334</td>
    </tr>
    <tr>
      <td>S&amp;P 500</td>
      <td>0.837908</td>
      <td>0.669612</td>
      <td>0.624023</td>
      <td>0.751297</td>
      <td>0.279366</td>
      <td>0.858828</td>
      <td>1.000000</td>
      <td>0.755508</td>
      <td>0.770717</td>
      <td>0.811437</td>
      <td>0.610252</td>
      <td>0.412715</td>
    </tr>
    <tr>
      <td>Equal Dollar Returns</td>
      <td>0.652593</td>
      <td>0.507858</td>
      <td>0.434836</td>
      <td>0.608966</td>
      <td>0.254907</td>
      <td>0.689425</td>
      <td>0.755508</td>
      <td>1.000000</td>
      <td>0.760458</td>
      <td>0.951012</td>
      <td>0.883686</td>
      <td>0.667428</td>
    </tr>
    <tr>
      <td>GMV Portfolio</td>
      <td>0.585070</td>
      <td>0.484871</td>
      <td>0.511190</td>
      <td>0.490328</td>
      <td>0.168138</td>
      <td>0.596409</td>
      <td>0.770717</td>
      <td>0.760458</td>
      <td>1.000000</td>
      <td>0.919201</td>
      <td>0.588059</td>
      <td>0.373795</td>
    </tr>
    <tr>
      <td>Equal Risk Contribution</td>
      <td>0.667221</td>
      <td>0.529521</td>
      <td>0.499944</td>
      <td>0.606944</td>
      <td>0.231482</td>
      <td>0.691693</td>
      <td>0.811437</td>
      <td>0.951012</td>
      <td>0.919201</td>
      <td>1.000000</td>
      <td>0.800646</td>
      <td>0.571468</td>
    </tr>
    <tr>
      <td>Momentum</td>
      <td>0.529510</td>
      <td>0.390066</td>
      <td>0.336028</td>
      <td>0.503816</td>
      <td>0.197859</td>
      <td>0.572155</td>
      <td>0.610252</td>
      <td>0.883686</td>
      <td>0.588059</td>
      <td>0.800646</td>
      <td>1.000000</td>
      <td>0.938277</td>
    </tr>
    <tr>
      <td>Momentum 100%</td>
      <td>0.360676</td>
      <td>0.245803</td>
      <td>0.213387</td>
      <td>0.352034</td>
      <td>0.125923</td>
      <td>0.401334</td>
      <td>0.412715</td>
      <td>0.667428</td>
      <td>0.373795</td>
      <td>0.571468</td>
      <td>0.938277</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### All of my alternatively weighted portfolios are moderately correlated with the 4 'Whale' Funds and Algo 2.Algo 1 seems to be only lightly correlated with every other portfolio. An allocation to Algo 1 would make a lot of sense assuming we can be reasonably sure that its great returns are reproducible

### Compute Drawdowns of the portfolios. Drawdowns are calculated cumulatively with the following function


```python
def drawdown(return_series: pd.Series):
    """
    This function takes a return Series and calculates the Wealth at each period as well as the cumulative high
    watermarks. It then subtracts these high watermarks from each month's returns to calculate drawdowns. Returns a
    dictionary.
    """
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks=wealth_index.cummax()
    drawdowns=(wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })
```


```python
eq_draw=drawdown(all_returns['Equal Dollar Returns'])
gmv_draw=drawdown(all_returns['GMV Portfolio'])
risk_draw=drawdown(all_returns['Equal Risk Contribution'])
momentum_draw=drawdown(all_returns['Momentum'])
momentum_100_draw=drawdown(all_returns['Momentum 100%'])
```


```python
ax=eq_draw['Drawdown'].plot(figsize=(18,12), label='EQ $', legend=True)
gmv_draw['Drawdown'].plot(ax=ax, label='GMV', legend=True)
risk_draw['Drawdown'].plot(ax=ax, label='EQ Risk', legend=True)
momentum_draw['Drawdown'].plot(ax=ax, label='Momentum', legend=True)
momentum_100_draw['Drawdown'].plot(ax=ax, label='Momentum 100%', legend=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26b4c07f748>




![png](output_154_1.png)


### No surprises here: The momentum portfolios have the biggest drawdowns. The more the portfolio is concentrated in a single stock, the worse the losses are. The GMV Portfolio and the Equal Risk Contribution Portfolio both have much lower drawdowns


```python
drawdown_df=pd.concat([pd.DataFrame(eq_draw['Drawdown']),
           pd.DataFrame(gmv_draw['Drawdown']), pd.DataFrame(risk_draw['Drawdown']), pd.DataFrame(momentum_draw['Drawdown']), pd.DataFrame(momentum_100_draw['Drawdown'])], axis=1)
drawdown_df.columns=['EQ $', 'GMV', 'EQ Risk', 'Momentum', 'Momentum 100%']
```


```python
drawdown_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EQ $</th>
      <th>GMV</th>
      <th>EQ Risk</th>
      <th>Momentum</th>
      <th>Momentum 100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>1042.000000</td>
      <td>1042.000000</td>
      <td>1042.000000</td>
      <td>1042.000000</td>
      <td>1042.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>-0.063698</td>
      <td>-0.056522</td>
      <td>-0.054254</td>
      <td>-0.101225</td>
      <td>-0.190089</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.058028</td>
      <td>0.050004</td>
      <td>0.051160</td>
      <td>0.084358</td>
      <td>0.135363</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-0.256000</td>
      <td>-0.234390</td>
      <td>-0.236743</td>
      <td>-0.384365</td>
      <td>-0.577554</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>-0.093643</td>
      <td>-0.093987</td>
      <td>-0.095569</td>
      <td>-0.144085</td>
      <td>-0.272320</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>-0.053693</td>
      <td>-0.043392</td>
      <td>-0.035291</td>
      <td>-0.089897</td>
      <td>-0.186964</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>-0.013518</td>
      <td>-0.015583</td>
      <td>-0.009860</td>
      <td>-0.030781</td>
      <td>-0.077668</td>
    </tr>
    <tr>
      <td>max</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Style Analysis for all of the Portfolios. I will run a regression using the 3 Fama-French factors

### The data I have for the Fama-French factor returns is monthly data and it only goes through December 2018. So I will modify my portfolios so that they have monthly data and they match the timeframe.


```python
all_returns_modified=all_returns.resample('1M').sum()
all_returns_modified.index=all_returns_modified.index.strftime('%Y-%m')
all_returns_modified=all_returns_modified.loc[:'2018-12']
```


```python
X2=exp_var.values
Y2=all_returns_modified.values
# Half of the stock returns data will be used as the training set, and the other half as the testing set.
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.5, random_state=0)
my_regressor2=LinearRegression()
my_regressor2.fit(X2_train, Y2_train)
df_regression2=pd.DataFrame(my_regressor2.coef_, index=all_returns_modified.columns, columns=exp_var.columns)
```


```python
df_regression2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mkt-RF</th>
      <th>SMB</th>
      <th>HML</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SOROS FUND MANAGEMENT LLC</td>
      <td>0.721749</td>
      <td>0.057939</td>
      <td>-0.249356</td>
    </tr>
    <tr>
      <td>PAULSON &amp; CO.INC.</td>
      <td>0.448366</td>
      <td>0.166409</td>
      <td>-0.099763</td>
    </tr>
    <tr>
      <td>TIGER GLOBAL MANAGEMENT LLC</td>
      <td>0.779079</td>
      <td>-0.201926</td>
      <td>0.130369</td>
    </tr>
    <tr>
      <td>BERKSHIRE HATHAWAY INC</td>
      <td>1.087713</td>
      <td>-0.032260</td>
      <td>-0.255324</td>
    </tr>
    <tr>
      <td>Algo 1</td>
      <td>0.111509</td>
      <td>0.090812</td>
      <td>0.021389</td>
    </tr>
    <tr>
      <td>Algo 2</td>
      <td>0.901029</td>
      <td>0.522216</td>
      <td>0.177007</td>
    </tr>
    <tr>
      <td>S&amp;P 500</td>
      <td>0.973418</td>
      <td>-0.122235</td>
      <td>-0.026302</td>
    </tr>
    <tr>
      <td>Equal Dollar Returns</td>
      <td>1.229274</td>
      <td>-0.790660</td>
      <td>0.582779</td>
    </tr>
    <tr>
      <td>GMV Portfolio</td>
      <td>0.781285</td>
      <td>-0.562540</td>
      <td>0.037641</td>
    </tr>
    <tr>
      <td>Equal Risk Contribution</td>
      <td>1.031611</td>
      <td>-0.672459</td>
      <td>0.308762</td>
    </tr>
    <tr>
      <td>Momentum</td>
      <td>2.285051</td>
      <td>-0.853294</td>
      <td>1.833753</td>
    </tr>
    <tr>
      <td>Momentum 100%</td>
      <td>4.037981</td>
      <td>-0.915706</td>
      <td>3.916067</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_regression2.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mkt-RF</th>
      <th>SMB</th>
      <th>HML</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>1.199005</td>
      <td>-0.276142</td>
      <td>0.531418</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.033308</td>
      <td>0.468473</td>
      <td>1.203500</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.111509</td>
      <td>-0.915706</td>
      <td>-0.255324</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>0.764747</td>
      <td>-0.702009</td>
      <td>-0.044667</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>0.937223</td>
      <td>-0.162081</td>
      <td>0.084005</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>1.123103</td>
      <td>0.066157</td>
      <td>0.377267</td>
    </tr>
    <tr>
      <td>max</td>
      <td>4.037981</td>
      <td>0.522216</td>
      <td>3.916067</td>
    </tr>
  </tbody>
</table>
</div>



### Some observations: Soros and Paulson both have a small growth tilt, and their exposure to the Market factor is significantly less than 1. Tiger Global has more of a large value tilt. Tiger's poor performance makes sense now, since this data is from between 2015 and 2018 and value stocks have been underperforming for about 10 years now. 
* ### My Equal Dollar Weighted Portfolio has a huge value bias which is expected, since equally weighted indices rebalance away from high growth stocks.
* ### I'm not sure what to make of the 2 momentum portfolios. Both have huge exposure to the market factor, and are also heavily skewed towards large cap and value. Momentum traditionally is a growth strategy. It could be due to my particular stock selection, or the fairly short period of sample returns.
* ### Algo 1 is very mysterious. It has very low Market exposure, a slight small tilt, and it's almost neutral between value and growth. I wonder how it achieved its high returns. It's possible that using other factors in the regression would have resulted in higher coefficients for Algo 1. Perhaps those factors would be more macro in nature.

# Other Sources used:
## [Scikit-Learn Regression Guide](https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f)
## [Random pages on StackOverFlow](https://stackoverflow.com/questions/50997339/convert-daily-data-in-pandas-dataframe-to-monthly-data)
