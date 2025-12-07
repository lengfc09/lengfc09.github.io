---
layout: mysingle
date: 2025-02-21 19:28:16 +0800
title: Time-varing models for estimating Value-at-Risk(vars) and volatility
categories: Quantitative_Financial_Risk_Management
excerpt: "Modeling volatility is crucial for asset pricing. Empirical evidence indicates that volatility is time-varying. Moreover, it exhibits time-dependence and volatility clustering. To capture the dynamics of volatility, it is necessary to construct models that transform random variables into independent and identically distributed (i.i.d.) variables, with which we are more familiar with."
header:
    # overlay_color: "#333"
    overlay_color: "#2f4f4f" #暗岩灰
    # overlay_color: "#e68ab8" #火鹤红
classes: wide
tags: statistics risk_management vars arch

toc: true
---

## Time-varing volatility modeling


Check this website for more theoretical details: [Vlab of NYU](https://vlab.stern.nyu.edu/docs).


### Example 1: real GDP growth rate in USA

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17378762703630.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="600px"}{: .align-center}

![](https://bens-2-pics.oss-cn-shanghai.aliyuncs.com/uPic/2025/02/21/1935image-20250221193527325.png){:width="600px"}{: .align-center}

### Example 2: S&P500 Index Return
* Daily Returns

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17378766412658.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="600px"}{: .align-center}


In fact, by ploting the ACF(Autocorrelation Funciton) and PACF(Partial Autocorrelation Function), we can find strong autocorrelations of the squared returns, absolute returns and High-Low returns ${ln(Y_t^H/Y_t^L)}$.


* $r_t$: not significant autocorrelation
![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17378813766988.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="600px"}{: .align-center}

* $r_t^2$:significant autocorrelation
![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17378813953235.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="600px"}{: .align-center}

* $|r_t|$:significant autocorrelation
![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17378814083903.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="600px"}{: .align-center}

* ${ln(Y_t^H/Y_t^L)}$:significant autocorrelation
![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17378814282787.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="600px"}{: .align-center}

Codes:
```python
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import scipy.stats as stats
import matplotlib.pyplot as plt
import var_ntime # my library

from WindPy import w
w.start()
error_code, price = w.wsd("000001.SH", "close", "2001-01-01", "2021-01-05", "PriceAdj=F",usedf=True)

price=price.dropna(how="any")
price.index=pd.to_datetime(price.index)
logr=100*np.log(price).diff().dropna()
# or returns=100*price.pct_change().dropna()
# Here the returns are multiplied by 100, so that the number is around 1-100, which helps the following model fitting process.

# check the normality of r
qplot1=qqplot(logr['CLOSE'],line='s')
# or use my hist gragh
var_ntime.myhist(logr,bins=100)

# Test the serial correlation with ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
acfs=plot_acf(logr)
pacfs=plot_pacf(logr)

# check volatility clusering
# squared return
acfs=plot_acf(logr.pow(2))
pacfs=plot_pacf(logr.pow(2))
# aboslute return
acfs=plot_acf(logr.abs())
pacfs=plot_pacf(logr.abs())
```

## Simple specications for conditional variance

Two models that are data-driven specifications:
* Rolling window average or moving average (MA)
    $$\hat{\sigma}_{t+1 \mid t}^2=\frac{1}{n} \sum_{i=1}^n\left(r_{t+1-i}-\mu\right)^2=\frac{1}{n} \times\left(\left(r_t-\mu\right)^2+\left(r_{t-1}-\mu\right)^2+\cdots+\left(r_{t-n}-\mu\right)^2\right)$$

* Exponentially Weighted Moving Average (EWMA) or RiskMetrics model.

$$x_{t \mid t-1}^2=\frac{x_{t-1}^2+\lambda^1 x_{t-2}^2+\lambda^2 x_{t-3}^2+\cdots+\lambda^{\infty} x_{\infty}^2}{1+\lambda^1+\lambda^2+\cdots+\lambda^{n-1}+\cdots+\lambda^{\infty}} ; \text{ } \lambda \epsilon(0,1)$$

$$\hat{\sigma}_{t+1 \mid t}^2=(1-\lambda)\left(r_t-\mu\right)^2+\lambda \hat{\sigma}_{t \mid t-1}^2$$

```python
def historical_volatility(returns, window=100,min_periods=50,align='origin'):
    """MA standard deviation"""
    if align=='origin':
        return returns.rolling(window,min_periods).std()
    elif align=='target':
        return returns.rolling(window,min_periods).std().shift(1)
    else:
        print("plese choose the align way: origin or target")
        return

def ewma_volatility(returns, lambda_param=0.94,align='origin'):
    """EWMA standard deviation, Assume 𝜆=0.94 as in RiskMetrics for daily returns"""
    returns2=returns-returns.mean()
    returns2=returns2.pow(2)
    var = returns2.ewm(alpha=1-lambda_param).mean()
    if align=='origin':
        return np.sqrt(var)
    elif align=='target':
        return np.sqrt(var).shift(1)
    else:
        print("plese choose the align way: origin or target")
        return


def ewma_volatility2(returns, lambda_param=0.94,min_periods=50,align='origin'):
    """EWMA standard deviation with EWMA mean"""
    #alternaively use the ewm().var(), in which the mean is the EWM mean
    var = returns.ewm(alpha=1-lambda_param,min_periods=min_periods).var()
    if align=='origin':
        return np.sqrt(var)
    elif align=='target':
        return np.sqrt(var).shift(1)
    else:
        print("plese choose the align way: origin or target")
        return
```

Example:
```python
import datetime as dt
import sys
import numpy as np
import pandas as pd
from arch import arch_model
import arch.data.sp500
import scipy as sp
data = arch.data.sp500.load()
returns=100*data['Adj Close'].apply(np.log).diff().dropna()


sigma1=historical_volatility(returns,align='target')
sigma2=ewma_volatility(returns,align='target')
sigma=pd.concat([sigma1,sigma2],1)
sigma.columns=['MA-std','EWMA-std']
sigma.plot()
# check the outliers
q=sp.stats.norm.ppf(0.05)
print((returns<q*sigma1).mean())
#0.056262425447316106
print((returns<q*sigma2).mean())
#0.05745526838966203
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17401273830114.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}

Set the window for MA-std to be 50 instead of 100 may give the similar result to EWMA($\lambda$ =0.94)

```python
sigma1=historical_volatility(returns,window=50).shift(1)
sigma2=ewma_volatility(returns).shift(1)
sigma=pd.concat([sigma1,sigma2],1)
sigma.columns=['MA-std-50D','EWMA-std']
sigma.plot()
```


![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17401274839669.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}


## Introduction to GARCH models

A basic GARCH model is specified as:
$$
\begin{eqnarray} r_t & = & \mu + \epsilon_t \\ \epsilon_t & = & \sigma_t e_t \\ \sigma^2_t & = & \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma^2_{t-1} \end{eqnarray}
$$

A complete ARCH model is divided into three components:

* a **mean** model, e.g., a constant mean or an ARX;
* a **volatility** process, e.g., a GARCH or an EGARCH process; and
* a **distribution** for the standardized residuals.

The standard steps of using:[ARCH Package](https://bashtage.github.io/arch/doc/)

1. Construct by specifying the params
2. Train (fit) and backtest with train-test split
3. Forecast: the volatility
4. Forecast: the VAR

`arch.univariate.arch_model(y, x=None, mean='Constant', lags=0, vol='Garch', p=1, o=0, q=1, power=2.0, dist='Normal', hold_back=None, rescale=None)`

By default, it is a GARCH(p=1,o=0,q=1) with constant mean model.
$$\sigma^2_t = \omega + \alpha \epsilon_{t-1}^2 + \gamma \epsilon_{t-1}^2 I_{[\epsilon_{t-1}<0]}+ \beta \sigma_{t-1}^2$$

Parameters

- **y** (*{ndarray**,* *Series**,* *None}*) -- The dependent variable

- **x** (*{np.array**,* *DataFrame}**,* *optional*) -- Exogenous regressors. Ignored if model does not permit exogenous regressors.

- **mean** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,* *optional*) -- Name of the mean model. Currently supported options are: 'Constant', 'Zero', 'LS', 'AR', 'ARX', 'HAR' and 'HARX'

- **lags** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list) *(*[*int*](https://docs.python.org/3/library/functions.html#int)*)**,* *optional*) -- Either a scalar integer value indicating lag length or a list of integers specifying lag locations.

- **vol** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,* *optional*) -- Name of the volatility model. Currently supported options are: 'GARCH' (default), 'ARCH', 'EGARCH', 'FIARCH' and 'HARCH'

- **p** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) -- Lag order of the symmetric innovation

- **o** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) -- Lag order of the asymmetric innovation

- **q** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) -- Lag order of lagged volatility or equivalent

- **power** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) -- Power to use with GARCH and related models

- **dist** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) --

  Name of the error distribution. Currently supported options are:

  > - Normal: 'normal', 'gaussian' (default)
  > - Students's t: 't', 'studentst'
  > - Skewed Student's t: 'skewstudent', 'skewt'
  > - Generalized Error Distribution: 'ged', 'generalized error"



- **hold_back** ([*int*](https://docs.python.org/3/library/functions.html#int)) -- Number of observations at the start of the sample to exclude when estimating model parameters. Used when comparing models with different lag lengths to estimate on the common sample.

- **rescale** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) -- Flag indicating whether to automatically rescale data if the scale of the data is likely to produce convergence issues when estimating model parameters. If False, the model is estimated on the data without transformation. If True, than y is rescaled and the new scale is reported in the estimation results.


Alternative mean and volatility processes can be directly specified

```python
am = arch_model(returns, mean='AR', lags=2, vol='harch', p=[1, 5, 22]
```

This example demonstrates the construction of a zero mean process with a TARCH volatility process and Student t error distribution

```python
am = arch_model(returns, mean='zero', p=1, o=1, q=1,power=1.0, dist='StudentsT')
```



### Construction

```python
import datetime as dt
import sys
import numpy as np
import pandas as pd
import arch.data.sp500
data = arch.data.sp500.load()
returns=100*data['Adj Close'].apply(np.log).diff().dropna()

from arch import arch_model
am = arch_model(returns,dist='StudentsT',p=1,q=1,o=0)

```

### Fitting/Training

#### The general method

```python
split_date=dt.datetime(2014,1,1) # will not be inlcuded in the samole
res = am.fit(update_freq=5,last_obs=split_date)
res.summary()
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17395191682932.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="600px"}{: .align-center}


```python
plt.rcParams["figure.figsize"] = (12, 6)
plt.figure(dpi=100)
fig=res.plot(annualize='D')
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17395191880736.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}

some params in fitting process:

* `update_freq=1`: controls the frequency of output form the optimizer
* `disp=on` or `disp=off`: controls whether convergence information is returned
* `last_obs`: it follow Python sequence rules so that the **actual date in last_obs** is ***not*** in the sample. 8Therefore, the last result in the `res.resid.dropna()` will be of the date `2013-12-31`.


#### Fixing Parameters

In some circumstances, fixed rather than estimated parameters might be of interest.

```python
fix_res=am.fix([0.06,0.01,0.08,0.9,8])
print(fix_res.summary())
```

Checked the fitted params vs fixed params model performance:

```python
df=pd.concat([res.conditional_volatility,fix_res.conditional_volatility],1)
df.columns=['estemated','fixed']
plt.style.use('ggplot')
df.plot()
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17396173959962.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}


```python
fig1=res.plot()
fig2=fix_res.plot()
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17396175595244.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}
![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17396175683558.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}

Notice that, the fixed parameters created model by `fix_res=am.fix([parameters...])` will cover the whole sample.

#### Build from components

Create the mean model, and then add the volatility model, and change the distribution model in the end.

```python
import arch.data.core_cpi
core_cpi = arch.data.core_cpi.load()
ann_inflation = 100 * core_cpi.CPILFESL.pct_change(12).dropna()
#Here pct_change(n): n means the periods to shift for forming percent change.
fig = ann_inflation.plot()
```

* Step 1: Auto-regression model to estimate the mean:

```python
from arch.univariate import ARX
ar = ARX(ann_inflation, lags=[1, 3, 12])
print(ar.fit().summary())
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17396182676693.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="600px"}{: .align-center}


* Step 2:Add a volatility model:

```python
from arch.univariate import ARCH, GARCH

ar.volatility = ARCH(p=5)
res = ar.fit(update_freq=0, disp='off')
print(res.summary())
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17396182859761.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="600px"}{: .align-center}

* Step 3: Finally the distribution can be changed from the default normal to a standardized Student's T using the distribution property of a mean model.

```python
from arch.univariate import StudentsT
ar.distribution = StudentsT()
res = ar.fit(update_freq=0, disp='off')
print(res.summary())
```

### return of the fitted model

```python
from arch import arch_model
am = arch_model(returns,dist='StudentsT',p=1,q=1,o=0)
res = am.fit(update_freq=5,last_obs=dt.datetime(2014,1,1))
help(res)
```

* `conditional_volatility` $\sigma$: square root of conditional variance. The values are aligned with the input data so that the value in the t-th position is the variance of t-th error, which is computed using time-(t-1) information.
* `params` : Estimated parameters
* `resid`: Residuals from model
* `volatility`: volatility from model
* `model`:The model object used to estimate the parameters
* `fit_start`: Start of sample used to estimate parameters
* `fit_stop`: End of sample used to estimate parameters
* `summary()`: Constructs a summary of the results from a fit model.
* `forecast()`:  Construct forecasts from estimated model.

### Forecasts

#### Three mothods

Forecasts here mean the conditional expectation of the future Mean, Volatility, and etc. All arch_models here support three methods of forecasting:

* Analytical(Parameter method):
  * Always available for the 1-step
  * Multi-step analytical forecasts are only available for GARCH or HARCH and certain linear models.These forecasts exploit the relationship $E_t[\epsilon_{t+1}^2] = \sigma_{t+1}$ to recursively compute forecasts

* Simulation(Monte-carlo simulation):
  * only useful for horizons larger than 1
  * the standardized residuals are **simulated by the assumed distribution** of residuals, e.g., a Normal or Student's t.

* Bootstrap(Historical method):
  * similar to Simulation method, except:
  * the **standardized residuals from the actual data are used** in the estimation rather than assuming a specific distribution.


#### call of forecaste()

The [`forecast()`](https://bashtage.github.io/arch/doc/univariate/generated/generated/arch.univariate.base.ARCHModelResult.forecast.html#arch.univariate.base.ARCHModelResult.forecast) method is attached to a model fit result.

- `params` - The model parameters used to forecast the mean and variance. If not specified, the parameters estimated during the call to `fit` the produced the result are used.
- `horizon` - A positive integer value indicating the maximum horizon to produce forecasts.
- `start` - A positive integer or, if the input to the mode is a DataFrame, a date (string, datetime, datetime64 or Timestamp). Forecasts are produced from `start` until the end of the sample. If not provided, `start` is set to the length of the insample-data minus 1.
- `method` - One of 'analytic' (default), 'simulation' or 'bootstrap' that describes the method used to produce the forecasts. Not all methods are available for all horizons.
- `simulations` - A non-negative integer indicating the number of simulation to use when `method` is 'simulation' or 'bootstrap'
- `align='origin'`: if forecast is called with `align="target"`, the forecasts are already aligned with the target and so do not need further shifting.

```python
forecasts=res.forecast(horizon=5)
forecasts.variance.dropna()
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17396201142958.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="600px"}{: .align-center}

The forecasts above are made with the data available in Day T for the following days. `2013-12-31`:

* h.1 $\to$ `2014-01-02`
* h.2 $\to$ `2014-01-03`

In order to make predicion, we need further shifting.

If forecast is called with `align="target"`, then the forecasts will already be aligned with the target day,and no further shifting is needed.

```python
forecasts=res.forecast(horizon=1,align='target')
forecasts.residual_variance.dropna()
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17401030474516.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="200px"}{: .align-center}



```python
# from the given start date
forecasts=res.forecast(horizon=5,start=dt.datetime(2014,12,31))
forecasts.variance.dropna()
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17396201606196.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}


#### return or forecast()

Forecasts are contained in an `ARCHModelForecast` object which has 4 attributes:

*   `mean` \- The forecast means

*   `residual_variance` \- The forecast residual variances, that is $E_t[\epsilon_{t+h}^2]$

*   `variance` \- The forecast variance of the process, $E_t[r_{t+h}^2]$. The variance will differ from the residual variance whenever the model has mean dynamics, e.g., in an AR process.

*   `simulations` \- An object that contains detailed information about the simulations used to generate forecasts. Only used if the forecast `method` is set to `'simulation'` or `'bootstrap'`. If using `'analytical'` (the default), this is `None`.

Lest's check the performance of the model:

```python
forecasts=res.forecast(horizon=5)
forecasts.variance.dropna()
plt.rcParams["figure.figsize"] = (12, 6)
plt.figure(dpi=100)

temp1=pd.concat([returns[returns.index.year>2013],forecasts.variance.iloc[:,0].shift(1).apply(lambda t: np.sqrt(t))],1)
temp1.columns=['returns','GARCH-std']
temp1.plot()
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17399568975243.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}

Combine all the forecasts

```python
fma=historical_volatility(returns,align='target')
fma=fma['2014':]

fewma=ewma_volatility2(returns,align='target')
fewma=fewma["2014":]

all_fores=pd.concat([temp1,fma,fewma],1)
all_fores.columns=['returns','GARCH-std','MA-std','EWMA-std']
all_fores.plot()
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17399562837746.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}

```python
all_fores['2014-12':'2015-12'].plot()
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17401285095257.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}


Check the forecasts of GARCH and EWMA

```python
all_fores.loc[:,['GARCH-std','EWMA-std']].plot()
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17399571613804.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}

#### Rolling Window Forecasting

Rolling window forecasts use a fixed sample length and then produce one-step from the final observation. These can be implemented using `first_obs` and `last_obs`.

```python
index = returns.index
start_loc = 0
end_loc = np.where(index >= '2010-1-1')[0].min()
forecasts = {}
for i in range(20):
    sys.stdout.write('.')
    sys.stdout.flush()
    res = am.fit(first_obs=i, last_obs=i + end_loc, disp='off')
    temp = res.forecast(horizon=3).variance
    fcast = temp.iloc[i + end_loc - 1]
    forecasts[fcast.name] = fcast
print()
print(pd.DataFrame(forecasts).T)
```


#### Recursive Forecast Generation

Recursive is similar to rolling except that the initial observation does not change. This can be easily implemented by dropping the `first_obs` input.

#### Simulation forecasts

For non-linear models (say TARCH), forecasts have no closed-form solution for horizons larger than 1. In this case we have to use the `forecast(mothod='simulation')`.


```python
res.conditional_volatility['2013'].plot()
# Simulation forecasts
forecasts_sim = res.forecast(horizon=5, method='simulation')
sims = forecasts_sim.simulations

# Bootstrap forecasts
forecasts_bootstrap= res.forecast(horizon=5, method='bootstrap')
sims = forecasts_bootstrap.simulations
```

Check the simulated lines:

```python
x = np.arange(1, 6)
dd=5 # show one line every dd lines.
lines = plt.plot(x, sims.residual_variances[-1, ::5].T, color='#9cb2d6', alpha=0.5)
lines[0].set_label('Simulated path')
line = plt.plot(x, forecasts.variance.iloc[-1].values, color='#002868')
line[0].set_label('Expected variance')
plt.gca().set_xticks(x)
plt.gca().set_xlim(1,5)
legend = plt.legend()
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17399640892269.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}


#### Value-at-Risk Forecasting

Value-at-Risk (VaR) forecasts from GARCH models depend on the conditional mean, the conditional volatility and the quantile of the standardized residuals：

$$VaR_{t+1|t} = \mu_{t+1|t} +q_{\alpha}  \sigma_{t+1|t} $$

or take the absolute value of it:

$$VaR_{t+1|t} = - \mu_{t+1|t} - q_{\alpha}  \sigma_{t+1|t} $$


where $q_{\alpha}$ is the $\alpha$ quantile of the standardized residuals, e.g., 5%.



```python
forecasts=res.forecast(horizon=1,align='target')
cond_mean=forecasts.mean.dropna()
cond_var=forecasts.variance.dropna()
q=am.distribution.ppf([0.01,0.05],res.params[-1:])
#here for student only 1 parameter nu

value_at_risk = cond_mean.values + np.sqrt(cond_var).values * q[None, :]
value_at_risk=pd.DataFrame(value_at_risk,columns=['1%','5%'],index=cond_var.index)
ax=value_at_risk.plot()
ax.set_title('GARCH VAR')
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17401289798734.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}

Notice:

`scipy.t.ppf(alpha,df,loc,scale)` is the Inverse CDF of student's t distribution. Therefore:

`sp.stats.t.std(df=nu,loc=0,scale=1)` = $\sqrt\frac{v}{v-2}$

`am.ditribution` is Standardized Student's t distribution with only one parameter, which is `df` or $\nu$ (nu), with mean=0, std=1.

`am.ditribution.ppf` is the Inverse CDF of **standardiazed** student's t distribution. The following are equivalent for any v greater than 2

* `am.distribution.ppf(alpha,v)` for `dist='StudentsT'`

* `sp.stats.t.ppf(alpha,df=v,loc=0,scale=np.sqrt((v-2)/v))`


Let's also consider Vars given by the MA and EWMA method:

```python
import scipy as sp
qq=sp.stats.norm.ppf([0.01,0.05])
mu=cond_mean.values.mean()

var_MA=pd.DataFrame(all_fores['MA-std']).dropna().values* qq[None,:] + mu
var_MA=pd.DataFrame(var_MA,columns=['1%','5%'],index=all_fores['MA-std'].dropna().index)
ax=var_MA.plot()
ax.set_title('MA VAR')

var_EWMA=pd.DataFrame(all_fores['EWMA-std']).dropna().values* qq[None,:] + mu
var_EWMA=pd.DataFrame(var_EWMA,columns=['1%','5%'],index=all_fores['EWMA-std'].dropna().index)
ax=var_EWMA.plot()
ax.set_title('EWMA VAR')
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17401289637554.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17401289706026.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}


### Performance

#### Var Scatter

```python
def myVarScatter(rets,value_at_risk):
    """
    A simple scatter plot for given returns and Value-at-Risks: column_names=alphas,['1%','5%']
    alphas are expected to be ranked upwards: 1%, 5%; not 5%,1%
    Returns and Vars are expecetd to be indexed with similar dates.
    """
    alphas=value_at_risk.columns
    ax = value_at_risk.plot(legend=False)
    xl = ax.set_xlim(value_at_risk.index[0], value_at_risk.index[-1])
    c = []
    for idx in value_at_risk.index:
        if rets[idx] > value_at_risk.loc[idx, alphas[1]]:
            c.append("#000000")
        elif rets[idx] > value_at_risk.loc[idx, alphas[0]]:
            c.append("#BB0000")
        else:
            c.append("#BB00BB")
    c = np.array(c, dtype="object")
    labels = {
        "#BB00BB": f"{alphas[0]} Exceedence",
        "#BB0000": f"{alphas[1]} Exceedence",
        "#000000": "No Exceedence",
    }
    markers = {"#BB0000": "x", "#BB00BB": "s", "#000000": "o"}
    sizes={"#BB0000": 80, "#BB00BB": 100, "#000000": 20}
    for color in np.unique(c):
        sel = c == color
        ax.scatter(
            rets.index[sel],
            rets.loc[sel],
            marker=markers[color],
            c=c[sel],
            s=sizes[color],
            label=labels[color],
        )
    ax.set_title("VaR Scatter")
    leg = ax.legend(frameon=False, ncol=3)
    for color in np.unique(c):
        print(f"{labels[color]}: { (c==color).mean()}")
    return
```

#### GARCH model

```python
myVarScatter(returns["2015":],value_at_risk["2015":])
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17401260894215.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}

* No Exceedence: 94.83%
* 5% Exceedence: 3.58%
* 1% Exceedence: 1.59%

#### MA

```python
myVarScatter(returns['2015':],var_MA['2015':])
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17401261460487.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}

* No Exceedence: 93.24%
* 5% Exceedence: 3.78%
* 1% Exceedence: 2.98%

#### EWMA

```python
myVarScatter(returns['2015':],var_EWMA['2015':])
```

![](http://bens-2-pics.oss-cn-shanghai.aliyuncs.com/2025/02/21/17401261460487.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="500px"}{: .align-center}

* No Exceedence: 94.33%
* 5% Exceedence: 3.28%
* 1% Exceedence: 2.39%

#### Conclusion

By comparison, the GARCH model have better performance in estimating 1%-VaR.



### Codes

```python
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import scipy.stats as stats
import matplotlib.pyplot as plt


def historical_volatility(returns, window=100, min_periods=50, align='origin'):
    """MA standard deviation"""
    if align == 'origin':
        return returns.rolling(window, min_periods).std()
    elif align == 'target':
        return returns.rolling(window, min_periods).std().shift(1)
    else:
        print("plese choose the align way: origin or target")
        return


def ewma_volatility(returns, lambda_param=0.94, align='origin'):
    """EWMA standard deviation, Assume 𝜆=0.94 as in RiskMetrics for daily returns"""
    returns2 = returns - returns.mean()
    returns2 = returns2.pow(2)
    var = returns2.ewm(alpha=1 - lambda_param).mean()
    if align == 'origin':
        return np.sqrt(var)
    elif align == 'target':
        return np.sqrt(var).shift(1)
    else:
        print("plese choose the align way: origin or target")
        return


def ewma_volatility2(returns, lambda_param=0.94, min_periods=50, align='origin'):
    """EWMA standard deviation with EWMA mean"""
    # alternaively use the ewm().var(), in which the mean is the EWM mean
    var = returns.ewm(alpha=1 - lambda_param, min_periods=min_periods).var()
    if align == 'origin':
        return np.sqrt(var)
    elif align == 'target':
        return np.sqrt(var).shift(1)
    else:
        print("plese choose the align way: origin or target")
        return


def myVarScatter(rets, value_at_risk):
    """
    A simple scatter plot for given returns and Value-at-Risks: column_names=alphas,['1%','5%']
    alphas are expected to be ranked upwards: 1%, 5%; not 5%,1%
    Returns and Vars are expecetd to be indexed with similar dates.
    """
    alphas = value_at_risk.columns
    ax = value_at_risk.plot(legend=False)
    xl = ax.set_xlim(value_at_risk.index[0], value_at_risk.index[-1])
    c = []
    for idx in value_at_risk.index:
        if rets[idx] > value_at_risk.loc[idx, alphas[1]]:
            c.append("#000000")
        elif rets[idx] > value_at_risk.loc[idx, alphas[0]]:
            c.append("#BB0000")
        else:
            c.append("#BB00BB")
    c = np.array(c, dtype="object")
    labels = {
        "#BB00BB": f"{alphas[0]} Exceedence",
        "#BB0000": f"{alphas[1]} Exceedence",
        "#000000": "No Exceedence",
    }
    markers = {"#BB0000": "x", "#BB00BB": "s", "#000000": "o"}
    sizes = {"#BB0000": 80, "#BB00BB": 100, "#000000": 20}
    for color in np.unique(c):
        sel = c == color
        ax.scatter(
            rets.index[sel],
            rets.loc[sel],
            marker=markers[color],
            c=c[sel],
            s=sizes[color],
            label=labels[color],
        )
    ax.set_title("VaR Scatter")
    leg = ax.legend(frameon=False, ncol=3)
    for color in np.unique(c):
        print(f"{labels[color]}: { (c==color).mean()}")
    return


if __name__ == "__main__":
    # get the data
    import arch.data.sp500
    data = arch.data.sp500.load()
    returns = 100 * data['Adj Close'].apply(np.log).diff().dropna()

    # modelling and fitting
    from arch import arch_model
    am = arch_model(returns, dist='StudentsT', p=1, q=1, o=0)
    split_date = dt.datetime(2014, 1, 1)  # will not be inlcuded in the sample
    res = am.fit(update_freq=5, last_obs=split_date)
    fig = res.plot()

    # fixed params fit
    fix_res = am.fix([0.06, 0.01, 0.08, 0.9, 8])
    fig2 = fix_res.plot()

    df = pd.concat([res.conditional_volatility,
                    fix_res.conditional_volatility], 1)
    df.columns = ['estemated', 'fixed']
    plt.style.use('ggplot')
    df.plot()

    # forecasts
    forecasts = res.forecast(
        horizon=1, start=dt.datetime(2014, 12, 31), align='target')

    # Returns vs Standard deviation
    temp1 = pd.concat([returns[returns.index.year > 2013],
                       forecasts.variance.apply(lambda t: np.sqrt(t))], 1)
    temp1.columns = ['returns', 'GARCH-std']
    temp1.plot()

    # Returns vs all forecasts
    fma = historical_volatility(returns, align='target')
    fma = fma['2014':]

    fewma = ewma_volatility2(returns, align='target')
    fewma = fewma["2014":]

    all_fores = pd.concat([temp1, fma, fewma], 1)
    all_fores.columns = ['returns', 'GARCH-std', 'MA-std', 'EWMA-std']
    all_fores['2014-12':'2015-12'].plot()

    # rolling-window forecasts
    index = returns.index
    start_loc = 0
    end_loc = np.where(index >= '2010-1-1')[0].min()
    forecasts = {}
    for i in range(20):
        sys.stdout.write('.')
        sys.stdout.flush()
        res = am.fit(first_obs=i, last_obs=i + end_loc, disp='off')
        temp = res.forecast(horizon=3).variance
        fcast = temp.iloc[i + end_loc - 1]
        forecasts[fcast.name] = fcast
    print()
    print(pd.DataFrame(forecasts).T)

    # Var forecasts - GARCH
    forecasts = res.forecast(horizon=1, align='target')
    cond_mean = forecasts.mean.dropna()
    cond_var = forecasts.variance.dropna()
    q = am.distribution.ppf([0.01, 0.05], res.params[-1:])
    # here for student only 1 parameter nu
    value_at_risk = cond_mean.values + np.sqrt(cond_var).values * q[None, :]
    value_at_risk = pd.DataFrame(value_at_risk, columns=[
                                 '1%', '5%'], index=cond_var.index)
    ax = value_at_risk.plot()
    ax.set_title('GARCH VAR')

    # Var forecassts-MA and EWMA
    qq = sp.stats.norm.ppf([0.01, 0.05])
    mu = cond_mean.values.mean()

    var_MA = pd.DataFrame(
        all_fores['MA-std']).dropna().values * qq[None, :] + mu
    var_MA = pd.DataFrame(
        var_MA, columns=['1%', '5%'], index=all_fores['MA-std'].dropna().index)
    ax = var_MA.plot()
    ax.set_title('MA VAR')

    var_EWMA = pd.DataFrame(
        all_fores['EWMA-std']).dropna().values * qq[None, :] + mu
    var_EWMA = pd.DataFrame(
        var_EWMA, columns=['1%', '5%'], index=all_fores['EWMA-std'].dropna().index)
    ax = var_EWMA.plot()
    ax.set_title('EWMA VAR')

    # Check the performance
    myVarScatter(returns["2015":], value_at_risk["2015":])
    myVarScatter(returns['2015':], var_MA['2015':])
    myVarScatter(returns['2015':], var_EWMA['2015':])

```
