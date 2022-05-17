---
layout: mysingle
date: 2022-05-16 22:34:16 +0800
title: Quantitative Trading - 1. Introduction
categories: Quantitative Trading
# excerpt: "Notes for The Power of Macroeconomics: Economic Principles in the Real World from coursera."
header:
    # overlay_color: "#333"
    # overlay_color: "#2f4f4f" #暗岩灰
    # overlay_color: "#e68ab8" #火鹤红
    overlay_color: "#0000ff" #蓝色
classes: wide
tags: linear_regression ols qmj mkt factor

toc: true
---


[TOC]

## Quant roles

* Data manager
* Strategist
* Back-tester
* Execution
* Portfolio management
* Risk analysis

## What is the quant trading skillset?

* Intuition about trading
    * What strategies are there?
    * What do people do?
    * What do we, and don’t we know?
    * How do I develop an edge?
* Statistical back-testing – take an idea and test if it works
    * Econometrics
    * Machine learning + alternative data
* Executing trades
    * Programming
    * Thinking about market rules
    * Game theory / strategy


The required skills are too many for one person. Therefore we must focus on a subset.

This class is designed for quant strategists CS, you focus on building a back-test, light on economic intuition.

In some asset classes, execution trading is essentially irrelevant and quantitative valuation is all that matters.

For example my former life was traded via auctions and OTC.  Most of the quant trading workflow is on the strategy, data and back-testing side.



## Definition of Quant Trading strategy

<div  class="definition">
Quant trading strategies are repeatable, rules-based strategies.
</div>

**Defined by these key features:**

*  Cross-sectional versus time-series
    * “relative outperformance” versus “up or down performance”
*  Horizon / frequency of trading
    *  “week or month’
*  Asset class “equity, bond…”
*  Instruments
*  Strategy

## Why Quant

![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135571452782.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)
![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135571589988.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)


![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135571871550.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

* Discretionary decisions are increasingly being fed by quant data
    * Alternative data, for example
* Understanding how to be fluent in quant strategies helps you make discretionary trades too


## Useful resources

* [Quantocracy](https://quantocracy.com/)
    * Daily blogs – goal is to get you fluent in this discourse

![-w1294](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135574072897.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)


* Textbooks – Antan Ilmamen’s “Expected Returns”
* Machine learning - Marcos De Lo Prado’s book, can also look at Ernest Can + Kaggle.
* [Quantpedia](https://quantpedia.com/)
    * Has a list of academic papers per trading strategy available for free
    * $300USD for a quarterly subscription – which is cheap for institutional asset managers
* [Quantopian](https://en.wikipedia.org/wiki/Quantopian)

# Basics of Quant Trading

A quantitative trading strategy aims to “outperform” and achieve high “return on risk”.

These are terms you probably have heard, let’s unpack them slowly

* What is a stock return?
* What is risk?
    * How do we think about risk?
    * What statistical methodology do we use to calculate “outperform”

## Difference between price and return

* Stock price returns, not prices
* Transaction costs and liquidity
    * The return you get is the return minus the transaction has a cost
    * Expected trading costs

![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135615904374.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

Price changes are wrong because of the following problems.

### Stock split

Apple's stock has split four times since the company went public. The stock split on a 7-for-1 basis on June 9, 2014 and split on a 2-for-1 basis on February 28, 2005,June 21, 2000, and June 16, 1987.


### Dividends

The price of a stock should fall by the amount of the dividend. (Not exactly true due to taxes, see Ivo Welch’s work)

![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135617253419.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

***Payouts are repurchases and dividends.***

About 1/4th of stocks pay dividends, important to consider:

![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135617563477.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

### Definition of return

A stock return includes both dividends and price appreciation:

$$r_{t+1}=\frac{P_{t+1}-P_t + D_{t+1}}{P_t}$$

After adjusting the price for splits – your financial database should do this for you.

Another thing to consider is taxes:

$$r_{t+1}=\frac{P_{t+1}-P_t + D_{t+1}}{P_t}(1-T)$$

### Financial Database in practice

Many databases have a field called stock return. The implicit assumption:

* Dividends are reinvested immediately, with zero transaction costs
* Of course, this may not be true. if I get a 1% dividend and I cannot buy fractional shares.
* I may also need to wait 3 days to receive the money

Some databases, if you’re lucky, calculate returns. Most of the time, they give you, for example with Factset:
* share price
* split adjustment factor
* Dividends
* Merger adjustment factor (rare)

**Example: Compustat**

Professional and Industry use.

![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135620695590.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)


**Example: CRSP**

Where will we get data from? CRSP! (Academic Use)

* University of Chicago developed the  Center for Research on Security Prices (1961) with a grant from Merill Lynch
    *  the first ever stock return database
*  Why was it good?
    *  Accounted for historical de-listings and listings (survivorship bias)
    *  Calculated price returns that included dividends
    *  Included data from all available domestic exchanges
* Roughly all stock data is structured similarly after CRSP set the standard

For more info, see [Wharton Research Data Services](mweblib://16053223786055).

**Tiingo**

> Made with Love Across the East Coast
>
> Tiingo was formed in 2014 and holds that belief that love is the ideal way to conduct business. We are a team made up of artists, engineers, and algorithmic hedge fund traders. Some of us have been professional photographers, and others have created trading algos managing hundreds of millions of dollars. We are united with the same goal: to make everyone's life easier in the ways we know how.
>
> Our client base is made up of individuals, hedge funds, quant funds, fintech companies, academic institutions, and others. To see our complete product offering, please visit the Tiingo Welcome Page.


![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135667234399.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135667489846.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

**Bottom Line**
We should care about the quality of the data.

Even professional databases may have bad data.

But always inspect very strange returns from a new database.
Generally, remove suspect observations or diagnose them and create a ruleset. So, for example any daily movements above 80%


### Simple and compound interest


#### Log return and standard return

One of the most common applications of logarithms in finance is computing log returns. Log returns are defined as follows:

$$\begin{equation}
r_{t} \equiv \ln \left(1+R_{t}\right) \quad \text { where } \quad R_{t}=\frac{P_{t}-P_{t-1}}{P_{t-1}}
\end{equation}$$

Alternatively:

$$ e^{r_t}= 1+R_{t}=\frac{P_{t}}{P_{t-1}}$$


To get a more precise estimate of the relationship between standard returns and log returns, we can use the following approximation:

$$r\approx R-\frac{1}{2}R^2$$

#### Compounding

To get the return of a security for two periods using simple returns, we have to do something that is not very intuitive, namely adding one to each of the returns, multiplying, and then subtracting one:

$$\begin{equation}
R_{2, t}=\frac{P_{t}-P_{t-2}}{P_{t-2}}=\left(1+R_{1, t}\right)\left(1+R_{1, t-1}\right)-1
\end{equation}$$

and for the log return:

$$r_{2,t}=r_{1,t}+r_{1,t-1}$$



#### Log return and log price

Define $p_t$ as the log of price $P_t$, then we have:

$$r_t=ln(P_t/P_{t-1})=p_t-p_{t-1}$$

As a result, we can check the log(price)-Time plot, to see whether the return is increasing:

![-w600](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/15938365361053.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

According to the graph above, the log return is constant, even though the price is increasing faster than a linear speed.


#### Cumulative return

Cumulative return: compound every return before it. Keep compounding and keep track of the total wealth.

Assume $R_t$ is the standard return (with dividend reinvested), the compounded return is:

$$\Pi_{t=0}^N  (R_t + 1) -1 $$

Assume the initial investment is 1 dollar, the Wealth process is defined as :

$$W_t= \Pi_{t=0}^N  (R_t + 1) $$

![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135680658087.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

## Risks

Many definitions of risk…

People may have different risk preferences:

* Risk-neutral: who cares?
* Risk-loving
* Risk-averse

$$\text{Sharpe ratio SR_p}=\frac{r_p-r_f}{\sigma_p} $$

Notice the difference between standard return and log-return:

$$r_t=log(1+R_t)$$

$$\sigma_r \text{ vs } \sigma_R$$

we may need to change the standard risk-free rate $R_f$ into the continuously compounded log-return.

Other metrics in industry (source : JPM)


**Maxdrawdown:**

$$MaxDrawDown=\frac{Peak-Worst}{Peak}$$

Other metrics in industry (source : JPM):

![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135744633001.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135745305901.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135747358360.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135747404056.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

**Key Insight:**

* The modern industry uses a more advanced statistical technique based on linear regression

* There are more advanced measures, but the baseline you need to fluently understand whitepapers is based on this


## Risk-adjustment

**Roadmap:**

* What is risk-adjustment? What is risk?

* What is the role of a regression? How do we perform a regression and interpret a regression?

* Next module: cross-sectional strategies

Given a stock return series r you want to know if it’s a good return on risk.

Sharpe ratio is a start…

* The market has a Sharpe ratio of .4 and expected return of .06
* Suppose we have two strategies with a Sharpe ratio of .4
* How can we tell if strategy #2 is different than the market?
* How can we tell if combining them is a good idea or if they’re just correlated with each other?


**Theories:**

* Market efficiency
* Market beta
* “Alpha”

It turns out they are all related!  And the framework here will provide the foundation for risk calculations

### Market Efficiency Theory

The efficient-market hypothesis (EMH) is a hypothesis in financial economics that states that asset prices reflect all available information. A direct implication is that it is impossible to “beat the market” consistently on a risk-adjusted basis since market prices should only react to new information

![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135753555761.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

Proposed in Fama (1970), by Nobel Prize winner Eugene Fama.

Three types of efficiency:

* **Strong form efficiency** – you can never make money that isn’t compensation for risk “read: the market prices in all private info”

* **Semi-strong form** –  you can never make money that isn’t compensation for risk unless you have private information “read: the market prices in all public info”

* **Weak form efficiency** – you can never make money exceeding factor risk exposure using technical info; fundamentals-based alpha exists


### Basic Takeaways of Risk-adjustment return analysis

Three attitudes towards the relationship between return and risk:

* **Market Efficiency** $\iff$ Return on risk: so you are only  compensated for risk

* **Weaker Market Efficiency** $\iff$ mispriced temporarily – once realized then eliminated

* **Alternative** $\iff$ mispricing – the market is stupid persistently

You never really know which one it is.

* For example: Nobel Prize was shared with Robert Shiller, who claimed that stock markets are irrational. The first finding of behavioral finance

However, academics have this debate still today

### The classic risk model -CAPM

In finance, the **capital asset pricing model (CAPM)** is a model used to determine a theoretically appropriate required rate of return of an asset, to make decisions about adding assets to a well-diversified portfolio.

**Covariance-adjustment (CAPM framework)**

Basic idea: If the market tanks, and your stock tanks, it is risky.

**But there might be multiple risks**.:

* You can have a 3 factor, four factor, five factor risk model.
* What happens if the market doesn’t fall but firms become less productive?
* What happens if consumers just stop demanding things?
* What happens if financial institutions have to unwind?

Even if the market doesn’t fall, there are risks

**CAPM framework for risk-adjustment**

Capital asset pricing model (CAPM) is the baseline model. Since it involves a regression, it is trivial to extend it from the CAPM to a multi-factor CAPM:

Under the CAPM:

$$
E\left[R_{t}^{s}-R_{t}^{f}\right]=\underbrace{\beta_{i}}_{\text { market risk }} * \underbrace{E\left[R_{t}^{m k t}-R_{t}^{f}\right]}_{\text {Market risk Premium}}
$$

On average, a security’s return is proportional to its market risk sensitivity.


![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16135766893199.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

* Under the CAPM, idiosyncratic risk is not rewarded
    * If you don’t like a stock, you sell it and buy another stock
    * In equilibrium, everyone is fully diversified

* In reality, stocks may outperform its beta-implied expected return. Because:
    * Not all risk factors captured
    * Market beta mis-estimated
    * Market not perfectly efficient, and there may be alpha

* If we find that dots don’t line up then … you can find stocks whose returns are higher than the sensitivity implied by CAPM

$$𝑟_{𝑠𝑡}= 𝛼+𝛽_1 (𝑟_{𝑚𝑘𝑡}−𝑟_𝑓)$$

Here $r_{st}$ is the excess return of the strategy.


### Variation of CAPM


* **Spoiler**: the CAPM is too simple to explain average returns so we tried lots of different things

* Different types of CAPM models … read Cochrane’s Discount Rates (2008)
    * Production-based asset pricing , innovation risk,  labor risk, Liquidity risk , Intermediary health risk
    * can stock prices be explained by their exposure to workers leaving the industry?
    * can stock prices be explained by their exposure to being made obsolete by innovation

* While interesting, these papers / debates uncover new economic reasonings for why stocks might produce return

So now we have a “zoo” of factors in academia.

* Three factor (1993) – Fama French: market, size, value model
* Five-factor (mid 2000s) – Fama French: market, size, investment, profitability model
* Even on a 7 factor model now

They are called different things in industry. **Barra model** for example is one standard risk factor library.

<div  class=“definition”>
A factor is a simple tradable strategy you put in this equation.

$$𝑟_{𝑠𝑡}=𝛼+𝛽_1 (𝑟_{𝑚𝑘𝑡}−𝑟_𝑓)  + 𝛽_2 𝑓𝑎𝑐𝑡𝑜𝑟$$
</div>

**Different views on risk factors:**

**Academically**: risk factors are non-diversifiable sources of risk based on a description of how the economy works. These are factors that seem to explain what investors care about evidenced by the fact they earn high returns and nobody wants to buy the stock

**In practice**: If you don’t have alpha relative to this “benchmark” model, then you lose zero opportunity cost.
Also, the betas can tell you your exposure to other known strategies


Whichever religion you have, you still need the same tool.

**Run the regression for k risk factors:**

$$r_t^{strategy}=\alpha+ \sum_{i=1}^k \beta_i \text{. risk-factor}_i+\epsilon_t^{strategy}$$


### Another perspective of risk-factor models

Forget about the economic meaning of risk-factors, as well as the return compensated. Just regard these risk factor as investable strategy.

Remember, a risk factor is a simple tradable strategy: **a self-financing excess return of long/short strategy for the risk factor**.

The portfolio just invests in these risk factors according to the regressed $\{\beta_i\}_{i=1}^k$. We can calibrate the weights of each **self-funding strategy**, so that the portfolio can replicate the regressed result.

From the perspective above:

$$
\begin{cases}
\text{If $\alpha>0$} &\text{great!}\\
\text{If $\alpha=0$} &\text{no cost still!}
\end{cases}
$$



### Example: QMJ


AQR Capital Management, LLC — Quality Minus Junk: Factors, Daily.

This QMJ sheet contains daily **self-financing excess returns of long/short Quality Minus Junk (QMJ) factors.**

```python
import pandas as pd
import numpy as np
import urllib
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"  # multiple output per jupyter notebook code block
%matplotlib inline

x = pd.read_excel(
    'qmj.xlsx',
    sheet_name='QMJ Factors',
    skiprows=18
)

market = pd.read_excel(
    'qmj.xlsx',
    sheet_name='MKT',
    skiprows=18
)


# merge the QMJ table and market table
x2 = pd.merge(x[['DATE', 'USA']], market[['DATE', 'USA']], on='DATE') # Excel sheet has QMJ data
x2 = x2.rename(columns={'USA_x':'qmj', 'USA_y':'mkt'}) # Market has

x2.head()

x2['DATE'] = pd.to_datetime(x2['DATE'])


# calculate the cumulated wealth process
x2 = x2.set_index('DATE', drop=False)

x2['cum_ret_mkt'] = (x2['mkt'] + 1).cumprod() - 1
x2['cum_ret_qmj'] = (x2['qmj'] + 1).cumprod() - 1
x2.head()

# Draw the wealth process

plt.rcParams["figure.figsize"] = (12,8)
ax = (
    x2
    .assign(date=x2['DATE'], mkt=x2['cum_ret_mkt']+1, qmj=x2['cum_ret_qmj']+1)
    .plot(x='date', y=['mkt', 'qmj'], logy=True)
).set_ylabel('cumulative return')

```


![-w719](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16136579743602.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)

```python
# calculate the sharpe ratio
## since it is daily data, we need to annualize it

{
    'Sharpe MKT': x2['mkt'].mean() / x2['mkt'].std() * 255**0.5,
    'Sharpe QMJ': x2['qmj'].mean() / x2['qmj'].std() * 255**0.5
===
Result:
{'Sharpe MKT': 0.41714780189609474, 'Sharpe QMJ': 0.6714456894831614}

}


# Regression QMT~MKT
olsm=smf.ols('qmj ~ mkt', data=x2).fit()
olsm.summary()
# We can also use print(olsm.summary().as_latex())
# or: print(olsm.summary().as_html())
```

![-w974](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16136585746166.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)


**What did we learn from this QMJ example?**
* sharpe ratio is higher
* it is a negative $\beta$ strategy. ON average, when the market goes up by 1%, QMJ does down by .18% $\implies$ QMJ factor will diversify the MKT factor.
* QMJ has positive $\alpha$ with respect to market
    * it is very significant
* Therefore, a $\beta$ neutral strategy e.g. short market by .18cents for every dollar invested in QMJ will earn, on average, .02 basis points per day or 5.4 % per year

* Because they are diversified and one has alpha with respect to another strategy, a blend should give diversification benefits and a higher Sharpe ratio

![-w948](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16136600380440.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)


```python
import sklearn.metrics as metrics
r2=metrics.r2_score(x2['qmj'],olsm.fittedvalues)
rho=np.sqrt(r2)
```

$\rho=0.45, SR_{QMJ}=0.67, SR_{MKT}=0.42$
We have the incremental Sharpe ratio of $S_{QMJ}^*$ is :
$$S_{QMJ}^* =  SR_{QMJ} - SR_{MKT}*\rho=0.481$$

As a result, adding QMJ to MKT will increase the sharpe ratio of MKT.


**In fact:**

```python
{
    'Sharpe 1/2 QMJ + 1/2 MKT': ((x2['mkt']+x2['qmj'])/2).mean() / ((x2['mkt']+x2['qmj'])/2).std() * 255**0.5,
    'Sharpe MKT': x2['mkt'].mean() / x2['mkt'].std() * 255**0.5,
    'Sharpe QMJ': x2['qmj'].mean() / x2['qmj'].std() * 255**0.5
}


===
#Result:
{'Sharpe 1/2 QMJ + 1/2 MKT': 0.776779377105961,
 'Sharpe MKT': 0.41714780189609474,
 'Sharpe QMJ': 0.6714456894831614}
```


```python

x2['halfsies'] = (x2['mkt']+x2['qmj'])/2

smf.ols('halfsies ~ mkt + qmj', data=x2).fit().summary()
```

![-w699](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16136588079666.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)


```python
#compare the half-size strategy and mkt

x2['cum_halfsies'] = (x2['halfsies'] + 1).cumprod() - 1


plt.rcParams["figure.figsize"] = (12,8)

ax = (
    x2
    .assign(date=x2['DATE'], mkt=x2['cum_ret_mkt']+1, halfsies=x2['cum_halfsies']+1)
    .plot(x='date', y=['mkt', 'halfsies'], logy=True)
).set_ylabel('cumulative return')

```

![-w718](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2022/05/17/16136588442967.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10)


Note:

pd.assign是直接向DataFrame对象添加新的一列

```python

import numpy as np
import pandas as pd

data = {'name':['lily','jack','hanson','bom'],'age':np.random.randint(15,25,size=4),'gerder':['F','M','M','F']}
df = pd.DataFrame(data)
df.assign(score=np.random.randint(60,100,size=4))
```



## My own replication in Github

https://github.com/lengfc09/quantitative_trading/blob/main/replication-1.ipynb
