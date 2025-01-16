---
layout: mysingle
date: 2020-10-03 18:02:16 +0800
title: Return to RiskMetrics (notes)
categories: Quantitative_Financial_Risk_Management
excerpt: "An update and restatement of the mathematical models in the 1996 RiskMetrics Technical Document, now known as RiskMetrics Classic. RiskMetrics Classic was the fourth edition, with the original document having been published in 1994. Since the initial publication, the model has become the standard in the field and is used extensively in practice, in academic studies, and as an educational tool."
header:
    overlay_color: "#333"
    # overlay_color: "#2f4f4f" #暗岩灰
    # overlay_color: "#e68ab8" #火鹤红
classes: wide
tags: statistics risk_management

toc: true
---


## Risk factors


Risk management systems are based on models that describe potential changes in the factors affecting portfolio value.

In other words, RM models function as:

$$
\begin{aligned}
\text{Risk fators} \to  \text{Prices of Instruments} \to \text{P&L of Portfolios}
\end{aligned}
$$

**By generating future scenarios for each risk factor**, we can infer changes in portfolio value and reprice the portfolio accordingly for different “states of the world.”



<div btit="--Two approaches to generate future scenarios for risk factors" blab="2method" class="proposition">

    <!-- Two approaches to generate future scenarios for risk factors: -->
$$
\begin{aligned}
&\textbf{1. Assume its probability distribution function} \\
&\textbf{2. Assume that future behavior will be similar to the past behavior}
\end{aligned}
$$
</div>
{: #2method}


Method 1 is similar to Bayesian Analysis in the sense that, we assume a distribution function for the risk factor before we incorporating the past information. Therefore, the past information is only used to calibrate the parameters of the distribution function.

Method 2, however, totally relies on the past information, and bears no prior belief.



<div btit="The risk management usually works in following way:"  class="attention">
$$
\begin{aligned}

&\text{Step 1: Identify the relevant risk factors }\\
&\text{Step 2: Specify the distribution of the risk factors.}\\
&\text{Step 3: Apply the pricing models with the risk factors given. }\\
&\text{Step 4: Produce P&L scenarios for the portfolio.}

\end{aligned}
$$

</div>

## Predict risk factors: with Models based on Distributional Assumptions

### Classic methodology presented in RiskMetrics Classic

In the classic methodology presented in RiskMetrics Classic, a distribution function is assumed for the log-return of the risk factors. This prior distribution is indeed called **conditionally normal distribution**. Here is the definition:

<div  class="definition">
Conditionally normal distribution means the logarithmic returns on the risk factors follow a normal distribution conditional on the current volatility estimate.
</div>

Moreover, the classic model updates the return volatility estimates based on the arrival of new information, where the importance of old observations *diminishes exponentially with time*.



The model for the distribution of future returns is based on the notion that log-returns, when standardized by an appropriate measure of volatility, are independent across time and normally distributed.

### How to get calibrate the conditionally normal distribution

With this so-called "conditionally normal distribution" in our mind, we still need to estimate the **volatility and correlation** of the risk factors before we can apply this model.

**Luckily, information about volatility and correlation are all included in the covariance matrix!** It means we just need to estimate the covariance matrix $\sum $ for the risk factors.



### For single risk factor model

Let us start by defining the logarithmic return of a risk factor as

$$
r_{t, T}=\log \left(\frac{P_{T}}{P_{t}}\right)=p_{T}-p_{t}
$$

where $r_{t, T}$ denotes the logarithmic return from time t to time T. $P_T$ is the level/price of the risk factor, $p_t=log(P_T )$ is the log-price.

Given a volatility estimate $\sigma$, the process generating the returns follows a geometric random walk:

$$
\frac{dP_t}{P_t}=\mu dt + \sigma dW_t $$

By Ito's formula, this means that the return from time t to time T can be written as:

$$
r_{t, T}=\left(\mu-\frac{1}{2} \sigma^{2}\right)(T-t)+\sigma \varepsilon \sqrt{T-t} \tag{1}\label{1}
$$

where $\varepsilon \sim N (0, 1)$.

There are two parameters that need to be estimated in \eqref{1}: the drift $\mu$ and the volatility $\sigma$.

<div  class="info">

For horizons shorter than three months are not likely to produce accurate predictions of future returns. In fact, most forecasts are not even likely to predict the sign of returns for a horizon shorter than three months. In addition, since volatility is much larger than the expected return at short horizons, the forecasts of future distribution of returns are dominated by the volatility estimate $\sigma$.

</div>


In other words, when we are dealing with short horizons, **using a zero expected return assumption** is as good as any mean estimate one could provide, except that we do not have to worry about producing a number for $\mu$. Hence, from this point forward, we will make the explicit assumption that the expected return is zero, or equivalently that $\mu=\frac{1}{2}\sigma^2$ .

We can incorporate the zero mean assumption in \eqref{1} and express the return as

$$
r_{t, T}=\sigma \varepsilon \sqrt{T-t} \tag{2}\label{2}
$$


The next question is how to estimate the volatility $\sigma$.

We use an exponentially weighted moving average (EWMA) of squared returns as an estimate of the volatility. If we have a history of m + 1 one-day returns from time t − m to time t , we can write the one-day volatility estimate at time t as

$$
\sigma=\frac{1-\lambda}{1-\lambda^{m+1}} \sum_{i=0}^{m} \lambda^{i} r_{t-i}^{2}=R^{\top} R
$$

where $0<λ≤1$ is the decay factor,$r_t$ denotes the return from day t to day t+1, and

$$
R=\sqrt{\frac{1-\lambda}{1-\lambda^{m+1}}}\left(\begin{array}{c}
r_{t} \\
\sqrt{\lambda} r_{t-1} \\
\vdots \\
\sqrt{\lambda^{m}} r_{t-m}
\end{array}\right)
$$

#### How to estimate the decay factor

By using the idea that the magnitude of future returns corresponds to the level of volatility, one approach to select an appropriate decay factor is to compare the volatility obtained with a certain λ to the magnitude of future returns.

According to RiskMetrics Classic, we formalize this idea and obtain an optimal decay factor by minimizing the mean squared differences between the variance estimate and the actual squared return on each day. Using this method, it is showed that each time series (corresponding to different countries and asset classes), has a different optimal decay factor ranging from 0.9 to 1.

In addition, is is found that the optimal λ to estimate longer-term volatility is usually larger than the optimal λ used to forecast one-day volatility.

The conclusion of the discussion in RiskMetrics Classic is that on average λ = 0.94 produces a very good forecast of one-day volatility, and λ = 0.97 results in good estimates for one-month volatility.

It is worth mentioning that, the information captured with decay factor $\lambda $ and number of observation $m$ is:

$$1-\lambda ^m$$

If we take m to the limit, we can prove that

$$\sigma^2_t=\lambda \sigma^2_{t-1}+(1-\lambda)r_t^2$$

#### Conditional normal distribution may produce heavy-tails

The assumption behind this model is that one-day returns *conditioned on the current level of volatility* are independent across time and normally distributed. It is important to note that this assumption does not preclude a *heavy-tailed unconditional distribution of returns*.

![-w793](/media/16017179858557/16017251984095.jpg){:width="800px"}{: .align-center}


Figure 2.1 compares the unconditional distribution of returns described above to a normal distribution with the same unconditional volatility. One can see that the unconditional distribution of returns has much heavier tails than those of a normal distribution.



### For multiple risk factor model

Suppose that we have n risk factors. Then, the process generating the returns for each risk factor can be written as

$$
\frac{d P_{t}^{(i)}}{P_{t}^{(i)}}=\mu_{i} d t+\sigma_{i} d W_{t}^{(i)}, \quad i=1, \ldots, n \tag{3}\label{3}
$$

where $Var(dW^{(i)})=dt, Cov(dW^{(i)},dW^{(j)})=\rho_{i,j}d_t$.

From \eqref{3} it follows that the return on each asset from time t to time t + T can be written as

$$
r_{t, T}^{(i)}=\left(\mu_{i}-\frac{1}{2} \sigma_{i}^{2}\right)(T-t)+\sigma_{i} \varepsilon_{i} \sqrt{T-t}
$$

where $\varepsilon_i \sim N (0, 1)$ and $Cov[\varepsilon_i,\varepsilon_j]=\rho_{i,j}$.

If we incorporate the zero mean assumption we get that

$$
r_{t, T}^{(i)}=\sigma_{i} \varepsilon_{i} \sqrt{T-t}
$$

Then with EWMA model for estimation of covariances, the covariance between returns on asset i and asset j can be written as:

$$
{\sum}_{i,j}=\sigma_i \sigma_j \rho_{i,j}=\frac{1-\lambda}{1-\lambda^{m+1}} \sum_{k=0}^{m} \lambda^{k} r_{t-k}^{(i)}r_{t-k}^{(j)}
$$

We can also write the covariance matrix 􏰁 as

$$\sum=R^T R$$

where R is an m × n matrix of weighted returns:

$$
R=\sqrt{\frac{1-\lambda}{1-\lambda^{m+1}}}\left(\begin{array}{cccc}
r_{t}^{(1)} & r_{t}^{(2)} & \cdots & r_{t}^{(n)} \\
\sqrt{\lambda} r_{t-1}^{(1)} & \cdots & \cdots & \sqrt{\lambda} r_{t-1}^{(n)} \\
\vdots & \vdots & \vdots & \vdots \\
\vdots & \vdots & \vdots & \vdots \\
\sqrt{\lambda^{m}} r_{t-m}^{(1)} & \sqrt{\lambda^{m}} r_{t-m}^{(2)} & \cdots & \sqrt{\lambda^{m}} r_{t-m}^{(n)}
\end{array}\right) \tag{EWMA}\label{EWMA}
$$

### Utilize the conditionally normal distribution model

Till this point, we have calibrated the parameters in our *conditionally normal distribution model*. As a result, we can use this model to predict the risk factors in various scenarios.

We have two approach to utilize this assumed model, which are:

* Monte Carlo simulation
* Parametric methods

#### Monte Carlo simulation


In order to understand the process to generate random scenarios, it is helpful to write \eqref{3} in terms of independent Brownian increments $d\widetilde{W}_{t}^{(i)}$:

$$
\frac{d P_{t}^{(i)}}{P_{t}^{(i)}}=\mu_{i} d t+\sum_{j=1}^{n} c_{j i} d \widetilde{W}_{t}^{(j)} . \quad i=1, \ldots, n
$$

In other words, we use linear combination of **independent brownian motions** to express the correlated brownian motions of the risk factors.

We can gain more intuition about the coefficients $c_{ji}$ if we write (2.14) in vector form:

$$\frac{\mathcal{dP_t}}{\mathcal{P_t}}=\mu dt+ C^T d\widetilde{W}_{t}$$

where $$\left\{\frac{\mathrm{d} \mathbf{P}_{t}}{\mathbf{P}_{t}}\right\}_{i}=\frac{d P_{t}^{(i)}}{P_{t}^{(i)}}(i=1,2, \ldots, n)$$ is a $n\times 1$ vector, $d\widetilde{W}_{t}$ is a vector of n independent Brownian increments.

This means that the vector of returns for every risk factor from time t to time T can be written as

$$\mathcal{r}_{t,T}=(\mu -\frac{1}{2}\mathcal{\sigma}^2)(T-t)+ C^T \mathcal{z} \sqrt{T-t} \tag{4}\label{4}$$

where $\mathcal{r}$ is a vector of returns from from time t to time T. $\mathcal{\sigma}^2(T-t)$  is a n × 1 vector equal to the diagonal of the covariance matrix $\sum$, $\mathcal{z} \sim MVN (0,I)$, a multi-variant normal distribution.


Following our assumption that $\mu_i = \frac{1}{2} \sigma_i^2 $, we can rewrite \eqref{4} as

$$
\mathbf{r}_{t, T}=C^{\top} \mathbf{z} \sqrt{T-t}
$$

We can calculate the covariance of r as

$$
\begin{aligned}
\text { Covariance } &=\mathbf{E}\left[C^{\top} \mathbf{z z}^{\top} C\right] \\
&=C^{\top} \mathbf{E}\left[\mathbf{z z}^{\top}\right] C \\
&=C^{\top} \mathbf{I} C \\
&=\Sigma
\end{aligned}
$$

<div btit="Summary for Generating Risk Factor Scenarios for Monte-Carlo Smulation"  class="attention">

</div>


Step 1: Use \eqref{EWMA} to estimate the covariance matrix for the joint returns of multiple risk factors

Step 2: Use SVD or Cholesky decomposition to find a matrix C such that $\sum =C^T C$.

Step 3: We generate n independent standard normal variables that we store in a column vector $\mathbf{z}$.

Step 4:  we use $\mathbf{r}=\sqrt{T}C^T \mathbf{z}$ to produce a T-day joint returns.($\mathbf{r}$ is a n × 1 vector)

Step 5: Obtain the price of each risk factor T day from now using the formula $P_T=P_0 e^{\mathbf{r}}$

Step 6: Get the portfolio P&L as $\sum [V(P_T)-V(P_0)]$

We can use Cholesky decomposition or the Singular Value decomposition (SVD) to get a decomposition of $\sum$.



#### Parametric methods

If we are willing to sacrifice some accuracy, and incorporate additional assumptions about the behaviour of the **pricing functions**, we can avoid some of the computational burden of Monte Carlo methods and come up with a simple parametric method for risk calculations.

As a result, the parametric approach represents an alternative to Monte Carlo simulation to calculate risk measures. Parametric methods present us with a tradeoff between accuracy and speed. They are much faster than Monte Carlo methods, but not as accurate unless the pricing function can be approximated well by a linear function of the risk factors.

The idea behind parametric methods is to approximate the pricing functions of every instrument in order to obtain an analytic formula for VaR and other risk statistics.

Let us assume that we are holding a single position dependent on n risk factors denoted by $P^{(1)},P^{(2)},\ldots,P^{(n)}$.

To calculate VaR, we approximate the present value V of the position using a first order Taylor series expansion:

$$
V(\mathbf{P}+\Delta \mathbf{P}) \approx V(\mathbf{P})+\sum_{i=1}^{n} \frac{\partial V}{\partial P^{(i)}} \Delta P^{(i)} \tag{5}\label{5}
$$

Even though $r^{(i)}$ is the log-return for $P^{(i)}$, for short-horizon cases, we can approximate that

$$\Delta P^{(i)}\approx P^{(i)}*r^{(i)}$$

If we denote $\delta_i$ as

$$\delta_i = P^{(i)} \frac{\partial V}{\partial P^{(i)}}$$

\eqref{5} can be denoted as the following matrix form:

$$\Delta V = \delta ^T \mathbf{r} \tag{Parametric Method}\label{Parametric Method}$$

Since risk factor returns $\mathbf{r}$ are normally distributed, it turns out that the P&L distribution under our parametric assumptions is also normally distributed with mean zero and variance $\delta^T \sum \delta$. In other words,

$$\Delta V \sim N(0,\delta^T \sum \delta) \tag{Distribution for Parametric Method}\label{Distribution for Parametric Method}$$


<div  class="info">
The entries of the δ vector are called the “delta equivalents” for the position, and they can be interpreted as
the set of sensitivities of the present value of the position with respect to percent changes in each of the risk factors.
</div>


In the inference above, we approximate the percentage returns $\frac{\Delta P }{P}$ by the log-return $r$. The reason we still model the log-return is that:

Percentage returns have nice properties when we want to aggregate across assets. For example, if we have a portfolio consisting of a stock and a bond, we can calculate the return on the portfolio as a weighted average of the returns on each asset:

$$
\frac{P_{1}-P_{0}}{P_{0}}=w r^{(1)}+(1-w) r^{(2)}
$$

where $w$ is the weight and $r$ is the percentage return.

But this percentage return can not be aggregated easily across time.

$$
\frac{P_{2}-P_{0}}{P_{0}}\neq \frac{P_{1}-P_{0}}{P_{0}}+ \frac{P_{2}-P_{1}}{P_{1}}
$$

In contrast with percentage returns, logarithmic returns aggregate nicely across time.

$$
r_{t, T}=\log \left(\frac{P_{T}}{P_{t}}\right)=\left(\log \frac{P_{\tau}}{P_{t}}\right)+\left(\log \frac{P_{T}}{P_{\tau}}\right)=r_{t, \tau}+r_{\tau, T}
$$

This is why we still model the log-return, so that the volatility of periods can be easily calculated as $\sqrt{T}*\sigma_1$.




## Predict risk factors: with Models Based on Empirical Distributions


A shared feature of the methods of last chapter is that they rely on the assumption of **a conditionally normal distribution** of returns. However, it has often been argued that the true distribution of returns (even after standardizing by the volatility) implies *a larger probability of extreme returns* than that implied from the normal distribution. But up until now academics as well as practitioners have not agreed on an industry standard heavy-tailed distribution.

Instead of trying to explicitly specify the distribution of returns, we can let historical data dictate the shape of the distribution.

It is important to emphasize that while we are not making **direct assumptions** about the likelihood of certain events, those likelihoods are determined by the historical period chosen to construct the empirical distribution of risk factors.

### Trade off between long sample time periods and short time periods

The choice of the length of the historical period is a critical input to the historical simulation model.

- long sample periods which potentially violate the assumption of i.i.d. observations (due to regime changes) and
- short sample periods which reduce the statistical precision of the estimates (due to lack of data).

One way of mitigating this problem is to scale past observations by an estimate of their volatility. Hull and White (1998) present a volatility updating scheme; instead of using the actual historical changes in risk factors, they use historical changes that have been adjusted to reflect the ratio of the current volatility to the volatility at the time of the
observation.

 As a general guideline, if our horizon is short (one day or one week), we should use the shortest possible history that provides enough information for reliable statistical estimation.

### Historical simulation

Suppose that we have n risk factors, and that we are using a database containing m daily returns. Let us also define the m × n matrix of historical returns as

$$
R=\left(\begin{array}{cccc}
r_{t}^{(1)} & r_{t}^{(2)} & \cdots & r_{t}^{(n)} \\
r_{t-1}^{(1)} & \cdots & \cdots & r_{t-1}^{(n)} \\
\vdots & \vdots & \vdots & \vdots \\
\vdots & \vdots & \vdots & \vdots \\
r_{t-m} & r_{t-m}^{(2)} & \cdots & r_{t-m}^{(n)}
\end{array}\right)
$$

Then, as each return scenario corresponds to a day of historical returns, we can think of a specific scenario r as a row of R.

Now, if we have M instruments in a portfolio, where the present value of each instrument is a function of the n risk factors $V_j(\mathbf{P})$ with j = 1,... ,M and $\mathbf{P}=(P^{(1)},P^{(2)},\ldots,P^{(n)})$, we obtain a T-day P&L scenario for the portfolios as follows:

1. Take a row $\mathbf{r}$ from R corresponding to a return scenario for each risk factor.
2. Obtain the price of each risk factor T days from now using the formula $P_T = P_0e^{r\sqrt{T}}$
3. Price each instruments with current price $\mathbf{P}_0$ and also using the T-day price scenarios $\mathbf{P}_T$.
4. Get the portfolio P&L as $\sum_j (V_j(\mathbf{P}_T)-V_j(\mathbf{P}_0))$


<div  class="info">

Note that in order to obtain a T -day return scenario in step 2, we multiplied the one-day return scenario r by square root of T. This guarantees that the volatility of returns scales with the square root of time.

$$$$


In general, this scaling procedure will not exactly result in the true T -day return distribution, but is a practical rule of thumb consistent with the scaling in Monte Carlo simulation.

$$$$


An alternative method would be to create a set of T -day non-overlapping returns from the daily return data set. This procedure is theoretically correct, but it is only feasible for relatively short horizons because the use of non-overlapping returns requires a long data history.$\tag{scarcity of historical data}\label{overlapping}$

</div>

<div  class="exampl">
If we have two years of data and want to estimate the distribution of one-month returns, the data set would be reduced to 24 observations, which are not sufficient to provide a reliable estimate. It is important to mention that in this case, the use of overlapping returns does not add valuable information for the analysis and introduces a bias in the estimates.

$$$$

The intuition is that by creating overlapping returns, a large observation persists for T − 1 windows, thus creating T − 1 large returns from what otherwise would be a single large return. However, the total number of observations increases by roughly the same amount, and hence the relative frequency of large returns stays more or less constant. In addition, the use of overlapping returns introduces artificial autocorrelation, since large T -day overlapping returns will tend to appear successively.
</div>



## Stress Testing

In this chapter we introduce stress testing as a complement to the statistical methods presented in Chapters 2 and 3. The advantage of stress tests is that they are not based on statistical assumptions about the distribution of risk factor returns. Since any statistical model has inherent flaws in its assumptions, stress tests are considered a good companion to any statistically based risk measure.

Stress tests are intended to explore a range of low probability events that lie outside of the predictive capacity of any statistical model.

The estimation of the potential economic loss under hypothetical extreme changes in risk factors allows us to obtain a sense of our exposure in abnormal market conditions.


Stress tests can be done in two steps.


1. **Selection of stress events.** This is the most important and challenging step in the stress testing process. The goal is to come up with credible scenarios that expose the potential weaknesses of a portfolio under particular market conditions.


2. **Portfolio revaluation.** This consists of marking-to-market the portfolio based on the stress scenarios for the risk factors, and is identical to the portfolio revaluation step carried out under Monte Carlo and historical simulation for each particular scenario. Once the portfolio has been revalued, we can calculate the P&L as the difference between the current present value and the present value calculated under the stress scenario.


The most important part of a stress test is the selection of scenarios. Unfortunately, there is not a standard or systematic approach to generate scenarios and the process is still regarded as more of an art than a science.

Given the importance and difficulty of choosing scenarios, we present three options that facilitate the process:

- historical scenarios,
- simple scenarios, and
- predictive scenarios.



### Historical Scenarios

A simple way to develop stress scenarios is to replicate past events.


One can select a historical period spanning a financial crisis (e.g., Black Monday (1987), Tequila crisis (1995), Asian crisis (1997), Russian crisis (1998)) and use the returns of the risk factors over that period as the stress scenarios. In general, if the user selects the period from time t to time T , then following (2.1) we calculate the historical returns as

$$
r=\log \left(\frac{P_{T}}{P_{t}}\right)
$$


and calculate the P&L of the portfolio based on the calculated returns

$$\text{P&L}=V(P_0e^{r})-V( P_0)$$

<div  class="info">
The r here is the historical returns on the relevant markets between time 0 and time T.

$$$$

It is not an annualized rate! It is the historical holding period return rate.
</div>

### User-defined simple scenarios

We have seen that historical extreme events present a convenient way of producing stress scenarios. However, historical events need to be complemented with user-defined scenarios in order to span the entire range of potential stress scenarios, and possibly incorporate expert views based on current macroeconomic and financial information.


In the simple user-defined stress tests, the user changes the value of **some risk factors** by specifying either a percentage or absolute change, or by setting the risk factor to a specific value. The risk factors which are unspecified remain unchanged. Then, the portfolio is revalued using the new risk factors (some of which will remain unchanged), and the P&L is calculated as the difference between the present values of the portfolio and the revalued portfolio.


### User-defined predictive scenarios

Since market variables tend to move together, we need to take into account the correlation between risk factors in order to generate realistic stress scenarios. For example, if we were to create a scenario reflecting a sharp devaluation for an emerging markets currency, we would expect to see a snowball effect causing other currencies in the region to lose value as well.


Given the importance of including expert views on stress events and accounting for potential changes in **every risk factor**, we need to come up with user-defined scenarios for every single variable affecting the value of the portfolio. To facilitate the generation of these comprehensive user-defined scenarios, we have developed a framework in which we can express expert views by defining changes for a subset of risk factors (**core factors**), and then make predictions for the rest of the factors (**peripheral factors**) based on the user-defined variables. The predictions for changes in the peripheral factors correspond to their expected change, given the changes specified for the core factors.

What does applying this method mean? If the core factors take on the user-specified values, then the values for the peripheral risk factors will follow accordingly. Intuitively, if the user specifies that the three-month interest rate will increase by ten basis points, then the highly correlated two-year interest rate would have an increase equivalent to its average change on the days when the three-month rate went up by ten basis points.

<div  class="exampl">
For example, let us say that we have invested USD 1,000 in the Indonesian JSE equity index, and are interested in the potential scenario of a 10% IDR devaluation. Instead of explicitly specifying a return scenario for the JSE index, we would like to estimate the potential change in the index as a result of a 10% currency devaluation. In the predictive framework, we would specify the change in the JSE index as

$$\Delta JSE = \mu_{JSE} + \beta (\Delta \text{FX rate} -\mu_{IDR})$$

When FX rate = -10%

$$\Delta JSE = \mu_{JSE} + \beta (-10\% -\mu_{IDR})$$

$\beta$ denotes the beta of the equity index with respect to
the FX rate. In this example, JSE IDR $\beta$ = 0.2, so that if the IDR drops 10%, then the JSE index would decrease on average by 2%.

</div>



In the example above, it is illustrated how to predict **peripheral factors** when we have only one **core factor**. We can generalize this method to incorporate changes in multiple core factors. We define the predicted
returns of the peripheral factors as their conditional expectation given that the returns specified for the core
assets are realized. We can write the unconditional distribution of risk factor returns as

$$
\left[\begin{array}{l}
\mathbf{r}_{1} \\
\mathbf{r}_{2}
\end{array}\right] \sim N\left(\left[\begin{array}{l}
\mu_{1} \\
\mu_{2}
\end{array}\right],\left[\begin{array}{ll}
\boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\
\boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22}
\end{array}\right]\right)
$$

where $\mathbf{r}_2$ is a vector of core factor returns, $\mathbf{r}_1$ is the vector of peripheral factor returns, and the covariance matrix has been partitioned. It can be shown that the expectation of the peripheral factors $\mathbf{r}_1$  conditional on the core factors $\mathbf{r}_2$  is given by

$$
E\left[\mathbf{r}_{1} \mid \mathbf{r}_{2}\right]=\mu_{1}+\Sigma_{12} \Sigma_{22}^{-1}\left(\mathbf{r}_{2}-\mu_{2}\right)
\tag{4.7}\label{4.7}$$

Setting $\mu_1=\mu_2=0$ reduces \eqref{4.7} to

$$
E\left[\mathbf{r}_{1} \mid \mathbf{r}_{2}\right]=\Sigma_{12} \Sigma_{22}^{-1}\mathbf{r}_{2}\tag{4.8}\label{4.8}$$

where $\sum_{12}$ is the covariance matrix between core and peripheral factors, and $\sum_{22}$ is the covariance matrix of the core risk factors.

## Pricing Considerations


## Statistics


VaR is widely perceived as a useful and valuable measure of total risk that has been used for internal risk management as well as to satisfy regulatory requirements. In this chapter, we define VaR and explain its calculation using three different methodologies: closed-form parametric solution, Monte Carlo simulation, and historical simulation.

However, in order to obtain a complete picture of risk, and introduce risk measures in the decision making process, we need to use additional statistics reflecting the interaction of the different pieces (positions, desks, business units) that lead to the total risk of the portfolio, as well as potential changes in risk due to changes in the composition of the portfolio.

**Marginal and Incremental VaR are related risk measures that can shed light on the interaction of different pieces of a portfolio.** We will also explain some of the shortcomings of VaR and introduce a family of “coherent” risk measures—including Expected Shortfall—that fixes those problems. Finally, we present a section on risk statistics that measure underperformance relative to a benchmark. These relative risk statistics are of particular interest to asset managers.

Finally, we present a section on risk statistics that measure underperformance relative to a benchmark. These relative risk statistics are of particular interest to asset managers.

### Marginal VaR

The Marginal VaR of a position with respect to a portfolio can be thought of as the amount of risk that the position is adding to the portfolio. In other words, Marginal VaR tells us how the VaR of our portfolio would change if we sold a specific position.

<div  class="definition">
Marginal VaR can be formally defined as the difference between the VaR of the total portfolio and the VaR of the portfolio without the position:

$$\text{Marginal VaR for position p}= VaR(P )-
VaR(P -p)$$
</div>

According to this definition, Marginal VaR will depend on the correlation of the position with the rest of the portfolio. For example, using the parametric approach (Var is defined as m $\times $ standard deviation), we can calculate the Marginal VaR of a position p with respect to portfolio P as:


$$
\begin{aligned}
\operatorname{VaR}(P)-\operatorname{VaR}(P-p) &=\sqrt{\operatorname{VaR}^{2}(P-p)+\operatorname{VaR}^{2}(p)+2 \rho \operatorname{VaR}(P-p) \operatorname{VaR}(P)}-\operatorname{VaR}(P-p) \\
&=\operatorname{VaR}(p ) \frac{1}{\xi}\left(\sqrt{\xi^{2}+2 \rho \xi+1}-1\right)
\end{aligned}
$$

where $\rho$ is the correlation between position $\mathbf{p}$ and the rest of the portofolio $\mathbf{P}-\mathbf{p}$, and $\xi =Var(P )/Var(P-p)$.

when the VaR of the position is much smaller than the VaR of the portfolio, Marginal VaR is approximately equal to the VaR of the position times ρ. That is,

$$\text{Marginal VaR} \to Var(p)\cdot \rho \quad as \quad \xi \to 0$$

To get some intuition about Marginal VaR we will examine three extreme cases:

1. If ρ = 1, Marginal Var = Var(p).
2. If ρ = −1, Marginal Var = -Var(p)
3. if ρ = 0, Marginal Var = $VaR(p) \frac{\sqrt{1+\xi^2}-1}{\xi} $


### Definition of Incremental VaR


In the previous section we explained how Marginal VaR can be used to compute the amount of risk added by a position or a group of positions to the total risk of the portfolio.

However, we are also interested in the potential effect that buying or selling a **relatively small portion of a position** would have on the overall risk. For example, in the process of rebalancing a portfolio, *we often wish to decrease our holdings by a small amount rather than liquidate the entire position*. Since Marginal VaR can only consider the effect of selling the whole position, it would be an inappropriate measure of risk contribution for this example.

<div  class="definition">


For a position with exposure $w_i$, we define the iVaR of the position as

$$iVaR_i=\frac{d(VaR)}{dw_i} w_i \tag{7.4}\label{7.4}$$


Here VaR is the total VaR of the portfolio. It is easier to get an intuition for iVaR if we rearrange Equation 7.4 as

$$d(VaR)=\frac{dw_i}{w_i}  iVaR_i\tag{7.5}\label{7.5}$$

Notice that $w_i$ is the actual exposure in real currency (say USD), not the weight or proportion!

</div>



If we have 200 of a security, and we add 2 to the position, then dwi/wi is 2/200 = 1%. On the left-hand side of the equation, d(VaR) is just the change in the VaR of the portfolio. Equation 7.5 is really only valid for infinitely small changes in wi, but for small changes it can be used as an approximation.

<div btit="Sum of iVaR" blab="Prop2" class="proposition">
The sum of the iVaRs in a portfolio are equal to the total VaR of the portfolio.
</div>

That iVaR is additive is true no matter how we calculate VaR, but it is easiest to prove for the parametric case, where we define our portfolio’s VaR as a multiple, m, of the portfolio’s standard deviation, $\sigma_P$.

<div  class="proof">

Without loss of generality, we can divide the portfolio into two positions: first, the position for which we are calculating the iVaR with size and standard deviation $w_1$ and $\sigma_1$, and second, the rest of the portfolio with size and standard deviation $w_2$ and $\sigma_2$. If the correlation between the two parts of the portfolio is $\rho$, we have

$$\mathrm{VaR}=m \sigma_{p}=m\left(w_{1}^{2} \sigma_{1}^{2}+w_{2}^{2} \sigma_{2}^{2}+2 \rho w_{1} w_{2} \sigma_{1} \sigma_{2}\right)^{1 / 2}$$

Taking the derivative with respect to $w_1$, we have

$$\frac{d(\mathrm{VaR})}{d w_{1}}=\frac{m}{\sigma_{p}}\left(w_{1} \sigma_{1}^{2}+\rho w_{2} \sigma_{1} \sigma_{2}\right)$$

We then multiply this result by the weight of the position to get

$$iVaR_1=w_1 \frac{d(\mathrm{VaR})}{d w_{1}}=\frac{m}{\sigma_{p}}\left(w_{1}^2 \sigma_{1}^{2}+\rho w_{1} w_{2} \sigma_{1} \sigma_{2}\right)$$

Adding together the iVaRs of both parts of the portfolios, we have

$$\begin{aligned} \mathrm{iVaR}_{1}+\mathrm{iVaR}_{2} &=\frac{m}{\sigma_{p}}\left(w_{1}^{2} \sigma_{2}^{2}+w_{2}^{2} \sigma_{2}^{2}+2 \rho w_{1} w_{2} \sigma_{1} \sigma_{2}\right) \\ &=\frac{m}{\sigma_{p}} \sigma_{p}^{2} \\ &=m \sigma_{p} \\ &=\mathrm{VaR} \end{aligned}$$


</div>

### Calculation of IVaR

#### For Parametric methods

To calculate VaR using the parametric approach, we simply note that VaR is a percentile of the P&L distribution, and that percentiles of the normal distribution are always multiples of the standard deviation. Hence, we can use \eqref{Distribution for Parametric Method} to compute the T -day (1 - α)% VaR as


$$
\mathrm{VaR}=-z_{\alpha} \sqrt{T \delta^{\top} \Sigma \delta}\tag{6.4}\label{6.4}
$$

For a portfolio as
$$V=S_1 + S_2 + \ldots + S_n $$


Since the size of a position in equities (in currency terms) is equal to the delta equivalent for the position

$$\delta_i=\frac{\partial V}{\partial S_i}*S_i =1*S_i =S_i=w_i$$
we can express the VaR of the portfolio in \eqref{6.4} as

$$\mathrm{VaR}=-z_{\alpha} \sqrt{ w^T \Sigma w}$$

We can then calculate IVaR for the i-th position as

$$
\begin{aligned}
\mathrm{IVaR}_{i} &=w_{i} \frac{\partial \mathrm{VaR}}{\partial w_{i}} \\
&=w_{i}\left(-z_{\alpha} \frac{\partial \sqrt{w^{\top} \Sigma w}}{\partial w_{i}}\right) \\
&=w_{i}\left(-\frac{z_{\alpha}}{\sqrt{w^{\top} \Sigma w} \sum_{j} w_{j} \Sigma_{i j}}\right)
\end{aligned}
$$

Hence

$$IVaR_i=w_i \nabla_i$$

where

$$
\nabla=-z_{\alpha} \frac{\Sigma w}{\sqrt{w^{\top} \Sigma w}}\tag{6.13}\label{6.13}
$$


<div  class="info">


The vector $\nabla$ can be interpreted as the gradient of sensitivities of VaR with respect to the risk factors. Therefore, \eqref{6.13} has a clear interpretation as the product of the exposures of the position with respect to each risk factor ($w_i$), and the sensitivity of the VaR of the portfolio with respect to changes in each of those risk factors ($\nabla$).


</div>




#### For Simulation methods

The parametric method described above produces exact results for linear positions such as equities. However, if the positions in our portfolio are not exactly linear, we need to use simulation methods to compute an exact IVaR figure.

We might be tempted to compute IVaR as a **numerical derivative** of VaR using a predefined set of scenarios and shifting the investments on each instrument by a small amount. While this method is correct in theory, in practice the simulation error is usually too large to permit a stable estimate of IVaR.

In light of this problem, we will use a different approach to calculate IVaR using simulation methods. Our method is based on the fact that we can write IVaR in terms of a conditional expectation.

<div  class="exampl">


To gain some intuition, let us say that we have calculated VaR using Monte Carlo simulation. Table 6.1 shows a few of the position scenarios corresponding to the portfolio P&L scenarios.

$$
\text{Table 6.1: Incremental VaR as a conditional expectation}\\
\begin{array}{cccccc|c|cccc}
\text { Scenario # } & 1 & 2 & \cdots & 932 & \cdots & 950 & \cdots & 968 & \cdots & 1000 \\
\hline \text { P\&L on position 1 } & 37 & 35 & \cdots & -32 & \cdots & -31 & \cdots & 28 & \cdots & -12 \\
\text { P\&L on position 2 } & -12 & 39 & \cdots & -10 & \cdots & 31 & \cdots & 23 & \cdots & -34 \\
\vdots & \vdots & \vdots & & \vdots & & \vdots & & \vdots & & \vdots \\
\text { P\&L on position N } & 60 & -57 & \cdots & 62 & \cdots & -54 & \cdots & 53 & \cdots & -110 \\
\hline \text { Total P\&L } & 1250 & 1200 & \cdots & -850 & \cdots & -865 & \cdots & -875 & \cdots & -1100
\end{array}
$$


</div>

In our example, since we have 1,000 simulations, the 95% VaR corresponds to the 950th ordered P&L scenario (−􏰀V(950) = EUR 865). Note that VaR is the sum of the P&L for each position on the 950th scenario. Now, if we increase our holdings in one of the positions by a small amount while keeping the rest constant, the resulting portfolio P&L will still be the 950th largest scenario and hence will still correspond to VaR.

**In other words, changing the weight of one position by a small amount will not change the order of the scenarios.**

Therefore, the change in VaR given a small change of size h in position i is $VaR = hx_i$ , where $$x_i=\frac{\text{P&L of i-th position}}{\text{Exposure of i-th position}}$$ is the P&L of the i-th position in the 950th scenario divided by the exposure of i-th risk factor. Assuming that VaR is realized only in the 950th scenario we can write:

$$
\begin{aligned}
w_{i} \frac{\partial \mathrm{VaR}}{\partial w_{i}} &=\lim _{h \rightarrow 0} w_{i} \frac{h x_{i}}{h} \\
&=w_{i} x_{i}
\end{aligned} \tag{6.15}\label{6.15}
$$

We can then make a loose interpretation of Incremental VaR for a position as the position P&L in the scenario corresponding to the portfolio VaR estimate. The Incremental VaR for the first position in the portfolio would then be roughly equal to EUR 31 (its P&L on the 950th scenario).

Since VaR is in general realized in more than one scenario, we need to average over all the scenarios where the value of the portfolio is equal to VaR. We can use \eqref{6.15} and apply our intuition to derive a formula for IVaR:

$$
\mathrm{IVaR}_{i}=\mathbf{E}\left[w_{i} x_{i} \mid w^{\top} x=\mathrm{VaR}\right]
$$

In other words, IVaRi is the expected P&L of instrument i given that the total P&L of the portfolio is equal to VaR.

While this interpretation of IVaR is rather simple and convenient, there are two caveats.

- The first is that there is simulation error around the portfolio VaR estimate, and the position scenarios can be sensitive to the choice of portfolio scenario.
- The second problem arises when we have more than one position in a portfolio leading to more than one scenario that produces the same portfolio P&L.



### Expected Shortfall

Although VaR is the most widely used statistic in the marketplace, it has a few shortcomings.

The most criticized drawback is that VaR is not a sub-additive measure of risk. Subadditivity means that the risk of the sum of subportfolios is smaller than the sum of their individual risks.

Another criticism of VaR is that it does not provide an estimate for the size of losses in those scenarios where the VaR threshold is indeed exceeded.

Expected Shortfall is a subadditive risk statistic that describes how large losses are on average when they exceed the VaR level, and hence it provides further information about the tail of the P&L distribution. Mathematically, we can define Expected Shortfall as the conditional expectation of the portfolio losses given that they are greater than VaR. That is

$$
\text { Expected Shortfall }=\mathbf{E}[-\Delta V \mid-\Delta V>\text { VaR }]
$$

Expected Shortfall also has some desirable mathematical properties that VaR lacks. For example, under some technical conditions, Expected Shortfall is a convex function of portfolio weights, which makes it extremely useful in solving optimization problems when we want to minimize the risk subject to certain constraints. (See Rockafellar and Uryasev (2000).)


## Reports

The main goal of risk reports is to facilitate the clear and timely communication of risk exposures from the risk takers to senior management, shareholders, and regulators.

Risk reports must summarize the risk characteristics of a portfolio, as well as highlight risk concentrations[^1].

[^1]: For a detailed explanation of risk reporting practices see Laubsch (1999).

The objective of this chapter is to give an overview of the basic ways in which we can visualize and report the risk characteristics of a portfolio using the statistics described in last chapter.

We will show

- how to study the risk attributes of a portfolio through its distribution
- how to identify the existence of risk concentrations in specific groups of positions.
- how to investigate the effect of various risk factors on the overall risk of the portfolio.


### An overview of risk reporting

At the most aggregate level, we can depict in a histogram the entire distribution of future P&L values for our portfolio.

We can construct a histogram using any of the methods described in Chapters 2 and 3 (i.e., Monte Carlo simulation, parametric, and historical simulation).  The resulting distribution will depend on the assumptions made for each method.

Figure 7.1 shows the histograms under each method for a one sigma out-of-the-money call option on the S&P 500. Note that the parametric distribution is symmetric, while the Monte Carlo and historical distributions are skewed to the right. Moreover, the historical distribution assigns positive probability to high return scenarios not likely to be observed under the normality assumption for risk factor returns.


![-w826](/media/16017179858557/16024857017341.jpg){:width="800px"}{: .align-center}


At a lower level of aggregation, we can use any of the risk measures described in Chapter 6 to describe particular features of the P&L distribution in more detail. For example, we can calculate the 95% VaR and expected shortfall from the distributions in Figure 7.1. Table 7.1 shows the results.

$$
\begin{aligned}
&\text { Table } 7.1: 95 \% \text { VaR and Expected Shortfall }\\
&\begin{array}{lcc}
& \text { VaR } & \text { Expected Shortfall } \\
\hline \text { Parametric } & -39 \% & -49 \% \\
\text { Monte Carlo } & -34 \% & -40 \% \\
\text { Historical } & -42 \% & -53 \%
\end{array}
\end{aligned}
$$



#### How to choose between different methods

The comparison of results from different methods is useful to study the effect of our distributional assumptions, and estimate the potential magnitude of the error incurred by the use of a model. However, in practice, it is often necessary to select from the parametric, historical, and Monte Carlo methods to facilitate the flow of information and consistency of results throughout an organization.

The selection of the calculation method should depend on the **specific portfolio** and the **choice of distribution of risk factor returns**.

- If the portfolio consists mainly of linear positions and we choose to use a normal distribution of returns:
    - choose the parametric method due to its speed and accuracy under those circumstances
- If the portfolio consists mainly of non-linear positions,
    -  then we need to use either Monte Carlo or historical simulation depending on the desired distribution of returns.


The selection between the normal and empirical distributions is usually done based on practical considerations rather than through statistical tests.

The main reason to use the empirical distribution is to assign greater likelihood to large returns which have a small probability of occurrence under the normal distribution. Problems associated with empirical distribution:

- Length of Time Periods: The first problem is the difficulty of selecting the historical period used to construct the distribution.
- In addition, the scarcity of historical data makes it difficult to extend the analysis horizon beyond a few days[^3].

[^3]: These issues are discussed in \eqref{overlapping}


$$
\begin{aligned}
&\text { Table 7.2: Selecting a methodology }\\
&\begin{array}{l|cc}
\text { Distribution } & \text { Linear Portfolio  } & \text { Non-linear Portfolio } \\
\hline \text { Normal } & \text { Parametric } & \text { Monte Carlo } \\
\text { Non-normal } & \text { Historical } & \text { Historical }
\end{array}
\end{aligned}
$$


### Risk concentrations in specific groups of positions

When dealing with complex or large portfolios, we will often need finer detail in the analysis. We can use risk measures to “dissect” risk across different dimensions and identify the sources of portfolio risk.

This is useful to identify risk concentrations by business unit, asset class, country, currency, and maybe even all the way down to the trader or instrument level.

For example, we can create a VaR table, where we show the risk of every business unit across rows, and counterparties across columns.

These different choices of rows and columns are called “drilldown dimensions”.

![-w1034](/media/16017179858557/16024917818351.jpg){:width="800px"}{: .align-center}

The next section describes drilldowns in detail and explains how to calculate statistics in each of the buckets defined by a drilldown dimension.

#### Drilldowns

We refer to the categories in which you can slice the risk of a portfolio as “drilldown dimensions”.

Examples of drilldown dimensions are: position, portfolio, asset type, counterparty, currency, risk type (e.g., foreign exchange, interest rate, equity), and yield curve maturity buckets.


**Drilldowns using simulation methods**

To produce a drilldown report for any statistic, we have to simulate changes in the risk factors contained in each bucket while keeping the remaining risk factors constant.

Once we have the change in value for each scenario on each bucket, we can calculate risk statistics using the 􏰀$\Delta V$ information per bucket. In the following example, we illustrate the calculation of 􏰀$\Delta V$ per bucket for one scenario.

<div  class="exampl">

Example Drilldown by risk type and position/currency.

In this example, we will calculate the change in value for a specific scenario in a portfolio consisting of a cash position of EUR one million, 13,000 shares of IBM, and a short position consisting of a one year at-the-money call on 20,000 shares of IBM with an implied volatility of 45.65%.

The current values and the new scenario for the risk factors are:

$$
\begin{array}{lcc}
& \text { Current Values } & \text { New Scenario } \\
\hline \text { IBM } & \text { USD 120 } & \text { USD 130 } \\
\text { EUR } & \text { USD 0.88 } & \text { USD 0.80 } \\
\text { Discount Rate } & 6.0 \% & 6.5 \%
\end{array}
$$

Table 7.4 shows the original value of each instrument in the portfolio as well as their values under the new scenario. The last column shows the change in value (􏰀􏰀$\Delta V$).

$$
\begin{aligned}
&\text { Table 7.4: Portfolio valuation under a new scenario }\\
&\begin{array}{lrrr}
& \text { Original Value } & \text { New Value } & \Delta V \\
\text { Position } & \text { (USD) } & {\text { (USD) }} & {\text { (USD) }} \\
\hline \text { Cash } & 880,000 & 800,000 & -80,000 \\
\text { Equity } & 1,560,000 & 1,690,000 & 130,000 \\
\text { Option } & -493,876 & -634,472 & -140,596 \\
\text { Total } & 1,946,123 & 1,855,527 & -90,596
\end{array}
\end{aligned}
$$


We can drilldown the value changes in Table 7.4 by risk type. For example, to calculate the change in value due to equity changes, we would simply price the portfolio under a USD 130 IBM price scenario keeping the rest of the risk factors constant.

Note that the total P&L of the portfolio is made up from the changes in value due to the equity, foreign exchange, and interest rate risk factors.


We can also drilldown 􏰀$\Delta V$ by risk type and currency by selecting the risk factors that would change for each risk type/currency bucket. The risk factors that we would move to revalue the portfolio for each bucket are

$$
\begin{aligned}
&\text { Table } 7.6: \Delta V \text { drilldown by risk type and currency }\\
&\begin{array}{lcccc}
& & & \text { Foreign } & \text { Interest } \\
& \text { Total } & \text { Equity } & \text { Exchange } & \text { Rate } \\
\hline \text { USD } & -10,596 & -4,581 & & -5,227 \\
\text { EUR } & -80,000 & & -80,000 & \\
\text { Total } & -90,596 & -4,581 & -80,000 & -5,227
\end{array}
\end{aligned}
$$

</div>


**Drilldowns using parametric methods**

Drilldowns using the parametric approach are based on delta equivalents rather than scenarios, that is, to calculate a risk statistic for each bucket, we set the delta equivalents falling outside the bucket equal to zero, and proceed to calculate the statistic as usual. This procedure is best explained with an example.

<div  class="exampl">


Example 7.2 Using delta equivalents in VaR drilldowns.

$$$$

In this example, we will use parametric methods to calculate a VaR report by risk type and currency for the portfolio in Example 7.1. The risk factors for the portfolio are IBM, the EUR/USD exchange rate, and a one-year zero-coupon bond. Table 7.7 shows the delta equivalents for the portfolio by position and risk factor. The columns in Table 7.7 contain the delta equivalent vectors for each position, as well as the total for the portfolio, while the rows contain the delta equivalents with respect to the corresponding risk factor broken down by position. Note that the sum of the delta equivalent vectors of the individual positions is equal to the delta equivalent vector of the portfolio, as explained in Section 2.3.

$$
\begin{aligned}
&\text { Table 7.7: Delta equivalents for the portfolio }\\
&\begin{array}{lrrrr}
\text { Risk Factors } & \text { Total } & \text { Cash } & \text { Equity } & \text { Option } \\
\hline \text { IBM } & 22,956 & 0 & 1,560,000 & -1,537,043 \\
\text { EUR } & 880,000 & 880,000 & 0 & 0 \\
\text { 1Y Bond } & 1,043,167 & 0 & 0 & 1,043,167
\end{array}
\end{aligned}
$$

Let us assume that the covariance matrix of risk factor returns is:

$$
\Sigma=\left(\begin{array}{ccc}
92.13 & -1.90 & 0.02 \\
-1.90 & 55.80 & -0.23 \\
0.02 & -0.23 & 0.09
\end{array}\right) \times 10^{-6}
$$

From \eqref{6.4}, we can calculate the one-day 95% VaR of the portfolio as $1.64\sqrt{\delta^T \sum \delta}=USD 10,768$, where

$$\delta =\left(\begin{array}{c}
22,956 \\
880,000 \\
1,043,167
\end{array}\right) $$

Table 7.8 shows the one-day 95% VaR drilled down by risk type and currency.


$$
\begin{aligned}
&\text { Table } 7.8 \text { : VaR drilldown by risk type and currency }\\
&\begin{array}{lrrcc}
& & & \text { Foreign } & \text { Interest } \\
& \text { Total } & \text { Equity } & \text { Exchange } & \text { Rate } \\
\hline \text { USD } & 625 & 362 & & 506 \\
\text { EUR } & 10,814 & & 10,814 & \\
\text { Total } & 10,768 & 362 & 10,814 & 506
\end{array}
\end{aligned}
$$

Note that the sum of the VaRs for the USD and EUR buckets is greater than the total VaR of the portfolio due to diversification benefit (625 + 10,814 > 10,768).



$$
\begin{aligned}
&\text { Table 7.9: Incremental VaR drill down by currency and asset type }\\
&\begin{array}{lrllr}
& \text { Total } & \text { AUD } & \text { JPY } & \text { USD } \\
\hline \text { Bond } & 1,016 & & -79 & 1,095 \\
\text { Bond Option } & 7,408 & 7,408 & & \\
\text { Callable Bond } & 1,473 & & & 1,473 \\
\text { Cap } & -6,165 & & & -6,165 \\
\text { Collar } & 18 & & & 18 \\
\text { Commodity } & 1,567,757 & & & 1,567,757 \\
\text { Convertible Bond } & -29,293 & & & -29,293 \\
\text { Equity } & 490,454 & & 283,806 & 206,647 \\
\text { Equity Option } & -462 & & & -462 \\
\text { Floor } & 25 & & \\
\text { FRA } & 8,703 & & \\
\text { FRN } & 3 & & & 25 \\
\text { FX Option } & 3,712 & 2,659 & & 1,054 \\
\text { Zero Coupon Bond } & 1,959 & & & 1,959 \\
\text { Total } & 2,046,609 & 10,067 & 283,728 & 1,752,814
\end{array}
\end{aligned}
$$


Table 7.9 shows the one-day 95% incremental VaR drilled down by asset type and currency. Using this information, we can identify the positions that most contribute to the risk of the portfolio. For example, we can see that the largest contributor to risk in our portfolio are commodities, while the group of convertible bonds in our portfolio is diversifying risk away. We can also see that the risk factors denominated in JPY account for USD 283,728 of the total risk of the portfolio. Note that in this report we have combined a proper dimension (asset type) with an improper one (currency).

</div>

Up to this point, we have presented some of the most common and effective ways of presenting the risk information as well as the methods to break down the aggregate risk in different dimensions. We have also emphasized the importance of looking at risk in many different ways in order to reveal potential exposures or concentrations to groups of risk factors. In the following section, we present a case study that provides a practical application of the reporting concepts we have introduced.


### Global Bank case study

Risk reporting is one of the most important aspects of risk management. Effective risk reports help understand the nature of market risks arising from different business units, countries, positions, and risk factors in order to prevent or act effectively in crisis situations. This section presents the example of a fictitious bank, ABC, which is structured in three organizational levels: corporate level, business units, and trading desks. Figure 7.2 presents the organizational chart of ABC bank. The format and content of each risk report is designed to suit the needs of each organizational level.

![](/media/16017179858557/16031117762568.jpg){:width="800px"}{: .align-center}

**Corporate Level**
At the corporate level, senior management needs a firmwide view of risk, and they will typically focus on market risk concentrations across business units as well as global stress test scenarios.

![](/media/16017179858557/16031119461962.jpg){:width="800px"}{: .align-center}


Table 7.10 reports VaR by business unit and risk type.

-  We can see that the one-day 95% VaR is USD 2,247,902.
- Among the business units, proprietary trading has the highest VaR level (USD 1,564,894), mainly as a result of their interest rate and equity exposures.
- However, the equity exposures in proprietary trading offset exposures in global equities resulting in a low total equity risk for the bank (USD 595,424).
- Similarly, the interest rate exposures taken by the proprietary trading unit are offsetting exposures in the emerging markets, fixed income, and foreign exchange units.
- We can also observe that the foreign exchange unit has a high interest rate risk reflecting the existence of FX forwards, futures, and options in their inventory.

**Business units Level**
Business units usually need to report risk by trading desk, showing more detail than the corporate reports. For example, a report at the business unit level might contain information by trading desk and country or currency. Table 7.11 reports VaR for the Foreign Exchange unit by trading desk and instrument type. For each instrument type the risk is reported by currency.

![](/media/16017179858557/16031121107792.jpg){:width="800px"}{: .align-center}


- We can see that most of ABC’s FX risk is in cash (USD 148,102) with the single largest exposure denominated in JPY (USD 85,435).
- Also note that the FX Europe trading desk creates a concentration in EUR across cash, FX forward, and FX option instruments which accounts for most of its USD 93,456 at risk.

**Trading desk Level**

At the trading desk level, risk information is presented at the most granular level. Trading desk reports might include detailed risk information by trader, position, and drilldowns such as yield curve positioning. Table 7.12 reports the VaR of the Government Bonds desk by trader and yield curve bucket.

![-w815](/media/16017179858557/16036122759531.jpg){:width="800px"}{: .align-center}

- We can observe that Trader A is exposed only to fluctuations in the short end of the yield curve, while Trader C is well diversified across the entire term structure of interest rates.
- We can also see that trader B has a barbell exposure to interest rates in the intervals from six months to three years and fifteen years to thirty years.
- Note that the risk of the desk is diversified across the three traders.
- We can also see that the VaR of the Government Bonds desk (USD 122,522) is roughly one third of the VaR for the Fixed Income unit (USD 393,131).

### Summary

This document provides an overview of the methodology currently used by RiskMetrics in our market risk management applications. Part I discusses the mathematical assumptions of the multivariate normal model and the empirical model for the distribution of risk factor returns, as well as the parametric and simulation methods used to characterize such distributions. In addition, Part I gives a description of stress testing methods to complement the statistical models. Part II illustrates the different pricing approaches and the assumptions made in order to cover a wide set of asset types. Finally, Part III explains how to calculate risk statistics using the methods in Part I in conjunction with the pricing functions of Part II. We conclude the document by showing how to create effective risk reports based on risk statistics.

The models, assumptions, and techniques described in this document lay a solid methodological foundation for market risk measurement upon which future improvements will undoubtedly be made. We hope to have provided a coherent framework for understanding and using risk modeling techniques.
