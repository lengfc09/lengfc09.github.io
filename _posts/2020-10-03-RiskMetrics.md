---
layout: mysingle
date: 2020-10-03 18:02:16 +0800
title: Notes for "Return to RiskMetrics -- The Evolution of a Standard"
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

$$\mathcal{r}_{t,T}=(\mu -\frac{1}{2}\mathcal{\sigma}^2)(T-t)+ C^T \mathcal{z} \sqrt{T-t}$$

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

<div btit="Summary for Generating Risk Factor Scenarios for Monte-Carlo sSmmulation"  class="attention">

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

$$\Delta V = \delta ^T \mathbf{r}$$

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







