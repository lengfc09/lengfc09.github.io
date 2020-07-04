---
layout: mysingle
date: 2020-07-04 11:47:16 +0800
title: Mathematics and Statistics for Financial Risk Management by Michael B. Miller
categories: Quantitative_Financial_Risk_Management
excerpt: "Notes for the book by Michael B. Miller. It includes basic concepts in mathemtacs and statistics which are commonly used in the risk mangement process."
header:
    overlay_color: "#036" #午夜蓝
classes: wide
tags: risk_management mathematics statistics

toc: true
---

# Mathematics and Statistics for Financial Risk Management by Michael B. Miller

## Chapter 1: Some Basic Math

### Log return and standard return

One of the most common applications of logarithms in finance is computing log returns. Log returns are defined as follows:

$$\begin{equation}
r_{t} \equiv \ln \left(1+R_{t}\right) \quad \text { where } \quad R_{t}=\frac{P_{t}-P_{t-1}}{P_{t-1}}
\end{equation}$$

Alternatively:

$$ e^{r_t}= 1+R_{t}=\frac{P_{t}}{P_{t-1}}$$


To get a more precise estimate of the relationship between standard returns and log returns, we can use the following approximation:

$$r\approx R-\frac{1}{2}R^2$$

### Compounding

To get the return of a security for two periods using simple returns, we have to do something that is not very intuitive, namely adding one to each of the returns, multiplying, and then subtracting one:

$$\begin{equation}
R_{2, t}=\frac{P_{t}-P_{t-2}}{P_{t-2}}=\left(1+R_{1, t}\right)\left(1+R_{1, t-1}\right)-1
\end{equation}$$

and for the log return:

$$r_{2,t}=r_{1,t}+r_{1,t-1}$$

### Log return and log price

Define $p_t$ as the log of price $P_t$, then we have:

$$r_t=ln(P_t/P_{t-1})=p_t-p_{t-1}$$

As a result, we can check the log(price)-Time plot, to see whether the return is increasing:

![-w600](media/15938358383413/15938365361053.jpg)

According to the graph above, the log return is constant, even though the price is increasing faster than a linear speed.

## Chapter 2: Basic Statistics

### Population and Sample Data

The state of absolute certainty is, unfortunately, quite rare in finance. More often, we are faced with a situation such as this: estimate the mean return of stock ABC, given the most recent year of daily returns. In a situation like this, we assume there is some underlying data-generating process, whose statistical properties are constant over time.

We can only estimate the true mean based on our limited data sample. In our example, assuming n returns, we estimate the mean using the same formula as before:

$$\begin{equation}
\hat{\mu}=\frac{1}{n} \sum_{i=1}^{n} r_{i}=\frac{1}{n}\left(r_{1}+r_{2}+\cdots+r_{n-1}+r_{n}\right)
\end{equation}$$

where $\hat{\mu}$ (pronounced “mu hat”) is our estimate of the true mean, μ, based on our sample of n returns. We call this the sample mean.

### Variance and Standard Deviation

The variance of a random variable measures how noisy or unpredictable that random variable is. Variance is defined as the expected value of the difference between the variable and its mean squared:

$$\sigma^2=E[(X-\mu)^2 ]$$

The square root of variance, typically denoted by σ , is called standard deviation. In finance we often refer to standard deviation as volatility. This is analogous to referring to the mean as the average. Standard deviation is a mathematically precise term, whereas volatility is a more general concept.

In the previous example, we were calculating the population variance and standard deviation. **All of the possible outcomes for the derivative were known**.

To calculate the sample variance of a random variable X based on n observations, $x_1,x_2,...,x_n$ and the sample mean $\hat{\mu}$, we can use the following formula:

$$\begin{equation}
\begin{array}{l}
\hat{\sigma}_{x}^{2}=\frac{1}{n-1} \sum_{i=1}^{n}\left(x_{i}-\hat{\mu}_{x}\right) \\
E\left[\hat{\sigma}_{x}^{2}\right]=\sigma_{x}^{2}
\end{array}
\end{equation}$$

We divide the sum of squared errors by n-1 to arrive at a unbiased estimator of true variance. If the mean is known or we are calculating the population variance, then we divide by n. If instead the mean is also being estimated, then we divide by n − 1.


It can easily be rearranged as follows:

$$
\sigma^{2}=E\left[X^{2}\right]-\mu^{2}=E\left[X^{2}\right]-E[X]^{2}
$$

### Covariance and Correlation

Covariance is analogous to variance, but instead of looking at the deviation from the mean of one variable, we are going to look at the relationship between the deviations of two variables:

$$
\sigma_{X Y}=E\left[\left(X-\mu_{X}\right)\left(Y-\mu_{Y}\right)\right]
$$

Just as we rewrote our variance equation earlier, we can rewrite the previous equation as follows:

$$
\sigma_{X Y}=E\left[\left(X-\mu_{X}\right)\left(Y-\mu_{Y}\right)\right]=E[X Y]-\mu_{X} \mu_{Y}=E[X Y]-E[X] E[Y]
$$

In order to calculate the covariance between two random variables, X and Y, **assuming the means of both variables are known**, we can use the following formula:

$$
\hat{\sigma}_{X, Y}=\frac{1}{n} \sum_{i=1}^{n}\left(x_{i}-\mu_{X}\right)\left(y_{i}-\mu_{Y}\right)
$$

If the means are unknown and must also be estimated, we replace n with (n − 1):

$$
\hat{\sigma}_{X, Y}=\frac{1}{n-1} \sum_{i=1}^{n}\left(x_{i}-\hat{\mu_{X}}\right)\left(y_{i}-\hat{\mu_{Y}}\right)
$$

To get rid of the the impact of each variable's deviation on the covariance, we can use the correlation:

$$\rho_{XY}=\frac{\sigma_{XY}}{\sigma_X \sigma_Y} $$

If two variables are highly correlated, it is often the case that one variable causes the other variable, or that both variables share a common underlying driver.

However, **Correlation does not prove causation**. Similarly, if two variables are uncorrelated, it does not necessarily follow that they are unrelated.

For example, assume X is a random variable, and Y = $X^2$. X has an equal probability of being −1, 0, or +1. The correlation between X and Y is 0.

### Application: Portfolio Variance and Hedging

For example, if we have two securities with random returns $X_A$ and $X_B$, with means $\mu_A$ and $\mu_B$ and standard deviations $\sigma_A$ and $\sigma_B$, respectively, we can calculate the variance of $X_A+X_B$ as follows:

$$\sigma^2_{A+B}=\sigma^2_{A}+\sigma^2_{B}+2\rho_{AB}\sigma_A \sigma_B$$

Since we know that the $\rho \in [-1,1]$, the portfolio has a minimum of variance if $\rho = -1$.

If the variance of both securities is equal, then Equation simplifies to:

$$\sigma^2_{A+B}=2 \sigma^2 (1+\rho_{AB})$$

In the special case where the correlation between the two securities is zero, we can further simplify our equation. For the standard deviation

$$\rho_{AB}=0 \Longrightarrow \sigma_{A+B}=\sqrt{2} \sigma$$

 If Y is a linear combination of $X_A$ and $X_B$, such that:

 $$Y=aX_A+bX_B$$

 then, using our standard notation, we have:

 $$\sigma^2_Y=a^2 \sigma^2_A +b^2 \sigma^2_B + 2ab \rho_{AB}\sigma_A \sigma_B $$

Correlation is central to the problem of hedging. Using the same notation as before, imagine we have **1 dollar** of Security A, and we wish to hedge it with **h dollar** of Security B.

$$
\begin{aligned}
P &=X_{A}+h X_{B} \\
\sigma_{P}^{2} &=\sigma_{A}^{2}+h^{2} \sigma_{B}^{2}+2 h \rho_{A B} \sigma_{A} \sigma_{B}
\end{aligned}
$$

As a risk manager, we might be interested to know what hedge ratio would achieve the portfolio with the least variance:

$$
\begin{aligned}
\frac{d \sigma_{P}^{2}}{d h} &=2 h \sigma_{B}^{2}+2 \rho_{A B} \sigma_{A} \sigma_{B} \\
h^{*} &=-\rho_{A B} \frac{\sigma_{A}}{\sigma_{B}}
\end{aligned}
$$

Substituting $h^*$ back into our original equation, we see that the smallest variance we can achieve is:

$$
\min \left[\sigma_{P}^{2}\right]=\sigma_{A}^{2}\left(1-\rho_{A B}^{2}\right)
$$

This risk that we cannot hedge is referred to as idiosyncratic risk.

### Moments

Previously, we defined the mean of a variable X as:

$$\mu = E[X]$$

It turns out that we can generalize this concept as follows:

$$m_k= E[X^k]$$

We refer to $m_k$ as the k-th moment of X. The mean of X is also the first moment of X.

Similarly, we can generalize the concept of variance as follows:

$$\mu_k = E[(X-\mu)^k]$$

We refer to $μ_k$ as the kth **central moment** of X.

While we can easily calculate any central moment, in risk management it is very rare that we are interested in anything beyond the fourth central moment.

#### Skeness

