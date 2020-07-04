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
