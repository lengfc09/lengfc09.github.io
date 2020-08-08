---
layout: mysingle
date: 2020-08-02 16:56:16 +0800
title: Quantitative Financial Risk Management
categories: Quantitative_Financial_Risk_Management
excerpt: "Anoter book from Michael B. Miller, which is organized around topics in risk management, such as market risk, credit risk and liquidity risk."
header:
    overlay_color: "#2f4f4f" #暗岩灰
classes: wide
tags: risk_management mathematics statistics
toc: true

---


## Market Risk: Standard Deviation

### Dollar Standard Deviation

In risk management, when we talk about the standard deviation of a security, we are almost always talking about the *standard deviation of the returns* for that security. While this is almost always the case, there will be times when we want to express standard deviation and other risk parameters in terms of dollars (or euros, yen, etc.).

In order to calculate the expected dollar standard deviation of a security, we begin by calculating the expected return standard deviation, and then multiply this value by the **current dollar value of the security** (or, in the case of futures, the nominal value of the contract). It’s that simple. If we have  200 worth of ABC stock, and the stock’s return standard deviation is 3%, then the expected dollar standard deviation is 3% × 200 = 6.

What you should not do is calculate the dollar standard deviation directly from past dollar returns or price changes. The difference is subtle, and it is easy to get confused. The reason that we want to calculate the return standard deviation first is that percentage returns are stable over time, whereas dollar returns rarely are. This may not be obvious in the short term, but consider what can happen to a security over a long period of time. Take, for example, IBM. In 1963, the average split-adjusted closing price of IBM was 2.00, compared to 127.53 in 2010.

Even though the share price of IBM grew substantially over those 47 years, the daily return standard deviation was relatively stable, 1.00% in 1963 versus 1.12% in 2010.

### Annualization

Up until now, we have not said anything about the frequency of returns. Most of the examples have made use of daily data. Daily returns are widely available for many financial assets, and daily return series often serve as the starting point for risk management models. That said, **in has become common practice in finance and risk management to present standard deviation as an annual number**.

For example, if somebody tells you that the option-implied standard deviation of a Microsoft one-month at-the-money call option is 18%—unless they specifically tell you otherwise—this will almost always mean that the annualized standard deviation is 18%.

It doesn’t matter that the option has one month to expiration, or that the model used to calculate the implied standard deviation used daily returns; the standard deviation quoted is annualized.

#### Square-root rule for I.I.D.s

If the returns of a security meet certain requirements, namely that the returns are independently and identically distributed, converting a daily standard deviation to an annual standard deviation is simply a matter of multiplying by the square root of days in the year.

For example, if we estimate the standard deviation of daily returns as 2.00%, and there are 256 business days in a year, then the expected standard deviation of annual returns is simply $32 \% = 2 \% \times \sqrt{256}$.

If we have a set of non-overlapping weekly returns, we could calculate the standard deviation of weekly returns and multiply by the square root of 52 to get the expected standard deviation of annual returns. This square-root rule only works if returns are i.i.d.

#### Non I.I.D Case

If the distribution of returns is changing over time, as with our GARCH model, or returns display serial correlation, which is not uncommon, then the standard deviation of annual returns could be higher or lower than the square-root rule would predict.

If returns are not i.i.d. and we are really interested in trying to predict the expected standard deviation of annual returns, then we should not use the square-root rule. We need to use a more sophisticated statistical model, or use actual annual returns as the basis of our calculation.

That said, in many settings, such as quoting option-implied standard deviation or describing the daily risk of portfolio returns, annualization is mostly about presentation, and making comparison of statistics easier. **In these situation, we often use the square root rule, even when returns are not i.i.d**. In these settings, doing anything else would actually cause more confusion. As always, if you are unsure of the convention, it is best to clearly explain how you arrived at your calculation.

## Market Risk: Value At Risk

### Defined as Losses or Returns

VaR was popularized by J.P. Morgan in the 1990s. The executives at J.P. Morgan wanted their risk managers to generate one statistic that summarized the risk of the firm’s entire portfolio, at the end of each day. What they came up with was VaR.

In order to formally define VaR, we begin by defining a random variable L, which represents the loss to our portfolio. L is simply the opposite of the return to our portfolio. If the return of our portfolio is −600, then the loss, L, is +600. For a given confidence level, $\gamma$, then, value at risk is defined as

$$P[L\geq VaR_{\gamma}]=1-\gamma $$

We can also define VaR directly in terms of returns. If we multiply both sides of the inequality in the preceding equation by −1, and replace −L with R, we come up with

$$P[L\leq -VaR_{\gamma}]=1-\gamma $$

While these two equations are equivalent, defining VaR in terms of losses is more common. It has the advantage that, for most portfolios for reasonable confidence levels, VaR will almost always be a positive number.

In practice, rather than saying that your VaR is 400, it is often best to resolve any ambiguity by stating that your VaR is a loss of 400.

### Advantages of VaR as a Risk Measure

One of the primary appeals of VaR is its *simplicity*. The concept of VaR is intuitive, even to those not versed in statistics. Because it boils risk down to a single number, VaR also provides us with a convenient way to track the risk of a portfolio over time.

Another appealing feature of VaR is that is focuses on *losses*. This may seem like an obvious criterion for a risk measure, but variance and standard deviation treat positive and negative deviations from the mean equally. For many risk managers, VaR also seems to strike the right balance, by focusing on losses that are significant, but not too extreme. We’ll have more to say about this at the end of the chapter, when we discuss backtesting.

VaR also allows us to *aggregate* risk across a portfolio with many different types of securities (e.g., stocks, bonds, futures, options, etc.). Prior to VaR, risk managers were often forced to evaluate different segments of a portfolio separately. For example, for the bonds in a port- folio they may have looked at the interest rate sensitivities, and for the equities they may have looked at how much exposure there was to different industries.

Finally, *VaR is robust to outliers*. As is true of the median or any quantile measure, a {: .align-center}single large event in our data set (or the absence of one) will usually not change our estimate of VaR. This advantage of VaR is a direct consequence of one of its deepest flaws, that it ignores the tail of the distribution. As we will see in the next chapter, expected shortfall, a closely related measure, has exactly the opposite problem: It incorporates the tail of the distribution, but it is not robust to outliers.

### Delta-Normal VaR

One of the simplest and easiest ways to calculate VaR is to make what are known as delta-normal assumptions. For any underlying asset, we assume that the log returns are normally distributed and we approximate the returns of any option using its *delta-adjusted exposure*. The delta-normal model includes additional assumptions when multiple securities are involved, which we will cover when we begin to look at portfolio risk measures.

To calculate the delta-normal VaR of a security, we start by calculating the standard deviation of returns for the security or, in the case of an option, for the returns of the option’s underlying security.

For regular securities, we then multiply the return standard deviation by the absolute market value or notional of our position to get the position’s standard deviation. For options we multiply by the absolute delta-adjusted exposure. The delta adjusted exposure of a single option being the underlying security’s price multiplied by the option’s delta.

For Black-Scholes model, the standard return has the following distribution:

$$\frac{ds_t}{s_t}=r dt + \sigma dw_t$$

$$dln(s_t ) =\frac{ds_t }{s_t} - \frac{1}{2}\frac{ds_t \cdot d_st }{s_t^2}$$

Since

$$E (ds_t\cdot ds_t )= \sigma^2 dt$$

we know that the log-return satisfies:

$$
\begin{aligned}
dln(s_t ) =&\frac{ds_t }{s_t} - \frac{1}{2}\frac{ds_t \cdot d_st }{s_t^2}\\
=& (r - \frac{1}{2}\sigma^2) dt + \sigma dw_t
\end{aligned}
$$


Notice that we have not said anything about the expected return. In practice, most VaR models assume that the distribution of returns has a mean of zero. At longer horizons this assumption may no longer be reasonable. Some practitioners will also assume that the theta for options is also zero. While this assumption may also be valid in many situations, it can fail even over short time horizons. In what follows, unless otherwise stated, assume security returns have zero mean but include theta in calculating VaR.

For call option, the greeks are defined as:

$$\Delta_t = \frac{dC_t}{dS_t}$$

$$\Gamma = \frac{d^2 C_t}{dS^2_t}$$

$$\Theta_t =\frac{d C_t}{dt}$$

$$Vega = \frac{dC_t}{d \sigma_t}$$

### Historical VaR

Another very simple model for estimating VaR is historical simulation or the historical method. In this approach we calculate VaR directly from past returns.

For example, suppose we want to calculate the one-day 95% VaR for an equity using 100 days of data. The 95th percentile would correspond to the least worst of the worst 5% of returns. In this case, because we are using 100 days of data, the VaR simply corresponds to the fifth worst day.

#### Choose.between Interpolation or Conservativeness

![-w600](/media/15963510819286/15963544305717.jpg){:width="600px"}{: .align-center}

Now suppose we have 256 days of data, sorted from lowest to highest as in Table 3.1. We still want to calculate the 95% VaR, but 5% of 256 is 12.8. Should we choose the 12th day? The 13th? The more conservative approach is to take the 12th point, −15.0%. Another alternative is to interpolate between the 12th and 13th points, to come up with –14.92%. Unless there is a strong justification for choosing the interpolation method, the conservative approach is recommended.

#### No Maturity Vs Finite Lifespans

For securities with no maturity date such as stocks, the historical approach is incredibly simple. For derivatives, such as equity options, or other instruments with finite lifespans, such as bonds, it is slightly more complicated.

For a derivative, we do not want to know what the actual return series was, we want to know what the return series would have been had we held exactly the same derivative in the past.

For example, suppose we own an at-the-money put with two days until expiration. Two-hundred-fifty days ago, the option would have had 252 days until expiration, and it may have been far in or out of the money. **We do not want to know what the return would have been for this option with 252 days to expiration. We want to know what the return would have been for an at-the-money put with two days to expiration, given conditions in the financial markets 250 days ago.**

What we need to do is to generate a constant maturity or backcast series. These constant maturity series, or backcast series, are quite common in finance.

The easiest way to calculate the backcast series for an option would be to use a delta approximation. If we currently hold a put with a delta of −30%, and the underlying return 250 days ago was 5%, then our backcast return for that day would be −1.5% = −30% × 5%.

A more accurate approach would be to fully reprice the option, taking into account not just changes in the underlying price, but also changes in implied volatility, the risk-free rate, the dividend yield, and time to expiration. Just as we could approximate option returns using delta, we could approximate bond returns using DV01, but a more accurate approach would be to fully reprice the bond based on changes in the relevant interest rates and credit spreads.

#### Parametric Vs Historical Approach

The delta-normal approach is an example of what we call a parametric model. This is because it is based on a mathematically defined, or parametric, distribution (in this case, the normal distribution). By contrast, the historical approach is non-parametric. We have not made any assumptions about the distribution of historical returns.

There are advantages and disadvantages to both approaches. The historical approach easily reproduces all the quirks that we see in historical data: changing standard deviation, skewness, kurtosis, jumps, etc. Developing a parametric model that reproduces all of the observed features of financial markets can be very difficult. At the same time, models based on distributions often make it easier to draw general conclusions. In the case of the historical approach, it is difficult to say if the data used for the model are unusual because the model does not define usual.

### Hybrid VaR (Historical & Decay Factor)

The historical model is easy to calculate and reproduces all the quirks that we see in historical data, but it places equal weight on all data points. If risk has increased recently, then the historical model will likely underestimate current risk. The delta-normal model can place more weight on more recent data by calculating standard deviation using a decay factor. The calculation with a decay factor is somewhat more difficult, but not much. Even with the decay factor, the delta-normal model still tends to be inaccurate because of the delta and normal assumptions.

Can we combine the advantages of the historical model with the advantages of a decay factor? The hybrid approach to VaR does exactly this. Just as we did with standard deviation in Chapter 2, we can place more weight on more recent points when calculating VaR by using a decay factor. If we were calculating standard historical VaR at the 95% confidence level, then 5% of the data points would be on one side of our VaR estimate, and 95% would be on the other side.

**With hybrid VaR it is not the number of points that matters, but their weight.** With hybrid VaR, 5% of the total weight of the data points is on one side, and 95% is on the other.

Mechanically, in order to calculate hybrid VaR, we start by assigning a weight to each data point. If our decay factor was 0.98, then we would assign a weight of 1 to the most recent date, 0.98 to the previous day, 0.982 to the day before that, and so on. Next, we divide these weights by the sum of all the weights, to get a percentage weight for each day. Then we sort the data based on returns. Finally, we determine our estimate of VaR, by adding up the percentage weights, starting with the worst return, until we get to the desired confidence level. If our confidence level was 95%, then we would stop when the sum of the percentage weights was equal to 5%.

![-w850](/media/15963510819286/15963549524100.jpg){:width="850px"}{: .align-center}

### Monte Carlo Simulation

Monte Carlo simulations are widely used throughout finance, and they can be a very powerful tool for calculating VaR.

For simple scenarios, MCS is less efficient as parametric methods like delta-normal method. The real power of Monte Carlo simulations is in more complex settings, where instruments are nonlinear, prices are path dependent, and distributions do not have well-defined inverses. Also, as we will see in subsequent chapters when we extend our VaR framework to portfolios of securities, even very simple portfolios can have very complex distributions.

Monte Carlo simulations can also be used to calculate VaR over multiple periods. In the preceding example, if instead of being interested in the one-day VaR, we wanted to know the four-day VaR, we could simply generate four one-day log returns, using the same distribution as before, and add them together to get one four-day return. We could repeat this process 1,000 times, generating a total of 4,000 one-day returns. As with the one-day example, in this particular case, there are more efficient ways to calculate the VaR statistic. That said, it is easy to imagine how multi-day scenarios could quickly become very complex. What if your policy was to reduce your position by 50% every time you suffered a loss in excess of 3%? What if returns exhibited serial correlation?

**Working with Multiple Periods:**

Monte Carlo simulations are usually based on parametric distributions, but we could also use nonparametric methods, randomly sampling from historical returns. Continuing with our gold example, if we had 500 days of returns for gold, and we wanted to calculate the four-day VaR, we could randomly pick a number from 1 to 500, and select the corresponding historical return. We would do this four times, to create one four-day return. We can repeat this process, generating as many four-day returns as we desire. The basic idea is very simple, but there are some important details to keep in mind.

First, generating multi-period returns this way involves what we call *sampling with replacement*. Pretend that the first draw from our random number generator is a 10, and we select the 10th historical return. We don’t remove that return before the next draw. If, on the second draw, our random number generator produces 10 again, then we select the same return. If we end up pulling 10 four time in a row, then our four-day return will be composed of the same 10th return repeated four times.

Even though we only have 500 returns to start out with, there are $500^4$, or 62.5 billion, possible four-day returns that we can generate this way. This method of estimating parameters, using sampling with replacement, is often referred to as **bootstrapping**.

The second detail that we need to pay attention to is **serial correlation and changes in the distribution over time.** We can only generate multi-period returns in the way just described if single-period returns are independent of each other and volatility is constant over time.

Suppose that this was not the case, and that gold tends to go through long periods of high volatility followed by long periods of low volatility.

A simple solution to this problem: Instead of generating a random number from 1 to 500, generate a random number from 1 to 497, and then select four successive returns. If our random number generator generates 125, then we create our four-day return from returns 125, 126, 127, and 128.

While this method will more accurately reflect the historical behavior of volatility, and capture any serial correlation, it greatly reduces the number of possible returns from 62.5 billion to 497, which effectively **reduces the Monte Carlo simulation to the historical simulation method.**

Another possibility is to try to **normalize the data to the current standard deviation.** If we believe the current standard deviation of gold is 10% and that the standard deviation on a certain day in the past was 5%, then we would simply multiply that return by two. While this approach gets us back to the full 62.5 billion possible four-day returns for our 500-day sample, it requires us to make a number of assumptions in order to calculate the standard deviation and normalize the data.

### Cornish-Fisher VaR

The delta-normal VaR model approximate the underlying returns with only the greek Delta. The Cornish-Fisher VaR model also incorporates gamma and theta.

To start with, we introduce some notation. Define the value of an option as V, and the value of the option’s underlying security as U. Next, define the option’s exposure-adjusted Black-Scholes-Merton Greeks as

$$\tilde{\Delta}=\frac{d V}{d U} U=\Delta U$$

$$\tilde{\Gamma}=\frac{d^{2} V}{d U^{2}} U^{2}=\Gamma U^{2}$$

$$\theta=\frac{d V}{d t}$$

Given a return on the underlying security, R, we can approximate the change in value of the option using the exposure-adjusted Greeks as

$$d V \approx \tilde{\Delta} R+\frac{1}{2} \tilde{\Gamma} R^{2}+\theta d t$$

If the returns of the underlying asset are normally distributed with a mean of zero and a standard deviation of $\sigma$, then we can calculate the moments of dV. The first three central moments and skewness of dV are

$$\begin{aligned} \mu_{d V} &=\mathrm{E}[d V]=\frac{1}{2} \tilde{\Gamma} \sigma^{2}+\theta d t \\ \sigma_{d V}^{2} &=\mathrm{E}\left[(d V-\mathrm{E}[d V])^{2}\right]=\tilde{\Delta}^{2} \sigma^{2}+\frac{1}{2} \tilde{\Gamma}^{2} \sigma^{4} \\ \mu_{3, d V} &=3 \tilde{\Delta}^{2} \tilde{\Gamma} \sigma^{4}+\tilde{\Gamma}^{3} \sigma^{6} \\ s_{d V} &=\frac{\mu_{3, d V}}{\sigma_{d V}^{3}} \end{aligned}$$

$\mu_{3, d V}$ is the third central moment, and $s_{dV}$ is the skewness. Notice that even though the distribution of the underlying security’s returns is symmetrical, the distribution of the change in value of the option is skewed ($s_{dV} \neq 0$ ).

This makes sense, given the asymmetrical nature of an option payout function. That the Cornish-Fisher model captures the asymmetry of options is an advantage over the
delta-normal model, which produces symmetrical distributions for options.

The central moments can be combined to approximate a confidence interval using a Cornish-Fisher expansion, which can in turn be used to calculate an approximation for VaR.

The Cornish-Fisher VaR (*defined as return*) of the an option is given by:

$$V a R=-\mu_{d V}-\sigma_{d V}\left[m+\frac{1}{6}\left(m^{2}-1\right) s_{d V}\right]$$

where m corresponds to the distance in standard deviations for our VaR confidence level based on a normal distribution (e.g., m = −1.64 for 95% VaR).

As we would expect, increasing the standard deviation, $\sigma_{dV}$ , will tend to increase VaR (remember m is generally negative).

As we would also expect, the more negative the skew of the distribution the higher the VaR tends to be (in practice, $m^2 − 1$ will tend to be positive).

Unfortunately, beyond this, the formula is far from intuitive, and its derivation is beyond the scope of this book. As is often the case, the easiest way to understand the approximation, may be to use it. The following sample problem provides an example.

---

**Sample Problem**

Question:
You are asked to evaluate the risk of a portfolio containing a single call option with a strike price of 110 and three months to expiration. The underlying price is 100, and the risk-free rate is 3%. The expected and implied standard deviations are both 20%. Calculate the one-day 95% VaR using both the delta-normal method and the Cornish-Fisher method. Use 365 days per year for theta and 256 days per year for standard deviation.

Answer:

To start with we need to calculate the Black-Scholes-Merton delta, gamma, and theta. These can be calculated in a spreadsheet or using other financial applications.

$$
\begin{aligned}
\Delta=& 0.2038\\
\Gamma=& 0.0283\\
\theta=&  -6.2415
\end{aligned}
$$

The one-day standard deviation and theta can be found as follows:

$$\sigma_{d}=\frac{20 \%}{\sqrt{256}}=1.25 \%$$

$$\theta_{d}=\left(\frac{-6.2415}{365}\right)=-0.0171$$

Using m = −1.64 for the 5% confidence level of the normal distribution, the delta-normal approximation is

$$-V a R=m \sigma_{d} S \Delta+\theta_{d}=-1.64 \times 1.25 \% \times 100 \times 0.2038-0.0171=-0.4361$$


For the Cornish-Fisher, approximation, we first calculate the exposure-adjusted Greeks:

$$
\begin{aligned}
\tilde{\Delta}&= \Delta U =20.3806\\
\tilde{\Gamma}&= \Gamma U^2 =283.1397
\end{aligned}
$$

Next, we calculate the mean, standard deviation, and skewness for the change in option value:

$$\begin{aligned} \mu_{d V} &=\frac{1}{2} \tilde{\Gamma} \sigma_{d}^{2}+\theta_{d}=\frac{1}{2} \times 283.1379 \times(1.25 \%)^{2}-0.0171=0.00503 \\ \sigma_{d V}^{2} &=\tilde{\Delta}^{2} \sigma_{d}^{2}+\frac{1}{2} \tilde{\Gamma}^{2} \sigma_{d}^{4}=20.3806^{2} 1.25 \%^{2}+\frac{1}{2} 283.1397^{2} 1.25 \%^{4}=0.06588 \\ \sigma_{d v} &=\sqrt{0.06588}=0.256672 \\ \mu_{3, d V} &=3 \tilde{\Delta}^{2} \tilde{\Gamma} \sigma_{d}^{4}+\tilde{\Gamma}^{3} \sigma_{d}^{6} \\ &=3 \times 20.3806^{2} \times 283.1379 \times 1.25 \%^{4}+283.1379^{3} \times 1.25 \%^{6} \\ &=0.0087 \\ s_{d V} &=\frac{\mu_{3, d V}}{\sigma_{d V}^{3}}=\frac{0.0087}{0.256672^{3}}=0.51453 \end{aligned}$$

We then plug the moments into our Cornish-Fisher approximation:

$$\begin{aligned}-V_{a} R &=\mu_{d V}+\sigma_{d V}\left[m+\frac{1}{6}\left(m^{2}-1\right) s_{d V}\right] \\ &=0.00502+0.256672\left[-1.64+\frac{1}{6}\left(-1.64^{2}-1\right) 0.514528\right] \\ &=-0.3796 \end{aligned}$$

The one-day 95% VaR for the Cornish-Fisher method is a loss of 0.3796, compared to a loss of 0.4316 for the delta-normal approximation. It turns out that in this particular case the exact answer can be found using the Black-Scholes-Merton equation. Given the assumptions of this sample problem, the actual one-day 95% VaR is 0.3759. In this sample problem, the Cornish-Fisher approximation is very close to the actual value, and provides a much better approximation than the delta-normal approach.

---

In certain instances, the Cornish-Fisher approximation can be extremely accurate. In practice, it is much more likely to be accurate if we do not go too far out into the tails of the distribution. If we try to calculate the 99.99% VaR using Cornish-Fisher, even for a simple portfolio, the result is unlikely to be as accurate as what we saw in the preceding sample problem. One reason is that our delta-gamma approximation will be more accurate for small returns. The other is that returns are much more likely to be well approximated by a normal distribution closer to the mean of a distribution. As we will see in the next section, there are other reasons why we might not want to calculate VaR too far out into the tails.

### Backtesting

**How it generally works?**

An obvious concern when using VaR is choosing the appropriate confidence interval. As mentioned, 95% has become a very popular choice in risk management. In some settings there may be a natural choice, but, most of the time, the specific value chosen for the confidence level is arbitrary.

A common mistake for newcomers is to choose a confidence level that is too high. Naturally, a higher confidence level sounds more conservative. It is tempting to believe that the risk manager using the 99.9% confidence level is concerned with more serious, riskier outcomes, and is therefore doing a better job.

The problem is that, as we go further and further out into the tail of the distribution, we become less and less certain of the shape of the distribution.

In most cases, the assumed distribution of returns for our portfolio will be based on historical data. If we have 1,000 data points, then there are 50 data points to back up our 95% confidence level, but only one to back up our 99.9% confidence level. As with any parameter, the variance of our estimate of the parameter decreases with the sample size. One data point is hardly a good sample size on which to base a parameter estimate.

A related problem has to do with backtesting. Good risk managers should regularly backtest their models. Backtesting entails checking the predicted outcome of a model against actual data. Any model parameter can be backtested.

In the case of VaR, backtesting is easy. When assessing a VaR model, each period can be viewed as a Bernoulli trial. Either we observe an exceedance or we do not. In the case of one-day 95% VaR, there is a 5% chance of an exceedance event each day, and a 95% chance that there is no exceedance. In general, for a confidence level (1 − p), the probability of observing an exceedance is p. If exceedance events are independent, then over the course of n days the distribution of exceedances will follow a binomial distribution

$$P[K=k]=C_n^k p^k (1-p)^{n-k}$$

We can reject the VaR model if the number exceedance is too high or too low.

Backtesting VaR models is extremely easy, but what would happen if we were trying to measure one-day VaR at the 99.9% confidence level. Four years is approximately 1,000 business days, so over four years we would expect to see just one exceedance. We could easily reject the model if we observed a large number of exceedances, but what if we did not observe any?

After four years, there would be a 37% probability of observing zero exceedances. Maybe our model is working fine, but maybe it is completely inaccurate and we will never see an exceedance. How long do we have to wait to reject a 99.9% VaR model if we have not seen any exceedances? Approximately 3,000 business days or 12 years.

At the 99.9% confidence level, the probability of not seeing a single exceedance after 3,000 business days is a bit less than 5%. It is still possible that the model is correct at this point, but it is unlikely.

Twelve years is too long to wait to know if a model is working or not. By contrast, for a 95% VaR model we could reject the model with the same level of confidence if we did not observe an exceedance after just three months. If our confidence level is too high, we will not be able to backtest our VaR model and we will not know if it is accurate.


The probability of a VaR exceedance should also be conditionally independent of all available information at the time the forecast is made. In other words, if we are calculating the 95% VaR for a portfolio, then the probability of an exceedance should always be 5%. The probability shouldn’t be different because today is Tuesday, because yesterday it was sunny, or because your firm has been having a good month. Importantly, the probability should not vary **because there was an exceedance the previous day or because risk levels are elevated.**

**Serial Correlation**

A common problem with VaR models in practice is that exceedances often end up being serially correlated. When exceedances are serially correlated, you are more likely to see another exceedance in the period immediately after an exceedance. {: .align-center}

To test for serial correlation in exceedances, we can look at the periods immediately {: .align-center}following any exceedance events. The number of exceedances in these periods should also follow a binomial distribution. For example, pretend we are calculating the one-day 95% VaR for a portfolio, and we observed 40 exceedances over the past 800 days. To test for serial correlation in the exceedances, we look at the 40 days immediately following the exceedance {: .align-center}events and count how many of those were also exceedances.

In other words, we count the number of back-to-back exceedances. Because we are calculating VaR at the 95% confidence level, of the 40 day-after days, we would expect that 2 of them (5% × 40 = 2) would also be exceedances. The actual number of these day-after exceedances should follow a binomial distribution with n = 40 and p = 5%.

**Independent with the current risk level**

Another common problem with VaR models in practice is that exceedances tend to be correlated with the level of risk. It may seem counterintuitive, but we should be no more or less likely to see VaR exceedances in years when market volatility is high compared to when it is low.

Positive correlation between exceedances and risk levels can happen when a model does not react quickly enough to changes in risk levels. Negative correlation can happen when model windows are too short, and the model over reacts.

To test for correlation between exceedances and the level of risk, we can divide our exceedances into two or more buckets, based on the level of risk. As an example, pretend we have been calculating the one-day 95% VaR for a portfolio over the past 800 days. We could divide the sample period in two, placing the 400 days with the highest forecasted VaR in one bucket and the 400 days with the lowest forecasted VaR in the other. We would expect each 400-day bucket to contain 20 exceedances: 5% × 400 = 20. The actual number of exceedances in each bucket should follow a binomial distribution with n = 400, and p = 5%.


## Coherent Risk Measures

At this point we have introduced two widely used measures of risk, standard deviation and value at risk (VaR). Before we introduce any more, it might be worthwhile to ask what qualities a good risk measure should have.

In 1999 Philippe Artzner and his colleagues proposed a set of axioms that they felt any logical risk measure should follow. They termed a risk measure that obeyed all of these axioms **coherent**. As we will see, while VaR has a number of attractive qualities, it is not a coherent risk measure.

A coherent risk measure is a function $\varrho$ that satisfies properties of monotonicity, sub-additivity, homogeneity, and translational invariance.
{: .notice}

### Properties of Coherent Risk Measures

Consider a random outcome ${\displaystyle X}$ viewed as an element of a linear space $\mathcal{L}$ of measurable functions, defined on an appropriate probability space. A functional  $$\varrho : \mathcal{L} → {\displaystyle \mathbb {R} \cup \{ +\infty \}}$$ is said to be coherent risk measure for $ \mathcal{L}$ if it satisfies the following properties:

**Normalized**


$$\varrho(0) = 0$$

That is, the risk of holding no assets is zero.

**Monotonicity**


$$
\mathrm{If}\; Z_1,Z_2 \in \mathcal{L} \;\mathrm{and}\; Z_1 \leq Z_2 \; \mathrm{a.s.} ,\; \mathrm{then} \; \varrho(Z_1) \geq \varrho(Z_2)
$$

**Sub-additivity**


$$
\mathrm{If}\; Z_1,Z_2 \in \mathcal{L} ,\; \mathrm{then}\; \varrho(Z_1 + Z_2) \leq \varrho(Z_1) + \varrho(Z_2)
$$

Indeed, the risk of two portfolios together cannot get any worse than adding the two risks separately: this is the diversification principle. In financial risk management, sub-additivity implies diversification is beneficial. The sub-additivity principle is sometimes also seen as problematic.

**Positive homogeneity**


$$
\mathrm{If}\; \alpha \ge 0 \; \mathrm{and} \; Z \in \mathcal{L} ,\; \mathrm{then} \; \varrho(\alpha Z) = \alpha \varrho(Z)
$$

Loosely speaking, if you double your portfolio then you double your risk. In financial risk management, positive homogeneity implies the risk of a position is proportional to its size.

**Translation invariance**

If A is a deterministic portfolio with guaranteed return a and $Z \in \mathcal{L}$ then


$$
\varrho(Z + A) = \varrho(Z) - a
$$


## Market Risk: Extreme Value Theory

For example, if you had 2,000 historical daily returns, then there would be 100 20-day periods, each with a worst day. You could then use this worst-day series to make a prediction for what the worst day will be over the next 20 days.

You could even use the distribution of worst days to construct a confidence interval. For example, if the 10th worst day in our 100 worst-day series is −5.80%, you could say that there is a 10% chance that the worst day over the next 20 days will be less than or equal to −5.80%. This is the basic idea behind extreme value theory (EVT), that extreme values (minimums and maximums) have distributions that can be used to make predictions about future events.

### Approaches to sampling historical Data for EVT

There are two basic approaches to sampling historical data for EVT.

The approach outlined above, where we divide the historical data into periods of equal length and determine the worst return in each period, is known as the **block-maxima approach**.

Another approach, known as the **peaks-over-threshold** (POT) approach, is similar, but only makes use of returns that exceed a certain threshold. The POT approach has some nice technical features, which has made it more popular in recent years, but the block-maxima approach is easier to understand. For this reason, we will focus primarily on the block-maxima approach.

Figure 4.2 shows the distribution of minimum daily returns generated in two Monte Carlo simulations. In each case, 40,000 returns were generated and divided into 20-day periods, producing 2,000 minimum returns. In the first simulation, the daily returns were generated by a normal distribution with a mean of 0.00% and a standard deviation of 1.00%. In the second simulation, daily returns were generated using a fat-tailed Student’s t-distribution, with the same mean and standard deviation. The median of both distributions is very similar, close to {: .align-center}−1.80%, but, as we might expect, the fat-tailed t-distribution generates more extreme minimums. For example, the normal distribution generated only one minimum less than −4.00%, while the t-distribution generated 82.

![-w853](/media/15963510819286/15963576502328.jpg){:width="850px"}{: .align-center}

While the two distributions in Figure 4.2 are not exactly the same, they are similar in some ways. **One of the most powerful results of EVT has to do with the shape of the distribution of extreme values for a random variable**.

Provided certain conditions are met, the distribution of extreme values will always follow one of three continuous distributions, either the Fréchet, Gumbel, or Weibull distribution. These three distributions can be considered special cases of a more general distribution, the generalized extreme value distribution.

That the distribution of extreme values will follow one of these three distributions is true for most parametric distributions. One note of caution: **This result will not be true if the distribution of returns is changing over time**. The distribution of returns for many financial variables is far from constant, making this an important consideration when using EVT.

Table 4.1 shows the formulas for the probability density function (PDF) and cumulative distribution function (CDF) for each of the three EVT distributions, as well as for the generalized Pareto distribution, which we will come to shortly. Here, s is a scale parameter, k is generally referred to as a shape parameter, and m is a location parameter. The location parameter, m, is often assumed to be zero.

![-w870](/media/15963510819286/15963578412034.jpg){:width="850px"}{: .align-center}

Figures 4.3, 4.4, and 4.5 show examples of the probability density function for each distribution. As specified here these are distributions for maximums, and m is the minimum for the distributions. If we want to model the distribution of minimums, we can either alter the formulas for the distributions, or simply reverse the signs of all of our data. The later approach is consistent with how we defined VaR and expected shortfall, working with losses instead of returns. This is the approach we will adopt for the remainder of this chapter.

![-w819](/media/15963510819286/15963579072331.jpg){:width="850px"}{: .align-center}

![-w837](/media/15963510819286/15963579272923.jpg){:width="850px"}{: .align-center}

![-w829](/media/15963510819286/15963579377492.jpg){:width="850px"}{: .align-center}

![-w829](/media/15963510819286/15963579401203.jpg){:width="850px"}{: .align-center}


If the underlying data-generating process is fat-tailed, then the distribution of the maximums will follow a Fréchet distribution. This makes the Fréchet distribution a popular choice in many financial risk management applications.

### Determination of the parameters

How do we determine which distribution to use and the values of the parameters? For block-maxima data, the most straightforward method is maximum likelihood estimation.

As an example, in Figure 4.6, 40,000 returns were generated using a Student’s t-distribution with a mean of 0.00% and a standard deviation of 1.00%. These were divided into 2,000 non-overlapping 20-day periods, just as in the previous example. The actual distribution in the figure shows the distribution of the 2,000 minimum points, with their signs reversed. (The distribution looks less smooth than in the previous figure because finer buckets were used.) On top of the actual distribution, we show the best-fitting Fréchet and Weibull distributions. As expected, the Fréchet distribution provides the best fit overall.

![-w827](/media/15963510819286/15963580433407.jpg){:width="850px"}{: .align-center}

One problem with the block-maxima approach, as specified, is that all of the EVT distributions have a defined minimum. In Figure 4.6, both EVT distributions have a minimum of 0.00%, implying that there is no possibility of a maximum below 0.00% (or, reversing the sign, a minimum above 0.00%). This is a problem because in theory, the minimum in any month could be positive. In practice, the probability is very low, but it can happen.

The POT approach avoids this problem at the outset by considering only the distribution of extremes beyond a certain threshold. This and some other technical features make the POT approach appealing. One drawback of the POT approach is that the parameters of the EVT distribution are usually determined using a relatively complex approach that relies on the fact that, under certain conditions, the EVT distributions will converge to a Generalized Pareto distribution.

---
**Sample Problem**

Question:
Based on the 2,000 maxima generated from the t-distribution, as described earlier, you have decided to model the maximum loss over the next 20 days using a Fréchet distribution with m = 0.000, s = 0.015, and k = 2.368. What is the probability that the maximum loss will be greater than 7.00%?

Answer:
We can use the CDF of the Fréchet distribution from Table 4.1 to solve this problem.

$$
P[L>0.07]=e^{-\left(\frac{x-m}{s}\right)^{-k}}=e^{-\left(\frac{0.07-0.00}{0.015}\right)^{-2.368}}=0.9743
$$


That is, given our distribution assumption, 97.43% of the distribution is less than 7.00%, meaning 2.57% of the distribution is greater or equal 7.00%. The probability that the maximum loss over the next 20 days will be greater than 7.00% is 2.57%.

---

Be careful how you interpret the EVT results. In the preceding example, there is a 2.57% chance that the maximum loss over the next 20 days will exceed 7%. It is tempting to believe that there is only a 2.57% chance that a loss in excess of 7% will occur tomorrow.

This is not the case. EVT is giving us a conditional probability, $P[L > 7\% \mid max]$, meaning the probability that the loss is greater than 7%, given that tomorrow is a maximum. This is not the same as the unconditional probability, P[L > 7%]. These two concepts are related, though. Mathematically,

$$\mathrm{P}[L>7 \%]=\mathrm{P}[L>7 \% \mid \max ] \mathrm{P}[\max ]+\mathrm{P}[L>7 \% \mid \max ] \mathrm{P}[\overline{\mathrm{max}}]$$

where we have used max to denote L not being the maximum. In our current example we are using a 20-day window, so the probability that any given day is a maximum is simply 1/20. Using this and the EVT probability, we have

$$\mathrm{P}[L>7 \%]=2.57 \% \frac{1}{20}+\mathrm{P}[L>7 \% \mid \max ] \frac{19}{20}$$

The second conditional probability, $P[L > 7\% \mid max]$, must be less than or equal to the first conditional probability. This makes Equation 4.5 a weighted average of 2.57% and some- thing less than or equal to 2.57%, so we know the unconditional probability must be less than or equal to 2.57%,

$$\mathrm{P}[L>7 \%]\leq 2.57\%$$

### Combination of EVT and Parametric model

A popular approach at this point is to assume a parametric distribution for the non-extreme events. This combined approach can be viewed as describing a mixture distribution. For example, we might use a normal distribution to model the non-extreme values and a Fréchet distribution to model the extreme values.

Continuing with our example, suppose we use a normal distribution and determine that the probability that L is greater than 7%, when L is not the maximum, is 0.05%. The unconditional probability would then be 0.17%,

$$\mathrm{P}[L>7 \%]=2.57 \% \frac{1}{20}+0.05 \% \frac{19}{20}=0.13 \%+0.04 \%=0.17 \%$$


If the parametric distribution does a good job at modeling the non-extreme values and the EVT distribution does a good job at modeling the extreme values, then we may be able to accurately forecast this unconditional probability using this combination approach. This combined approach also allows us to use EVT to forecast VaR and expected shortfall, something we cannot do with EVT alone.

There is a certain logic to EVT. If we are primarily concerned with extreme events, then it makes sense for us to focus on the shape of the distribution of extreme events, rather than trying to approximate the distribution for all returns. But, in the past some proponents of EVT have gone a step further and claimed that this focus allows them to be very certain about extremely rare events.

Once we have determined the parameters of our extreme value distribution, it is very easy to calculate the probability of losses at any confidence level. It is not uncommon to see EVT used as the basis of one-day 99.9% or even 99.99% VaR calculations, for example.

Recall the discussion on VaR back-testing, though. Unless you have a lot of data and are sure that the data generating process is stable, you are unlikely to be able to make these kinds of claims. This is a fundamental limit in statistics. There is no way to get around it.

The remarkable thing about EVT is that we can determine the distribution of the minimum or maximum even if we don’t know the distribution of the underlying data-generating process. But knowing the type of distribution is very different from knowing the distribution itself. It’s as if we knew that the distribution of returns for a security followed a lognormal distribution, but didn’t know the mean or standard deviation. If we know that extreme losses for a security follow a Fréchet distribution, but are highly uncertain about the parameters of the distribution, then we really don’t know much.

One requirement of EVT that is unlikely to be met in most financial settings is that the underlying data-generation process is constant over time. There are ways to work around this assumption, but this typically leads to more assumptions and more complication.

Though we did not mention it before, the EVT results are only strictly true in the limit, as the frequency of the extreme values approaches zero. In theory, the EVT distributions should describe the distribution of the maxima reasonably well as long as the frequency of the extreme values is sufficiently low. Unfortunately, defining “sufficiently low” is not easy, and EVT can perform poorly in practice for **frequencies as low as 5%.**

