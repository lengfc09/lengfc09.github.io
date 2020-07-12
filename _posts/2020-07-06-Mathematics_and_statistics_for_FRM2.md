---
layout: mysingle
date: 2020-07-01 22:02:16 +0800
title: Mathematics and Statistics for Financial Risk Management by Michael B. Miller --- Part 2
categories: Quantitative_Financial_Risk_Management
excerpt: "Notes for the book by Michael B. Miller. It includes basic concepts in mathemtacs and statistics which are commonly used in the risk mangement process."
header:
    overlay_color: "#036" #午夜蓝
classes: wide
tags: risk_management mathematics statistics

toc: true
---

## Chapter 4: Bayesian Analysis

Bayes’ theorem is often described as a procedure for **updating beliefs about the world when presented with new information.**

Say our prior belief for the probability of event B is $P(B)$. After we observe another event A which has happened, e.g. **new information**, we want to update our estimation of the probability of event B.

$$
P(B\|A)=\frac{P(A\|B)* P(B)}{P(A)}
$$


1. The conditional probability $$P(A\|B)$$ is called the likelihood or evidence.
2. $$P(B)$$ is the prior belief.
3. $$P(B\|A)$$ is the posterior belief.

### Bayes versus Frequentists

Pretend that as an analyst you are given daily profit data for a fund, and that the fund has had positive returns for 560 of the past 1,000 trading days. What is the probability that the fund will generate a positive return tomorrow? Without any further instructions, it is tempting to say that the probability is 56%, (560/1,000 = 56%).

In the previous sample problem, though, we were presented with a portfolio manager who beat the market three years in a row. Shouldn’t we have concluded that the probability that the portfolio manager would beat the market the following year was 100% (3/3 = 100%), and not 60%?

**The frequentist approach** based only on the observed frequency of positive results.

**The Bayesian approach** also counts the number of positive results. The conclusion is different because the Bayesian approach **starts with a prior belief about the probability**.


**Which approach is better?**

Situations in which there is very little data, or in which the signal-to-noise ratio is extremely low, often lend themselves to Bayesian analysis.

When we have lots of data, the conclusions of frequentist analysis and Bayesian analysis are often very similar, and the frequentist results are often easier to calculate.

### Continuous Distributions

$$
f(A \mid B)=\frac{g(B \mid A) f(A)}{\int_{-\infty}^{+\infty} g(B \mid A) f(A) d A}
$$

Here f(A) is the prior probability density function, and f(A\|B) is the posterior PDF. g(B\|A) is the likelihood.


### Conjugate Distribution

It is not always the case that the prior and posterior distributions are of the same type.

When both the prior and posterior distributions are of the same type, we say that they are **conjugate distributions**. As we just saw, the beta distribution is the conjugate distribution for the binomial likelihood distribution.

Here is another useful set of conjugates: The normal distribution is the conjugate distribution for the normal likelihood when we are trying to estimate the distribution of the mean, and the variance is known.

The real world is not obligated to make our statistical calculations easy. In practice, prior and posterior distributions may be nonparametric and require numerical methods to solve. While all of this makes Bayesian analysis involving continuous distributions more complex, these are problems that are easily solved by computers. One reason for the increasing popularity of Bayesian analysis has to do with the rapidly increasing power of computers in recent decades.


### Bayesian Networks

A Bayesian network illustrates the causal relationship between different random variables.

Exhibit 6.4 shows a Bayesian network with two nodes that represent the economy go up, E, and a stock go up, S.

Exhibit 6.4 also shows three probabilities: the probability that E is up, $P[E]$; the probability that S is up given that E is up, $P[S \| E]$; and the probability that S is up given that E is not up, $P[S \| E]$.


![-w600](/media/15940441031797/15944709528677.jpg){:width="600px"}
Using Bayes’ theorem, we can also calculate $P[E \| S]$. This is the probability that E is up given that we have observed S being up:

$$
P[E \mid S]=\frac{P[S \mid E] P[E]}{P[S]}=\frac{P[S \mid E] P[E]}{P[S \mid E] P[E]+P[S \mid \bar{E}] P[\bar{E}]}
$$

**Causal reasoning**, $P[S \| E]$, follows the cause-and-effect arrow of our Bayesian network. **Diagnostic reasoning**, $P[E \| S]$, works in reverse.

For most people, causal reasoning is much more intuitive than diagnostic reasoning. Diagnostic reasoning is one reason why people often find Bayesian logic to be confusing. Bayesian networks do not eliminate this problem, but they do implicitly model cause and effect, allowing us to differentiate easily between causal and diagnostic relationships.

Bayesian networks are extremely flexible. Exhibit 6.5 shows a network with seven nodes. Nodes can have multiple inputs and multiple outputs. For example, node B influences both nodes D and E, and node F is influenced by both nodes C and D.

![-w600](/media/15940441031797/15944711947804.jpg){:width="600px"}


In a network with n nodes, where each node can be in one of two states (for example, up or down), there are a total of $2^n$ possible states for the network. As we will see, an advantage of Bayesian networks is that we will **rarely have to** specify $2^n$ probabilities in order to define the network.

For example, in Exhibit 6.4 with two nodes, there are four possible states for the network, but we only had to define three probabilities.


### Bayesian networks versus Correlation matrices

![-w800](/media/15940441031797/15944712994630.jpg){:width="800px"}

Exhibit 6.6 shows two networks, each with three nodes. In each network E is the economy and S1 and S2 are two stocks. In the first network, on the left, S1 and S2 are directly influenced by the economy. In the second network, on the right, S1 is still directly influenced by the economy, but S2 is only indirectly influenced by the economy, being directly influenced only by S1.

**Exhibit 6.7 Probabilities of Networks**

$$
\begin{array}{ccccc}
\hline E & S 1 & S 2 & \text { Network 1 } & \text { Network 2 } \\
\hline 0 & 0 & 0 & 19.20 \% & 20.80 \% \\
0 & 0 & 1 & 12.80 \% & 11.20 \% \\
0 & 1 & 0 & 4.80 \% & 4.00 \% \\
0 & 1 & 1 & 3.20 \% & 4.00 \% \\
1 & 0 & 0 & 7.20 \% & 11.70 \% \\
1 & 0 & 1 & 10.80 \% & 6.30 \% \\
1 & 1 & 0 & 16.80 \% & 21.00 \% \\
1 & 1 & 1 & \frac{25.20 \%}{100.00 \%} & \frac{21.00 \%}{100.00 \%} \\
\hline
\end{array}
$$

We can calculate the covariance matrix for these two networks:

![-w800](/media/15940441031797/15944717041039.jpg){:width="800px"}

**Advantage 1: Fewer Parameters**

One advantage of Bayesian networks is that **they can be specified with very few parameters**, relative to other approaches. In the preceding example, we were able to specify each network using only five probabilities, **but each covariance matrix contains six nontrivial entries**, and the joint probability table, Exhibit 6.7, contains eight entries for each network. As networks grow in size, this advantage tends to become even more dramatic.

**Advantage 2: More Intuitive**

Another advantage of Bayesian networks is that **they are more intuitive**. It is hard to have much intuition for entries in a covariance matrix or a joint probability table.

An equity analyst covering the two companies represented by S1 and S2 might be able to look at the Bayesian networks and say that the linkages and probabilities seem reasonable, but the analyst is unlikely to be able to say the same about the two covariance matrices.

Because Bayesian networks are more intuitive, they might be easier to update in the face of a structural change or regime change. In the second network, where we have described S2 as being a supplier to S1, suppose that S2 announces that it has signed a contract to supply another large firm, thereby making it less reliant on S1? With the help of our equity analyst, we might be able to update the Bayesian network im/mediately (for example, by decreasing the probabilities $P[S2 \| S1]$ and $P[S2 \| S1]$), but it is not as obvious how we would directly update the covariance matrices.




## Chapter 5: Hypothesis testing and Confidence Intervals

In this chapter we explore two closely related topics, confidence intervals and hypothesis testing. At the end of the chapter, we explore applications, including value at risk (VaR).

### Confidence Intervals

For the sample mean, we have $$t=\frac{\hat{\mu}-\mu}{\hat{\sigma} / \sqrt{n}}$$ follows a Student’s t distribution with (n − 1) degrees of freedom.

By looking up the appropriate values for the t distribution, we can establish the probability that our t-statistic is contained within a certain range:

$$
P\left[x_{L} \leq \frac{\hat{\mu}-\mu}{\hat{\sigma} / \sqrt{n}} \leq x_{U}\right]=\gamma
$$

where $x_L$ and $x_U$ are constants, which, respectively, define the lower and upper bounds of the range within the t distribution, and $\gamma$ is the probability that our t-statistic will be found within that range.

Typically $\gamma$ is referred to as the **confidence level**.

Rather than working directly with the confidence level, we often work with the quantity $1-\gamma$ , which is known as **the significance level** and is often denoted by $\alpha$.

In practice, the population mean, $\mu$, is often unknown. By rearranging the previous equation we come to an equation with a more interesting form:

$$
P\left[\hat{\mu}-\frac{x_{L} \hat{\sigma}}{\sqrt{n}} \leq \mu \leq \hat{\mu}+\frac{x_{U} \hat{\sigma}}{\sqrt{n}}\right]=\gamma
$$

Looked at this way, we are now giving the probability that the population mean will be contained within the defined range.


### Hypothesis testing


One problem with confidence intervals is that they require us to settle on an arbitrary confidence level.

Rather than saying there is an x% probability that the population mean is contained within a given interval, we may want to know what the probability is that the population mean is greater than y.

Traditionally the question is put in the form of a null hypothesis. If we are interested in knowing whether the expected return of a portfolio manager is greater than 10%, we would write:

$$H_0: u_r >10 \%  $$

where $H_0$ is known as the null hypothesis.

With our null hypothesis in hand, we gather our data, calculate the sample mean, and form the appropriate t-statistic. In this case, the appropriate t-statistic is:

$$
t=\frac{\hat{\mu}-10 \%}{\hat{\sigma} / \sqrt{n}}
$$

We can then look up the corresponding probability from the t distribution.

In addition to the null hypothesis, we can offer an alternative hypothesis. In the previous example:

$$H_1: u_r \leq 10 \%  $$

#### One tail or two?

In many scientific fields where positive and negative deviations are equally important, two-tailed confidence levels are more prevalent. In risk management, more often than not we are more concerned with the probability of extreme negative outcomes, and this concern naturally leads to one-tailed tests.

A two-tailed null hypothesis could take the form:

$$
H_0: u_r = 0 \\
H_1: u_r \neq 0 \%  $$


A one-tailed test could be of the form:

$$
H_0: u_r > c  \\
H_1: u_r \leq c  $$

#### Chebyshev’s Inequality

An easy way to estimate the probability of **outliers**:

$$
P[|x-\mu |\geq n\sigma] \leq \frac{1}{n^2}
$$

Chebyshev’s inequality tells us that the probability of being greater than two standard deviations from the mean is less than or equal to **25%**. The exact probability for a standard normal variable is closer to **5%**, which is indeed less than 25%.

If a variable is normally distributed, the probability of a three standard deviation event is very small, 0.27%. If we assume normality, we will assume **that three standard deviation events are very rare**. For other distributions, though, Chebyshev’s inequality tells us that the probability could be as high as 1∕9, or approximately 11%.

### Value at Risk

alue at risk (VaR) is one of the most widely used risk measures in finance. VaR was popularized by J.P. Morgan in the 1990s.

In order to formally define VaR, we begin by defining a random variable L, which represents the loss to our portfolio. L is simply the negative of the return to our portfolio. If the return of our portfolio is −600, then the loss, L, is +600. For a given confidence level, $\gamma$, then, we can define value at risk as Equation 7.15:

$$P[L\geq VaR_{\gamma}]=1-\gamma $$

We can also define VaR directly in terms of returns. If we multiply both sides of the inequality in Equation 7.15 by −1, and replace −L with R, we come up with Equation 7.16:

$$P[R \leq -VaR_{\gamma}]=1-\gamma $$


While Equations 7.15 and 7.16 are equivalent, you should know that some risk managers go one step further and drop the negative sign from Equation 7.16. What we have described as a VaR of 400 they would describe as a VaR of −400.

The convention we have described is more popular. It has the advantage that for reasonable confidence levels for most portfolios, VaR will almost always be a positive number.

In practice, rather than just saying that your VaR is 400, it is often best to resolve any ambiguity by stating that your VaR is a loss of 400 or that your VaR is a return of −400.

#### Backtesting

In the case of VaR, backtesting is easy. When assessing a VaR model, each period can be viewed as a Bernoulli trial. In the case of one-day 95% VaR, there is a 5% chance of an exceedance event each day, and a 95% chance that there is no exceedance.

Because exceedance events are independent, over the course of n days the distribution of exceedances follows a binomial distribution:

$$
P[K=k]=\left(\begin{array}{l}
n \\
k
\end{array}\right) p^{k}(1-p)^{n-k}
$$

The probability of a VaR exceedance should be conditionally independent of all available information at the time the forecast is made. Importantly, the probability should not vary because there was an exceedance the previous day, or because risk levels are elevated.

#### 99.9% VS 95%

It is tempting to believe that the risk manager using the 99.9% confidence level is concerned with more serious, riskier outcomes, and is therefore doing a better job.

**The problem is that, as we go further and further out into the tail of the distribution, we become less and less certain of the shape of the distribution.** In most cases, the assumed distribution of returns for our portfolio will be based on historical data. If we have 1,000 data points, then there are 50 data points to back up our 95% confidence level, but **only one to back up our 99.9% confidence level**. As with any distribution parameter, the variance of our estimate of the parameter decreases with the sample size. One data point is hardly a good sample size on which to base a parameter estimate.

We should also follow this rule in our backtesting.

#### Problem with VaR

A common problem with VaR models in practice is that exceedances often end up being **serially correlated**.

Another common problem with VaR models in practice is that exceedances tend to be **correlated with the level of risk**.

There is a reason VaR has become so popular in risk management. The appeal of VaR is its simplicity. Because VaR can be calculated for any portfolio, it allows us to easily compare the risk of different portfolios. Because it boils risk down to a single number, VaR provides us with a convenient way to track the risk of a portfolio over time. Finally, the concept of VaR is intuitive, even to those not versed in statistics.

Because it is so popular, VaR has come under a lot of criticism. The criticism generally falls into one of three categories.

At a very high level, financial institutions have been criticized for being overly reliant on VaR. This is not so much a criticism of VaR as it is a criticism of financial institutions for trying to make risk too simple.

At the other end of the spectrum, many experts have criticized how VaR is measured in practice. This is not so much a criticism of VaR as it is a criticism of specific implementations of VaR.

As computing power became cheaper and more widespread, this approach quickly fell out of favor. **Today VaR models can be extremely complex, but many people outside of risk management still remember when delta-normal was the standard approach, and mistakenly believe that this is a fundamental shortcoming of VaR.**

In between, there are more sophisticated criticisms. One such criticism is that VaR is **not a subadditive risk measure**. If we have two risky portfolios, X and Y, then f is said to be subadditive if:

$$f(X+Y)\leq f(X) + f(Y)$$

In other words, the risk of the combined portfolio, X + Y, is less than or equal to the sum of the risks of the separate portfolios. Variance and standard deviation are subadditive risk measures.

For example: Imagine a portfolio with two bonds, each with a 4% probability of defaulting. Assume that default events are uncorrelated and that the recovery rate of both bonds is 0%. If a bond defaults, it is worth 0; if it does not, it is worth 100. What is the 95% VaR of each bond separately? What is the 95% VaR of the bond portfolio?

$$
\begin{array}{cr}
\hline P[x] & x \\
\hline 0.16 \% & -\$ 200 \\
7.68 \% & -\$ 100 \\
92.16 \% & \$ 0 \\
\hline
\end{array}
$$

As we can easily see, there are no defaults in only 92.16% of the scenarios, (1 – 4%)2 = 92.16%. In the other 7.84% of scenarios, the loss is greater than or equal to 100. The 95% VaR of the portfolio is therefore 100. For this portfolio, VaR is not subadditive.

### Expected Shortfall

Another criticism of VaR is that it does not tell us anything about the tail of the distribution.

Using the concept of conditional probability, we can define the expected value of a loss, given an exceedance, as follows:


$$
E[L|L\geq VaR_{\gamma}]=S
$$

We refer to this conditional expected loss, S, as the expected shortfall.

Expected shortfall does answer an important question. What’s more, expected shortfall turns out to be subadditive, thereby avoiding one of the major criticisms of VaR.

As our discussion on backtesting suggests, though, **because it is concerned with the tail of the distribution, the reliability of our expected shortfall measure may be difficult to gauge.**

