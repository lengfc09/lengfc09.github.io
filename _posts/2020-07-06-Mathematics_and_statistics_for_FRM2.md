---
layout: mysingle
date: 2020-07-06 22:02:16 +0800
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
P(B|A)=\frac{P(A|B)* P(B)}{P(A)}
$$

* The conditional probability $$P(A|B)$$ is called the *likelihood* or *evidence*.
* $$P(B)$$ is the prior belief.
* $$P(B|A)$$ is the posterior belief.

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

Here f(A) is the prior probability density function, and f(A|B) is the posterior PDF. g(B|A) is the likelihood.


### Conjugate Distribution

It is not always the case that the prior and posterior distributions are of the same type.

When both the prior and posterior distributions are of the same type, we say that they are **conjugate distributions**. As we just saw, the beta distribution is the conjugate distribution for the binomial likelihood distribution.

Here is another useful set of conjugates: The normal distribution is the conjugate distribution for the normal likelihood when we are trying to estimate the distribution of the mean, and the variance is known.

The real world is not obligated to make our statistical calculations easy. In practice, prior and posterior distributions may be nonparametric and require numerical methods to solve. While all of this makes Bayesian analysis involving continuous distributions more complex, these are problems that are easily solved by computers. One reason for the increasing popularity of Bayesian analysis has to do with the rapidly increasing power of computers in recent decades.


### Bayesian Networks

A Bayesian network illustrates the causal relationship between different random variables.

Exhibit 6.4 shows a Bayesian network with two nodes that represent the economy go up, E, and a stock go up, S.

Exhibit 6.4 also shows three probabilities: the probability that E is up, $P[E]$; the probability that S is up given that E is up, $P[S | E]$; and the probability that S is up given that E is not up, $P[S | E]$.


![-w600](/media/15940441031797/15944709528677.jpg){:width="600px"}
Using Bayes’ theorem, we can also calculate $P[E | S]$. This is the probability that E is up given that we have observed S being up:

$$
P[E \mid S]=\frac{P[S \mid E] P[E]}{P[S]}=\frac{P[S \mid E] P[E]}{P[S \mid E] P[E]+P[S \mid \bar{E}] P[\bar{E}]}
$$

**Causal reasoning**, $P[S | E]$, follows the cause-and-effect arrow of our Bayesian network. **Diagnostic reasoning**, $P[E | S]$, works in reverse.

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

Because Bayesian networks are more intuitive, they might be easier to update in the face of a structural change or regime change. In the second network, where we have described S2 as being a supplier to S1, suppose that S2 announces that it has signed a contract to supply another large firm, thereby making it less reliant on S1? With the help of our equity analyst, we might be able to update the Bayesian network im/mediately (for example, by decreasing the {:width="600px"}probabilities $P[S2 | S1]$ and $P[S2 | S1]$), but it is not as obvious how we would directly update the covariance matrices.
