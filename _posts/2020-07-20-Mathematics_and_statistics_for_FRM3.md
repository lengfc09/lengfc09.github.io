---
layout: mysingle
date: 2020-07-20 23:59:16 +0800
title: Mathematics and Statistics for Financial Risk Management by Michael B. Miller --- Part 3
categories: Quantitative_Financial_Risk_Management
excerpt: "Notes for the book by Michael B. Miller. It includes basic concepts in mathemtacs and statistics which are commonly used in the risk mangement process."
header:
    overlay_color: "#036" #午夜蓝
classes: wide
tags: risk_management mathematics statistics

toc: true
---

## Chapter 7: Linear Regression Analysis

###  Univariate Linear Regression (One Regressor)

$$Y=\alpha +\beta X  + \varepsilon $$

As specified, X is known as the **regressor** or independent variable. Similarly, Y is known as the **regressand** or dependent variable. As dependent implies, traditionally we think of X as causing Y. This relationship is not necessary, and in practice, especially in finance, this cause-and-effect relationship is either ambiguous or entirely absent. In finance, it is often the case that both X and Y are being driven by a common underlying factor.

Note that, even it is called univariate linear regression, there are actually two features (1,X).

In our regression model, Y is divided into a systematic component, $\alpha + \beta X$, and an idiosyncratic component, $\varepsilon$.

![-w300](/media/15951262104465/15952518112811.jpg){:width="300px"}{: .align-center}

#### Ordinary Least Square

The univariate regression model is conceptually simple. In order to **uniquely determine** the parameters in the model, though, we need to make some assumption about our variables.

By far the most popular linear regression model is ordinary least squares (OLS). OLS makes several assumptions about the form of the regression model, which can be summarized as follows:

A1: The relationship between the regressor and the regressand is linear.

A2: $$E[\varepsilon|X]=0$$

A3: $$Var[\varepsilon|X]=\sigma ^2$$

A4: $$Cov[\varepsilon_i , \varepsilon_j ] = 0~ ∀i \neq j$$

A5: $$\varepsilon_i  ∼ N(0,\sigma^2 )~\forall \varepsilon_i$$

A6: The regressor is nonstochastic.


##### A1: Linear

This assumption is not nearly as restrictive as it sounds.

1) Suppose we suspect that default rates are related to interest rates in the following way:

$$D=\alpha +\beta R^{3/4} + \varepsilon $$

Because of the exponent on R, the relationship between D and R is clearly nonlinear. Still, the relationship between D and $R^{3/4}$ is linear.

2) As specified, the model implies that the linear relationship should **be true for all values of D and R**. In practice, we often **only require that the relationship is linear within a given range**.

##### A2: independence between ε and X

Assumption A2 implies that the error term is independent of X, i.e.:

$$Cov[X,\varepsilon]=0$$

This because $E[\varepsilon X]=E[X* E_X [\varepsilon]]=E[X*0]=0$

##### A3: homoscedasticity

Assumption A3 states that the variance of the error term is constant. This property of constant variance is known as **homoscedasticity**, in contrast to **heteroscedasticity,** where the variance is nonconstant.

In finance, many models that appear to be linear often violate this assumption. As we will see in the next chapter, interest rate models often specify an error term that varies in relation to the level of interest rates.

##### A4: spherical errors

Assumption A4 states that the error terms for various data points should be uncorrelated with each other.

A random variable that has constant variance and is uncorrelated with itself is termed spherical. OLS assumes spherical errors.

##### A5: normally distributed

Assumption A5 states that the error terms in the model should be normally distributed. Many of the results of the OLS model are true, regardless of this assumption. This assumption is most useful when it comes to defining confidence levels for the model parameters.

##### A6: Nonstochastic

Finally, assumption A6 assumes that the **regressor is nonstochastic**, or nonrandom.

In reality, both the index’s return and the stock’s return are random variables, determined by a number of factors, some of which they might have in common. At some point, the discussion around assumption A6 tends to become deeply philosophical. From a practical standpoint, most of the results of OLS hold true, regardless of assumption A6. In many cases the conclusion needs to be modified only slightly.


##### An application of A2

$$
\beta=\frac{\operatorname{Cov}[X, Y]}{\sigma_{X}^{2}}=\rho_{X Y} \frac{\sigma_{Y}}{\sigma_{X}}
$$

This regression is so popular that we frequently speak of a stock’s beta, which is simply $\beta$ from the regression equation. While there are other ways to calculate a stock’s beta, the functional form given above is extremely popular, as it relates two values, $\sigma_X$ and $\sigma_Y$, with which traders and risk managers are often familiar, to two other terms, $\rho(X,Y)$ and $\beta$, which should be rather intuitive.


#### Estimating the parameters

$$
\mathrm{RSS}=\sum_{i=1}^{n} \varepsilon_{i}^{2}=\sum_{i=1}^{n}\left(y_{i}-\alpha-\beta x_{i}\right)^{2}
$$


where RSS is the commonly used acronym for the residual sum of squares (sum of squared residuals would probably be a more accurate description, but RSS is the convention).

With the framework of OLS, in order to minimize this equation, we first take its derivative with respect to $\alpha $ and $\beta$ separately.

We set the derivatives to zero and solve the resulting simultaneous equations. The result is the equations for OLS parameters:

$$
\alpha=\bar{Y}-\beta \bar{X}
$$



$$
\beta=\frac{\sum_{i=1}^{n} x_{i} y_{i}-n \bar{Y} \bar{X}}{\sum_{i=1}^{n} x_{i}^{2}-n \bar{X}^{2}}
$$

where $\bar{X}$ and $\bar{Y}$ are the sample mean of X and Y, respectively.


#### Evaluating the regression

##### Evaluation: R square

In finance it is rare that a simple univariate regression model is going to completely explain a large data set. In many cases, the data are so noisy that we must ask ourselves if the model is explaining anything at all. Even when a relationship appears to exist, we are likely to want some quantitative measure of just how strong that relationship is.

Probably the most popular statistic for describing linear regressions is the **coefficient of determination**, commonly known as R-squared, or just $R^2$.

To calculate the coefficient of determination, we need to define two additional terms: the **total sum of squares** (TSS) and the **explained sum of squares** (ESS). They are defined as:

$$
\mathrm{TSS}=\sum_{i=1}^{n}\left(y_{i}-\bar{Y}\right)^{2}
$$

$$
\mathrm{ESS}=\sum_{i=1}^{n}\left(\hat{y}_{i}-\bar{Y}\right)^{2}=\sum_{i=1}^{n}\left(\alpha+\beta x_{i}-\bar{Y}\right)^{2}
$$

These two sums are related to the previously encountered residual sum of
squares, as follows:

$$TSS=RSS+ESS$$

These sums can be used to compute $R^2$:

$$R^2=\frac{ESS}{TSS}=1-\frac{RSS}{TSS}$$

As promised, when there are no residual errors, when RSS is zero, $R^2$ is one. Also, when ESS is zero, or when the variation in the errors is equal to TSS, $R^2$ is zero.

It turns out that for the univariate linear regression model, R2 is also equal to the correlation between X and Y, squared.

$$R^2=\frac{Var(\alpha +\beta X)}{Var(Y)}$$

since we know that:

$$Var(\alpha +\beta X)=\beta^2 \sigma_X^2$$

and

$$\beta=\rho(X,Y) \frac{\sigma_Y}{\sigma_X}$$

we have

$$R^2= \rho^2(X,Y)$$


##### Evaluation: Significance of the parameters

In regression analysis, the most common null hypothesis is that the slope parameter, $\beta$, is zero. If $\beta$ is zero, then the regression model does not explain any variation in the regressand.

In finance, we often want to know if $\alpha$ is significantly different from zero, but for different reasons. In modern finance, alpha has become synonymous with the ability of a portfolio manager to generate excess returns.

In order to test the significance of the regression parameters,  we first need to calculate the variance of α and β, which we can obtain from the following formulas:


$$
\hat{\sigma}_{\varepsilon}^{2}=\frac{\sum_{i=1}^{n} \varepsilon_i^2}{n-2}
$$

$$
\hat{\sigma}_{\alpha}^{2}=\frac{\sum_{i=1}^{n} x_{i}^{2}}{n \sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}} \hat{\sigma}_{\varepsilon}^{2}
$$


$$
\hat{\sigma}_{\beta}^{2}=\frac{\hat{\sigma}_{\varepsilon}^{2}}{ \sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}
$$

**Proof:**

The first formula gives the variance of the error term, $\varepsilon$, which is simply the RSS divided by the **degrees of freedom for the regression (T-n)**.

$$
\hat{\beta}=\frac{\sum(X_i -\bar{X})(Y_i-\bar{Y})}{\sum (X_i-\bar{X})^2}
$$

since

$$Y_i=\alpha + \beta X_i +\varepsilon_i$$

$$\bar{Y}=\alpha + \beta \bar{X} $$

we have

$$Y_i-\bar{Y}=\beta (X_i-\bar{X}) + \varepsilon_i$$

Plug in the original equation, we have:

$$
\hat{\beta}=\frac{\sum(X_i -\bar{X})(\beta (X_i-\bar{X}) + \varepsilon_i)}{\sum (X_i-\bar{X})^2}=\beta+\frac{\sum(X_i -\bar{X})\varepsilon_i}{\sum (X_i-\bar{X})^2}
$$

As a result, we have


$$V(\hat{\beta})=V(\varepsilon) \frac{\sum(X_i -\bar{X})^2}{(\sum (X_i-\bar{X})^2)^2}=\frac{V(\varepsilon)}{\sum (X_i-\bar{X})^2}$$

For $\hat{\alpha}$:

$$
\begin{aligned}
\hat{\alpha}&= \bar{Y}-\hat{\beta} \bar{X} \\
&= \bar{Y}-\hat{\beta} \sum \frac{X_i}{n}
\end{aligned}
$$










Using the equations for the variance of our estimators, we can then form an appropriate t-statistic. For example, for $\beta$ we would have

$$
\frac{\hat{\beta}-\beta}{\hat{\sigma}_{\beta}} \sim t_{n-2}
$$

The most common null hypothesis when testing regression parameters is that the parameters are equal to zero. More often than not, we do not care if the parameters are significantly greater than or less than zero; **we just care that they are significantly different than 0**. Because of this, rather than using the standard t-statistics as in Equation 10.17, some practitioners prefer to use the absolute value of the t-statistic. Some software packages also follow this convention.


### Linear regression (Multivariate)

$$Y=X\beta + \varepsilon$$

#### No Multicollinearity
In the multivariate case, we require that all of the independent variables be linearly independent of each other. We say that the independent variables must lack multicollinearity:

**A7: The independent variables have no multicollinearity.**

To say that the independent variables lack multicollinearity means that it is impossible to express one of the independent variables as a linear combination of the others.

In other words, **the features has full rank.**

There is no well-accepted procedure for dealing with multicollinearity.

1) **eliminate a variable**
The easiest course of action is often simply to **eliminate a variable** from the regression. While easy, this is hardly satisfactory.

2) **transform the variables**
Another possibility is to transform the variables, to create uncorrelated variables out of linear combinations of the existing variables. In the previous example, even though $X_3$ is correlated with $X_2$, $X_3-\lambda X_2$ is uncorrelated with $X_2$.

3) **PCA**

One potential problem with this approach is similar to what we saw with **principal component analysis** (which is really just another method for creating uncorrelated variables from linear combinations of correlated variables).

**Make economic sense?**

If we are lucky, a linear combination of variables will have a simple economic interpretation. For example, if X2 and X3 are two equity indexes, then their difference might correspond to a familiar spread. Similarly, if the two variables are interest rates, their difference might bear some relation to the shape of the yield curve. Other linear combinations might be difficult to interpret, and if the relationship is not readily identifiable, then the relationship is more likely to be unstable or spurious.


#### Estimating the parameters

The result is our OLS estimator for $\beta $, $\hat{\beta}$:

$$
\hat{\boldsymbol{\beta}}=\left(\mathbf{X}^{\prime} \mathbf{X}\right)^{-1} \mathbf{X}^{\prime} \mathbf{Y}
$$

Where we had two parameters in the univariate case, now we have a vector of n parameters, which define our regression equation.

Given the OLS assumptions—actually, we don’t even need assumption A6, that the regressors are nonstochastic— $\hat{\beta}$ is the best linear unbiased estimator of $\beta$. This result is known as the Gauss-Markov theorem.


#### Evaluation of the regression

##### Evaluation: R square

One problem in the multivariate setting is that $R^2$ tends to increase as we add independent variables to our regression.

Clearly there should be some penalty for adding variables to a regression. An attempt to rectify this situation is the adjusted $R^2$, which is typically denoted by $R^2$, and defined as:

$$
\bar{R}^{2}=1-\left(1-R^{2}\right) \frac{t-1}{t-n}
$$

where t is the number of sample points and n is the number of regressors, including the constant term.

##### Evaluation: significance of parameters with t-statistics

Just as with the univariate model, we can calculate the variance of the error term. Given t data points and n regressors, the variance of the error term is:

$$
\hat{\sigma}_{\varepsilon}^{2}=\frac{\sum_{i=1}^{t} \varepsilon_{i}^{2}}{t-n}
$$

The variance of the i-th estimator is then:

$$
\hat{\sigma}_{i}^{2}=\hat{\sigma}_{\varepsilon}^{2}\left[\left(\mathbf{X}^{\prime} \mathbf{X}\right)^{-1}\right]_{i, i}
$$

**Proof**

If A=A' and A is invertible, we know that

$$(A^{-1})'=(A')^{-1}$$

Therefore:

$$((X'X)^{-1})'=(X'X)^{-1}$$

The inverse of a symmetric matrix is still symmetric.

For the BLUE $\hat{\beta}$, we have:

$$
\begin{aligned}
\hat{\beta}&=(X'X)^{-1}X'Y\\
&=(X'X)^{-1}X'(X\beta +\varepsilon)\\
&=\beta + (X'X)^{-1}X'\varepsilon
\end{aligned}
$$


For a random variable $X\in R^{n\times 1}$, the covariance matrix can be expressed as:

$$
Var(X)=E(X*X')-E(X)*E(X')
$$

Remember that,  for some random vector $X$ and some non-random matrix $A$, we have

$$Var(AX)=A*Var(X)*A'$$

we have:

$$
V(\hat{\beta})=(X'X)^{-1}X' V(\varepsilon)X(X'X)^{-1}
$$

Since $V(\varepsilon)=\sigma_{\varepsilon}^2 I$

$$
V(\hat{\beta})=\sigma_{\varepsilon}^2 (X'X)^{-1}X' X(X'X)^{-1}=\sigma_{\varepsilon}^2 (X'X)^{-1}
$$

Therefore, the variance of the i-th estimator is

$$
\hat{\sigma}_{i}^{2}=\hat{\sigma}_{\varepsilon}^{2}\left[\left(\mathbf{X}^{\prime} \mathbf{X}\right)^{-1}\right]_{i, i}
$$

We can then use this to form an appropriate t-statistic, with t −n degrees of freedom:

$$
\frac{\hat{\beta}_{i}-\beta_{i}}{\hat{\sigma}_{i}} \sim t_{t-n}
$$

##### Evaluation: significance of parameters with F-statistics

Instead of just testing one parameter, we can actually test the significance of all of the parameters, excluding the constant, using what is known as an F-test. The F-statistic can be calculated using $R^2$:

$$
\frac{ESS/(n-1)}{RSS/(t-n)}=\frac{R^{2} /(n-1)}{\left(1-R^{2}\right) /(t-n)} \sim F_{n-1, t-n}
$$

In general, we want to keep our models as simple as possible. We don’t want to add variables just for the sake of adding variables. This principle is known as **parsimony**.

$\bar{R}^2$ , t-tests, and F-tests are often used in deciding whether to include an additional variable in a regression.

In finance, even when the statistical significance of the betas is high, $R^2$ and $\bar{R}^2$ are often very low. For this reason, it is common to evaluate the addition of a variable on the basis of its t-statistic. If the t-statistic of the additional variable is statistically significant, then it is kept in the model. **It is less common, but it is possible to have a collection of variables, none of which are statistically significant by themselves, but which are jointly significant.**
This is why it is important to monitor the F-statistic as well.

When applied systematically, this process of adding or remov ing variables from a regression model is referred to as **stepwise regression.**


![-w600](/media/15951262104465/15952601866871.jpg){:width="600px"}


### Application of linear regression

#### Application: Factor analysis

In a large, complex portfolio, it is sometimes far from obvious how much exposure a portfolio has to a given factor. Depending on a portfolio manager’s objectives, it may be desirable to minimize certain factor exposures or to keep the amount of risk from certain factors within a given range. It typically falls to risk management to ensure that the factor exposures are maintained at acceptable levels.

The classic approach to factor analysis can best be described as risk taxonomy.

These kinds of obvious questions led to the development of various statistical approaches to factor analysis. One very popular approach is to associate each factor with an index, and then to use that index in a regression analysis to measure a portfolio’s exposure to that factor.

$$r_{\text {portfolio }}=\alpha+\beta r_{\text {index }}+\varepsilon$$

Another nice thing about factor analysis is that the factor exposures can be added across portfolios.

$$r_{\mathrm{A}}=\alpha_{\mathrm{A}}+\beta_{\mathrm{A}} r_{\text {index }}+\varepsilon_{\mathrm{A}}$$

$$r_{\mathrm{B}}=\alpha_{\mathrm{B}}+\beta_{\mathrm{B}} r_{\text {index }}+\varepsilon_{\mathrm{B}}$$

$$r_{\mathrm{A}+\mathrm{B}}=\left(\alpha_{\mathrm{A}}+\alpha_{\mathrm{B}}\right)+\left(\beta_{\mathrm{A}}+\beta_{\mathrm{B}}\right) r_{\text {index }}+\left(\varepsilon_{\mathrm{A}}+\varepsilon_{\mathrm{B}}\right)
$$

In addition to giving us the factor exposure, the factor analysis allows us to divide the risk of a portfolio into systematic and idiosyncratic components.

$$\sigma_{\text {portfolio }}^{2}=\beta^{2} \sigma_{\text {index }}^{2}+\sigma_{\varepsilon}^{2}$$


In theory, factors can be based on almost any kind of return series. The advantage of **indexes based on publicly traded securities** is that it makes hedging very straightforward.

At the same time, there might be **some risks that are not captured by any publicly traded index**. Some risk managers have attempted to resolve this problem by using statistical techniques, such as principal component analysis (PCA) or cluster analysis, to develop more robust factors.

#### Application: Stress Testing

In risk management, stress testing assesses the likely impact of an extreme, but plausible, scenario on a portfolio. There is no universally accepted method for perform- ing stress tests. One popular approach, which we consider here, is closely related to factor analysis.

The first step in stress testing is defining a scenario. Scenarios can be either ad hoc or based on a historical episode.

In the second step, we need to define **how all other underlying financial instruments react, given our scenario**. In order to do this, we construct multivariate regressions. We regress the returns of each underlying financial instrument against the returns of the instruments that define our scenario.

What might seem strange is that, even in the case of the historical scenarios, we use recent returns in our regression. In the case of the historical scenarios, why don’t we just use the actual returns from that period?

- some securities or companies may not exist
- the relationship between variables has changed significantly

In the final step, after we have generated the returns for all of the underlying financial instruments, we price any options or other derivatives. While using **delta approximations** might have been acceptable for calculating value at risk statistics at one point in time, **it should never have been acceptable for stress testing**. By definition, stress testing is the examination of extreme events, and the accurate pricing of nonlinear instruments is critical.

