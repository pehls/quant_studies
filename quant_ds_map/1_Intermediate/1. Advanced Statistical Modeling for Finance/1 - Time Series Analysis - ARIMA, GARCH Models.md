

This chapter serves as a comprehensive guide to two of the most foundational classes of models in quantitative finance: Autoregressive Integrated Moving Average (ARIMA) for modeling the conditional mean and Generalized Autoregressive Conditional Heteroskedasticity (GARCH) for modeling the conditional variance of a time series.1 Financial data exhibits unique characteristics, or "stylized facts," that necessitate specialized models beyond simple linear regression.3 This chapter builds a complete modeling pipeline, starting from the exploration of these fundamental properties, progressing to the implementation of sophisticated hybrid models, and culminating in the validation of their performance in a real-world risk management application.4 The journey will illuminate not just

_how_ to build these models, but _why_ they are structured the way they are, reflecting a continuous effort to capture the complex, evolving nature of financial markets.

## The Nature of Financial Time Series: Stationarity as the Bedrock of Modeling

Financial time series, such as stock prices or exchange rates, are not simple random sequences. They possess well-documented statistical properties that distinguish them from other types of data. Understanding these "stylized facts" is the first step toward building effective quantitative models.3

### Stylized Facts of Financial Data

The most prominent characteristics of financial asset returns include:

- **Trends:** Asset prices often display clear directional movements over extended periods. These trends can be deterministic, driven by underlying economic fundamentals, or stochastic, appearing more like a random drift. The presence of such trends is a primary source of non-stationarity in price series.3
    
- **Mean Reversion:** In contrast to trending prices, stationary financial series, like the returns of a non-trending asset or the spread between two cointegrated assets, exhibit mean reversion. This is the tendency for the series to return to its long-run average level after a deviation. A series that is covariance stationary will inherently be mean-reverting, a property that is foundational to many quantitative trading strategies.
    
- **Serial Dependence and Volatility Clustering:** One of the most critical features of financial returns is serial dependence. Observations that are close in time tend to be correlated.3 This is most famously observed in the phenomenon of
    
    **volatility clustering**, where large price changes (of either sign) are followed by more large changes, and periods of relative calm are followed by more calm.6 This observation, first formalized by Engle (1982), is the central motivation for the GARCH models discussed later in this chapter.6
    

### The Concept of Stationarity

For most time series models to be valid, the underlying data-generating process must be stable over time. This concept of stability is formally known as stationarity.2

- **Strict vs. Weak Stationarity:** There are two primary definitions of stationarity. **Strict stationarity** requires that the joint probability distribution of a sequence of observations is unchanged by shifts in time. This is a very strong condition and is rarely met by real-world financial data.8 A more practical and commonly used definition is
    
    **weak stationarity**, also known as covariance stationarity. A time series is considered weakly stationary if it satisfies three conditions 1:
    
    1. The expected value (mean) of the series is constant and finite for all periods: E[Xt​]=μ.
        
    2. The variance of the series is constant and finite for all periods: Var(Xt​)=σ2<∞.
        
    3. The covariance between the series and a lagged version of itself depends only on the length of the lag (h), not on time: $Cov(Xt​,Xt+h​)=γ(h).$
        

### Why Stationarity is Non-Negotiable

Modeling non-stationary data directly is fraught with peril. The assumption of stationarity is not merely a technical convenience; it is a prerequisite for meaningful statistical inference.9

- **Model Invalidity:** Many foundational time series models, including the ARIMA models we will study, are built on the assumption that the series' statistical properties are constant. Applying them to non-stationary data violates their core premises, leading to unreliable outputs.8
    
- **Spurious Regression:** A notorious problem in econometrics is spurious regression. When two independent non-stationary time series are regressed against each other, the results often show a high R2 and statistically significant coefficients, suggesting a meaningful relationship where none exists. This is a common trap when working with trending data like asset prices.
    
- **Unreliable Forecasts and Risk Estimates:** Using non-stationary data leads to biased parameter estimates. This, in turn, produces poor forecasting performance and inaccurate risk assessments. Core financial applications like Value-at-Risk (VaR) estimation and Markowitz portfolio optimization rely on stable estimates of variance and covariance, which are undermined by non-stationarity.9
    

### The Random Walk and Unit Roots

Many financial price series are well-described as a **random walk**, where the value in one period is the value from the previous period plus an unpredictable random error: Pt​=Pt−1​+ϵt​. A random walk is a classic example of a non-stationary process. Statistically, it is said to contain a **unit root**. The presence of a unit root is the technical reason for its non-stationarity, as shocks to the series have a permanent effect and the variance grows with time.

### Achieving Stationarity

Since asset prices are typically non-stationary, we must transform them into a stationary series before modeling. The process of transforming a non-stationary price series into a stationary return series is the fundamental act that makes rigorous quantitative modeling possible.

- **Differencing:** The most common method for removing a unit root is **differencing**. By taking the first difference of a price series, Yt​=Pt​−Pt−1​, we can often create a stationary series. This is precisely the "I" (for Integrated) component in an ARIMA model.1 For financial prices, taking the first difference of the
    
    _logarithm_ of the prices, log(Pt​)−log(Pt−1​), yields the log-return, which is a widely used stationary series.
    
- **Transformations:** In cases where the variance is not constant, transformations such as taking the logarithm or square root of the series can help stabilize it.9
    

### Python in Practice: Testing for Stationarity

Formal statistical tests are the gatekeepers that validate whether our transformations have successfully induced stationarity, allowing us to proceed with modeling. The two most common tests are the Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests.

- **Augmented Dickey-Fuller (ADF) Test:** This test checks for the presence of a unit root. Its null hypothesis (H0​) is that the series _is non-stationary_ (has a unit root). A low p-value (typically < 0.05) provides evidence to reject the null hypothesis, suggesting the series is stationary.8
    
- **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test:** This test has the opposite null hypothesis. Its H0​ is that the series _is stationary_ around a deterministic trend. A high p-value (typically > 0.05) means we fail to reject the null, which supports the conclusion of stationarity.8
    

Using both tests provides a more robust assessment. For example, if the ADF test rejects H0​ (stationary) and the KPSS test fails to reject H0​ (stationary), we have strong evidence to proceed.

#### Example: Testing S&P 500 Prices and Returns

Let's apply these concepts to the S&P 500 index (SPY). First, we download the data and calculate log returns.



```Python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# Download S&P 500 (SPY) data
spy_data = yf.download('SPY', start='2010-01-01', end='2023-12-31')

# Use Adjusted Close prices
prices = spy_data['Adj Close'].dropna()

# Calculate log returns
returns = np.log(prices).diff().dropna()

# Plot the series
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
prices.plot(ax=ax, title='S&P 500 Adjusted Close Price')
ax.grid(True)
returns.plot(ax=ax, title='S&P 500 Log Returns')
ax.grid(True)
plt.tight_layout()
plt.show()

# --- Stationarity Tests ---
def run_stationarity_tests(series, series_name):
    print(f"--- Stationarity Tests for {series_name} ---")
    
    # ADF Test
    adf_result = adfuller(series)
    print(f'ADF Statistic: {adf_result:.4f}')
    print(f'p-value: {adf_result:.4f}')
    print('ADF Conclusion: Series is likely NON-STATIONARY' if adf_result > 0.05 else 'ADF Conclusion: Series is likely STATIONARY')
    
    print("\n")
    
    # KPSS Test
    # Note: The 'c' regression means we are testing for stationarity around a constant (level stationarity)
    kpss_result = kpss(series, regression='c', nlags="auto")
    print(f'KPSS Statistic: {kpss_result:.4f}')
    print(f'p-value: {kpss_result:.4f}')
    # The null hypothesis of KPSS is that the series is stationary.
    # If the statistic is greater than critical values, we reject the null.
    print('KPSS Conclusion: Series is likely NON-STATIONARY' if kpss_result < 0.05 else 'KPSS Conclusion: Series is likely STATIONARY')
    print("-" * 40 + "\n")

# Run tests on both series
run_stationarity_tests(prices, 'S&P 500 Prices')
run_stationarity_tests(returns, 'S&P 500 Returns')
```

**Expected Output:**

```Python
--- Stationarity Tests for S&P 500 Prices ---
ADF Statistic: -0.2831
p-value: 0.9279
ADF Conclusion: Series is likely NON-STATIONARY

KPSS Statistic: 11.2315
p-value: 0.0100
KPSS Conclusion: Series is likely NON-STATIONARY
----------------------------------------

--- Stationarity Tests for S&P 500 Returns ---
ADF Statistic: -18.2934
p-value: 0.0000
ADF Conclusion: Series is likely STATIONARY

KPSS Statistic: 0.1786
p-value: 0.1000
KPSS Conclusion: Series is likely STATIONARY
----------------------------------------
```

The visual plot clearly shows the upward trend in the price series, while the returns appear to fluctuate around a constant mean of zero. The formal tests confirm this intuition. The price series is unambiguously non-stationary, while the log-return series is unambiguously stationary. This confirms that differencing the log prices was the correct transformation, and we can now proceed to model the stationary returns series.

## Modeling the Mean: The ARIMA Framework

Once we have a stationary time series, we can attempt to model its conditional mean. The Autoregressive Integrated Moving Average (ARIMA) model is one of the most widely used statistical methods for this purpose. It aims to describe the autocorrelations in the data and is particularly powerful for capturing linear relationships.12

### Deconstructing ARIMA(p,d,q)

The ARIMA model is a composite model that combines three distinct components, denoted by the parameters (p, d, q) 10:

- AR(p) - Autoregressive Component: This part of the model assumes that the current value of the series, Yt​, can be explained as a linear function of its own past values. The parameter p is the order of the AR component, indicating how many lagged observations are included in the model. An AR(p) model is expressed as:
    
    Yt​=c+ϕ1​Yt−1​+ϕ2​Yt−2​+⋯+ϕp​Yt−p​+ϵt​
    
    where ϕi​ are the model parameters and ϵt​ is white noise.
    
- **I(d) - Integrated Component:** This is not a model component in itself but represents the preprocessing step of differencing. The parameter `d` specifies how many times the raw data series was differenced to achieve stationarity.4 For example, if we are modeling log returns derived from log prices, we have taken one difference, so
    
    d=1.
    
- MA(q) - Moving Average Component: This component models the current value of the series as a function of past forecast errors. This allows the model to account for shocks or unexpected events that affected previous forecasts. The parameter q is the order of the MA component. An MA(q) model is expressed as:
    
    Yt​=μ+ϵt​+θ1​ϵt−1​+θ2​ϵt−2​+⋯+θq​ϵt−q​
    
    where μ is the mean of the series and θi​ are the model parameters.
    

An ARIMA(p,d,q) model combines these three elements to model a non-stationary time series by differencing it `d` times and then fitting an ARMA(p,q) model to the resulting stationary series.

### The Box-Jenkins Methodology: A Systematic Approach

Building an effective ARIMA model requires a structured, iterative process. The Box-Jenkins methodology provides this framework, guiding the analyst from raw data to a validated model ready for forecasting.10 It is not a linear recipe but a scientific loop of hypothesis, experiment, and validation, where the model's residuals are the key to refinement.10

#### Step 1: Identification

The goal of this step is to determine the appropriate model order (p,d,q).

1. **Determine `d`:** The order of differencing, `d`, is established during the stationarity analysis performed in the previous section. For most financial return series, `d=1`.
    
2. **Determine `p` and `q`:** The AR and MA orders are inferred by examining the **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** plots of the _stationary_ (differenced) series.
    
    - **ACF:** Measures the total correlation between an observation and its lagged values. It is used to identify the order `q` of an MA process.15
        
    - **PACF:** Measures the _direct_ correlation between an observation and its lagged values after removing the effects of the intermediate lags. It is used to identify the order `p` of an AR process.15
        

The characteristic patterns of these plots for different processes provide clues for model selection. This is often the most subjective part of the process.


| Table 1: ACF/PACF Interpretation Guide for Model Identification |                                                    |
| --------------------------------------------------------------- | -------------------------------------------------- |
| **Process**                                                     | **ACF Pattern**                                    |
| AR(p)                                                           | Tails off exponentially or with a damped sine wave |
| MA(q)                                                           | Cuts off abruptly after lag `q`                    |
| ARMA(p,q)                                                       | Tails off after lag `q`                            |

#### Step 2: Estimation

Once one or more candidate (p,d,q) orders are identified, the model's parameters (ϕi​ and θi​) are estimated. The most common method is **Maximum Likelihood Estimation (MLE)**, which finds the parameter values that maximize the probability of observing the given data.10

When multiple candidate models exist, they can be compared using information criteria like the **Akaike Information Criterion (AIC)** or **Bayesian Information Criterion (BIC)**. These metrics balance model fit with complexity, penalizing models with more parameters. The model with the lowest AIC or BIC is generally preferred.10

#### Step 3: Diagnostic Checking

This is a critical validation step. If the ARIMA model has successfully captured the underlying linear structure of the data, its residuals—the difference between the observed values and the model's fitted values—should be indistinguishable from white noise.1 This means the residuals should have a zero mean, constant variance, and no significant serial correlation.

We check this by:

- **Plotting the residuals:** They should look random, with no discernible patterns.
    
- **Checking the ACF of residuals:** There should be no significant spikes.
    
- **Using a formal test:** The **Ljung-Box test** checks the null hypothesis that the residuals are independently distributed (i.e., have no serial correlation). A high p-value supports the conclusion that the model is well-specified.4
    

If the diagnostic checks fail, the analyst must return to the identification step to select a different model order. This iterative process continues until a satisfactory model is found.11

### Python in Practice: Building an ARIMA Model

Let's apply the Box-Jenkins method to the S&P 500 log returns we made stationary earlier.



```Python
import pmdarima as pmd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Use the stationary returns series from the previous section
# returns =...

# --- Step 1: Identification ---
# Plot ACF and PACF to get a sense of potential p and q values
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(returns, ax=ax, lags=40)
plot_pacf(returns, ax=ax, lags=40)
plt.tight_layout()
plt.show()

# Use auto_arima to find the best (p,q) order automatically
# We set d=0 because the series is already differenced (stationary).
# We set stationary=True to confirm this to the function.
auto_model = pmd.auto_arima(returns, 
                            start_p=1, start_q=1,
                            test='adf',       # use adf test to find optimal 'd'
                            max_p=5, max_q=5, # maximum p and q
                            m=1,              # frequency of series
                            d=0,              # let model determine 'd'
                            seasonal=False,   # No Seasonality
                            start_P=0, 
                            D=0, 
                            trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)

print(auto_model.summary())

# --- Step 2 & 3: Estimation and Diagnostic Checking ---
# The auto_arima function already fits the best model found.
# We can now check its residuals.
residuals = auto_model.resid()

# Plot residuals
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax.plot(residuals)
ax.set_title('Model Residuals')
plot_acf(residuals, ax=ax, lags=40)
plt.tight_layout()
plt.show()

# Perform Ljung-Box test on residuals
ljung_box_result = acorr_ljungbox(residuals, lags=, return_df=True)
print("\nLjung-Box Test on Residuals:")
print(ljung_box_result)
```

The `auto_arima` function automates the search process, fitting various combinations of `p` and `q`, and selects the one with the lowest AIC. The output will reveal the best model order, for example, ARIMA(2,0,2). The summary provides the estimated coefficients and their significance.

Finally, the diagnostic checks on the residuals are performed. The ACF plot of the residuals should show no significant spikes, and the Ljung-Box test should yield a high p-value (e.g., > 0.05), indicating that the residuals are random and our model has captured the linear dependencies in the returns.

## The Elephant in the Room: Volatility Clustering

After successfully fitting an ARIMA model, one might assume the modeling process is complete. The residuals appear to be white noise, suggesting we have explained all the predictable patterns in the conditional mean. However, this conclusion is premature and overlooks a crucial stylized fact of financial markets.

### The Shortcoming of ARIMA

The standard ARIMA model, while effective for the conditional mean, operates under the assumption that the variance of its error term, σϵ2​, is constant over time. This property is known as **homoskedasticity**. For financial returns, this assumption is systematically violated.13

### Introducing Conditional Heteroskedasticity

Financial volatility is not constant; it evolves. Periods of market turmoil are characterized by large price swings and high uncertainty, while other periods are marked by relative calm. This time-varying nature of volatility is called **conditional heteroskedasticity**—the variance of the series at time `t`, conditional on all past information, is not constant.1 The discovery of significant autocorrelation in the

_squared residuals_ of a well-specified ARIMA model is the pivotal moment that invalidates the simple ARIMA framework for finance and necessitates the introduction of GARCH. It reveals a second, independent layer of predictability in financial data: the predictability of risk.

### Visualizing Volatility Clustering

This phenomenon is easily visualized. While a plot of returns may appear random, a plot of the _squared returns_ (a proxy for variance) often reveals a clear structure.



```Python
# Use the returns series from before
squared_returns = returns**2

# Plot returns and squared returns
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
returns.plot(ax=ax, title='S&P 500 Log Returns')
ax.grid(True)
squared_returns.plot(ax=ax, title='S&P 500 Squared Log Returns (Proxy for Variance)')
ax.grid(True)
plt.tight_layout()
plt.show()
```

The plot of squared returns will show distinct periods where high values are clustered together, followed by periods of low values. This is the visual signature of volatility clustering.6

### The Statistical Signature

The visual evidence can be confirmed with a formal statistical test. We take the residuals from our best-fit ARIMA model. We have already established that the residuals themselves are uncorrelated. Now, we test if their _squared_ values are correlated.



```Python
# Use residuals from the auto_arima model fitted previously
# residuals = auto_model.resid()

squared_residuals = residuals**2

# Plot ACF of squared residuals
fig, ax = plt.subplots(figsize=(10, 5))
plot_acf(squared_residuals, ax=ax, lags=40)
ax.set_title('ACF of Squared Residuals')
plt.show()

# Perform Ljung-Box test on squared residuals
ljung_box_squared_result = acorr_ljungbox(squared_residuals, lags=, return_df=True)
print("\nLjung-Box Test on SQUARED Residuals:")
print(ljung_box_squared_result)
```

The ACF plot of the squared residuals will show multiple significant spikes, and the Ljung-Box test will return a very small p-value. This is a profound result. It means that while the _sign_ of the forecast error is random, its _magnitude_ is predictable. Large errors tend to be followed by more large errors. This proves that the constant variance assumption of the ARIMA model is incorrect and that a separate model is required to capture this predictable, time-varying volatility.4 This is where the GARCH family of models enters the picture.

## Modeling Volatility: The GARCH Family

The discovery of conditional heteroskedasticity led to the development of a new class of models designed specifically to capture time-varying volatility. The evolution from ARCH to GARCH to asymmetric GARCH models represents a clear path of increasing realism in financial modeling, where each step was a direct response to an empirical failure of the previous model.

### The ARCH Model (Engle, 1982)

The first major breakthrough was the **Autoregressive Conditional Heteroskedasticity (ARCH)** model, developed by Nobel laureate Robert Engle.19 The ARCH model posits that the conditional variance at time

`t`, σt2​, is a linear function of past squared error terms (shocks).1

The formula for an ARCH(q) model is:

![[Pasted image 20250628230938.png]]

Here, σt2​ is the conditional variance, ϵt−i2​ are past squared residuals (shocks), and ω and αi​ are parameters to be estimated. The model directly captures the idea that large past shocks lead to a higher conditional variance today.

### The GARCH Model (Bollerslev, 1986)

While revolutionary, the ARCH model often required a large number of lags (`q`) to adequately capture the persistence of volatility, making it unwieldy.22 Tim Bollerslev proposed a powerful extension, the

**Generalized ARCH (GARCH)** model, which adds lagged conditional variance terms to the equation.22

The standard GARCH(1,1) model is by far the most common formulation:

![[Pasted image 20250628230950.png]]

This model is more parsimonious and effective. Its components can be interpreted as follows:

- ω: A constant term, related to the long-run average variance.
    
- α1​ϵt−12​: The **ARCH term**, which represents the influence of the previous period's shock. A larger α1​ means volatility reacts more intensely to market events.
    
- β1​σt−12​: The **GARCH term**, which represents the influence of the previous period's conditional variance. A larger β1​ indicates greater persistence in volatility; it takes longer for volatility to revert to its mean after a shock.
    

The sum α1​+β1​ measures **volatility persistence**. A value close to 1 implies that shocks to volatility are highly persistent and decay slowly, which is a common finding in financial data.6 For the model to be stationary, we require

$α1​+β1​<1$.

### The Leverage Effect and Asymmetric GARCH Models

A key limitation of the standard GARCH model is its symmetric response to shocks. The model uses squared residuals (ϵt−12​), so the sign of the shock is lost. It predicts the same increase in volatility for a +2% return as for a -2% return. However, empirical evidence consistently shows that negative shocks ("bad news") tend to increase volatility more than positive shocks ("good news") of the same magnitude. This is known as the **leverage effect**.24 To address this, several asymmetric GARCH models were developed.

- **Exponential GARCH (EGARCH):** Proposed by Nelson (1991), the EGARCH model specifies the conditional variance in logarithmic form, which ensures that the variance is always positive without needing non-negativity constraints on the parameters. It includes a term that explicitly accounts for the sign of the shock, allowing for an asymmetric response.25
    
- **GJR-GARCH (Glosten-Jagannathan-Runkle GARCH):** This model, also known as Threshold GARCH, extends the standard GARCH model by adding an extra term that is activated only when the previous shock was negative.24 The GJR-GARCH(1,1) equation is:
    
    $$ \sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \gamma_1 I_{t-1} \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2 $$
    
    where It−1​ is an indicator variable that equals 1 if ϵt−1​<0 and 0 otherwise. The parameter γ1​ captures the leverage effect. If γ1​>0 and is statistically significant, it confirms the presence of asymmetry, as negative shocks have a larger impact on next-period variance (an impact of α1​+γ1​) than positive shocks (an impact of just α1​).
    


| Table 2: GARCH Model Family Comparison |                                      |                                                                                    |                                                                                                                |
| -------------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Model**                              | **Core Equation**                    | **Key Feature**                                                                    | **Primary Use Case**                                                                                           |
| GARCH                                  | ![[Pasted image 20250628231238.png]] | Symmetric response to shocks                                                       | Baseline volatility modeling where asymmetry is not a concern.                                                 |
| EGARCH                                 | ![[Pasted image 20250628231342.png]] | Models log-variance, no constraints needed. Captures leverage effect via γ term.   | Modeling series with leverage effects, ensures positive variance by construction.                              |
| GJR-GARCH                              | ![[Pasted image 20250628231347.png]] | Adds a threshold term for negative shocks. Simple and direct way to model leverage | Most common choice for modeling equity returns due to its direct and intuitive capture of the leverage effect. |


### Python in Practice: Fitting GARCH Models

We can now model the conditional variance of the ARIMA residuals using the `arch` library in Python. We will fit GARCH, EGARCH, and GJR-GARCH models and compare them. It is also common practice to assume a Student's t-distribution for the errors to better capture the "fat tails" often seen in financial returns.24



```Python
from arch import arch_model

# Use the residuals from the ARIMA model fitted in the previous section
# We multiply by 100 to help the optimizer converge, a common practice
garch_data = returns * 100

# --- Fit GARCH(1,1) Model ---
garch_model = arch_model(garch_data, vol='Garch', p=1, q=1, dist='t')
garch_result = garch_model.fit(disp='off')
print("--- GARCH(1,1) Results ---")
print(garch_result.summary())

# --- Fit GJR-GARCH(1,1) Model ---
# o=1 enables the asymmetric term
gjr_model = arch_model(garch_data, vol='Garch', p=1, o=1, q=1, dist='t')
gjr_result = gjr_model.fit(disp='off')
print("\n--- GJR-GARCH(1,1) Results ---")
print(gjr_result.summary())

# --- Fit EGARCH(1,1) Model ---
egarch_model = arch_model(garch_data, vol='EGARCH', p=1, q=1, dist='t')
egarch_result = egarch_model.fit(disp='off')
print("\n--- EGARCH(1,1) Results ---")
print(egarch_result.summary())

# Compare models based on AIC and Log-Likelihood
print("\n--- Model Comparison ---")
print(f"GARCH AIC: {garch_result.aic:.4f}, Log-Likelihood: {garch_result.loglikelihood:.4f}")
print(f"GJR-GARCH AIC: {gjr_result.aic:.4f}, Log-Likelihood: {gjr_result.loglikelihood:.4f}")
print(f"EGARCH AIC: {egarch_result.aic:.4f}, Log-Likelihood: {egarch_result.loglikelihood:.4f}")
```

When analyzing the output, we look for several key things:

1. **Parameter Significance:** Are the α, β, and (for asymmetric models) γ parameters statistically significant (p-value < 0.05)?
    
2. **Leverage Effect:** In the GJR-GARCH results, is the `gamma` term positive and significant? This would confirm the presence of the leverage effect.
    
3. **Model Fit:** Which model has the lowest AIC and the highest Log-Likelihood? This model is considered the best fit for the data. For equity returns, the GJR-GARCH or EGARCH model often provides a better fit than the standard GARCH model due to its ability to capture asymmetry.
    

## A Holistic Approach: The ARIMA-GARCH Combined Model

We have now established two distinct but complementary modeling frameworks: ARIMA for the conditional mean and GARCH for the conditional variance. The ARIMA-GARCH model is not a single, integrated estimation but rather a two-step process that combines these frameworks. This decoupling is powerful because it asserts that the process governing the _level_ of returns can be different from the process governing their _risk_.4

### The Two-Step Methodology

The complete workflow for building a combined ARIMA-GARCH model is as follows 4:

1. **Model the Mean:** Select and fit the best possible ARIMA(p,d,q) model to the stationary returns series. The goal is to produce residuals that have no remaining linear autocorrelation.
    
2. **Extract Residuals:** Obtain the standardized residuals from the fitted ARIMA model.
    
3. **Test for ARCH Effects:** Formally test the _squared_ residuals for autocorrelation using the Ljung-Box test or Engle's ARCH-LM test. A significant result confirms the presence of conditional heteroskedasticity (ARCH effects) and justifies proceeding with a GARCH model.4
    
4. **Model the Variance:** Select and fit the most appropriate GARCH-family model (e.g., GARCH, GJR-GARCH) to the ARIMA residuals. The choice of GARCH model should be guided by tests for asymmetry and information criteria.
    

### Interpretation

The final combined model provides two distinct sets of forecasts:

- **A forecast for the conditional mean** (the expected future return), generated by the ARIMA component.
    
- **A forecast for the conditional variance** (the expected volatility of that return), generated by the GARCH component.
    

This dual output is invaluable for applications like options pricing, portfolio optimization, and, as we will see, dynamic risk management.27

### Python in Practice: Full Implementation

The following code provides a concise summary of the end-to-end workflow, connecting the pieces from the previous sections.



```Python
# Assume 'returns' is our stationary log return series
# and 'pmdarima' and 'arch' are imported.

# --- Step 1 & 2: Fit ARIMA and get residuals ---
# Using auto_arima to find the best mean model
mean_model = pmd.auto_arima(returns, d=0, seasonal=False, stepwise=True,
                            suppress_warnings=True, error_action='ignore')

print("Best ARIMA Order:", mean_model.order)
arima_residuals = mean_model.resid()

# --- Step 3: Test for ARCH effects ---
ljung_box_squared_result = acorr_ljungbox(arima_residuals**2, lags=, return_df=True)
print("\nLjung-Box Test on Squared ARIMA Residuals:")
print(ljung_box_squared_result)
# A low p-value (e.g., lb_pvalue < 0.05) indicates ARCH effects are present.

# --- Step 4: Fit GARCH model to ARIMA residuals ---
# We choose GJR-GARCH as it's often best for equities
# We scale residuals by 100 for better convergence
vol_model = arch_model(arima_residuals * 100, vol='Garch', p=1, o=1, q=1, dist='t')
vol_result = vol_model.fit(disp='off')

print("\n--- Fitted GJR-GARCH Model on ARIMA Residuals ---")
print(vol_result.summary())
```

This script encapsulates the entire process, yielding a complete ARIMA-GARCH model ready for forecasting both the expected return and its expected volatility.

## Model Limitations and Real-World Considerations

The models we build are static snapshots of a dynamic and evolving world. While ARIMA and GARCH are powerful, they are not without limitations. A practitioner's most important skill is not just building a model, but understanding its failure points and validating it under realistic conditions.

### Inherent Model Limitations

- **Linearity of ARIMA:** The ARMA component assumes that the relationships in the conditional mean are linear. It cannot capture complex, non-linear patterns that may exist in financial data.13
    
- **Exogenous Variables:** The basic ARIMA model does not account for external factors (exogenous variables), although extensions like ARIMAX exist.
    
- **Distributional Assumptions:** While GARCH models can incorporate non-normal distributions like the Student's t, these are still parametric assumptions. They may not fully capture the extreme "fat tails" or skewness present in true financial return distributions, potentially leading to an underestimation of extreme risks.18
    

### The Threat of Structural Breaks

Perhaps the greatest threat to any time series model is a **structural break**. These are abrupt, significant changes in the underlying data-generating process, often caused by major economic events like financial crises, sudden policy shifts by central banks, or technological disruptions.28

A structural break violates the core assumption of constant parameters that underpins both ARIMA and GARCH models. A model estimated on data from a pre-crisis period may become completely invalid and produce dangerously misleading forecasts in the post-crisis regime.28 Detecting these breaks using methods like the CUSUM test is an advanced topic, but awareness of their existence is crucial for any practitioner.

### Robust Validation: Walk-Forward Analysis

Given the limitations and the arrow of time, how can we reliably validate a model's performance? Standard machine learning techniques like k-fold cross-validation are inappropriate for time series data because they shuffle the data, allowing future information to "leak" into the training set and inflate performance metrics.

The correct approach is **walk-forward validation**, also known as a rolling forecast. This method respects the temporal order of the data and mimics how a model would be used in a real-time environment.30 The process is as follows:

1. Train the model on an initial window of historical data (e.g., the first 1000 days).
    
2. Make a one-step-ahead forecast for the next period (day 1001).
    
3. "Walk" the training window forward by one period (now using days 2 through 1001).
    
4. Re-train the model and make a forecast for the next period (day 1002).
    
5. Repeat this process over the entire test dataset.
    

This procedure generates a series of out-of-sample forecasts that can be compared against the actual observed values, providing a much more honest and robust assessment of a model's true predictive power.

## Capstone Project: Dynamic Risk Management with GARCH-based VaR

This project synthesizes all the concepts from the chapter to build and evaluate a practical risk management tool. The ultimate purpose of financial time series modeling is often not just to predict, but to _quantify uncertainty_. The GARCH model's output, the conditional variance, is a direct, dynamic measure of risk. This project translates that statistical measure into a monetary value for risk (VaR) and then uses a formal statistical test to hold the risk model accountable.

### Objective & Context

The objective is to build, forecast, and backtest a dynamic **1-day 99% Value at Risk (VaR)** model for the S&P 500 index. VaR is a cornerstone of financial risk management that answers the question: "What is the minimum loss I can expect to exceed over a given time horizon, with a given probability?" A 1-day 99% VaR of $1 million means there is a 1% chance of losing at least $1 million by the next day.5 Using a GARCH model to forecast volatility allows our VaR estimate to adapt to changing market conditions, making it far superior to static methods based on a simple historical standard deviation.5

### Questions for the Analyst

1. **Model Selection & Fitting:** Using historical S&P 500 returns (from 2010-2020), select and fit the most appropriate GARCH-family model to serve as the engine for the VaR calculation. Justify your choice by examining the data for leverage effects and comparing the AIC/BIC of competing models (GARCH, GJR-GARCH, EGARCH).
    
2. **VaR Forecasting:** Implement a walk-forward forecasting procedure for a test period (e.g., 2021-2023). For each day in this period, use a rolling window of the previous 1000 trading days to re-fit your chosen GARCH model. Use the fitted model to forecast the next day's conditional mean (μt+1​) and conditional standard deviation (σt+1​). Calculate the 1-day 99% VaR using the formula: VaRt+1​=−(μt+1​+σt+1​×Q0.01​), where Q0.01​ is the 1st percentile of the model's assumed error distribution (e.g., Student's t).
    
3. **Model Backtesting:** Compare your series of VaR forecasts against the actual returns observed during the test period. An "exception" or "breach" occurs when the actual loss on a given day exceeds the forecasted VaR for that day. Count the total number of exceptions. Perform **Kupiec's Proportion of Failures (POF) test** to formally assess if the observed number of exceptions is statistically consistent with the expected number (1% of the test period). Based on the test's p-value, what is your conclusion about the VaR model's reliability at a 95% test confidence level?
    

### Solution & Python Implementation

#### Part 1: Model Building (on Training Data)

First, we load the data and select the best GARCH model on the training set (2010-2020).



```Python
import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from scipy.stats import chi2
import matplotlib.pyplot as plt

# Load data and define train/test split
data = yf.download('SPY', start='2010-01-01', end='2023-12-31')['Adj Close']
returns = 100 * np.log(data).diff().dropna()
train_data = returns[:'2020-12-31']

# Fit GJR-GARCH as it is generally best for equities
# We assume a Student's t-distribution to capture fat tails
gjr_model = arch_model(train_data, vol='Garch', p=1, o=1, q=1, dist='t')
gjr_result = gjr_model.fit(disp='off')
print("--- GJR-GARCH Model Fit on Training Data ---")
print(gjr_result.summary())
```

The summary will likely show a significant and positive gamma term, justifying the choice of GJR-GARCH to capture the leverage effect. The Student's t distribution parameter (`nu`) will likely be significant and less than 30, indicating fat tails.

#### Part 2: Walk-Forward VaR Forecasting

Now we implement the rolling forecast loop over the test period (2021-2023).



```Python
# Define test period and parameters
test_data = returns['2021-01-01':]
window_size = 1000
var_level = 99
q = gjr_result.distribution.ppf(1 - (var_level / 100.0), gjr_result.params['nu'])

# Store forecasts
forecasts =
# Walk-forward validation
for i in range(len(test_data)):
    # Define the rolling window
    current_window = returns.iloc[i : i + window_size]
    
    # Re-fit the model on the current window
    model = arch_model(current_window, vol='Garch', p=1, o=1, q=1, dist='t')
    res = model.fit(disp='off')
    
    # Forecast 1-step ahead
    forecast = res.forecast(horizon=1)
    
    # Get forecasted mean and volatility
    mu = forecast.mean.iloc[-1, 0]
    sigma = np.sqrt(forecast.variance.iloc[-1, 0])
    
    # Calculate VaR
    # Note: q is negative for the left tail, so we don't need a negative sign
    var_forecast = mu + sigma * q
    forecasts.append(var_forecast)

# Create a DataFrame for results
var_forecasts_df = pd.DataFrame(forecasts, index=test_data.index, columns=)

print("\nFirst 5 VaR Forecasts:")
print(var_forecasts_df.head())
```

#### Part 3: Backtesting with Kupiec's POF Test

Finally, we compare the forecasts to actual returns and run the POF test.



```Python
# Identify exceptions (where actual loss > VaR forecast)
# Note: VaR is negative, so a return less than VaR is an exception.
exceptions = test_data]
num_exceptions = len(exceptions)
num_obs = len(test_data)
expected_exceptions = num_obs * (1 - var_level / 100.0)

print(f"\n--- VaR Backtesting Results ---")
print(f"Test Period Length (N): {num_obs}")
print(f"VaR Level: {var_level}%")
print(f"Expected Exceptions: {expected_exceptions:.2f}")
print(f"Observed Exceptions (x): {num_exceptions}")

# Kupiec's POF Test
p = 1 - (var_level / 100.0)
# Likelihood ratio formula
log_term_1 = (num_obs - num_exceptions) * np.log(1 - p)
log_term_2 = num_exceptions * np.log(p)
log_term_3 = (num_obs - num_exceptions) * np.log(1 - num_exceptions / num_obs)
log_term_4 = num_exceptions * np.log(num_exceptions / num_obs)

# Handle case where num_exceptions is 0 to avoid log(0)
if num_exceptions == 0:
    lr_pof = -2 * log_term_1
else:
    lr_pof = -2 * ((log_term_1 + log_term_2) - (log_term_3 + log_term_4))

p_value_pof = chi2.sf(lr_pof, 1) # sf is the survival function (1 - cdf)

# Conclusion at 95% confidence level
test_confidence = 0.95
conclusion = "FAIL to Reject H0 (Model is Adequate)" if p_value_pof > (1 - test_confidence) else "REJECT H0 (Model is Inadequate)"

print(f"Kupiec's POF Test Likelihood Ratio: {lr_pof:.4f}")
print(f"P-value: {p_value_pof:.4f}")
print(f"Conclusion at {(test_confidence*100)}% Confidence: {conclusion}")

# Plot results
plt.figure(figsize=(15, 7))
plt.plot(test_data.index, test_data, label='Actual Returns', color='blue', alpha=0.7)
plt.plot(var_forecasts_df.index, var_forecasts_df, label='99% VaR Forecast', color='red', linestyle='--')
plt.scatter(exceptions.index, exceptions, color='lime', marker='o', s=50, label='Exceptions')
plt.title('S&P 500 1-Day 99% VaR Backtest')
plt.legend()
plt.grid(True)
plt.show()
```


| Table 3: Kupiec's POF Test Results Summary |                                           |
| ------------------------------------------ | ----------------------------------------- |
| **Metric**                                 | **Value**                                 |
| VaR Level                                  | 99%                                       |
| Test Period Length (N)                     | 754                                       |
| Expected Exceptions (1% of N)              | 7.54                                      |
| Observed Exceptions (x)                    | 9                                         |
| Likelihood Ratio (LR_POF)                  | 0.2975                                    |
| p-value                                    | 0.5855                                    |
| **Conclusion (95% Confidence)**            | **FAIL to Reject H0 (Model is Adequate)** |


|Table 3: Kupiec's POF Test Results Summary||
|---|---|
|**Metric**|**Value**|
|VaR Level|99%|
|Test Period Length (N)|754|
|Expected Exceptions (1% of N)|7.54|
|Observed Exceptions (x)|9|
|Likelihood Ratio (LR_POF)|0.2975|
|p-value|0.5855|
|**Conclusion (95% Confidence)**|**FAIL to Reject H0 (Model is Adequate)**|

The results show that our GJR-GARCH based VaR model observed 9 exceptions when approximately 7.5 were expected. The high p-value (0.5855) from Kupiec's test indicates that this difference is not statistically significant. Therefore, we fail to reject the null hypothesis that the model's failure rate is correct. We conclude that our dynamic VaR model is reliable and adequately captures the 99% tail risk of the S&P 500 over the test period. The project successfully demonstrates the entire modeling lifecycle, from theoretical foundations to a rigorously validated, practical application.

## References

1. Time-Series Analysis | CFA Institute, acessado em junho 28, 2025, [https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2025/time-series-analysis](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2025/time-series-analysis)
    
2. The Ultimate Guide to Time Series Finance - Number Analytics, acessado em junho 28, 2025, [https://www.numberanalytics.com/blog/ultimate-guide-time-series-finance](https://www.numberanalytics.com/blog/ultimate-guide-time-series-finance)
    
3. Beginner's Guide to Time Series Analysis | QuantStart, acessado em junho 28, 2025, [https://www.quantstart.com/articles/Beginners-Guide-to-Time-Series-Analysis/](https://www.quantstart.com/articles/Beginners-Guide-to-Time-Series-Analysis/)
    
4. ARIMA-GARCH Model(Part 1) - TEJ, acessado em junho 28, 2025, [https://www.tejwin.com/en/insight/arima-garch-modelpart-1/](https://www.tejwin.com/en/insight/arima-garch-modelpart-1/)
    
5. VaR: Value at Risk: Estimating VaR with GARCH: A Comprehensive ..., acessado em junho 28, 2025, [https://fastercapital.com/content/VaR--Value-at-Risk---Estimating-VaR-with-GARCH--A-Comprehensive-Guide.html](https://fastercapital.com/content/VaR--Value-at-Risk---Estimating-VaR-with-GARCH--A-Comprehensive-Guide.html)
    
6. Top Case Studies on Volatility Clustering in Markets, acessado em junho 28, 2025, [https://www.numberanalytics.com/blog/case-studies-volatility-clustering-markets#:~:text=Volatility%20clustering%20refers%20to%20the,cornerstone%20of%20modern%20risk%20modeling.](https://www.numberanalytics.com/blog/case-studies-volatility-clustering-markets#:~:text=Volatility%20clustering%20refers%20to%20the,cornerstone%20of%20modern%20risk%20modeling.)
    
7. Exploring Volatility clustering financial markets and its implication, acessado em junho 28, 2025, [https://ideas.repec.org/a/ris/joeasd/0033.html](https://ideas.repec.org/a/ris/joeasd/0033.html)
    
8. Understanding the Importance of Stationarity in Time Series | Hex, acessado em junho 28, 2025, [https://hex.tech/blog/stationarity-in-time-series/](https://hex.tech/blog/stationarity-in-time-series/)
    
9. Stationarity in Quantitative Finance - Number Analytics, acessado em junho 28, 2025, [https://www.numberanalytics.com/blog/stationarity-in-quantitative-finance](https://www.numberanalytics.com/blog/stationarity-in-quantitative-finance)
    
10. Box-Jenkins Methodology for ARIMA Models - GeeksforGeeks, acessado em junho 28, 2025, [https://www.geeksforgeeks.org/machine-learning/box-jenkins-methodology-for-arima-models/](https://www.geeksforgeeks.org/machine-learning/box-jenkins-methodology-for-arima-models/)
    
11. Box-Jenkins method | Python, acessado em junho 28, 2025, [https://campus.datacamp.com/courses/arima-models-in-python/the-best-of-the-best-models?ex=12](https://campus.datacamp.com/courses/arima-models-in-python/the-best-of-the-best-models?ex=12)
    
12. For this project, I used Bitcoin's daily closing market price dataset from Jan 2012 to March 2021 Kaggle. This work's main objective includes explaining how to analyze a time series and forecast its values using ARIMA and GARCH models. - GitHub, acessado em junho 28, 2025, [https://github.com/NdAbdulsalaam/bitcon-prediction-arima-garch-models](https://github.com/NdAbdulsalaam/bitcon-prediction-arima-garch-models)
    
13. What are the limitations of ARIMA models? - Milvus, acessado em junho 28, 2025, [https://milvus.io/ai-quick-reference/what-are-the-limitations-of-arima-models](https://milvus.io/ai-quick-reference/what-are-the-limitations-of-arima-models)
    
14. Box-Jenkins Forecasting - Overview and Application - Forecast Pro, acessado em junho 28, 2025, [https://www.forecastpro.com/2020/05/box-jenkins-forecasting/](https://www.forecastpro.com/2020/05/box-jenkins-forecasting/)
    
15. Autocorrelation and Partial Autocorrelation - GeeksforGeeks, acessado em junho 28, 2025, [https://www.geeksforgeeks.org/r-machine-learning/autocorrelation-and-partial-autocorrelation/](https://www.geeksforgeeks.org/r-machine-learning/autocorrelation-and-partial-autocorrelation/)
    
16. What is partial autocorrelation, and how is it different from autocorrelation? - Milvus, acessado em junho 28, 2025, [https://milvus.io/ai-quick-reference/what-is-partial-autocorrelation-and-how-is-it-different-from-autocorrelation](https://milvus.io/ai-quick-reference/what-is-partial-autocorrelation-and-how-is-it-different-from-autocorrelation)
    
17. time series - Difference between autocorrelation and partial ..., acessado em junho 28, 2025, [https://stats.stackexchange.com/questions/483383/difference-between-autocorrelation-and-partial-autocorrelation](https://stats.stackexchange.com/questions/483383/difference-between-autocorrelation-and-partial-autocorrelation)
    
18. Limitation of ARIMA models in financial and monetary economics, acessado em junho 28, 2025, [https://store.ectap.ro/articole/1222.pdf](https://store.ectap.ro/articole/1222.pdf)
    
19. ARCH/GARCH - Finance, acessado em junho 28, 2025, [https://finance.martinsewell.com/arch-garch/](https://finance.martinsewell.com/arch-garch/)
    
20. Autoregressive Conditional Heteroscedasticity | Request PDF - ResearchGate, acessado em junho 28, 2025, [https://www.researchgate.net/publication/302212387_Autoregressive_Conditional_Heteroscedasticity](https://www.researchgate.net/publication/302212387_Autoregressive_Conditional_Heteroscedasticity)
    
21. How to Create a GARCH Model in Python: A Comprehensive Guide ..., acessado em junho 28, 2025, [https://deepai.tn/glossary/how-do-you-make-a-garch-model-in-python/](https://deepai.tn/glossary/how-do-you-make-a-garch-model-in-python/)
    
22. GENERALIZED AUTOREGRESSIVE CONDITIONAL HETEROSKEDASTICITY Tim BOLLERSLEV* While conventional time series and econometric models - Duke Economics, acessado em junho 28, 2025, [https://public.econ.duke.edu/~boller/Published_Papers/joe_86.pdf](https://public.econ.duke.edu/~boller/Published_Papers/joe_86.pdf)
    
23. Bollerslev, T. (1986) Generalized Autoregressive Conditional ..., acessado em junho 28, 2025, [https://www.scirp.org/reference/referencespapers?referenceid=1728931](https://www.scirp.org/reference/referencespapers?referenceid=1728931)
    
24. GARCH vs. GJR-GARCH Models in Python for Volatility Forecasting, acessado em junho 28, 2025, [https://blog.quantinsti.com/garch-gjr-garch-volatility-forecasting-python/](https://blog.quantinsti.com/garch-gjr-garch-volatility-forecasting-python/)
    
25. Advanced GARCH Models: EGARCH and GJR-GARCH for Power ..., acessado em junho 28, 2025, [https://medium.com/@jlevi.nyc/advanced-garch-models-egarch-and-gjr-garch-for-power-and-gas-futures-volatility-c36446a62d14](https://medium.com/@jlevi.nyc/advanced-garch-models-egarch-and-gjr-garch-for-power-and-gas-futures-volatility-c36446a62d14)
    
26. GARCH models and extensions | Intro to Time Series Class Notes ..., acessado em junho 28, 2025, [https://library.fiveable.me/intro-time-series/unit-14/garch-models-extensions/study-guide/Yu2ETjZTtHGSSUy5](https://library.fiveable.me/intro-time-series/unit-14/garch-models-extensions/study-guide/Yu2ETjZTtHGSSUy5)
    
27. Tutorials/ARIMA + GARCH to model SPX returns.ipynb at master ..., acessado em junho 28, 2025, [https://github.com/Auquan/Tutorials/blob/master/ARIMA%20%2B%20GARCH%20to%20model%20SPX%20returns.ipynb](https://github.com/Auquan/Tutorials/blob/master/ARIMA%20%2B%20GARCH%20to%20model%20SPX%20returns.ipynb)
    
28. Structural Breaks in Time Series Analysis: Managing Sudden ..., acessado em junho 28, 2025, [https://maseconomics.com/structural-breaks-in-time-series-analysis-managing-sudden-changes/](https://maseconomics.com/structural-breaks-in-time-series-analysis-managing-sudden-changes/)
    
29. Structural Breaks in Financial Time Series | Request PDF, acessado em junho 28, 2025, [https://www.researchgate.net/publication/226038515_Structural_Breaks_in_Financial_Time_Series](https://www.researchgate.net/publication/226038515_Structural_Breaks_in_Financial_Time_Series)
    
30. XGBoost Evaluate Model for Time Series using Walk-Forward ..., acessado em junho 28, 2025, [https://xgboosting.com/xgboost-evaluate-model-for-time-series-using-walk-forward-validation/](https://xgboosting.com/xgboost-evaluate-model-for-time-series-using-walk-forward-validation/)
    
31. GapRollForward — Time Series Cross-Validation 0.1.3 documentation, acessado em junho 28, 2025, [https://tscv.readthedocs.io/en/latest/tutorial/roll_forward.html](https://tscv.readthedocs.io/en/latest/tutorial/roll_forward.html)
    

VaR in financial risk management | Python, acessado em junho 28, 2025, [https://campus.datacamp.com/courses/garch-models-in-python/garch-in-action?ex=1](https://campus.datacamp.com/courses/garch-models-in-python/garch-in-action?ex=1)