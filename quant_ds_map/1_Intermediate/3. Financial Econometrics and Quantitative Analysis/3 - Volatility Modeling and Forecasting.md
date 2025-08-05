### 3.3.1 The Nature of Volatility: Quantifying Financial Risk

Volatility is a foundational concept in finance, serving as a primary metric for quantifying risk and uncertainty in asset prices. It measures the magnitude of price fluctuations over a given period, with higher volatility indicating greater dispersion of returns and, consequently, higher perceived risk.2 A stock that experiences wide price swings, for instance, trading between $20 and $40, is considered more volatile than one with a stable price range, such as $25 to $30, over the same timeframe.2 Understanding and forecasting volatility is crucial for a wide range of financial applications, including options pricing, portfolio construction, and, most critically, risk management.3

There are two principal approaches to measuring volatility: one is backward-looking, analyzing past data, while the other is forward-looking, gauging market expectations.5

#### Historical Volatility (HV): Learning from the Past

Historical Volatility (HV), also referred to as realized or statistical volatility, is a measure derived directly from the historical price movements of a security or index.1 It is fundamentally a backward-looking metric that quantifies the degree of price variation over a specified past period. Investors and analysts use HV to assess an asset's past risk profile and potential for large price swings.5

The most common method for calculating HV is the "close-to-close" approach, which involves the following steps:

1. **Calculate Logarithmic Returns**: The first step is to compute the daily returns of the asset. Logarithmic returns are preferred over simple returns because they are additive over time, a property that simplifies time series modeling.6 The log return (
    
    rt​) on day t is calculated from the closing price (Pt​) as:
    
    ![[Pasted image 20250702001513.png]]
2. **Compute Rolling Standard Deviation**: Next, the standard deviation of these log returns is calculated over a specific lookback window, such as 30, 90, or 180 days. This is typically done on a rolling basis to observe how volatility changes over time.
    
3. **Annualize the Volatility**: Since the standard deviation is calculated from daily returns, it represents daily volatility. To make it comparable across different time frames, it is annualized by multiplying by the square root of the number of trading periods in a year, which is typically assumed to be 252 for equities.7 The formula for annualized historical volatility (σannual​) over an N-day window is:
    
    ![[Pasted image 20250702001738.png]]​

**Python Example: Calculating and Plotting Rolling HV for Apple (AAPL)**

The following Python code demonstrates how to download price data for Apple Inc. (AAPL) using the `yfinance` library and calculate its 30-day rolling annualized historical volatility.



```Python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Download historical data for AAPL
symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-12-31'
aapl_data = yf.download(symbol, start=start_date, end=end_date)

# 2. Calculate daily logarithmic returns
aapl_data = np.log(aapl_data['Adj Close'] / aapl_data['Adj Close'].shift(1))

# 3. Calculate 30-day rolling annualized historical volatility
window = 30
aapl_data['Historical_Volatility'] = aapl_data.rolling(window=window).std() * np.sqrt(252)

# 4. Plot the results
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot closing price on the primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('AAPL Adjusted Close Price ($)', color=color)
ax1.plot(aapl_data.index, aapl_data['Adj Close'], color=color, label='Adj Close Price')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_title('AAPL Price and 30-Day Rolling Historical Volatility')

# Create a second y-axis for volatility
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Annualized Volatility', color=color)
ax2.plot(aapl_data.index, aapl_data['Historical_Volatility'], color=color, label='30-Day HV')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

# Display the last few rows of the DataFrame
print(aapl_data].tail())
```

The resulting plot clearly illustrates the concept of volatility clustering: periods of significant price turbulence in AAPL stock correspond to noticeable spikes in its historical volatility.

#### Implied Volatility (IV): The Market's Expectation

In contrast to the backward-looking nature of HV, Implied Volatility (IV) is a forward-looking metric that encapsulates the market's collective expectation of an asset's future price fluctuations.1 IV is not calculated from historical price data; rather, it is

_implied_ by the current market prices of options contracts.

The price of an option depends on several factors, including the underlying asset's price, the strike price, time to expiration, interest rates, and the expected volatility of the underlying asset. If all other factors are known, an options pricing model, such as the Black-Scholes model, can be used to solve for the one unknown: volatility. The value of volatility that makes the model's theoretical price equal to the option's current market price is the implied volatility.

Therefore, IV reflects the market's consensus on how volatile an asset is likely to be in the future. It is heavily influenced by supply and demand for options, which in turn is driven by market participants' uncertainty about upcoming events like earnings reports, regulatory decisions, or major economic data releases.1 When uncertainty is high, demand for options (as a form of insurance) increases, driving up their prices and, consequently, the implied volatility.2

The most widely cited measure of implied volatility is the CBOE Volatility Index, or VIX. Often dubbed the market's "Fear Gauge," the VIX is calculated from a weighted average of the prices of S&P 500 index options and represents the market's expectation of 30-day forward-looking volatility.2

#### Comparing HV and IV: The Volatility Risk Premium

The relationship between historical and implied volatility is dynamic and provides valuable information about market sentiment. A key observation is that IV often acts as a leading indicator for HV.9 Because options are forward-looking instruments, their prices (and thus IV) will rise in

_anticipation_ of a volatile event. HV, being a measure of past price changes, will only spike _after_ the event has occurred and caused significant price movement.

This dynamic often leads to IV overshooting the volatility that is subsequently realized (HV). This phenomenon, where implied volatility tends to trade at a premium over historical volatility, is known as the **Volatility Risk Premium (VRP)**. It can be interpreted as the premium that options buyers are willing to pay for protection against future uncertainty, or the compensation that options sellers demand for taking on that risk.

This spread between IV and HV creates strategic opportunities for traders. When IV is substantially higher than HV, options may be considered relatively "expensive," creating favorable conditions for strategies that involve selling volatility (e.g., writing covered calls or cash-secured puts). Conversely, when IV is unusually low compared to HV, options may be seen as "cheap," presenting an advantage for options buyers.1

### 3.3.2 Stylized Facts: The Empirical Behavior of Asset Returns

The development of sophisticated volatility models like GARCH was motivated by the consistent observation that financial asset returns do not conform to the simple assumptions of classical models, such as the random walk hypothesis. Instead, they exhibit several persistent statistical properties known as "stylized facts".6 These empirical regularities are observed across a wide range of assets, markets, and time periods.

#### Introduction: Why the Normal Distribution Fails

Many traditional financial theories are built on the assumption that asset returns are independent and identically distributed (i.i.d.) according to a normal (Gaussian) distribution. However, empirical analysis of real-world financial data consistently refutes this assumption.6 Asset returns display characteristics that are fundamentally different from a normal distribution, necessitating more advanced modeling techniques.

#### Fact 1: Volatility Clustering

Perhaps the most crucial stylized fact is volatility clustering. This is the tendency for periods of high volatility to be followed by more high volatility, and periods of low volatility to be followed by more low volatility.6 In other words, volatility is not constant over time (a property known as heteroskedasticity) but arrives in clusters.

This clustering implies that volatility is, to some extent, predictable. While the direction of the next price move may be random, the magnitude of that move is not. Evidence for this is found by examining the autocorrelation of returns. While the returns themselves (rt​) show little to no significant serial correlation, their squared values (rt2​) or absolute values (∣rt​∣) exhibit positive and statistically significant autocorrelation that decays slowly over time.11 This indicates that a large price move today (positive or negative) increases the probability of a large price move tomorrow.

**Python Example: Testing for ARCH Effects**

We can test for volatility clustering (also known as ARCH effects) in S&P 500 returns. We will first visualize the autocorrelation of returns and squared returns, then perform a formal statistical test.



```Python
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Download S&P 500 data
gspc_data = yf.download('^GSPC', start='2010-01-01', end='2023-12-31')
returns = 100 * gspc_data['Adj Close'].pct_change().dropna()

# Prepare squared returns
squared_returns = returns**2

# Plot ACF of returns and squared returns
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(returns, ax=ax1, lags=40, title='ACF of S&P 500 Returns')
ax1.set_xlabel('Lag')
ax1.set_ylabel('Autocorrelation')

plot_acf(squared_returns, ax=ax2, lags=40, title='ACF of S&P 500 Squared Returns')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Autocorrelation')
plt.tight_layout()
plt.show()

# Perform Ljung-Box test on squared returns
ljung_box_results = acorr_ljungbox(squared_returns, lags=)
print("Ljung-Box Test on Squared Returns:")
print(ljung_box_results)
```

The ACF plot of raw returns will show insignificant correlations for most lags, consistent with efficient market theories. In stark contrast, the ACF plot for squared returns will display significant positive correlations that persist for many lags, visually confirming volatility clustering. The Ljung-Box test on the squared returns will yield a very small p-value, leading to the rejection of the null hypothesis of no autocorrelation and formally confirming the presence of ARCH effects.12

#### Fact 2: Fat Tails (Leptokurtosis)

Another key stylized fact is that the distribution of financial returns exhibits "fat tails," or leptokurtosis.6 This means that extreme price movements (both large gains and large losses) occur much more frequently in reality than would be predicted by a normal distribution. The unconditional distribution of returns is sharply peaked around the mean and has heavier tails.11

The presence of volatility clustering is a primary driver of this observed leptokurtosis. If returns were drawn from a single normal distribution with constant variance, extreme events would be exceedingly rare. However, in reality, the variance itself changes over time. During periods of high volatility (a cluster), returns are effectively drawn from a distribution with a much larger variance, making large outcomes more probable. When these periods are combined with periods of calm (draws from a low-variance distribution), the resulting overall, or unconditional, distribution has fatter tails than any single normal distribution.13

**Python Example: Normality Testing**

We can demonstrate the non-normality of returns both visually and statistically.



```Python
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, jarque_bera

# Use the same S&P 500 returns data
gspc_data = yf.download('^GSPC', start='2010-01-01', end='2023-12-31')
returns = 100 * gspc_data['Adj Close'].pct_change().dropna()

# Plot histogram against a normal distribution
plt.figure(figsize=(10, 6))
plt.hist(returns, bins=100, density=True, alpha=0.6, label='S&P 500 Returns')
mu, std = norm.fit(returns)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
plt.title('Distribution of S&P 500 Returns vs. Normal Distribution')
plt.legend()
plt.show()

# Calculate Skewness and Kurtosis
skewness = returns.skew()
kurtosis = returns.kurt() # Pandas calculates excess kurtosis (K-3)
print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis (actual): {kurtosis + 3:.4f}") # Add 3 to get actual kurtosis

# Perform Jarque-Bera test
jb_statistic, jb_pvalue = jarque_bera(returns)
print(f"\nJarque-Bera Test:")
print(f"JB Statistic: {jb_statistic:.4f}")
print(f"P-value: {jb_pvalue}")
```

The histogram will show a distribution that is more peaked at the center and has fatter tails than the overlaid normal curve. The calculated kurtosis will be significantly greater than 3 (the kurtosis of a normal distribution).12 The Jarque-Bera test formally evaluates the null hypothesis of normality based on skewness (

S) and kurtosis (K). Its test statistic is defined as:

![[Pasted image 20250702001939.png]]

where n is the number of observations.12 The resulting p-value will be extremely small, providing strong statistical evidence to reject the hypothesis that returns are normally distributed.

#### Fact 3: The Leverage Effect

The leverage effect describes the tendency for an asset's volatility to be negatively correlated with its returns.13 Put simply, negative news or price drops tend to increase future volatility more than positive news or price gains of the same magnitude.12 This phenomenon is also known as volatility asymmetry.

The name "leverage" comes from an early explanation proposed by Black (1976), who argued that as a company's stock price falls, its debt-to-equity ratio increases, making the firm financially more leveraged and thus riskier, which in turn increases its stock's volatility. However, subsequent research has shown that this mechanical effect is too small to fully account for the observed asymmetry.13 A more widely accepted explanation is the "volatility feedback effect," which posits that the anticipation of increased volatility (risk) can itself depress stock prices, creating a feedback loop where falling prices lead to higher volatility.14

**Python Example: Testing for Leverage Effect**

A straightforward way to test for the leverage effect is to compute the correlation between lagged returns (rt−1​) and current squared returns (rt2​). A statistically significant negative correlation provides evidence of the effect.12



```Python
import yfinance as yf
import numpy as np
import pandas as pd

# Use the same S&P 500 returns data
gspc_data = yf.download('^GSPC', start='2010-01-01', end='2023-12-31')
returns = 100 * gspc_data['Adj Close'].pct_change().dropna()

# Create a DataFrame for the test
leverage_df = pd.DataFrame({
    'returns_t': returns,
    'squared_returns_t': returns**2,
    'returns_t-1': returns.shift(1)
}).dropna()

# Calculate the correlation
leverage_correlation = leverage_df['squared_returns_t'].corr(leverage_df['returns_t-1'])

print(f"Correlation between lagged returns and current squared returns: {leverage_correlation:.4f}")
```

The output will show a negative correlation coefficient, confirming that for the S&P 500, past negative returns are associated with higher current volatility (as proxied by squared returns).

### 3.3.3 Econometric Volatility Models: The GARCH Family

To address the stylized facts of financial returns—namely volatility clustering, fat tails, and the leverage effect—econometricians have developed a specialized class of time series models. These models, known collectively as the ARCH/GARCH family, are designed to capture the dynamic, time-varying nature of volatility.

#### The ARCH(q) Model: Autoregressive Conditional Heteroskedasticity

The Autoregressive Conditional Heteroskedasticity (ARCH) model, introduced by Robert F. Engle in a seminal 1982 paper for which he was awarded the Nobel Prize, was the first to formalize the concept of time-varying volatility.16 The core idea of the ARCH model is to link the conditional variance of the error term directly to the magnitude of recent past error terms. It explicitly models volatility clustering by assuming that large past shocks lead to a higher variance in the current period.17

The mathematical specification of an ARCH(q) model is as follows:

Let the return series be rt​=μt​+ϵt​, where μt​ is the conditional mean and ϵt​ is the error term. The error term is defined as ϵt​=σt​zt​, where zt​ is a white noise process, often assumed to be standard normal (zt​∼N(0,1)). The conditional variance, σt2​, is modeled as a function of the past q squared error terms:

![[Pasted image 20250702001957.png]]

To ensure that the conditional variance is always positive, the parameters must satisfy the constraints ω>0 and αi​≥0 for i=1,…,q.17

**Python Example: Fitting an ARCH Model**

While ARCH was a groundbreaking model, in practice it often requires a large number of lags (q) to adequately capture the persistence of volatility, making it less parsimonious. We can fit an ARCH model using the Python `arch` library.18



```Python
import yfinance as yf
from arch import arch_model

# Use the same S&P 500 returns data
gspc_data = yf.download('^GSPC', start='2010-01-01', end='2023-12-31')
returns = 100 * gspc_data['Adj Close'].pct_change().dropna()

# Fit an ARCH(5) model
arch_model_spec = arch_model(returns, vol='ARCH', p=5, mean='Constant')
arch_result = arch_model_spec.fit(update_freq=10)

print(arch_result.summary())
```

The summary will show the estimated coefficients for ω and the αi​ terms. One would typically observe that several lags are statistically significant, hinting at the need for a more efficient model structure.

#### The GARCH(p,q) Model: A More Powerful Generalization

The Generalized ARCH (GARCH) model, introduced by Tim Bollerslev in 1986, is a more efficient and widely used extension of the ARCH model.17 The GARCH model improves upon ARCH by including not only past shocks (the ARCH terms) but also past conditional variances (the GARCH terms) in the variance equation. This structure is more parsimonious and often provides a better fit to financial data, analogous to how an ARMA model can be more efficient than a pure AR model for modeling the mean of a time series.20

The conditional variance equation for a GARCH(p,q) model is:
![[Pasted image 20250702002015.png]]

Here, p is the order of the GARCH terms (lagged variances) and q is the order of the ARCH terms (lagged squared errors).22

The most common and often sufficient specification is the GARCH(1,1) model 19:

![[Pasted image 20250702002026.png]]

The parameters of the GARCH(1,1) model have intuitive interpretations:

- ω: A constant term, representing the baseline or long-run variance component.
    
- α1​ (ARCH term): This coefficient measures the reaction of volatility to market shocks or "news" from the previous period. A larger α1​ implies that volatility is more sensitive to recent market events.
    
- β1​ (GARCH term): This coefficient measures the persistence of volatility. It captures how much of yesterday's volatility carries over to today. A larger β1​ indicates that volatility shocks are long-lasting and die out slowly.
    

The sum α1​+β1​ is known as the _persistence_ of the model. This value indicates the rate at which the effect of a shock on conditional variance decays. If the sum is close to 1, shocks are highly persistent. If α1​+β1​≥1, the model is non-stationary (known as Integrated GARCH or IGARCH), and the long-run variance is not well-defined.19 For a stationary GARCH(1,1) model, the unconditional or long-run average variance (VL​) is given by ![[Pasted image 20250702002051.png]]

**Python Example: Fitting a GARCH(1,1) Model**

We can fit a GARCH(1,1) model to the S&P 500 returns and analyze its output.



```Python
import yfinance as yf
from arch import arch_model

# Use the same S&P 500 returns data
gspc_data = yf.download('^GSPC', start='2010-01-01', end='2023-12-31')
returns = 100 * gspc_data['Adj Close'].pct_change().dropna()

# Specify and fit a GARCH(1,1) model
# We assume a constant mean and normal distribution for the residuals
garch_model_spec = arch_model(returns, vol='GARCH', p=1, q=1, mean='Constant', dist='normal')
garch_result = garch_model_spec.fit(update_freq=10)

print(garch_result.summary())
```

The summary output provides a wealth of information. The "Volatility Model" section shows the estimated coefficients for `omega` (ω), `alpha` (α1​), and `beta` (β1​), along with their standard errors and p-values. Typically for stock market indices, we expect to see a small but significant α1​ and a large, highly significant β1​ (often > 0.8), with their sum being close to 1, indicating high volatility persistence. The model's log-likelihood, AIC (Akaike Information Criterion), and BIC (Bayesian Information Criterion) are also provided, which are useful for comparing different model specifications.

#### Modeling Asymmetry: GJR-GARCH and EGARCH

A key limitation of the standard GARCH model is its symmetric response to shocks; it assumes that positive and negative shocks of the same magnitude have an identical impact on future volatility (ϵt−12​). This contradicts the well-documented leverage effect. To address this, several asymmetric GARCH models have been proposed, with the GJR-GARCH and EGARCH models being the most popular.25

**The GJR-GARCH Model**

The GJR-GARCH model, named after its creators Glosten, Jagannathan, and Runkle, extends the standard GARCH model by adding a term to capture the leverage effect directly.25 It introduces an indicator function that "activates" only when the previous shock was negative.

The conditional variance equation for a GJR-GARCH(1,1) model is:

![[Pasted image 20250702002110.png]]

where It−1​ is an indicator function that equals 1 if ϵt−1​<0 (bad news) and 0 otherwise (good news).25

In this model, the impact of a positive shock on volatility is governed by α, while the impact of a negative shock is (α+γ). A statistically significant and positive γ coefficient is direct evidence of the leverage effect, indicating that negative shocks have a greater impact on volatility than positive shocks.25

**The EGARCH Model**

The Exponential GARCH (EGARCH) model, proposed by Nelson (1991), provides an alternative way to capture asymmetry. It models the logarithm of the conditional variance, which offers two main advantages:

1. Since log(σt2​) is modeled, the variance σt2​ is guaranteed to be positive regardless of the parameter values, eliminating the need for non-negativity constraints.29
    
2. The specification naturally allows for an asymmetric response to shocks.
    

The EGARCH(1,1) variance equation is:

![[Pasted image 20250702002130.png]]

Here, the term with coefficient α captures the magnitude effect (symmetric), while the term with coefficient γ captures the sign effect (asymmetric or leverage effect).30 If

γ<0, a negative shock (ϵt−1​<0) will have a larger impact on the log-variance than a positive shock of the same magnitude, consistent with the leverage effect.30

**Python Example: Fitting and Comparing Models**

We can fit all three models—GARCH, GJR-GARCH, and EGARCH—to the S&P 500 data and compare their performance using information criteria like AIC and BIC. A lower AIC or BIC value suggests a better model fit, penalizing for model complexity.



```Python
import yfinance as yf
from arch import arch_model
import pandas as pd

# Use the same S&P 500 returns data
gspc_data = yf.download('^GSPC', start='2010-01-01', end='2023-12-31')
returns = 100 * gspc_data['Adj Close'].pct_change().dropna()

# Fit GARCH(1,1)
garch = arch_model(returns, vol='GARCH', p=1, q=1, mean='Constant', dist='normal')
garch_res = garch.fit(disp='off')

# Fit GJR-GARCH(1,1) (o=1 for the asymmetric term)
gjr_garch = arch_model(returns, vol='GARCH', p=1, o=1, q=1, mean='Constant', dist='normal')
gjr_res = gjr_garch.fit(disp='off')

# Fit EGARCH(1,1)
egarch = arch_model(returns, vol='EGARCH', p=1, q=1, mean='Constant', dist='normal')
egarch_res = egarch.fit(disp='off')

# Create a summary table
results_df = pd.DataFrame({
    'GARCH(1,1)': [garch_res.aic, garch_res.bic, '-', '-'],
    'GJR-GARCH(1,1)': [gjr_res.aic, gjr_res.bic, f"{gjr_res.params['gamma']:.4f}", f"{gjr_res.pvalues['gamma']:.4f}"],
    'EGARCH(1,1)': [egarch_res.aic, egarch_res.bic, f"{egarch_res.params['gamma']:.4f}", f"{egarch_res.pvalues['gamma']:.4f}"]
}, index=)

print("GARCH Model Comparison on S&P 500 Data")
print(results_df)

```

**Table 1: GARCH Model Comparison on S&P 500 Data**

|Metric|GARCH(1,1)|GJR-GARCH(1,1)|EGARCH(1,1)|
|---|---|---|---|
|AIC|8919.58|8843.83|8846.88|
|BIC|8945.71|8876.58|8879.63|
|Leverage Term (γ)|-|0.1248|-0.1064|
|P-value for γ|-|0.0000|0.0000|

_Note: The numerical values in this table are illustrative and will vary based on the exact data and time period used._

The results in the table provide a clear, data-driven justification for using asymmetric models. Both GJR-GARCH and EGARCH show significantly lower AIC and BIC values compared to the standard GARCH model, indicating a superior fit to the data. Furthermore, the leverage term (γ) is highly statistically significant (p-value near zero) in both models. This confirms that the leverage effect is a crucial feature of S&P 500 returns and that models explicitly accounting for it provide a more accurate representation of volatility dynamics.

### 3.3.4 Forecasting Future Volatility

After fitting and selecting an appropriate GARCH model, its primary use is to forecast future volatility. This is essential for risk management, derivative pricing, and portfolio optimization.

#### One-Step vs. Multi-Step Ahead Forecasting

GARCH models can produce both one-step-ahead and multi-step-ahead forecasts.24

- **One-step-ahead forecast**: Predicts the conditional variance for the very next time period (t+1) based on all information available up to time t.
    
- **Multi-step-ahead forecast**: Predicts the conditional variance for multiple periods into the future (t+2,t+3,…,t+h).
    

A key property of forecasts from a stationary GARCH model is **mean reversion**. For forecasts far into the future, the predicted conditional variance will converge to the model's long-run unconditional variance.19 The speed of this convergence is dictated by the model's persistence (

α1​+β1​). A higher persistence means the forecast will revert to the long-run mean more slowly.

The formula for an h-step-ahead variance forecast from a GARCH(1,1) model highlights this property. The 1-step forecast is known at time t:

![[Pasted image 20250702002206.png]]

For h>1, we take the conditional expectation. Since ![[Pasted image 20250702002237.png]], the recursive forecast formula becomes:

![[Pasted image 20250702002215.png]]

This shows that each future forecast is a weighted average of the previous forecast and the long-run variance, pulling it towards the mean over time.34

#### Generating and Visualizing Forecasts in Python

The `arch` library provides a straightforward `.forecast()` method to generate these predictions.36

**Python Example: Producing a 30-day Volatility Forecast**

Using the GJR-GARCH model, which was determined to be the best fit for the S&P 500 data, we can generate and visualize a 30-day forecast.



```Python
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np

# Use the same S&P 500 returns data and fitted GJR-GARCH model
gspc_data = yf.download('^GSPC', start='2010-01-01', end='2023-12-31')
returns = 100 * gspc_data['Adj Close'].pct_change().dropna()
gjr_garch = arch_model(returns, vol='GARCH', p=1, o=1, q=1, mean='Constant', dist='normal')
gjr_res = gjr_garch.fit(disp='off')

# Generate a 30-day forecast from the end of the sample
forecast_horizon = 30
forecasts = gjr_res.forecast(horizon=forecast_horizon, start=returns.index[-1])

# Extract the forecasted variance
forecasted_variance = forecasts.variance.iloc[-1]

# Convert variance to annualized volatility
forecasted_volatility = np.sqrt(forecasted_variance) * np.sqrt(252)

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(range(1, forecast_horizon + 1), forecasted_volatility, marker='o', linestyle='--')
plt.title(f'GJR-GARCH {forecast_horizon}-Day Volatility Forecast for S&P 500')
plt.xlabel('Forecast Horizon (Days)')
plt.ylabel('Annualized Volatility (%)')
plt.grid(True)
plt.show()

print("Forecasted Annualized Volatility:")
print(forecasted_volatility)
```

The plot will show the forecasted volatility for the next 30 days. It will typically start near the last observed conditional volatility and then gradually revert towards the model's long-run average level, visually demonstrating the mean-reversion property.

### 3.3.5 Capstone Project: Dynamic Value-at-Risk (VaR) Forecasting and Backtesting

This capstone project integrates the chapter's concepts into a practical risk management application. The objective is to construct and validate a dynamic Value-at-Risk (VaR) model for a notoriously volatile stock, Tesla (TSLA). VaR is a statistical measure that estimates the maximum potential loss a portfolio could face over a specific time horizon at a given confidence level.38 While simple historical VaR models are common, they are static and fail to adapt to changing market conditions. A dynamic VaR, powered by a GARCH model's volatility forecast, can provide a much more realistic and responsive risk measure.40

#### Question 1: Model Selection and Dynamic VaR Calculation

**Task:**

1. Download daily price data for Tesla (TSLA) from January 1, 2018, to the present.
    
2. Calculate log returns and confirm the presence of stylized facts (volatility clustering, fat tails) to justify using a GARCH model.
    
3. Fit GARCH(1,1), GJR-GARCH(1,1), and EGARCH(1,1) models. To better account for the fat tails observed in financial returns, assume a Student's-t distribution for the innovations.40
    
4. Select the best-fitting model based on the AIC or BIC.
    
5. Using the chosen model, implement a rolling one-step-ahead forecast for the conditional volatility over an out-of-sample period (e.g., the last year of data).
    
6. Calculate the corresponding 1-day 95% dynamic VaR for this out-of-sample period.
    

**Response:**

The calculation of GARCH-based VaR combines the forecasted conditional mean and volatility with the appropriate quantile from the assumed distribution of the standardized residuals. The formula for the 1-day VaR at a confidence level of (1−α) is:

![[Pasted image 20250702002257.png]]

where μt+1​ is the one-step-ahead forecast of the conditional mean, σt+1​ is the one-step-ahead forecast of the conditional volatility (the square root of the forecasted variance), and F−1(α) is the inverse cumulative distribution function (CDF), or quantile, of the assumed standardized residual distribution at the α probability level (e.g., 0.05 for a 95% VaR).39

The following Python code implements the entire workflow.



```Python
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t
import matplotlib.pyplot as plt

# 1. Download and prepare data for TSLA
tsla_data = yf.download('TSLA', start='2018-01-01', end='2024-05-31')
returns = 100 * np.log(tsla_data['Adj Close'] / tsla_data['Adj Close'].shift(1)).dropna()

# --- Model Selection (Conceptual - run to determine best model) ---
# garch = arch_model(returns, vol='GARCH', p=1, q=1, dist='t').fit(disp='off')
# gjr = arch_model(returns, vol='GARCH', p=1, o=1, q=1, dist='t').fit(disp='off')
# egarch = arch_model(returns, vol='EGARCH', p=1, q=1, dist='t').fit(disp='off')
# print(f"GARCH AIC: {garch.aic}, GJR-GARCH AIC: {gjr.aic}, EGARCH AIC: {egarch.aic}")
# Based on typical results for equities, we will proceed with GJR-GARCH.

# 2. Implement a rolling window forecast
split_date = '2023-01-01'
train_data = returns[:split_date]
test_data = returns[split_date:]

window_size = len(train_data)
forecasts =

# Loop through the test data
for i in range(len(test_data)):
    current_window = returns.iloc[i : i + window_size]
    
    # Fit the GJR-GARCH model on the rolling window
    model = arch_model(current_window, vol='GARCH', p=1, o=1, q=1, mean='Constant', dist='t')
    res = model.fit(disp='off')
    
    # Forecast one step ahead
    forecast = res.forecast(horizon=1)
    
    # Store the forecast
    forecasts.append(forecast)

# 3. Calculate Dynamic VaR
var_forecasts =
nu = res.params['nu'] # Degrees of freedom from the last model fit
alpha = 0.05
q = t.ppf(alpha, df=nu) # Quantile from Student's-t distribution

for f in forecasts:
    cond_mean = f.mean.iloc
    cond_vol = np.sqrt(f.variance.iloc)
    var = -(cond_mean + cond_vol * q)
    var_forecasts.append(var)

# Create a DataFrame for plotting
var_df = pd.DataFrame({'Returns': test_data, 'VaR_95': var_forecasts}, index=test_data.index)

# 4. Plot the results
plt.figure(figsize=(14, 7))
plt.plot(var_df, label='TSLA Returns', color='blue', alpha=0.7)
plt.plot(var_df, label='95% Dynamic VaR (GJR-GARCH)', color='red', linestyle='--')
plt.title('TSLA Daily Returns and Dynamic 1-Day 95% VaR')
plt.xlabel('Date')
plt.ylabel('Percentage Return')
plt.legend()
plt.show()

# Identify breaches
var_df = var_df < -var_df
print(f"Number of VaR breaches: {var_df.sum()}")
print(f"Total observations in test set: {len(var_df)}")
```

The plot generated by this code will show the daily returns of TSLA for the out-of-sample period. Overlaid on this is the dynamic VaR forecast. It is expected that the red VaR line will widen during periods of high turbulence and narrow during periods of calm, demonstrating its adaptive nature, which is a significant improvement over a static VaR measure.

#### Question 2: Backtesting the VaR Model

**Task:** A VaR model's credibility hinges on its accuracy. We must backtest our dynamic VaR model to ensure it is reliable. Implement two standard regulatory backtests: Kupiec's Unconditional Coverage (POF) test and Christoffersen's Conditional Coverage test.

**Response:**

Backtesting involves systematically comparing the forecasted VaR with the actual profit and loss outcomes to see if the model's performance aligns with its design.43

- **Kupiec's Proportion of Failures (POF) Test**: This test, also known as the unconditional coverage test, checks whether the observed frequency of VaR breaches is statistically consistent with the expected frequency given the confidence level (α).44 For a 95% VaR (
    
    α=0.05), we expect breaches to occur on 5% of the days. The test statistic is a likelihood ratio (LRPOF​) that follows a χ2 distribution with one degree of freedom.
    
- **Christoffersen's Conditional Coverage Test**: This is a more comprehensive test that examines two properties simultaneously: (1) whether the total number of breaches is correct (like the Kupiec test), and (2) whether the breaches are independent of each other.44 A good VaR model should not produce clustered breaches, as this indicates it is too slow to react to changing market conditions. The test statistic (
    
    LRCC​) follows a χ2 distribution with two degrees of freedom.
    

**Python Implementation:**

The following Python code implements both tests from scratch.



```Python
from scipy.stats import chi2

def kupiec_pof_test(breaches, alpha):
    """
    Kupiec's Proportion of Failures (POF) test.
    Checks if the number of VaR breaches is consistent with the confidence level.
    """
    T = len(breaches)
    N = breaches.sum()
    
    if N == 0:
        # Avoid log(0) if no breaches
        return 0.0, 1.0

    pi_hat = N / T
    
    log_likelihood_unrestricted = N * np.log(pi_hat) + (T - N) * np.log(1 - pi_hat)
    log_likelihood_restricted = N * np.log(alpha) + (T - N) * np.log(1 - alpha)
    
    lr_pof = -2 * (log_likelihood_restricted - log_likelihood_unrestricted)
    p_value = 1 - chi2.cdf(lr_pof, df=1)
    
    return lr_pof, p_value

def christoffersen_cc_test(breaches, alpha):
    """
    Christoffersen's Conditional Coverage test.
    Checks for both correct number of breaches and independence of breaches.
    """
    breaches = breaches.astype(int)
    T = len(breaches)
    N = breaches.sum()
    
    if N < 2: # Need at least two breaches to check for clustering
        print("Not enough breaches for Christoffersen test.")
        return np.nan, np.nan, np.nan, np.nan

    # Transition counts
    n00 = ((breaches.shift(1) == 0) & (breaches == 0)).sum()
    n01 = ((breaches.shift(1) == 0) & (breaches == 1)).sum()
    n10 = ((breaches.shift(1) == 1) & (breaches == 0)).sum()
    n11 = ((breaches.shift(1) == 1) & (breaches == 1)).sum()

    # Transition probabilities
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / T

    # Independence Likelihood Ratio (LR_ind)
    log_likelihood_ind_unrestricted = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    log_likelihood_ind_restricted = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    
    lr_ind = -2 * (log_likelihood_ind_restricted - log_likelihood_ind_unrestricted)
    p_value_ind = 1 - chi2.cdf(lr_ind, df=1)
    
    # Conditional Coverage Likelihood Ratio (LR_cc)
    lr_pof, _ = kupiec_pof_test(breaches, alpha)
    lr_cc = lr_pof + lr_ind
    p_value_cc = 1 - chi2.cdf(lr_cc, df=2)
    
    return lr_pof, lr_ind, lr_cc, p_value_cc

# Perform the backtests on the TSLA VaR model
breaches = var_df
alpha = 0.05

lr_pof, p_value_pof = kupiec_pof_test(breaches, alpha)
print("\n--- Kupiec's POF Test ---")
print(f"LR Statistic: {lr_pof:.4f}")
print(f"P-value: {p_value_pof:.4f}")
if p_value_pof > alpha:
    print("Result: Fail to reject H0. The number of breaches is acceptable.")
else:
    print("Result: Reject H0. The number of breaches is not acceptable.")

lr_pof_cc, lr_ind_cc, lr_cc, p_value_cc = christoffersen_cc_test(breaches, alpha)
print("\n--- Christoffersen's Conditional Coverage Test ---")
print(f"LR_ind Statistic: {lr_ind_cc:.4f}")
print(f"LR_cc Statistic: {lr_cc:.4f}")
print(f"P-value: {p_value_cc:.4f}")
if p_value_cc > alpha:
    print("Result: Fail to reject H0. The model provides correct conditional coverage.")
else:
    print("Result: Reject H0. The model fails on conditional coverage (either incorrect breach count or clustering).")
```

**Interpretation:** The output of these tests provides a formal assessment of the VaR model. For a model to be considered reliable, it should ideally pass both tests (i.e., have p-values greater than the significance level, typically 0.05). A high p-value means we fail to reject the null hypothesis of a correctly specified model.

#### Question 3: Practical Implications and Model Refinement

**Task:** From the perspective of a risk manager, discuss the performance of the VaR model based on the backtesting results. If the model were to fail one or both tests, what would be the practical implications and the logical next steps for refinement?

**Response:**

The results of the backtesting have direct and significant practical implications for a financial institution. The choice of a VaR model influences capital requirements, risk limits, and strategic decisions.47 A flawed model can have severe financial consequences.

- **If the Kupiec (POF) Test Fails:** A failure here (p-value < 0.05) means the model is miscalibrated in terms of its unconditional coverage.
    
    - **Implication:** If there are too many breaches, the model is systematically underestimating risk. This is highly dangerous, as it could lead to insufficient capital reserves to cover actual losses, potentially resulting in catastrophic financial distress during a market downturn. If there are too few breaches, the model is systematically overestimating risk. While safer, this is inefficient, as it leads to the firm holding excessive, unproductive capital that could otherwise be deployed to generate profit.
        
    - **Next Steps:** The first step is to re-examine the distributional assumption. Financial returns are known to have fatter tails than a normal distribution. If a normal distribution was used, switching to a Student's-t or a Skewed Student's-t distribution is a logical refinement, as these are better able to capture extreme events.40 One might also consider if the GARCH specification itself is adequate or if a more flexible model is needed.
        
- **If the Christoffersen Test Fails (but Kupiec passes):** This is a more subtle but equally important failure. It implies that while the _total number_ of breaches over the period might be correct, their timing is not random. The breaches are clustered together.
    
    - **Implication:** This indicates that the VaR model is not adapting quickly enough to changes in market conditions. It performs adequately during calm periods but fails to adjust its risk estimate upward fast enough when a volatility cluster begins, leading to a series of consecutive breaches. This is a critical flaw, as a risk model's primary purpose is to protect against losses during periods of market stress, which is precisely when this model is failing.
        
    - **Next Steps:** The issue here lies with the model's dynamic specification. The GARCH parameters (α and β) may not be capturing the true persistence of volatility correctly. The next steps would involve:
        
        1. Re-evaluating the GARCH model order (e.g., trying GARCH(1,2) or GARCH(2,1)).
            
        2. Ensuring an asymmetric model (GJR-GARCH or EGARCH) was used, as the leverage effect is a key driver of volatility dynamics. If a symmetric GARCH was used, switching to an asymmetric one is a necessary step.47
            
        3. Considering more advanced models that can capture different volatility components (e.g., short-term vs. long-term) if the issue persists.
            

In conclusion, VaR modeling is not a one-time exercise of fitting a model. It is an iterative process of specification, forecasting, rigorous backtesting, and refinement. The choice of the underlying volatility model and its distributional assumptions has profound implications for the accuracy and reliability of risk estimates, directly impacting a firm's financial stability and profitability.24

## References
**

1. Implied Volatility vs. Historical Volatility: What's the Difference?, acessado em julho 1, 2025, [https://www.investopedia.com/articles/investing-strategy/071616/implied-vs-historical-volatility-main-differences.asp](https://www.investopedia.com/articles/investing-strategy/071616/implied-vs-historical-volatility-main-differences.asp)
    
2. Implied vs historical volatility: what's the difference? | Fidelity Hong Kong, acessado em julho 1, 2025, [https://www.fidelity.com.hk/en/start-investing/learn-about-investing/what-is-volatility/implied-vs-historical-volatility](https://www.fidelity.com.hk/en/start-investing/learn-about-investing/what-is-volatility/implied-vs-historical-volatility)
    
3. GARCH Models for Volatility Forecasting: A Python-Based Guide | by The AI Quant | Medium, acessado em julho 1, 2025, [https://theaiquant.medium.com/garch-models-for-volatility-forecasting-a-python-based-guide-d48deb5c7d7b](https://theaiquant.medium.com/garch-models-for-volatility-forecasting-a-python-based-guide-d48deb5c7d7b)
    
4. What is an ARCH model? | CQF, acessado em julho 1, 2025, [https://www.cqf.com/blog/quant-finance-101/what-is-an-arch-model](https://www.cqf.com/blog/quant-finance-101/what-is-an-arch-model)
    
5. Implied Volatility vs Historical Volatility Compared - SoFi, acessado em julho 1, 2025, [https://www.sofi.com/learn/content/implied-vs-historical-volatility/](https://www.sofi.com/learn/content/implied-vs-historical-volatility/)
    
6. Stylized Facts - Financial Data - Portfolio Optimization Book, acessado em julho 1, 2025, [https://portfoliooptimizationbook.com/slides/slides-stylized-facts.pdf](https://portfoliooptimizationbook.com/slides/slides-stylized-facts.pdf)
    
7. Volatility And Measures Of Risk-Adjusted Return With Python - QuantInsti Blog, acessado em julho 1, 2025, [https://blog.quantinsti.com/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/](https://blog.quantinsti.com/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/)
    
8. Volatility Calculations in Python; Estimate the Annualized Volatility of Historical Stock Prices, acessado em julho 1, 2025, [https://medium.com/@polanitzer/volatility-calculation-in-python-estimate-the-annualized-volatility-of-historical-stock-prices-db937366a54d](https://medium.com/@polanitzer/volatility-calculation-in-python-estimate-the-annualized-volatility-of-historical-stock-prices-db937366a54d)
    
9. Implied Volatility vs Historical Volatility | Blog - Option Samurai, acessado em julho 1, 2025, [https://optionsamurai.com/blog/implied-volatility-vs-historical-volatility-in-options-trading-unveiling-patterns-and-insights/](https://optionsamurai.com/blog/implied-volatility-vs-historical-volatility-in-options-trading-unveiling-patterns-and-insights/)
    
10. Figure 7. Implied and Historical Volatility in Equity Markets, acessado em julho 1, 2025, [https://www.imf.org/-/media/Websites/IMF/imported-flagship-issues/external/pubs/ft/GFSR/2010/01/sa/_safigure7pdf.ashx](https://www.imf.org/-/media/Websites/IMF/imported-flagship-issues/external/pubs/ft/GFSR/2010/01/sa/_safigure7pdf.ashx)
    
11. Stylized facts in financial time series - Wolfram Cloud, acessado em julho 1, 2025, [https://www.wolframcloud.com/objects/summerschool/pages/2017/CarlosManuelRodriguezMartinez_TE](https://www.wolframcloud.com/objects/summerschool/pages/2017/CarlosManuelRodriguezMartinez_TE)
    
12. 2.1 Stylized facts of returns | Financial econometrics using R, acessado em julho 1, 2025, [https://bookdown.org/jarneric/financial_econometrics/2.1-stylized-facts-of-returns.html](https://bookdown.org/jarneric/financial_econometrics/2.1-stylized-facts-of-returns.html)
    
13. Relationships Between Stylized Facts - Finance, acessado em julho 1, 2025, [https://finance.martinsewell.com/stylized-facts/intra/](https://finance.martinsewell.com/stylized-facts/intra/)
    
14. More stylized facts of financial markets: Leverage effect and downside correlations | Request PDF - ResearchGate, acessado em julho 1, 2025, [https://www.researchgate.net/publication/222574804_More_stylized_facts_of_financial_markets_Leverage_effect_and_downside_correlations](https://www.researchgate.net/publication/222574804_More_stylized_facts_of_financial_markets_Leverage_effect_and_downside_correlations)
    
15. More stylized facts of financial markets: leverage effect and downside correlations - CFM, acessado em julho 1, 2025, [https://www.cfm.com/wp-content/uploads/2022/12/200-2001-more-stylized-facts-of-financial-markets-leverage-effect-and-downside-correlations.pdf](https://www.cfm.com/wp-content/uploads/2022/12/200-2001-more-stylized-facts-of-financial-markets-leverage-effect-and-downside-correlations.pdf)
    
16. Autoregressive Conditional Heteroskedasticity (ARCH) Explained - Investopedia, acessado em julho 1, 2025, [https://www.investopedia.com/terms/a/autoregressive-conditional-heteroskedasticity.asp](https://www.investopedia.com/terms/a/autoregressive-conditional-heteroskedasticity.asp)
    
17. Autoregressive conditional heteroskedasticity - Wikipedia, acessado em julho 1, 2025, [https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity)
    
18. arch - PyPI, acessado em julho 1, 2025, [https://pypi.org/project/arch/](https://pypi.org/project/arch/)
    
19. Building A GARCH(1,1) Model in Python, Step by Step | by Roi Polanitzer - Medium, acessado em julho 1, 2025, [https://medium.com/@polanitzer/building-a-garch-1-1-model-in-python-step-by-step-f8503e868efa](https://medium.com/@polanitzer/building-a-garch-1-1-model-in-python-step-by-step-f8503e868efa)
    
20. Time Series Analysis, Lecture 24: The GARCH Process - YouTube, acessado em julho 1, 2025, [https://www.youtube.com/watch?v=Nmy6IOHF4ho](https://www.youtube.com/watch?v=Nmy6IOHF4ho)
    
21. GARCH Time Series Models: - An Application to Retail Livestock Prices, acessado em julho 1, 2025, [https://www.card.iastate.edu/products/publications/pdf/88wp29.pdf](https://www.card.iastate.edu/products/publications/pdf/88wp29.pdf)
    
22. Autoregressive conditional heteroskedasticity - Wikipedia, acessado em julho 1, 2025, [https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity#GARCH](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity#GARCH)
    
23. Mastering GARCH Models in Time Series - Number Analytics, acessado em julho 1, 2025, [https://www.numberanalytics.com/blog/mastering-garch-models-time-series](https://www.numberanalytics.com/blog/mastering-garch-models-time-series)
    
24. An Introduction to the Use of ARCH/GARCH models in Applied Econometrics - NYU Stern, acessado em julho 1, 2025, [https://www.stern.nyu.edu/rengle/GARCH101.PDF](https://www.stern.nyu.edu/rengle/GARCH101.PDF)
    
25. GARCH vs. GJR-GARCH Models in Python for Volatility Forecasting - QuantInsti Blog, acessado em julho 1, 2025, [https://blog.quantinsti.com/garch-gjr-garch-volatility-forecasting-python/](https://blog.quantinsti.com/garch-gjr-garch-volatility-forecasting-python/)
    
26. Advanced GARCH Models: EGARCH and GJR-GARCH for Power and Gas Futures Volatility | by Jonathan | Medium, acessado em julho 1, 2025, [https://medium.com/@jlevi.nyc/advanced-garch-models-egarch-and-gjr-garch-for-power-and-gas-futures-volatility-c36446a62d14](https://medium.com/@jlevi.nyc/advanced-garch-models-egarch-and-gjr-garch-for-power-and-gas-futures-volatility-c36446a62d14)
    
27. Volatility models for asymmetric shocks | Python, acessado em julho 1, 2025, [https://campus.datacamp.com/courses/garch-models-in-python/garch-model-configuration?ex=8](https://campus.datacamp.com/courses/garch-models-in-python/garch-model-configuration?ex=8)
    
28. Comparison of the GARCH, EGARCH, GJR-GARCH and TGARCH model in times of crisis for the S&P500, NASDAQ and Dow-Jones, acessado em julho 1, 2025, [https://thesis.eur.nl/pub/59759/Thesis-Misha-Dol-final-version.pdf](https://thesis.eur.nl/pub/59759/Thesis-Misha-Dol-final-version.pdf)
    
29. Exponential GARCH Volatility Documentation - V-Lab, acessado em julho 1, 2025, [https://vlab.stern.nyu.edu/docs/volatility/EGARCH](https://vlab.stern.nyu.edu/docs/volatility/EGARCH)
    
30. The Ultimate Guide to EGARCH Modeling in Finance - Number Analytics, acessado em julho 1, 2025, [https://www.numberanalytics.com/blog/ultimate-egarch-modeling-finance](https://www.numberanalytics.com/blog/ultimate-egarch-modeling-finance)
    
31. Specify EGARCH Models - MATLAB & Simulink - MathWorks, acessado em julho 1, 2025, [https://www.mathworks.com/help/econ/specify-egarch-models-using-egarch.html](https://www.mathworks.com/help/econ/specify-egarch-models-using-egarch.html)
    
32. ARCH, GARCH, EGARCH. How to measure volatility in equity… | by Terrill Toe - Medium, acessado em julho 1, 2025, [https://medium.com/@NNGCap/arch-garch-egarch-92dd7277a966](https://medium.com/@NNGCap/arch-garch-egarch-92dd7277a966)
    
33. Optimal Multi-Step-Ahead Prediction of ARCH/GARCH Models and NoVaS Transformation, acessado em julho 1, 2025, [https://www.mdpi.com/2225-1146/7/3/34](https://www.mdpi.com/2225-1146/7/3/34)
    
34. Volatility Forecasting: GARCH(1,1) Model - Portfolio Optimizer, acessado em julho 1, 2025, [https://portfoliooptimizer.io/blog/volatility-forecasting-garch11-model/](https://portfoliooptimizer.io/blog/volatility-forecasting-garch11-model/)
    
35. Multistep ahead forecasts in GARCH equations - Quantitative Finance Stack Exchange, acessado em julho 1, 2025, [https://quant.stackexchange.com/questions/73540/multistep-ahead-forecasts-in-garch-equations](https://quant.stackexchange.com/questions/73540/multistep-ahead-forecasts-in-garch-equations)
    
36. Volatility Forecasting - arch 7.2.0, acessado em julho 1, 2025, [https://arch.readthedocs.io/en/stable/univariate/univariate_volatility_forecasting.html](https://arch.readthedocs.io/en/stable/univariate/univariate_volatility_forecasting.html)
    
37. GARCH Modeling in Python: Building Volatility Forecasts ..., acessado em julho 1, 2025, [https://fastercapital.com/content/GARCH-Modeling-in-Python--Building-Volatility-Forecasts.html](https://fastercapital.com/content/GARCH-Modeling-in-Python--Building-Volatility-Forecasts.html)
    
38. Forecasting Value-at-Risk using GARCH and Extreme-Value-Theory Approaches for Daily Returns - Scientific & Academic Publishing, acessado em julho 1, 2025, [http://article.sapub.org/10.5923.j.statistics.20170702.10.html](http://article.sapub.org/10.5923.j.statistics.20170702.10.html)
    
39. VaR: Value at Risk: Estimating VaR with GARCH: A Comprehensive ..., acessado em julho 1, 2025, [https://fastercapital.com/content/VaR--Value-at-Risk---Estimating-VaR-with-GARCH--A-Comprehensive-Guide.html](https://fastercapital.com/content/VaR--Value-at-Risk---Estimating-VaR-with-GARCH--A-Comprehensive-Guide.html)
    
40. (PDF) The use of GARCH models in VaR estimation - ResearchGate, acessado em julho 1, 2025, [https://www.researchgate.net/publication/222531664_The_use_of_GARCH_models_in_VaR_estimation](https://www.researchgate.net/publication/222531664_The_use_of_GARCH_models_in_VaR_estimation)
    
41. Application of GARCH Type Models in Forecasting Value at Risk - Scholarship at UWindsor, acessado em julho 1, 2025, [https://scholar.uwindsor.ca/cgi/viewcontent.cgi?article=1200&context=major-papers](https://scholar.uwindsor.ca/cgi/viewcontent.cgi?article=1200&context=major-papers)
    
42. VaR in financial risk management | Python, acessado em julho 1, 2025, [https://campus.datacamp.com/courses/garch-models-in-python/garch-in-action?ex=1](https://campus.datacamp.com/courses/garch-models-in-python/garch-in-action?ex=1)
    
43. (PDF) Backtesting Value at Risk Forecast: the Case of Kupiec Pof-Test - ResearchGate, acessado em julho 1, 2025, [https://www.researchgate.net/publication/308899080_Backtesting_Value_at_Risk_Forecast_the_Case_of_Kupiec_Pof-Test](https://www.researchgate.net/publication/308899080_Backtesting_Value_at_Risk_Forecast_the_Case_of_Kupiec_Pof-Test)
    
44. Overview of VaR Backtesting - MATLAB & - MathWorks, acessado em julho 1, 2025, [https://www.mathworks.com/help/risk/overview-of-var-backtesting.html](https://www.mathworks.com/help/risk/overview-of-var-backtesting.html)
    
45. Advanced Backtesting, Stress Testing & CVaR in VaR - Number Analytics, acessado em julho 1, 2025, [https://www.numberanalytics.com/blog/advanced-var-backtesting-stress-cvar](https://www.numberanalytics.com/blog/advanced-var-backtesting-stress-cvar)
    
46. 14.5 Backtesting With Independence Tests - Value-at-risk.net, acessado em julho 1, 2025, [https://www.value-at-risk.net/backtesting-independence-tests/](https://www.value-at-risk.net/backtesting-independence-tests/)
    
47. Which GARCH model is best for Value-at-Risk? - DiVA portal, acessado em julho 1, 2025, [https://www.diva-portal.org/smash/get/diva2:788857/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:788857/FULLTEXT01.pdf)
    

**