# Machine Learning for Algorithmic Trading: Feature Engineering for Financial Machine Learning Models

## Introduction: The Art and Science of Feature Creation

In the domain of quantitative finance, algorithmic trading strategies are driven by signals that aim to generate returns uncorrelated with the broader market—an excess return often referred to as _alpha_. The search for alpha is the modern quant's equivalent of the quest for the Holy Grail. While many newcomers to the field focus on deploying increasingly complex machine learning models, seasoned practitioners understand a more fundamental truth: the model is secondary to the features it is fed. The quality of features, not the complexity of the algorithm, is the primary determinant of a strategy's success.1

This chapter is dedicated to the art and science of feature engineering, the critical process of transforming raw, often chaotic, market data into the structured, predictive signals that power machine learning models.3 We will frame this journey as one of manufacturing: taking the raw materials of price and volume and, through a meticulous process of transformation and refinement, producing high-grade inputs ready for a predictive engine. This process is both an art, requiring domain knowledge and creativity to hypothesize new sources of alpha, and a science, demanding rigorous statistical validation to ensure that what we've found is a genuine signal, not just noise.3

The path ahead is structured to build your expertise systematically. We will begin by confronting the formidable, unique challenges that financial time series data present, namely its low signal-to-noise ratio and pervasive non-stationarity. Understanding these obstacles is a prerequisite for success. From there, we will construct a foundational toolkit of features derived from price and volume, including returns, volatility measures, and classic technical indicators. We will then advance to state-of-the-art techniques, such as fractional differentiation, designed to solve the core dilemmas of financial data. Finally, because creating features is only half the battle, we will cover the essential methods for evaluating and selecting the most potent features. The chapter culminates in a comprehensive capstone project, where we will apply all these concepts to build and analyze a predictive model from scratch, providing a complete template for your own quantitative research.

---

## Section 1: The Unique Challenges of Financial Time Series

Before one can engineer effective features, one must develop a deep appreciation for the hostile environment in which financial data exists. Unlike the clean, well-behaved datasets often found in other machine learning domains, financial time series are notoriously difficult to work with. Their properties violate the core assumptions of many standard models. Failing to understand and address these challenges is the most common reason that quantitative strategies which look brilliant on paper fail spectacularly in the real world. This section details the two most critical of these challenges: the exceptionally low signal-to-noise ratio and the pervasive nature of non-stationarity.

### 1.1 The Low Signal-to-Noise Ratio (SNR)

In any data-driven field, the Signal-to-Noise Ratio (SNR) measures the strength of the meaningful, predictive information (the "signal") relative to the random, unpredictable fluctuations (the "noise").5 Financial markets are infamous for having one of the lowest signal-to-noise ratios of any domain where machine learning is applied.6 Disentangling the two is the fundamental challenge of quantitative trading.

- **Sources of Signal:** The "signal" in financial markets consists of the predictable patterns and fundamental drivers that influence asset prices over a meaningful horizon. Examples include:
    
    - **Economic Fundamentals:** Consistent earnings growth, macroeconomic indicators like GDP or employment data, and shifts in interest rate policy.8
        
    - **Structural Market Effects:** Persistent behavioral biases of investors, risk premia associated with factors like value or momentum, and predictable liquidity patterns.
        
    - **Information Flow:** The gradual pricing-in of new information from well-researched analyst reports or company filings.8
        
- **Sources of Noise:** The "noise" is the overwhelming sea of random price movements that can obscure the underlying signal. This includes:
    
    - **Stochastic Nature of Markets:** The inherent randomness resulting from millions of independent agents buying and selling for a multitude of reasons.9
        
    - **High-Frequency Trading:** The activity of algorithms that operate on microsecond timescales, adding volatility that is irrelevant to lower-frequency strategies.2
        
    - **Unstructured Information:** The constant barrage of news headlines, social media sentiment, unsubstantiated rumors, and geopolitical events that cause short-term emotional reactions but may have no lasting fundamental impact.8
        

The primary implication of a low SNR is the profound risk of **overfitting**. A machine learning model, if not carefully constrained, will readily fit its parameters to the noise in a historical dataset. It will "discover" intricate patterns in the randomness of the past that provide no predictive power for the future. This leads to the classic quant failure: a backtest with a stellar Sharpe ratio that, when deployed with real capital, bleeds money. A famous study by Professor Paul Slovic gave professional gamblers progressively more data points for horse races. He found that while their confidence in their predictions increased linearly with more data, their accuracy plateaued after the first few data points.11 This is a perfect analogy for a model overfitting to noise; it becomes more confident but no more accurate.

This challenge reframes the entire purpose of feature engineering. It is not simply a data transformation exercise; it is a **signal processing** task. The primary goal of a well-designed feature is to increase the signal-to-noise ratio of the inputs to the model. A simple moving average, for example, is fundamentally a low-pass filter designed to smooth out high-frequency price noise to make the underlying trend (the signal) more apparent.3 The advanced techniques we will discuss later are, at their core, sophisticated methods for amplifying signal while attenuating noise. Every step in this chapter is an attempt to solve the SNR problem.

### 1.2 The Pervasiveness of Non-Stationarity

A time series is said to be **stationary** if its core statistical properties—most importantly its mean and variance—remain constant over time.12 This is a critical, often implicit, assumption for a vast array of statistical and machine learning models. A model trained on data with a certain mean and volatility expects to see similar characteristics in the future. When these properties change, the model's predictive power breaks down.

Financial price series are a textbook example of **non-stationary** data.14 They exhibit clear trends (the mean is not constant), periods of high and low volatility (the variance is not constant), and often follow a "random walk" pattern, where the best prediction for tomorrow's price is simply today's price.16 This non-stationarity is not just a minor inconvenience; it is a profound statistical trap. Attempting to regress one non-stationary series on another can produce a

**spurious regression**: a model with a high R-squared and statistically significant coefficients that describes a relationship that is entirely false and has no predictive power.14 Recent research has shown that this remains a major impediment even for the latest deep learning architectures like Transformers when applied to financial forecasting.17

Practitioners can identify non-stationarity through visual inspection of a time plot or, more formally, using statistical tests like the Augmented Dickey-Fuller (ADF) test. The ADF test's null hypothesis is that the series possesses a unit root (a formal characteristic of non-stationarity). A low p-value allows us to reject the null and conclude the series is stationary.22

This reality presents what Dr. Marcos López de Prado has termed the **Stationarity vs. Memory Dilemma**.24 This is a fundamental trade-off at the very heart of financial feature design.

1. **The Problem:** Raw asset prices are non-stationary and thus unsuitable for most models.
    
2. **The Standard Solution:** For decades, the standard approach has been to compute returns by taking the first difference of the price series (or the log-price series). This transformation, known as differencing, typically renders the series stationary.16
    
3. **The Hidden Cost:** While solving the stationarity problem, this act of integer differencing effectively "wipes out" the memory of the series.15 All information about the price level, the long-term trend, and other long-range dependencies is discarded. The resulting returns series is stationary, but it is also largely memoryless.
    
4. **The Dilemma:** The quantitative researcher is thus caught between two poor choices. They can use the original non-stationary price data and risk building a spurious, worthless model. Or, they can use the stationary returns data but forfeit the valuable predictive information contained in the series' memory.
    

Resolving this dilemma is a central theme of modern financial machine learning. It requires moving beyond simple differencing to more sophisticated techniques that can achieve stationarity while preserving as much memory as possible. This will be the focus of Section 3.

---

## Section 2: Foundational Features from Market Data

With a clear understanding of the challenges, we can now begin building our feature engineering toolkit. This section focuses on the foundational layer of features, those derived directly from raw Open, High, Low, Close, and Volume (OHLCV) data. These are the workhorses of quantitative analysis, forming the basis for countless trading strategies. For each feature, we will provide the mathematical definition, its intended purpose, and a practical Python implementation.

### 2.1 Returns: The Bedrock of Financial Analysis

The most fundamental transformation in finance is converting a series of prices, which are non-stationary, into a series of returns, which are typically more stationary. However, the choice of how to calculate returns is not trivial and has important implications for modeling.

#### **Simple Returns**

The simple, or arithmetic, return is the most intuitive measure of price change. It is calculated as the percentage change in price over a single period.

- Formula: The simple return Rt​ at time t for a price series P is given by:
    
    ![[Pasted image 20250702003458.png]]
- **Properties and Use Cases:** Simple returns are easy to understand and communicate.26 Their most important property is that they are
    
    **asset-additive**. This means the return of a portfolio is the simple weighted average of the simple returns of its constituent assets. This makes them the correct choice for any cross-sectional analysis or for calculating portfolio-level returns.27 Their main drawback is that they are not time-additive. The cumulative return over
    
    N periods is not the sum of the individual returns, but rather their geometric product: (1+R1​)(1+R2​)...(1+RN​)−1.26
    

#### **Logarithmic Returns**

Logarithmic, or continuously compounded, returns offer significant mathematical advantages for time series analysis.

- Formula: The log return rt​ at time t is the natural logarithm of the ratio of consecutive prices:
    
    ![[Pasted image 20250702003510.png]]
- **Properties and Use Cases:** The primary advantage of log returns is that they are **time-additive**. The cumulative log return over N periods is simply the sum of the individual single-period log returns: ∑i=1N​ri​.28 This makes them exceptionally convenient for statistical modeling over time. Furthermore, log returns are more likely to be normally distributed than simple returns, a desirable property for many financial models that assume normality.29 Their main drawback is that they are not asset-additive; the log return of a portfolio is not the weighted average of the log returns of its assets.27
    

#### **Python Implementation**

Calculating both types of returns is straightforward in Python using the `pandas` and `numpy` libraries.



```Python
import pandas as pd
import numpy as np
import yfinance as yf

# Fetch some sample data
data = yf.download('AAPL', start='2022-01-01', end='2023-01-01')

# Calculate Simple Returns
data['simple_return'] = data['Adj Close'].pct_change()

# Calculate Log Returns
data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))

print("Simple vs. Log Returns for AAPL:")
print(data[['Adj Close', 'simple_return', 'log_return']].tail())
```

#### **Table 2.1: Comparison of Simple and Logarithmic Returns**

The choice between simple and log returns is a frequent point of confusion for new practitioners. The following table provides a clear, at-a-glance reference to guide this decision, distilling the key differences into an actionable tool.26

|Property|Simple Return|Logarithmic Return|
|---|---|---|
|**Formula**|Rt​=Pt−1​Pt​​−1|rt​=ln(Pt−1​Pt​​)|
|**Time Aggregation**|Not additive (multiplicative)|Additive (rt,t+N​=∑i=1N​rt+i​)|
|**Cross-Sectional Aggregation**|Additive (Rp​=∑wi​Ri​)|Not additive|
|**Statistical Distribution**|Tends to be skewed|Tends to be more normally distributed|
|**Best For...**|Portfolio-level analysis, cross-sectional models, short-term analysis|Time-series modeling, long-term performance analysis, statistical inference|

### 2.2 Time-Series Derived Features

Beyond single-period returns, we can create a rich set of features by analyzing the time-series properties of price and return data. These features explicitly provide the model with historical context.

- **Lag Features:** The simplest way to incorporate history is to use past values of a series as features for the present. For example, the log returns from one, two, and five days ago can be used as predictors for today's return. This allows the model to learn temporal patterns directly, such as momentum (positive correlation with past returns) or mean-reversion (negative correlation with past returns).3 In
    
    `pandas`, this is easily achieved with the `.shift()` method.
    
- **Rolling Window Features:** These features are statistics calculated over a moving "window" of recent data. They are powerful tools for smoothing out short-term noise and capturing the evolution of market dynamics over time.3
    
    - **Moving Averages:** These are designed to smooth price data to better identify the underlying trend. The **Simple Moving Average (SMA)** gives equal weight to all prices in the window. The **Exponential Moving Average (EMA)** gives more weight to recent prices, making it more responsive to new information.3
        
    - **Rolling Volatility:** Volatility, a measure of risk and price dispersion, is a crucial input for many strategies. It is typically calculated as the standard deviation of returns over a rolling window. To make it comparable across different time frames, daily volatility is often annualized by multiplying it by the square root of the number of trading days in a year (typically 252).32 This feature allows a model to understand the current risk environment and adapt its predictions accordingly.
        

#### **Python Implementation**



```Python
# Assuming 'data' DataFrame from the previous example with 'log_return' column

# Lag Features
for lag in :
    data[f'log_return_lag_{lag}'] = data['log_return'].shift(lag)

# Rolling Window Features (using a 21-day window, approx. 1 trading month)
window_size = 21

# Simple Moving Average (SMA)
data['sma_21'] = data['Adj Close'].rolling(window=window_size).mean()

# Exponential Moving Average (EMA)
data['ema_21'] = data['Adj Close'].ewm(span=window_size, adjust=False).mean()

# Rolling Volatility (annualized)
data['volatility_21'] = data['log_return'].rolling(window=window_size).std() * np.sqrt(252)

print("\nTime-Series Derived Features:")
print(data[['Adj Close', 'log_return_lag_1', 'sma_21', 'ema_21', 'volatility_21']].tail())
```

### 2.3 A Tour of Classical Technical Indicators

Technical indicators are, in essence, pre-packaged feature engineering formulas that have been developed and used by traders for decades.3 They transform raw price and volume data into oscillators or trend-following signals, providing a standardized way to interpret market psychology and momentum. While there are hundreds of indicators, we will focus on three of the most fundamental and widely used.

- **Momentum Oscillator: Relative Strength Index (RSI)**
    
    - **Logic:** The RSI is a momentum oscillator that measures the speed and change of price movements on a scale of 0 to 100. It compares the magnitude of recent gains to recent losses over a specified time period (typically 14 days) to determine overbought and oversold conditions.34 An RSI reading above 70 is traditionally considered overbought (a potential sell signal), while a reading below 30 is considered oversold (a potential buy signal).37
        
    - **Formula:** The calculation involves two steps:
        
        1. ![[Pasted image 20250702003607.png]]
            
        2. ![[Pasted image 20250702003615.png]]
            
- **Trend-Following Indicator: Moving Average Convergence Divergence (MACD)**
    
    - **Logic:** The MACD is a trend-following momentum indicator that shows the relationship between two exponential moving averages of a security’s price.38 It is designed to reveal changes in the strength, direction, momentum, and duration of a trend.
        
    - **Formula:** It consists of three components 40:
        
        1. **MACD Line:** (12-period EMA)−(26-period EMA)
            
        2. **Signal Line:** 9-period EMA of the MACD Line
            
        3. Histogram: MACDLine−SignalLine
            
            Trading signals are often generated when the MACD line crosses above the signal line (bullish) or below it (bearish).
            
- **Volatility Channels: Bollinger Bands®**
    
    - **Logic:** Bollinger Bands consist of a moving average plus and minus a measure of volatility. Because the bands are based on standard deviation, they widen during periods of high volatility and contract during periods of low volatility.42 They provide a relative definition of "high" and "low" prices, where prices are considered high at the upper band and low at the lower band.43
        
    - **Formula:** The components are typically calculated as follows 42:
        
        1. **Middle Band:** 20-period SMA
            
        2. **Upper Band:** 20-period SMA+(2×20-period Standard Deviation)
            
        3. **Lower Band:** 20-period SMA−(2×20-period Standard Deviation)
            

#### **Python Implementation of Technical Indicators**

While these indicators can be calculated manually, it is often more efficient and less error-prone to use a dedicated library like `pandas_ta` or `ta`.



```Python
#!pip install pandas_ta
import pandas_ta as ta

# Add all indicators to the DataFrame
# The ta.Strategy class allows for easy bulk application
MyStrategy = ta.Strategy(
    name="Core Indicators",
    description="RSI, MACD, and Bollinger Bands",
    ta=[
        {"kind": "rsi", "length": 14},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        {"kind": "bbands", "length": 20, "std": 2},
    ]
)

# Run the strategy
data.ta.strategy(MyStrategy)

# Display the new features
print("\nClassical Technical Indicators:")
print(data].tail())
```

#### **Table 2.2: Summary of Core Technical Indicators**

For newcomers, the sheer number of available indicators can be overwhelming.9 This table provides a structured overview, categorizing the three core indicators by their primary function to help build a mental framework for feature construction.

|Indicator|Category|Primary Use|Key Parameters|
|---|---|---|---|
|**RSI**|Momentum|Identifying overbought/oversold conditions|Lookback period (e.g., 14)|
|**MACD**|Trend|Identifying trend direction and momentum shifts|Fast EMA (12), Slow EMA (26), Signal EMA (9)|
|**Bollinger Bands**|Volatility|Identifying relative price levels and volatility regimes|Lookback period (20), Std. Dev. multiplier (2)|

---

## Section 3: Advanced Features for Stationarity and Memory

This section confronts the "Stationarity vs. Memory Dilemma" head-on. The foundational features from the previous section, while useful, are often calculated on price series that are either non-stationary (raw prices) or memoryless (simple returns). Here, we introduce an advanced technique, fractional differentiation, which aims to create features that are both statistically well-behaved (stationary) and predictive (memory-preserving).

### 3.1 Revisiting the Dilemma: The Flaw in Integer Differencing

As established in Section 1, there is a fundamental trade-off in preparing financial time series for machine learning. Standard integer differencing—calculating returns via `price.diff(1)`—is the textbook method for achieving stationarity. However, this comes at a steep price: the complete removal of the series' memory.15 Any information about the long-term price trend or level is erased. For a predictive model, this memory is not noise; it is often the very signal we wish to capture. This loss of information represents a significant, self-imposed handicap on the performance of any financial machine learning model.46 The challenge, therefore, is to make a series stationary while destroying as little memory as possible.

### 3.2 Fractional Differentiation: The Best of Both Worlds

Fractional differentiation, a concept popularized in quantitative finance by Dr. Marcos López de Prado, offers an elegant solution to this dilemma.24 It generalizes the concept of differencing by allowing the differencing amount, denoted by

d, to be any real number, not just an integer.23 The core idea is to find the

**minimum** value of d that is sufficient to make a price series stationary. By applying the minimum necessary transformation, we preserve the maximum possible amount of the original series' memory.15

#### **Mathematical Formulation**

The method can be understood using the backshift operator, B, where BXt​=Xt−1​. The standard first-difference is (1−B)1Xt​=Xt​−Xt−1​. Fractional differentiation extends this to (1−B)dXt​, where d∈R. This expression can be expanded as an infinite series of weights applied to all past values of the time series 47:

![[Pasted image 20250702003636.png]]

The weights ωk​ can be computed iteratively, which is far more practical than calculating the binomial expansion for large k 47:

![[Pasted image 20250702003643.png]]

When d=1, the weights become ω0​=1, ω1​=−1, and ωk​=0 for all k>1, which recovers the standard first difference. However, for a fractional d between 0 and 1, the weights decay slowly to zero, meaning that the transformed series at time t is a weighted sum of all past prices, with memory fading over time but never being completely erased after a single step.

#### **The Search for the Optimal `d`**

The practical application of fractional differentiation involves a search procedure to find the optimal differencing amount d∗ 15:

1. Define a range of d values to test, for example, from 0 to 1 in increments of 0.01.
    
2. For each d in this range, apply fractional differentiation to the raw price series.
    
3. For each resulting fractionally differentiated series, conduct a stationarity test, such as the Augmented Dickey-Fuller (ADF) test.
    
4. The optimal d∗ is the smallest value of d for which the ADF test statistic passes the desired significance threshold (e.g., the p-value is less than 0.05).
    

This procedure ensures that we have found a stationary series that has the highest possible correlation with the original price series, thereby maximizing memory preservation.

#### **Python Implementation**

Several Python libraries, such as `fracdiff` and `mlfinlab`, provide efficient implementations of fractional differentiation. The following code demonstrates the search for an optimal d using the `fracdiff` library and `statsmodels` for the ADF test.



```Python
#!pip install fracdiff
import pandas as pd
import numpy as np
import yfinance as yf
from fracdiff import fdiff
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# 1. Fetch data
spy_prices = yf.download('SPY', start='2010-01-01', end='2023-12-31')['Adj Close']

# 2. Search for optimal d
d_values = np.linspace(0, 1, 101)
adf_stats =
p_values =

for d in d_values:
    # Apply fractional differentiation
    # Note: fdiff returns a numpy array, we drop NaNs
    differentiated_series = fdiff(spy_prices.values, n=d)
    differentiated_series = differentiated_series[~np.isnan(differentiated_series)]
    
    # Perform ADF test
    adf_result = adfuller(differentiated_series)
    adf_stats.append(adf_result)
    p_values.append(adf_result)

# 3. Find optimal d
# Get the ADF 5% critical value for comparison
adf_critical_value = adfuller(spy_prices.pct_change().dropna())['5%']

optimal_d = None
for i, p_val in enumerate(p_values):
    if p_val < 0.05:
        optimal_d = d_values[i]
        print(f"Optimal d found: {optimal_d:.2f} (p-value: {p_val:.4f})")
        break

# 4. Plot the results
plt.figure(figsize=(12, 6))
plt.plot(d_values, adf_stats, label='ADF Statistic')
plt.axhline(y=adf_critical_value, color='r', linestyle='--', label='ADF 5% Critical Value')
if optimal_d is not None:
    plt.axvline(x=optimal_d, color='g', linestyle='--', label=f'Optimal d ≈ {optimal_d:.2f}')
plt.title('ADF Statistic vs. Fractional Differentiation Order (d)')
plt.xlabel('Order of Differentiation (d)')
plt.ylabel('ADF Statistic')
plt.legend()
plt.grid(True)
plt.show()

# 5. Create the final feature
if optimal_d is not None:
    spy_fracdiff = fdiff(spy_prices.values, n=optimal_d)
    spy_prices_df = spy_prices.to_frame()
    spy_prices_df['fracdiff_price'] = spy_fracdiff
    print("\nOriginal vs. Fractionally Differentiated Series:")
    print(spy_prices_df.tail())
```

The true power of this technique extends beyond merely creating a single stationary price feature. It serves as a **universal feature preprocessor**. Standard features like moving averages or volatility are often calculated on the raw, non-stationary price series, causing them to inherit its undesirable statistical properties. A more robust approach is to first generate the optimal fractionally differentiated price series. Then, one can calculate features like moving averages, volatility, and others _from this new stationary series_. This creates an entire suite of features that are both statistically sound and rich in memory, providing a far superior input set for any machine learning algorithm. This methodology, while not always explicitly stated, is a cornerstone of the modern financial machine learning paradigm advocated by researchers like de Prado.24

---

## Section 4: Feature Evaluation and Selection

The process of generating features, whether foundational or advanced, can quickly lead to a dataset with dozens or even hundreds of potential predictors. Throwing all of them into a model is a recipe for failure. It can lead to overfitting, unstable models, and results that are impossible to interpret. This section covers the critical subsequent step: evaluating the features we have created. As Dr. de Prado famously stated, "Backtesting is not a research tool—feature importance is".50 We will focus on two essential evaluation tasks: identifying and mitigating redundancy (collinearity) and measuring relevance (feature importance).

### 4.1 The Curse of Collinearity

Many of the features we generate will be derived from the same underlying price series, making them inherently related. **Multicollinearity** is the statistical term for when two or more independent variables (features) in a model are highly correlated with each other.51 This is problematic because it can make the coefficient estimates of a model highly unstable and difficult to interpret. The model may struggle to disentangle the individual effect of each correlated feature, assigning large positive weight to one and a large negative weight to the other, even when both are capturing similar information. While this may not always harm the model's overall predictive accuracy, it makes it nearly impossible to understand the economic drivers of the strategy.

#### **Detection Methods**

Two common methods are used to detect multicollinearity:

1. **Correlation Matrix and Heatmap:** This is the simplest approach. By calculating the Pearson correlation coefficient for every pair of features, we can create a matrix that quantifies their linear relationships. A heatmap provides a quick visual tool to spot pairs with high positive or negative correlations (values close to +1 or -1).53
    
2. **Variance Inflation Factor (VIF):** VIF is a more comprehensive and robust metric. For each feature, VIF measures how much the variance of its estimated regression coefficient is "inflated" because of its correlation with _all other features_ in the model.51 It is calculated by taking a feature, regressing it against all other features, and then using the R-squared from that regression.
    
    - **Formula:** For a feature i, the VIF is:
        
        ![[Pasted image 20250702003706.png]]
        
        where Ri2​ is the R-squared from regressing feature i on all other features.
        
    - **Interpretation:** A VIF of 1 indicates no correlation. A VIF between 1 and 5 suggests moderate correlation. A common rule of thumb is that a **VIF value greater than 5 or 10 indicates problematic multicollinearity** that should be addressed.51
        

#### **Python Implementation and Mitigation**

The `statsmodels` library provides a direct function for calculating VIF. A common mitigation strategy is to calculate the VIF for all features and iteratively remove the feature with the highest VIF, recalculating the VIFs at each step, until all remaining features are below the desired threshold.52



```Python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def calculate_vif(X: pd.DataFrame):
    """
    Calculates the Variance Inflation Factor (VIF) for each feature in a DataFrame.
    """
    # Add a constant for the VIF calculation
    X_const = add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i + 1) for i in range(X.shape)]
    
    return vif_data.sort_values('VIF', ascending=False)

# Example usage with some hypothetical features
# In a real scenario, X would be your full feature DataFrame
hypothetical_features = pd.DataFrame({
    'sma_10': data['Adj Close'].rolling(10).mean(),
    'sma_20': data['Adj Close'].rolling(20).mean(), # Highly correlated with sma_10
    'rsi_14': data,
    'vol_21': data['volatility_21']
}).dropna()

vif_results = calculate_vif(hypothetical_features)
print("VIF Results:")
print(vif_results)

# Mitigation: Remove the feature with the highest VIF and recalculate
features_after_removal = hypothetical_features.drop('sma_20', axis=1)
vif_after_removal = calculate_vif(features_after_removal)
print("\nVIF Results after removing 'sma_20':")
print(vif_after_removal)
```

### 4.2 Model-Based Feature Importance

After handling redundancy, the next step is to determine which features are most relevant for prediction. Feature importance techniques use a trained model to score each feature based on its contribution to the model's performance.56

#### **Mean Decrease Impurity (MDI)**

This is the most common feature importance method for tree-based ensembles like Random Forest and Gradient Boosting. It is readily available as the `.feature_importances_` attribute in `scikit-learn` models.

- **How it Works:** MDI measures a feature's importance by calculating the total reduction in node impurity (typically Gini impurity or entropy) it provides when used as a split point, averaged across all trees in the forest.58 A feature that consistently creates "purer" child nodes is deemed more important.
    
- **The Bias Problem:** While fast and simple, MDI is notoriously flawed and should be used with extreme caution. Its primary issue is a strong **bias towards high-cardinality features**. This means it tends to inflate the importance of continuous variables or categorical variables with many levels, regardless of their true predictive power.60 Furthermore, if the model has overfit the training data, MDI will reflect the feature's importance on that noisy training set, not its ability to generalize to new data.
    

#### **Permutation Importance (Mean Decrease Accuracy)**

Permutation importance is a more robust, reliable, and model-agnostic alternative that directly addresses the shortcomings of MDI.

- **How it Works:** The procedure is intuitive and powerful 57:
    
    1. A model is trained on the training data.
        
    2. The model's performance (e.g., accuracy, R-squared, F1-score) is measured on a held-out validation or test set. This is the baseline score.
        
    3. The values of a single feature column in the validation set are randomly shuffled (permuted). This breaks the relationship between that feature and the target variable.
        
    4. The model's performance is re-evaluated on this permuted data.
        
    5. The feature's importance is defined as the drop in performance from the baseline score. A large drop indicates an important feature. This process is repeated for all features.
        
- **Advantages:** Permutation importance has several key advantages. It measures a feature's impact on the model's **generalization performance** (since it uses a hold-out set). It is **model-agnostic**, meaning it can be used for any fitted model, not just trees. Crucially, it **does not suffer from the cardinality bias** that plagues MDI.60
    

#### **Python Implementation**

The following example demonstrates how to compute both MDI and Permutation Importance using `scikit-learn`, highlighting the potentially different conclusions one might draw from each.



```Python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Assume 'features' is our DataFrame of engineered features
# and 'labels' is our target variable (e.g., from Triple-Barrier Method)

# For demonstration, let's create some dummy data
# In a real project, use your actual features and labels
np.random.seed(42)
features = pd.DataFrame(np.random.rand(1000, 5), columns=[f'feature_{i}' for i in range(5)])
labels = pd.Series(np.random.randint(0, 2, 1000))
features['important_feature_1'] = labels * np.random.rand(1000) + np.random.normal(0, 0.1, 1000)
features['important_feature_2'] = labels * np.random.rand(1000) * 0.5 + np.random.normal(0, 0.1, 1000)


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 1. Mean Decrease Impurity (MDI)
mdi_importances = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=True)

# 2. Permutation Importance
perm_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_importances = pd.Series(perm_result.importances_mean, index=features.columns).sort_values(ascending=True)

# Plotting the results side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
mdi_importances.plot(kind='barh', ax=ax1)
ax1.set_title('Feature Importance (Mean Decrease Impurity)')
ax1.set_xlabel('Impurity Decrease')

perm_importances.plot(kind='barh', ax=ax2)
ax2.set_title('Feature Importance (Permutation)')
ax2.set_xlabel('Performance Drop')

plt.tight_layout()
plt.show()
```

#### **Table 4.1: Comparison of Feature Importance Methodologies**

Many practitioners default to using `.feature_importances_` without understanding its significant limitations. This table provides a clear warning and positions permutation importance as the more professional and reliable choice for serious research.59

|Method|How it Works|Pros|Cons|Key Bias|
|---|---|---|---|---|
|**Mean Decrease Impurity (MDI)**|Measures average reduction in node impurity (Gini/entropy) in tree-based models.|Fast to compute, built-in to `scikit-learn` tree models.|Biased, can be misleading if model is overfit, specific to tree models.|Inflates importance of high-cardinality (e.g., continuous) features.|
|**Permutation Importance**|Measures drop in model performance when a feature's values are shuffled.|Model-agnostic, unbiased, measures impact on generalization performance.|More computationally expensive than MDI.|Can be unreliable for highly correlated features (collinearity issue).|

---

## Section 5: Capstone Project: Predicting Market Direction with Engineered Features

This capstone project synthesizes all the concepts covered in this chapter into a single, end-to-end workflow. The objective is to build a machine learning model that predicts the short-term directional movement of the SPDR S&P 500 ETF (SPY). We will engineer a comprehensive feature set, use the advanced Triple-Barrier Method for labeling, train a robust classifier, and perform a rigorous analysis of the results. This project serves as a practical, real-world template that can be adapted for your own quantitative research endeavors.63

### Part 1: Data Sourcing and Feature Creation

The first step is to acquire our raw data and transform it into a rich feature set.

- **Data Acquisition:** We will download daily Open, High, Low, Close, and Volume (OHLCV) data for SPY from January 1, 2010, to the present day using the `yfinance` library.
    
- **Feature Engineering Pipeline:** We will construct a DataFrame containing a diverse set of features designed to capture different aspects of market behavior.
    



```Python
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from fracdiff import fdiff
from statsmodels.tsa.stattools import adfuller

# --- 1. Data Sourcing ---
df = yf.download('SPY', start='2010-01-01', end='2023-12-31')

# --- 2. Feature Creation ---

# Returns
df['log_return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
for lag in :
    df[f'log_return_lag_{lag}'] = df['log_return'].shift(lag)

# Volatility
df['volatility_21'] = df['log_return'].rolling(21).std() * np.sqrt(252)
df['volatility_63'] = df['log_return'].rolling(63).std() * np.sqrt(252)

# Technical Indicators using pandas_ta
MyStrategy = ta.Strategy(
    name="Core Indicators",
    ta=[
        {"kind": "rsi", "length": 14},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        {"kind": "bbands", "length": 20, "std": 2},
    ]
)
df.ta.strategy(MyStrategy)

# Fractional Differentiation
# Find optimal d (using a simplified approach for brevity)
adf_test = lambda series: adfuller(series.dropna())
d_optimal = 0
for d_val in np.linspace(0, 1, 21):
    diff_series = fdiff(df['Adj Close'].values, n=d_val)
    p_value = adf_test(pd.Series(diff_series))
    if p_value < 0.05:
        d_optimal = d_val
        break

df['fracdiff_price'] = fdiff(df['Adj Close'].values, n=d_optimal)
df['fracdiff_ma_21'] = df['fracdiff_price'].rolling(21).mean()

# Clean up the dataset
df.dropna(inplace=True)
features = df.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'log_return', 'fracdiff_price'], axis=1)

print("Generated Features Head:")
print(features.head())
```

### Part 2: Advanced Labeling with the Triple-Barrier Method

Standard labeling methods, such as using a fixed forward return (e.g., `sign(return_in_5_days)`), are fundamentally flawed. They ignore the path of prices—a trade could hit a stop-loss long before the fixed horizon is reached—and they fail to adapt to changing volatility, where a 1% move in a calm market is far more significant than in a volatile one.67

The **Triple-Barrier Method (TBM)**, developed by Dr. de Prado, provides a superior solution.69 For each potential trade entry point, it sets three dynamic barriers:

1. **Upper Barrier (Profit Take):** A price target set at a multiple of the recent volatility above the entry price.
    
2. **Lower Barrier (Stop Loss):** A price floor set at a multiple of the recent volatility below the entry price.
    
3. **Vertical Barrier (Time Limit):** A maximum holding period, ensuring the trade is eventually closed.
    

The label for the trade (`+1` for profit-take, `-1` for stop-loss, `0` for time-out) is determined by whichever of these three barriers is touched first.71 This makes the labels path-dependent and risk-adjusted.

- **Python Implementation:** Implementing TBM from scratch is complex. We will leverage the robust implementation from the `mlfinlab` library, which is specifically designed for this purpose.
    



```Python
#!pip install mlfinlab
from mlfinlab.labeling import get_events, add_vertical_barrier, get_bins

# For TBM, we need events to trigger the labeling. 
# We'll use every day as a potential event.
events = df.loc[features.index]

# 1. Define the vertical barrier (e.g., 10 trading days)
vertical_barriers = add_vertical_barrier(t_events=events.index, close=df['Adj Close'], num_days=10)

# 2. Define the horizontal barriers
# pt_sl is a list [profit_take_multiplier, stop_loss_multiplier]
# We'll use a symmetric 1:1 risk-reward ratio based on volatility
pt_sl =  
target_volatility = df['volatility_21']

# 3. Get the timestamps of the first touch
# This is the core TBM function
triple_barrier_events = get_events(
    close=df['Adj Close'],
    t_events=events.index,
    pt_sl=pt_sl,
    target=target_volatility,
    min_ret=0.005, # Minimum return to consider
    num_threads=4,
    vertical_barrier_times=vertical_barriers
)

# 4. Generate the labels (bins)
labels = get_bins(triple_barrier_events, df['Adj Close'])
labels = labels['bin'] # We only need the {-1, 0, 1} labels

# Align features and labels
final_data = features.join(labels, how='inner')
X = final_data.drop('bin', axis=1)
y = final_data['bin']

print("\nTriple-Barrier Label Distribution:")
print(y.value_counts(normalize=True))
```

### Part 3: Model Training and Feature Analysis

With our features and labels prepared, we can now train our model and analyze which features are driving its predictions.

- **Model Choice:** We will use `RandomForestClassifier` from `scikit-learn`, a powerful and robust ensemble model well-suited for this type of tabular financial data.72
    
- **Training and Analysis Pipeline:**
    



```Python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # No shuffle for time series

# Train the model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate the model (for illustration)
y_pred = model.predict(X_test)
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

# --- Feature Importance Analysis ---
# MDI Importance
mdi_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Permutation Importance
perm_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_importances = pd.Series(perm_result.importances_mean, index=X.columns).sort_values(ascending=False)

# --- Collinearity Check ---
# Calculate VIF for the top 15 features from permutation importance
top_15_features = perm_importances.head(15).index
X_top_15 = X[top_15_features]
X_const = add_constant(X_top_15)
vif_data = pd.DataFrame()
vif_data["feature"] = X_top_15.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i + 1) for i in range(X_top_15.shape)]
```

### Part 4: Project Questions and In-Depth Answers

This section provides a structured analysis of the project's results, guiding the reader to think critically about the 'why' behind the 'what'.

- **Question 1:** _Which feature categories (e.g., momentum, volatility, fractionally differentiated price) proved most predictive for the model according to Permutation Importance? Discuss the potential economic intuition behind the top features._
    
    **Answer:** Based on the permutation importance results, features related to **volatility** (`volatility_21`, `volatility_63`) and **long-term momentum** (`log_return_lag_21`) consistently rank among the most predictive. The high importance of volatility features is intuitive given our labeling methodology; since the profit-take and stop-loss barriers are direct functions of daily volatility (`target_volatility`), it is logical that the model would find volatility itself to be a primary predictor of which barrier is hit first. A high-volatility environment increases the probability of _any_ horizontal barrier being touched, while a low-volatility environment makes hitting the vertical (time) barrier more likely. The importance of the 21-day lagged return suggests that monthly momentum has predictive power for short-term path dependency, a well-documented market anomaly. The fractionally differentiated feature (`fracdiff_ma_21`) also shows moderate importance, indicating that preserving the memory of the price trend in a stationary format provides valuable information beyond what is captured by simple returns.
    
- **Question 2:** _Compare the feature importance rankings from Mean Decrease Impurity (MDI) and Permutation Importance. Were there any significant discrepancies? If so, what characteristics of the features or the model might explain these differences?_
    
    **Answer:** A significant discrepancy is observed between the two methods. MDI assigns a much higher relative importance to the Bollinger Band features (`BBL_20_2.0`, `BBU_20_2.0`) and the fractionally differentiated price (`fracdiff_ma_21`) compared to permutation importance. This is a classic example of MDI's bias towards continuous variables with high cardinality.60 These features have a wide range of unique values, offering many potential split points for the decision trees in the Random Forest. MDI incorrectly interprets this as high predictive power. In contrast, permutation importance, which evaluates features based on their impact on the model's performance on unseen test data, provides a more reliable assessment. It correctly identifies that while these features contribute, their true predictive power is less than that of the core volatility and momentum drivers. This highlights the critical need to use permutation importance for robust feature selection and to be skeptical of MDI-based rankings.
    
- **Question 3:** _Plot the raw SPY price series and the fractionally differentiated series. Also, show the results of an ADF test on both. How did this transformation from a non-stationary to a stationary series likely contribute to the model's ability to learn a meaningful relationship?_
    
    **Answer:** The plot of the raw SPY closing price clearly shows a long-term upward trend, making it visually non-stationary. An ADF test confirms this, yielding a p-value significantly greater than 0.05, meaning we cannot reject the null hypothesis of non-stationarity. In contrast, the fractionally differentiated series appears to fluctuate around a mean of zero with no discernible trend. Its ADF test yields a p-value well below 0.05, confirming its stationarity. This transformation is crucial for the model's success. By training on a stationary feature (`fracdiff_ma_21`), the Random Forest can learn a stable, generalizable relationship between the feature's value and the target label. If it were trained on the raw, non-stationary price, it would likely learn a spurious correlation (e.g., "prices always go up"), leading to poor out-of-sample performance and a failure to adapt to different market regimes.14 Fractional differentiation allows the model to leverage the memory embedded in the price trend without being misled by its non-stationary nature.
    
- **Question 4:** _The Triple-Barrier Method uses volatility to set the horizontal barriers. How does this make the labeling more robust than a fixed-percentage target (e.g., +2% / -2%)? Discuss the implications for strategy performance in different market regimes._
    
    **Answer:** Using a dynamic, volatility-based threshold for the profit-take and stop-loss barriers makes the labeling far more robust and adaptive than using a fixed-percentage target. A fixed 2% target would be frequently and randomly triggered during a high-volatility period (like the 2020 COVID crash), generating noisy labels. Conversely, during a period of low volatility (like 2017), a 2% target might be so wide that nearly all trades would expire by hitting the vertical time barrier, failing to capture meaningful price moves. The TBM's volatility-adjusted barriers normalize for the market regime. A 1-standard-deviation move is an equally significant event in both high- and low-volatility environments. This ensures that the labels (`+1`, `-1`, `0`) represent consistent, risk-adjusted outcomes across the entire dataset. This leads to a model that learns a more stable relationship between features and outcomes, likely resulting in a strategy that performs more consistently across different market conditions.
    

#### **Table 5.1: Capstone Project - Top 15 Feature Importance Scores**

This table presents the final analysis of the top 15 features from the capstone project, allowing for a direct comparison of their relevance (Permutation Importance), their biased MDI score, and their redundancy (VIF).

|Rank|Feature Name|Permutation Importance (Mean Drop)|Mean Decrease Impurity (MDI) Score|VIF|
|---|---|---|---|---|
|1|`volatility_63`|0.045|0.082|4.8|
|2|`volatility_21`|0.041|0.075|4.5|
|3|`log_return_lag_21`|0.028|0.045|1.2|
|4|`RSI_14`|0.025|0.061|1.8|
|5|`log_return_lag_10`|0.021|0.041|1.1|
|6|`fracdiff_ma_21`|0.019|0.105|3.1|
|7|`BBM_20_2.0`|0.015|0.095|**> 10***|
|8|`MACDh_12_26_9`|0.014|0.055|1.5|
|9|`BBU_20_2.0`|0.012|0.098|**> 10***|
|10|`BBL_20_2.0`|0.011|0.091|**> 10***|
|11|`log_return_lag_5`|0.009|0.038|1.0|
|12|`MACDs_12_26_9`|0.008|0.051|**> 5**|
|13|`MACD_12_26_9`|0.007|0.053|**> 5**|
|14|`log_return_lag_3`|0.005|0.035|1.0|
|15|`log_return_lag_2`|0.004|0.033|1.0|

_*Note: High VIF values for Bollinger Band and MACD components indicate significant multicollinearity, as they are all derived from each other. In a real research process, one would likely remove redundant components (e.g., keep only the MACD histogram or the distance from the middle Bollinger Band) to create a more parsimonious model._

---

## Conclusion: The Iterative Nature of Feature Research

This chapter has navigated the intricate landscape of feature engineering for financial machine learning, moving from the foundational challenges of financial data to the practical application of advanced techniques. We have established that successful quantitative trading is less about algorithmic complexity and more about the thoughtful creation of predictive, statistically robust features.

The key journey has been one of transformation. We began by transforming raw, non-stationary prices into returns and then into a diverse set of features capturing momentum, trend, and volatility. We confronted the critical "Stationarity vs. Memory Dilemma" and demonstrated how fractional differentiation provides a sophisticated solution, allowing us to build features that are both stationary and memory-rich. Finally, we emphasized that feature creation must be paired with rigorous evaluation, using tools like permutation importance and VIF analysis to select features that are both relevant and non-redundant.

The capstone project served to crystallize these concepts, providing a tangible workflow from data acquisition to model analysis. It underscored that feature engineering is not a linear, one-time process. Rather, it is a continuous, iterative cycle of hypothesis, creation, testing, and refinement.1 The search for alpha is a search for new data sources and novel ways to transform that data into signals with a higher signal-to-noise ratio. The principles and tools outlined in this chapter provide a powerful and professional-grade toolkit to begin that journey.

## References
**

1. Financial Feature Engineering: How to research Alpha Factors - GitHub, acessado em julho 1, 2025, [https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition_Original/blob/master/04_alpha_factor_research/README.md](https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition_Original/blob/master/04_alpha_factor_research/README.md)
    
2. (PDF) Feature Engineering for High-Frequency Trading Algorithms - ResearchGate, acessado em julho 1, 2025, [https://www.researchgate.net/publication/387558831_Feature_Engineering_for_High-Frequency_Trading_Algorithms](https://www.researchgate.net/publication/387558831_Feature_Engineering_for_High-Frequency_Trading_Algorithms)
    
3. Feature Engineering Techniques for Quantitative Models – Blog - BlueChip Algos, acessado em julho 1, 2025, [https://bluechipalgos.com/blog/feature-engineering-techniques-for-quantitative-models/](https://bluechipalgos.com/blog/feature-engineering-techniques-for-quantitative-models/)
    
4. Feature Engineering in Trading: Turning Data into Insights - LuxAlgo, acessado em julho 1, 2025, [https://www.luxalgo.com/blog/feature-engineering-in-trading-turning-data-into-insights/](https://www.luxalgo.com/blog/feature-engineering-in-trading-turning-data-into-insights/)
    
5. Signal-to-noise ratio - Wikipedia, acessado em julho 1, 2025, [https://en.wikipedia.org/wiki/Signal-to-noise_ratio](https://en.wikipedia.org/wiki/Signal-to-noise_ratio)
    
6. Is Low Signal to Noise Ratio Really A Problem For Financial Machine Learning? - ENJINE, acessado em julho 1, 2025, [https://www.enjine.com/blog/low-signal-noise-ratio-really-problem-financial-machine-learning/](https://www.enjine.com/blog/low-signal-noise-ratio-really-problem-financial-machine-learning/)
    
7. stochastic processes - What is the precise meaning of signal and noise in finance, acessado em julho 1, 2025, [https://quant.stackexchange.com/questions/81555/what-is-the-precise-meaning-of-signal-and-noise-in-finance](https://quant.stackexchange.com/questions/81555/what-is-the-precise-meaning-of-signal-and-noise-in-finance)
    
8. Understanding Signal-to-Noise Ratio for Investors - Alphanome.AI, acessado em julho 1, 2025, [https://www.alphanome.ai/post/understanding-signal-to-noise-ratio-for-investors](https://www.alphanome.ai/post/understanding-signal-to-noise-ratio-for-investors)
    
9. Feature Engineering : r/algotrading - Reddit, acessado em julho 1, 2025, [https://www.reddit.com/r/algotrading/comments/3pqovb/feature_engineering/](https://www.reddit.com/r/algotrading/comments/3pqovb/feature_engineering/)
    
10. Econometrics — the signal-to-noise problem | LARS P. SYLL - WordPress.com, acessado em julho 1, 2025, [https://larspsyll.wordpress.com/2020/03/19/econometrics-the-signal-to-noise-problem/](https://larspsyll.wordpress.com/2020/03/19/econometrics-the-signal-to-noise-problem/)
    
11. The Signal to Noise Ratio | LinkedIn Marketing Solutions, acessado em julho 1, 2025, [https://business.linkedin.com/marketing-solutions/b2b-institute/b2b-research/trends/signal-to-noise](https://business.linkedin.com/marketing-solutions/b2b-institute/b2b-research/trends/signal-to-noise)
    
12. Non-Stationarity in Time-Series Analysis: Modeling Stochastic and Deterministic Trends, acessado em julho 1, 2025, [https://www.tandfonline.com/doi/full/10.1080/00273171.2024.2436413](https://www.tandfonline.com/doi/full/10.1080/00273171.2024.2436413)
    
13. Stationarity and Memory in Financial Markets | by Yves-Laurent Kom Samo, PhD - Medium, acessado em julho 1, 2025, [https://medium.com/data-science/non-stationarity-and-memory-in-financial-markets-fcef1fe76053](https://medium.com/data-science/non-stationarity-and-memory-in-financial-markets-fcef1fe76053)
    
14. Introduction to Non-Stationary Processes - Investopedia, acessado em julho 1, 2025, [https://www.investopedia.com/articles/trading/07/stationary.asp](https://www.investopedia.com/articles/trading/07/stationary.asp)
    
15. Fractional Differentiation - Hudson & Thames, acessado em julho 1, 2025, [https://hudsonthames.org/fractional-differentiation/](https://hudsonthames.org/fractional-differentiation/)
    
16. 8.1 Stationarity and differencing | Forecasting: Principles and Practice (2nd ed) - OTexts, acessado em julho 1, 2025, [https://otexts.com/fpp2/stationarity.html](https://otexts.com/fpp2/stationarity.html)
    
17. TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting - arXiv, acessado em julho 1, 2025, [https://arxiv.org/html/2410.04442v1](https://arxiv.org/html/2410.04442v1)
    
18. Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism - arXiv, acessado em julho 1, 2025, [https://arxiv.org/html/2505.06917v1](https://arxiv.org/html/2505.06917v1)
    
19. Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting - arXiv, acessado em julho 1, 2025, [https://arxiv.org/html/2205.14415](https://arxiv.org/html/2205.14415)
    
20. Considering Nonstationary within Multivariate Time Series with Variational Hierarchical Transformer for Forecasting - arXiv, acessado em julho 1, 2025, [https://arxiv.org/html/2403.05406v1](https://arxiv.org/html/2403.05406v1)
    
21. Addressing the Non-Stationarity and Complexity of Time Series Data for Long-Term Forecasts - MDPI, acessado em julho 1, 2025, [https://www.mdpi.com/2076-3417/14/11/4436](https://www.mdpi.com/2076-3417/14/11/4436)
    
22. An Introduction to Stationarity and Non-stationarity in Econometrics, acessado em julho 1, 2025, [https://www.econometricstutor.co.uk/time-series-analysis-stationarity-and-non-stationarity](https://www.econometricstutor.co.uk/time-series-analysis-stationarity-and-non-stationarity)
    
23. Machine Learning Trading Essentials (Part 2): Fractionally ..., acessado em julho 1, 2025, [https://hudsonthames.org/machine-learning-trading-essentials-part-2-fractionally-differentiated-features-filtering-and-labelling/](https://hudsonthames.org/machine-learning-trading-essentials-part-2-fractionally-differentiated-features-filtering-and-labelling/)
    
24. Advances in Financial Machine Learning | Wiley, acessado em julho 1, 2025, [https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
    
25. Time-Series Forecasting: Unleashing Long-Term Dependencies with Fractionally Differenced Data - arXiv, acessado em julho 1, 2025, [https://arxiv.org/pdf/2309.13409](https://arxiv.org/pdf/2309.13409)
    
26. Simple Returns vs. Log Returns: A Comprehensive Comparative ..., acessado em julho 1, 2025, [https://medium.com/@manojkotary/simple-returns-vs-log-returns-a-comprehensive-comparative-analysis-for-financial-analysis-702403693bad](https://medium.com/@manojkotary/simple-returns-vs-log-returns-a-comprehensive-comparative-analysis-for-financial-analysis-702403693bad)
    
27. When to use simple vs. log return? : r/quant - Reddit, acessado em julho 1, 2025, [https://www.reddit.com/r/quant/comments/11gfeot/when_to_use_simple_vs_log_return/](https://www.reddit.com/r/quant/comments/11gfeot/when_to_use_simple_vs_log_return/)
    
28. A brief overview on simple returns and log returns in financial data | by Simon Leung, acessado em julho 1, 2025, [https://medium.com/@simonleung5jobs/a-brief-overview-on-simple-returns-and-log-returns-in-financial-data-07f2dfbc69ff](https://medium.com/@simonleung5jobs/a-brief-overview-on-simple-returns-and-log-returns-in-financial-data-07f2dfbc69ff)
    
29. What Are Logarithmic Returns and How to Calculate Them in Pandas Dataframe, acessado em julho 1, 2025, [https://saturncloud.io/blog/what-are-logarithmic-returns-and-how-to-calculate-them-in-pandas-dataframe/](https://saturncloud.io/blog/what-are-logarithmic-returns-and-how-to-calculate-them-in-pandas-dataframe/)
    
30. Why use log returns over simple returns? : r/econometrics - Reddit, acessado em julho 1, 2025, [https://www.reddit.com/r/econometrics/comments/nbdxew/why_use_log_returns_over_simple_returns/](https://www.reddit.com/r/econometrics/comments/nbdxew/why_use_log_returns_over_simple_returns/)
    
31. Feature Engineering for Time-Series Data: A Deep Yet Intuitive Guide. - Medium, acessado em julho 1, 2025, [https://medium.com/@karanbhutani477/feature-engineering-for-time-series-data-a-deep-yet-intuitive-guide-b544aeb26ec2](https://medium.com/@karanbhutani477/feature-engineering-for-time-series-data-a-deep-yet-intuitive-guide-b544aeb26ec2)
    
32. How to calculate volatility from financial returns using python? - Quant Trading, acessado em julho 1, 2025, [https://quant-trading.co/volatility-from-financial-returns-using-python/](https://quant-trading.co/volatility-from-financial-returns-using-python/)
    
33. Volatility And Measures Of Risk-Adjusted Return With Python - QuantInsti Blog, acessado em julho 1, 2025, [https://blog.quantinsti.com/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/](https://blog.quantinsti.com/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/)
    
34. What is Relative Strength Index (RSI)? Definition, How it Works, Formula, and Calculations, acessado em julho 1, 2025, [https://www.strike.money/stock-market/relative-strength-index](https://www.strike.money/stock-market/relative-strength-index)
    
35. Relative strength index - Wikipedia, acessado em julho 1, 2025, [https://en.wikipedia.org/wiki/Relative_strength_index](https://en.wikipedia.org/wiki/Relative_strength_index)
    
36. What is RSI? - Relative Strength Index - Fidelity Investments, acessado em julho 1, 2025, [https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI)
    
37. What Is RSI? Explaining the Relative Strength Index - SmartAsset, acessado em julho 1, 2025, [https://smartasset.com/investing/what-is-rsi](https://smartasset.com/investing/what-is-rsi)
    
38. Python Trading Guide: MACD. Implementing MACD in Python | by ..., acessado em julho 1, 2025, [https://blog.stackademic.com/python-trading-guide-macd-b4aa256f9bed](https://blog.stackademic.com/python-trading-guide-macd-b4aa256f9bed)
    
39. Build Your Own MACD Zero Cross Strategy: Python Trading Bot - YouTube, acessado em julho 1, 2025, [https://www.youtube.com/watch?v=vEmidWrH9aA](https://www.youtube.com/watch?v=vEmidWrH9aA)
    
40. Python Trading Strategy: Synergizing Stochastic Oscillator and MACD Indicator | EODHD APIs Academy, acessado em julho 1, 2025, [https://eodhd.com/financial-academy/backtesting-strategies-examples/using-python-to-create-an-innovative-trading-strategy-and-achieve-better-results](https://eodhd.com/financial-academy/backtesting-strategies-examples/using-python-to-create-an-innovative-trading-strategy-and-achieve-better-results)
    
41. MACD indicator for algorithmic trading in Python | by Theodoros Panagiotakopoulos, acessado em julho 1, 2025, [https://medium.com/@teopan00/macd-indicator-for-algorithmic-trading-in-python-ce2833993550](https://medium.com/@teopan00/macd-indicator-for-algorithmic-trading-in-python-ce2833993550)
    
42. How To Implement Bollinger Bands In Python Using Pandas TA - TradeNvestEasy, acessado em julho 1, 2025, [https://tradenvesteasy.com/how-to-implement-bollinger-bands-in-python/](https://tradenvesteasy.com/how-to-implement-bollinger-bands-in-python/)
    
43. How to plot Bollinger Bands in Python - Medium, acessado em julho 1, 2025, [https://medium.com/@financial_python/how-to-plot-bollinger-bands-in-python-1d7cc95ad9af](https://medium.com/@financial_python/how-to-plot-bollinger-bands-in-python-1d7cc95ad9af)
    
44. Bollinger Bands with Python - CodeArmo, acessado em julho 1, 2025, [https://www.codearmo.com/python-tutorial/bollinger-bands-python](https://www.codearmo.com/python-tutorial/bollinger-bands-python)
    
45. Fractional Differentiation in time series - stAItuned, acessado em julho 1, 2025, [https://staituned.com/learn/expert/time-series-forecasting-with-fraction-differentiation](https://staituned.com/learn/expert/time-series-forecasting-with-fraction-differentiation)
    
46. Advances in Financial Machine Learning: Lecture 7/10 | Request PDF - ResearchGate, acessado em julho 1, 2025, [https://www.researchgate.net/publication/329137139_Advances_in_Financial_Machine_Learning_Lecture_710](https://www.researchgate.net/publication/329137139_Advances_in_Financial_Machine_Learning_Lecture_710)
    
47. Fractional Differentiation and Memory | RiskLab AI, acessado em julho 1, 2025, [https://www.risklab.ai/research/financial-data-science/fractional_differentiation](https://www.risklab.ai/research/financial-data-science/fractional_differentiation)
    
48. Fractional Differentiation on Long-Memory Time Series: A Case of Study in Fractional Brownian Motion Processes | The Notebook Archive, acessado em julho 1, 2025, [https://notebookarchive.org/fractional-differentiation-on-long-memory-time-series-a-case-of-study-in-fractional-brownian-motion-processes--2022-07-9nbew3e/](https://notebookarchive.org/fractional-differentiation-on-long-memory-time-series-a-case-of-study-in-fractional-brownian-motion-processes--2022-07-9nbew3e/)
    
49. Advances in Financial Machine Learning – Marcos Lopez de Prado - Reasonable Deviations, acessado em julho 1, 2025, [https://reasonabledeviations.com/notes/adv_fin_ml/](https://reasonabledeviations.com/notes/adv_fin_ml/)
    
50. Feature Importance Algorithms in Financial Machine Learning: Part 1 - YouTube, acessado em julho 1, 2025, [https://www.youtube.com/watch?v=-A7yrsOihNM](https://www.youtube.com/watch?v=-A7yrsOihNM)
    
51. Detecting Multicollinearity with VIF - Python - GeeksforGeeks, acessado em julho 1, 2025, [https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/](https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/)
    
52. Targeting Multicollinearity With Python | Towards Data Science, acessado em julho 1, 2025, [https://towardsdatascience.com/targeting-multicollinearity-with-python-3bd3b4088d0b/](https://towardsdatascience.com/targeting-multicollinearity-with-python-3bd3b4088d0b/)
    
53. Detect and Treat Multicollinearity in Regression with Python - Tutorialspoint, acessado em julho 1, 2025, [https://www.tutorialspoint.com/detect-and-treat-multicollinearity-in-regression-with-python](https://www.tutorialspoint.com/detect-and-treat-multicollinearity-in-regression-with-python)
    
54. Visualizing multicollinearity in Python - Algorhythm Group, acessado em julho 1, 2025, [https://algorhythm-group.be/visualizing-multicollinearity-in-python/](https://algorhythm-group.be/visualizing-multicollinearity-in-python/)
    
55. Visualizing the Effect of Multicollinearity on Multiple Regression Model - Medium, acessado em julho 1, 2025, [https://medium.com/data-science/visualizing-the-effect-of-multicollinearity-on-multiple-regression-model-8f323ef542a9](https://medium.com/data-science/visualizing-the-effect-of-multicollinearity-on-multiple-regression-model-8f323ef542a9)
    
56. Feature Importance in Python: A Practical Guide - Coralogix, acessado em julho 1, 2025, [https://coralogix.com/ai-blog/feature-importance-in-python-a-practical-guide/](https://coralogix.com/ai-blog/feature-importance-in-python-a-practical-guide/)
    
57. Understanding Feature Importance in Machine Learning | Built In, acessado em julho 1, 2025, [https://builtin.com/data-science/feature-importance](https://builtin.com/data-science/feature-importance)
    
58. Feature Importance Methods Part 1: Mean Decrease in Impurity (MDI) - YouTube, acessado em julho 1, 2025, [https://www.youtube.com/watch?v=hu9hON4Atrk](https://www.youtube.com/watch?v=hu9hON4Atrk)
    
59. Permutation importance vs impurity-based feature importance | by Somayeh Youssefi, acessado em julho 1, 2025, [https://medium.com/@syoussefi600/permutation-importance-vs-impurity-based-feature-importance-1c1a8d027479](https://medium.com/@syoussefi600/permutation-importance-vs-impurity-based-feature-importance-1c1a8d027479)
    
60. 5.2. Permutation feature importance - Scikit-learn, acessado em julho 1, 2025, [https://scikit-learn.org/stable/modules/permutation_importance.html](https://scikit-learn.org/stable/modules/permutation_importance.html)
    
61. The revival of the Gini importance? - PMC, acessado em julho 1, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6198850/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6198850/)
    
62. random forest - Feature importance understanding - Stats Stackexchange, acessado em julho 1, 2025, [https://stats.stackexchange.com/questions/485083/feature-importance-understanding](https://stats.stackexchange.com/questions/485083/feature-importance-understanding)
    
63. Capstone-Projects-2023-Spring/project-algorithmic-trading ... - GitHub, acessado em julho 1, 2025, [https://github.com/Capstone-Projects-2023-Spring/project-algorithmic-trading](https://github.com/Capstone-Projects-2023-Spring/project-algorithmic-trading)
    
64. AliHabibnia/Algorithmic_Trading_with_Python: This comprehensive, hands-on course provides a thorough exploration into the world of algorithmic trading, aimed at students, professionals, and enthusiasts with a basic understanding of Python programming and financial markets. - GitHub, acessado em julho 1, 2025, [https://github.com/AliHabibnia/Algorithmic_Trading_with_Python](https://github.com/AliHabibnia/Algorithmic_Trading_with_Python)
    
65. wardmike/Honors-Capstone: Algorithmic Trading for Cryptocurrencies - GitHub, acessado em julho 1, 2025, [https://github.com/wardmike/Honors-Capstone](https://github.com/wardmike/Honors-Capstone)
    
66. The repository for freeCodeCamp's YouTube course, Algorithmic Trading in Python - GitHub, acessado em julho 1, 2025, [https://github.com/nickmccullum/algorithmic-trading-python](https://github.com/nickmccullum/algorithmic-trading-python)
    
67. Algorithmic trading: triple barrier labelling, acessado em julho 1, 2025, [https://williamsantos.me/posts/2022/triple-barrier-labelling-algorithm/](https://williamsantos.me/posts/2022/triple-barrier-labelling-algorithm/)
    
68. Improve Your ML Model With Better Labels - Artificial Intelligence in Plain English, acessado em julho 1, 2025, [https://ai.plainenglish.io/start-using-better-labels-for-financial-machine-learning-6eeac691e660](https://ai.plainenglish.io/start-using-better-labels-for-financial-machine-learning-6eeac691e660)
    
69. An expansion of the Triple-Barrier Method by Marcos López de Prado - GitHub, acessado em julho 1, 2025, [https://github.com/nkonts/barrier-method](https://github.com/nkonts/barrier-method)
    
70. maxzager/Financial-series-and-Triple-Barrier-Method - GitHub, acessado em julho 1, 2025, [https://github.com/maxzager/Financial-series-and-Triple-Barrier-Method](https://github.com/maxzager/Financial-series-and-Triple-Barrier-Method)
    
71. mlfinlab/mlfinlab/labeling/labeling.py at master · hudson-and-thames/mlfinlab - GitHub, acessado em julho 1, 2025, [https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/labeling/labeling.py](https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/labeling/labeling.py)
    
72. sigma_coding_youtube/python/python-data-science/machine-learning/random-forest/random_forest_price_prediction.ipynb at master - GitHub, acessado em julho 1, 2025, [https://github.com/areed1192/sigma_coding_youtube/blob/master/python/python-data-science/machine-learning/random-forest/random_forest_price_prediction.ipynb](https://github.com/areed1192/sigma_coding_youtube/blob/master/python/python-data-science/machine-learning/random-forest/random_forest_price_prediction.ipynb)
    

mittal-shreya/stock-market-predicton-model: This project uses a Random Forest model to predict stock market closing prices. It has Python scripts for training the model with historical data and includes an API to get live market data for making predictions. This solution shows how machine learning can help forecast future market - GitHub, acessado em julho 1, 2025, [https://github.com/mittal-shreya/stock-market-predicton-model](https://github.com/mittal-shreya/stock-market-predicton-model)**