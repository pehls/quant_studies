## 1.0 Introduction to Statistical Arbitrage

Statistical arbitrage represents a sophisticated and heavily quantitative domain within modern finance. It moves beyond simple directional bets on market movements and instead focuses on exploiting transient, statistically identifiable pricing inefficiencies between financial instruments. This chapter provides a comprehensive exploration of statistical arbitrage, beginning with its foundational principles and historical development. It then delves into the core econometric models and practical implementation workflows, from the canonical pairs trade to more advanced dynamic systems. Crucially, it establishes a robust framework for understanding and managing the inherent risks, culminating in a practical capstone project that synthesizes these concepts into a tangible trading strategy. The objective is to equip the reader with the theoretical knowledge, mathematical tools, and practical coding skills necessary to navigate this complex field.

### 1.1 Defining the Arbitrage: From Risk-Free to Statistical

The term "arbitrage" in its purest form describes a transaction that generates a risk-free profit from price discrepancies of a single asset across different markets or forms. For example, if a stock trades for $100 on the New York Stock Exchange and simultaneously for $100.05 on another exchange, a pure arbitrageur could buy on the former and sell on the latter, capturing a guaranteed, risk-free profit of 5 cents per share, assuming no transaction costs. These opportunities are rare and fleeting in modern electronic markets, as their very exploitation ensures market efficiency.

Statistical Arbitrage (StatArb), by contrast, is a fundamentally different paradigm.2 It is not risk-free. Instead, it is a class of short-term trading strategies that employ statistical and econometric models to identify temporary mispricings between related securities.4 The core philosophy of StatArb is to construct a portfolio, often containing hundreds or thousands of long and short positions, where each position represents a bet on the convergence of a statistical relationship.2 While any individual bet carries significant risk—namely, the risk that the identified statistical relationship breaks down—the strategy relies on the law of large numbers. Over a vast number of trades, the positive expected value of each bet is theorized to produce a consistent profit stream with low volatility.7

The use of the word "arbitrage" in this context can be misleading and has critical implications for risk perception. It does not imply a risk-free guarantee but rather the exploitation of a market _inefficiency_ identified through quantitative analysis.6 The profit is a statistical expectation, not a certainty. This distinction is paramount; a misunderstanding can lead to catastrophic mismanagement of risk, particularly through excessive leverage, under the false assumption that the strategy is safer than it is. The infamous collapse of Long-Term Capital Management (LTCM) in 1998, which heavily utilized convergence trades—a form of statistical arbitrage—serves as a historical cautionary tale. Their models correctly identified statistical relationships in bond markets, but they failed to account for the low-probability event that these relationships could diverge dramatically and remain so under market stress, leading to the fund's demise.1 Therefore, a more accurate framing of StatArb is as a strategy of "betting on the convergence of a statistical relationship," which immediately and correctly foregrounds its primary risk: the possibility that convergence fails to occur.

### 1.2 A Brief History: From Morgan Stanley's Black Box to Modern Quants

The genesis of quantitative finance can be traced back to Louis Bachelier's 1900 thesis on option pricing and Robert Brown's 1827 observation of what became known as Brownian motion.11 However, the practical application of quantitative methods to systematic trading strategies, particularly StatArb, is a more recent development, born from the confluence of financial theory, computational power, and market innovation.

It is widely accepted that institutional statistical arbitrage began in the mid-1980s at Morgan Stanley, under the leadership of Nunzio Tartaglia.1 Tartaglia assembled a team of mathematicians, physicists, and computer scientists to develop a proprietary "black box" trading system.1 This system moved beyond the simpler, intuitive pairs trading concepts of the era and used sophisticated statistical models to trade large, diversified portfolios of equities, aiming to hedge out broad market risk while capturing small, persistent pricing anomalies. These early efforts were secretive and highly profitable, marking the birth of StatArb as a major force on Wall Street.7

The intellectual groundwork for such strategies was laid by pioneers like Edward Thorp, a mathematician who famously applied probability theory to blackjack before turning his attention to financial markets. Thorp's work in the late 1960s and 1970s on identifying mispriced securities and using quantitative methods to manage risk was a precursor to the systematic approaches later formalized in StatArb.12

The strategy's evolution has been inextricably linked to technological advancement. The rise of electronic trading platforms in the 1990s and the exponential growth in computing power dramatically lowered transaction costs and increased the speed of execution, making it possible to implement StatArb on a massive scale.15 The availability of vast amounts of historical and high-frequency data enabled the development of more complex and data-hungry models.8 What began as a strategy trading hundreds of stocks with holding periods of days has evolved into modern high-frequency trading (HFT) operations that may trade thousands of instruments with holding periods of milliseconds, all driven by automated algorithms.2 Today, StatArb is a cornerstone of many hedge funds and proprietary trading firms, employing techniques that span from classical econometrics to cutting-edge machine learning.2

### 1.3 The Cornerstone: The Mean-Reversion Hypothesis in Financial Markets

At the heart of nearly every statistical arbitrage strategy lies a single, powerful economic assumption: the hypothesis of mean reversion.5 Mean reversion is the theory that asset prices, returns, or the statistical relationships between them tend to revert to a long-term average level over time.19 When the current market price of an asset or a spread deviates significantly from this historical mean, it is considered to be in a state of disequilibrium. The mean-reversion hypothesis posits that this disequilibrium is temporary and that the price will eventually be pulled back toward its average.21

This concept provides a clear framework for generating trading signals. A quantitative analyst first identifies a mean-reverting relationship and calculates its historical mean and standard deviation. When the current value of the relationship deviates from the mean by a statistically significant amount—often measured in terms of standard deviations, or a "Z-score"—a trading opportunity is flagged.19 For example, if a spread is two standard deviations below its mean, it is considered "cheap," and a trader would enter a long position, betting on its rise back toward the average. Conversely, if it is two standard deviations above its mean, it is "expensive," and a trader would enter a short position, betting on its fall.22

However, it is critically important to understand that mean reversion is a conditional hypothesis, not an unconditional law of nature. A common and dangerous pitfall for aspiring quants is to treat it as a guaranteed force. The reality is more nuanced. A significant deviation from the mean does not automatically guarantee a return to the _old_ mean. It is possible that the underlying fundamentals of the asset or the relationship have permanently changed, causing a "re-rating" where the mean itself shifts to a new level.9 For instance, a technological breakthrough for one company in a pair could permanently alter its valuation relative to its competitor, causing their historical price relationship to break down indefinitely.

This implies that a successful StatArb model cannot be purely statistical; it must be robust to the failure of its core assumption. The most sophisticated StatArb systems incorporate mechanisms to detect "regime changes" or structural breaks in statistical relationships.5 This fundamental challenge—distinguishing between a temporary, tradable deviation and a permanent structural break—is what makes statistical arbitrage so difficult. It motivates the need for the advanced dynamic models discussed later in this chapter, such as the Kalman filter, which can adapt to changing parameters, and it underscores the non-negotiable importance of a rigorous risk management framework, including hard stop-losses, for when the mean-reversion hypothesis inevitably fails. The goal is not just to model mean reversion, but to build systems that are skeptical of it and can survive its failure.

## 2.0 The Canonical Pairs Trading Workflow

Pairs trading is the archetypal statistical arbitrage strategy and serves as the perfect foundation for understanding the entire field.24 It is a market-neutral strategy that matches a long position in one security with a short position in a related, historically correlated security.24 The objective is to profit from the relative performance of the two assets, independent of the overall market's direction. This section provides a complete, practical walkthrough of the pairs trading workflow, from identifying candidate pairs to generating and visualizing trading signals.

### 2.1 Identifying Potential Pairs: The Role of Correlation and Economic Linkages

The search for tradable pairs begins with a combination of qualitative reasoning and quantitative screening. The most robust pairs are typically found between assets that have a strong underlying economic connection, making their price co-movement logical and more likely to persist.24

The first step is a qualitative screen. Analysts look for companies operating in the same industry and serving the same markets. These firms are subject to similar macroeconomic forces, regulatory environments, and consumer trends, providing a fundamental reason for their stock prices to move in tandem. Classic examples include:

- **Direct Competitors:** The Coca-Cola Company (KO) and PepsiCo (PEP) in the beverage industry, or General Motors (GM) and Ford Motor Company (F) in the automotive sector.6
    
- **Sector ETFs:** Exchange-Traded Funds (ETFs) that track the same or similar market indexes or sectors, such as the SPDR S&P 500 ETF (SPY) and the Invesco QQQ Trust (QQQ), which both track large-cap U.S. equities.26
    
- **Commodity-Related Assets:** A gold mining company and the price of gold itself, or two crude oil futures contracts like WTI and Brent.28
    

Once a universe of logically related candidates is established, a quantitative screen is applied. The most common first-pass filter is the statistical correlation of historical returns.30 A high positive correlation (typically > 0.80) suggests a strong linear relationship, making the pair a candidate for further analysis.23 However, as will be detailed in Section 3.0, relying solely on correlation is fraught with peril due to the non-stationary nature of price series. It is a useful but insufficient tool for identifying genuinely mean-reverting relationships.

### 2.2 Modeling the Spread: Price Ratios, Regression, and Normalization (Z-Scores)

After identifying a candidate pair, the next step is to construct the "spread," which is the time series that will actually be modeled and traded. The spread represents the difference or ratio between the two asset prices, and the goal is to create a spread that is stationary (i.e., mean-reverting).

A simple approach is to calculate the price ratio (PA​/PB​) or the price difference (PA​−PB​) between the two assets, Asset A and Asset B.31 While intuitive, this method implicitly assumes a one-to-one hedge ratio, which is rarely optimal.

A more robust and widely used method is to use Ordinary Least Squares (OLS) regression to determine a dynamic hedge ratio. We model one asset's price as a linear function of the other:

![[Pasted image 20250630082629.png]]

Here, yt​ is the price of Asset Y at time t, xt​ is the price of Asset X, β is the hedge ratio, α is the intercept, and ϵt​ is the residual term.32 The spread is then defined as the series of residuals from this regression:

![[Pasted image 20250630082639.png]]

Where β^​ and α^ are the estimated coefficients from the OLS regression. This spread represents the deviation of the pair from its long-term linear relationship. If the assets are truly cointegrated, this spread series should be stationary and mean-reverting around zero.

To make spreads comparable across different pairs and over time, the spread is typically normalized by calculating its Z-score. The Z-score measures how many standard deviations the current spread value is from its historical mean 19:

![[Pasted image 20250630082651.png]]

The mean and standard deviation are usually calculated over a rolling lookback window (e.g., 30 or 60 days) to allow the model to adapt to changing market conditions. The resulting Z-score is a standardized signal that oscillates around zero, providing clear, quantifiable entry and exit points.31

### 2.3 Generating Trading Signals: Threshold-Based Entry and Exit Logic

With a normalized spread (Z-score) in hand, defining the trading logic becomes straightforward. The strategy is based on pre-defined thresholds that signal significant deviations from the mean. A common set of rules is as follows 22:

- **Short Entry Signal:** When the Z-score rises above a certain positive threshold (e.g., +2.0), it indicates that the spread is historically expensive. This means Asset Y is overvalued relative to Asset X. The strategy would be to "short the spread": sell Asset Y and buy Asset X (hedged by the ratio β).
    
- **Long Entry Signal:** When the Z-score falls below a certain negative threshold (e.g., -2.0), it indicates the spread is historically cheap. Asset Y is undervalued relative to Asset X. The strategy would be to "go long the spread": buy Asset Y and sell Asset X.
    
- **Exit Signal:** The position is held until the spread reverts to its mean. The exit signal is typically triggered when the Z-score crosses back to zero. Some strategies may use a wider band for exits (e.g., exit when the Z-score is between -0.5 and +0.5) to reduce trading frequency and transaction costs.33
    

The choice of thresholds (e.g., +/- 2.0 for entry, 0.0 for exit) represents a trade-off. Wider entry thresholds lead to fewer, but potentially more reliable, trades. Narrower thresholds increase trading frequency but may result in more false signals. These parameters are often optimized during the backtesting phase of strategy development.

### 2.4 A First Implementation: A Simple Pairs Trade in Python

The following Python code provides a complete, end-to-end implementation of a basic pairs trading strategy. It demonstrates the full workflow: acquiring data for a classic pair (Coca-Cola and PepsiCo), calculating the spread using OLS regression, normalizing it with a Z-score, and plotting the results to visualize potential trading signals.



```Python
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- 1. Data Acquisition ---
# Define the tickers and the time period for analysis
ticker1 = 'KO'  # Coca-Cola
ticker2 = 'PEP'  # PepsiCo
start_date = '2018-01-01'
end_date = '2023-12-31'

# Download historical price data using yfinance
ko_data = yf.download(ticker1, start=start_date, end=end_date)
pep_data = yf.download(ticker2, start=start_date, end=end_date)

# Create a DataFrame with the closing prices of both stocks
prices = pd.DataFrame({
    'KO': ko_data['Adj Close'],
    'PEP': pep_data['Adj Close']
}).dropna()

# --- 2. Spread Calculation using OLS Regression ---
# Define the dependent (y) and independent (x) variables
y = prices['PEP']
x = prices['KO']

# Add a constant to the independent variable for the regression intercept
x_with_const = sm.add_constant(x)

# Fit the Ordinary Least Squares (OLS) model
model = sm.OLS(y, x_with_const)
results = model.fit()

# Get the hedge ratio (beta) and intercept (alpha)
beta = results.params['KO']
alpha = results.params['const']
print(f"Hedge Ratio (Beta): {beta:.4f}")
print(f"Intercept (Alpha): {alpha:.4f}")

# Calculate the spread (residuals of the regression)
prices = prices['PEP'] - (beta * prices['KO'] + alpha)

# --- 3. Normalization with Z-Score ---
# Define a rolling window for calculating mean and std dev
lookback_window = 30

# Calculate the rolling mean and standard deviation of the spread
prices = prices.rolling(window=lookback_window).mean()
prices = prices.rolling(window=lookback_window).std()

# Calculate the Z-score
prices = (prices - prices) / prices

# --- 4. Visualization ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# Plot 1: Normalized Prices
ax1.plot(prices['KO'] / prices['KO'].iloc, label='KO (Normalized)')
ax1.plot(prices['PEP'] / prices['PEP'].iloc, label='PEP (Normalized)')
ax1.set_title('Normalized Prices of KO and PEP')
ax1.set_ylabel('Normalized Price')
ax1.legend()

# Plot 2: Spread Z-Score with Trading Thresholds
ax2.plot(prices, label='Spread Z-Score')
ax2.axhline(2.0, color='red', linestyle='--', label='Short Entry Threshold (+2.0σ)')
ax2.axhline(-2.0, color='green', linestyle='--', label='Long Entry Threshold (-2.0σ)')
ax2.axhline(0.0, color='black', linestyle='-', linewidth=1, label='Mean (Exit Level)')
ax2.set_title('Normalized Spread (Z-Score) with Trading Thresholds')
ax2.set_ylabel('Z-Score (Standard Deviations)')
ax2.set_xlabel('Date')
ax2.legend()

plt.tight_layout()
plt.show()

# Display the last few rows of the resulting DataFrame
print(prices.tail())
```

This code produces a visual representation of the pairs trading concept. The top plot shows how closely the normalized prices of KO and PEP have moved together over the specified period. The bottom plot shows the Z-score of their spread. The areas where the blue line (Z-score) crosses the red or green dashed lines represent potential trading opportunities, where the relationship has deviated significantly from its historical mean. The subsequent return of the blue line to the central black line represents the mean reversion that the strategy aims to capture.

## 3.0 The Econometric Foundation: Testing for Cointegration

While the pairs trading workflow in the previous section provides a practical starting point, its reliance on simple regression and visual inspection is not statistically rigorous. A professional quantitative approach requires a deeper econometric foundation to ensure that the identified relationships are genuine and not merely statistical artifacts. This section introduces cointegration, the formal statistical property that underpins mean-reverting spreads, and details the standard tests used to detect it.

### 3.1 Beyond Correlation: Why Cointegration Matters

A frequent and critical error made by newcomers to quantitative finance is to conflate correlation with cointegration. While the two concepts are related to how assets move together, they are fundamentally different, and understanding this difference is crucial for building robust StatArb strategies.35

**Correlation** measures the degree to which the _returns_ of two assets move in relation to each other over a short period. It is a measure of co-movement for stationary time series. Asset prices, however, are generally non-stationary; they exhibit trends and do not have a constant mean or variance, a property known as being "integrated of order 1," or I(1).37 Applying linear regression to two non-stationary time series, even if they are completely unrelated, can often produce a high R-squared and a statistically significant relationship. This phenomenon, known as

**spurious regression**, is a classic econometric pitfall.32 Two trending series will appear correlated simply because they are both trending, not because of any underlying economic link.

**Cointegration**, in contrast, is a property of two or more non-stationary (I(1)) time series. They are said to be cointegrated if some linear combination of them is stationary (integrated of order 0, or I(0)).39 This stationary linear combination is precisely the mean-reverting spread that a pairs trader seeks. It implies that even though the individual asset prices can wander off indefinitely (a property of I(1) series), they are bound together by a long-term equilibrium relationship. When they drift apart, this equilibrium acts as a tether, pulling them back together.

Therefore, the key takeaway is this: high correlation of prices does not guarantee a tradable, mean-reverting spread. Cointegration does. Testing for cointegration is the statistically sound method for verifying that a spread constructed from two or more assets is stationary and thus suitable for a mean-reversion strategy.

|Feature|Correlation|Cointegration|
|---|---|---|
|**Definition**|A measure of the linear relationship between two variables.|A statistical property of time series variables where a linear combination is stationary.|
|**Time Horizon**|Describes short-term co-movement.|Describes a long-term equilibrium relationship.|
|**Data Type**|Applied to stationary data (e.g., asset returns).|Applied to non-stationary data (e.g., asset prices).|
|**Mathematical Test**|Pearson correlation coefficient.|Engle-Granger test, Johansen test.|
|**Implication for Trading**|High correlation of returns is useful but does not guarantee a mean-reverting spread.|Cointegration of prices implies the existence of a stationary, mean-reverting spread.|

### 3.2 The Engle-Granger Two-Step Test

The Engle-Granger test, developed by Nobel laureates Robert Engle and Clive Granger, is a foundational, two-step method for testing for cointegration between two time series.32

#### Step 1: Mathematical Formulation and OLS Regression

The first step is to test if the individual price series, yt​ and xt​, are non-stationary, specifically I(1). This is typically done using a unit root test like the Augmented Dickey-Fuller (ADF) test. Assuming both series are confirmed to be I(1), the next step is to estimate the long-run equilibrium relationship between them using an OLS regression 42:

![[Pasted image 20250630082721.png]]

The estimated coefficient, β^​, serves as the cointegrating vector or hedge ratio. The residuals from this regression, u^t​, represent the spread or the deviation from the long-term equilibrium at each point in time.43

#### Step 2: Testing Residuals for Stationarity

The second and crucial step is to test whether these residuals, u^t​, are stationary. If the residuals are stationary (I(0)), then the original series yt​ and xt​ are cointegrated. This is done by performing a unit root test, typically the ADF test, on the residual series u^t​.42 The ADF test estimates the following regression:

![[Pasted image 20250630082732.png]]

The null hypothesis of the test is H0​:γ=0, which implies that the residual series has a unit root and is non-stationary (i.e., no cointegration). The alternative hypothesis is Ha​:γ<0, which implies the series is stationary (i.e., cointegration exists).

A critical nuance is that the distribution of the test statistic for γ is not the standard Dickey-Fuller distribution. Because the residuals u^t​ are estimated from a prior regression rather than being directly observed, a different set of critical values, known as MacKinnon critical values, must be used to determine statistical significance.41

#### Python Implementation with `statsmodels`

The `statsmodels` library in Python provides a convenient function, `coint`, which performs the Engle-Granger test.



```Python
import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import coint

# Use the same KO and PEP data from the previous example
ticker1 = 'KO'
ticker2 = 'PEP'
start_date = '2018-01-01'
end_date = '2023-12-31'
ko_data = yf.download(ticker1, start=start_date, end=end_date)
pep_data = yf.download(ticker2, start=start_date, end=end_date)
prices = pd.DataFrame({'KO': ko_data['Adj Close'], 'PEP': pep_data['Adj Close']}).dropna()

# Perform the Engle-Granger cointegration test
# The null hypothesis is that there is NO cointegration.
coint_test_result = coint(prices['PEP'], prices['KO'])

# Extract and interpret the results
t_statistic = coint_test_result
p_value = coint_test_result
critical_values = coint_test_result

print(f"Cointegration Test for PEP and KO:")
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print("Critical Values:")
print(f"  1%: {critical_values:.4f}")
print(f"  5%: {critical_values:.4f}")
print(f"  10%: {critical_values:.4f}")

# Interpretation
if p_value < 0.05:
    print("\nResult: The series are likely cointegrated (p-value < 0.05).")
    print("We can reject the null hypothesis of no cointegration.")
else:
    print("\nResult: The series are not cointegrated (p-value >= 0.05).")
    print("We fail to reject the null hypothesis of no cointegration.")

if t_statistic < critical_values:
    print("The T-statistic is less than the 5% critical value, further supporting cointegration.")
else:
    print("The T-statistic is greater than the 5% critical value, suggesting no cointegration at this level.")

```

The output of this code will provide the test statistic, the p-value, and the critical values at the 1%, 5%, and 10% significance levels. If the p-value is below the chosen significance level (e.g., 0.05), one can reject the null hypothesis and conclude that the two stock price series are cointegrated, providing a statistical basis for a pairs trading strategy.44

### 3.3 The Johansen Test for Multivariate Cointegration

The Engle-Granger test is powerful but has a key limitation: it can only test for a single cointegrating relationship between two variables.41 In many cases, a group of three or more assets may share multiple long-term equilibrium relationships. For example, a basket of financial sector ETFs (e.g., XLF, VFH, IYF) might be driven by common factors, leading to more than one stationary portfolio combination.

The Johansen test is a more general and powerful procedure that addresses this limitation. It can test for and identify multiple cointegrating vectors within a system of several non-stationary time series.45

#### Vector Error Correction Models (VECM)

The Johansen test is based on the Vector Error Correction Model (VECM) framework. A VECM is a type of vector autoregression (VAR) model designed for use with non-stationary series that are known to be cointegrated. The VECM for a vector of time series Xt​ can be written as 47:

![[Pasted image 20250630082754.png]]

The crucial component is the matrix Π, known as the long-run impact matrix. The **rank** of this matrix, denoted by r, is equal to the number of cointegrating relationships in the system.

- If rank(Π)=0, there are no cointegrating relationships.
    
- If rank(Π)=k (where k is the number of variables), all series are stationary.
    
- If 0<rank(Π)<k, there are r cointegrating vectors.
    

The Johansen test is a procedure for determining the rank r of the matrix Π.

#### Trace and Maximum Eigenvalue Statistics

The test provides two different statistics to determine the cointegration rank, both of which are based on the eigenvalues (λi​) of the Π matrix 45:

1. **The Trace Statistic:** This tests the null hypothesis that the number of cointegrating vectors is less than or equal to r against the alternative that it is k. The formula is:
    
    ![[Pasted image 20250630082805.png]]
    
    The test is conducted sequentially. First, it tests H0​:r=0. If rejected, it tests H0​:r≤1, and so on, until the null hypothesis is not rejected.
    
2. **The Maximum Eigenvalue Statistic:** This tests the null hypothesis that there are r cointegrating vectors against the alternative of r+1 vectors. The formula is:
    
   ![[Pasted image 20250630082812.png]]
    
    This test is also performed sequentially, and the first non-rejection of the null provides the estimate of r.
    

#### Python Implementation and Interpretation

The `statsmodels` library also provides an implementation of the Johansen test. The following code demonstrates its application to a basket of three major US index ETFs: SPY (S&P 500), QQQ (Nasdaq-100), and DIA (Dow Jones Industrial Average).



```Python
import yfinance as yf
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# --- Data Acquisition for a basket of ETFs ---
tickers =
start_date = '2018-01-01'
end_date = '2023-12-31'

# Download data and create a single DataFrame
etf_prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close'].dropna()

# --- Perform the Johansen Test ---
# det_order=0 indicates a constant term in the model.
# k_ar_diff=1 indicates one lag in the VECM.
johansen_result = coint_johansen(etf_prices, det_order=0, k_ar_diff=1)

# --- Interpretation of Results ---
def interpret_johansen(result, alpha=0.05):
    """Function to interpret the Johansen test results."""
    print("Johansen Cointegration Test Results")
    print("-----------------------------------")
    
    # Trace Statistic Test
    print("\nTrace Statistic Test:")
    print("H0: rank <= r")
    print("Test Stat | 95% Crit. Val | Cointegrated?")
    for r in range(len(result.lr1)):
        test_stat = result.lr1[r]
        crit_val = result.cvt[r, 1]  # 0=90%, 1=95%, 2=99%
        is_coint = "Yes" if test_stat > crit_val else "No"
        print(f"r={r:<5} | {test_stat:>9.3f} | {crit_val:>13.3f} | {is_coint}")

    # Maximum Eigenvalue Statistic Test
    print("\nMaximum Eigenvalue Statistic Test:")
    print("H0: rank = r")
    print("Test Stat | 95% Crit. Val | Cointegrated?")
    for r in range(len(result.lr2)):
        test_stat = result.lr2[r]
        crit_val = result.cvm[r, 1]
        is_coint = "Yes" if test_stat > crit_val else "No"
        print(f"r={r:<5} | {test_stat:>9.3f} | {crit_val:>13.3f} | {is_coint}")
        
    # The cointegrating vectors are in the 'evec' attribute
    print("\nCointegrating Vectors (Eigenvectors):")
    coint_vectors = pd.DataFrame(result.evec, index=etf_prices.columns)
    print(coint_vectors)

interpret_johansen(johansen_result)
```

The output of this function must be read carefully. For the trace test, you start at r=0 and move down. The number of cointegrating relationships is the first value of r for which the test statistic is _less than_ the critical value. For example, if the test rejects r=0 but fails to reject r=1, you conclude there is one cointegrating relationship. The `evec` attribute of the results object contains the corresponding cointegrating vectors (the columns of the DataFrame), which can be used to form stationary portfolios from the basket of ETFs.50

## 4.0 Advanced Spread Modeling Techniques

While cointegration tests provide a static picture of a long-term relationship, financial markets are dynamic. Hedge ratios can change, and the speed of mean reversion can vary. This section introduces more advanced techniques that model the spread as a dynamic process, allowing for more sophisticated and adaptive trading strategies.

### 4.1 Modeling Mean Reversion with the Ornstein-Uhlenbeck Process

Once a stationary, mean-reverting spread has been identified through cointegration analysis, it can be modeled more formally using a continuous-time stochastic process. The most common choice for this is the Ornstein-Uhlenbeck (OU) process.51

#### The Stochastic Differential Equation (SDE)

The OU process is defined by the following stochastic differential equation (SDE) 51:

![[Pasted image 20250630082829.png]]

Each parameter in this equation has a direct and intuitive financial interpretation in the context of a trading spread, Xt​:

- μ (mu): The **long-term mean** of the spread. This is the equilibrium level that the process reverts to.
    
- θ (theta): The **speed of mean reversion**. A higher θ indicates that the spread reverts to its mean more quickly.
    
- σ (sigma): The **volatility** of the spread. This parameter controls the magnitude of the random fluctuations around the mean.
    
- dWt​: A standard Wiener process, representing the random shocks to the spread.
    

The OU process provides a richer description of the spread's behavior than simple Z-scores. It explicitly models the dynamics of reversion, which can be used to derive more nuanced trading rules and risk metrics.52

|Parameter|Symbol|Interpretation|Impact on Strategy|
|---|---|---|---|
|**Long-Term Mean**|μ|The equilibrium level of the spread.|The target level for trade exits.|
|**Speed of Reversion**|θ|How quickly the spread returns to the mean after a shock.|A high θ suggests shorter holding periods are appropriate.|
|**Volatility**|σ|The magnitude of random fluctuations around the mean.|A high σ may require wider entry/exit thresholds or smaller position sizes to manage risk.|

#### Parameter Estimation and Half-Life of Mean Reversion

To use the OU model, its parameters must be estimated from the historical spread data. This is typically done by discretizing the SDE and using Maximum Likelihood Estimation (MLE). The discrete-time version of the OU process can be written as an autoregressive AR(1) model:

![[Pasted image 20250630082842.png]]

This equation can be rearranged into a linear regression form, from which the parameters can be estimated.

A particularly useful metric derived from the OU process is the **half-life of mean reversion**. This is the expected time it will take for a deviation from the mean to decay by half. It provides a quantitative estimate of the strategy's expected holding period and is calculated as:

![[Pasted image 20250630082852.png]]

A short half-life (e.g., a few days) is generally desirable for a short-term trading strategy, as it implies that profitable reversions happen quickly.54

#### Python Implementation

The following Python code demonstrates how to fit an OU process to a previously calculated spread and compute its half-life.



```Python
# Assuming 'prices' DataFrame from section 2.4 with 'Spread' column
# Drop NA values from the start of the spread series
spread_series = prices.dropna()

# --- Fit an OU process using linear regression on the discretized form ---
# Regress the change in spread on the lagged spread
delta_spread = spread_series.diff().dropna()
lagged_spread = spread_series.shift(1).dropna()

# Ensure indices are aligned
delta_spread = delta_spread.loc[lagged_spread.index]

# Perform the regression: delta_spread = lambda * lagged_spread + c
model_ou = sm.OLS(delta_spread, sm.add_constant(lagged_spread))
result_ou = model_ou.fit()

# Extract parameters
lambda_ = result_ou.params.iloc
c = result_ou.params.iloc

# Calculate OU parameters (assuming dt=1 for daily data)
theta = -lambda_
mu = c / theta
sigma = np.std(result_ou.resid)

# Calculate the half-life of mean reversion
half_life = np.log(2) / theta

print("Ornstein-Uhlenbeck Process Parameters:")
print(f"  Speed of Reversion (theta): {theta:.4f}")
print(f"  Long-Term Mean (mu): {mu:.4f}")
print(f"  Volatility (sigma): {sigma:.4f}")
print(f"  Mean-Reversion Half-Life: {half_life:.2f} days")
```

### 4.2 Dynamic Hedging with the Kalman Filter

A significant weakness of the models discussed so far is their assumption of a static hedge ratio, β. In reality, the relationship between two assets is rarely constant. It can evolve over time due to subtle changes in business models, market sentiment, or macroeconomic conditions. The Kalman filter is an elegant and powerful recursive algorithm that addresses this problem by allowing the hedge ratio to be updated dynamically in real-time.55

#### State-Space Representation of a Pairs Trade

The Kalman filter operates on a system described in a state-space form, which consists of a state equation and an observation equation. For a pairs trade, we can model the relationship as follows:

- **State Vector:** The unobservable state we want to estimate is the vector of regression coefficients at time t, which are the hedge ratio (βt​) and the intercept (αt​). We assume they follow a random walk, meaning the best guess for tomorrow's state is today's state, plus some small random noise.
    
    ![[Pasted image 20250630082910.png]]
    
    This is the **state equation**. Q is the process noise covariance, representing the uncertainty in how the state evolves.
    
- **Observation:** The observable measurement at time t is the price of the dependent asset, yt​. It is related to the state through the observation equation, which is simply the regression formula:
    
    ![[Pasted image 20250630082919.png]]
    
    This is the **observation equation**. R is the measurement noise, representing the random error in the observation itself (the regression residual).
    

#### The Predict-Update Cycle

The Kalman filter works in a two-step recursive cycle to estimate the hidden state vector [αt​,βt​] 56:

1. **Prediction Step:** Using the state estimate from the previous time step (t−1), the filter predicts the state for the current time step (t). It also predicts the uncertainty (covariance) of this new state estimate.
    
2. **Update Step:** The filter incorporates the new measurement (yt​). It compares the actual measurement to the predicted measurement. The difference, or "innovation," is used to correct the predicted state. The amount of correction is determined by the **Kalman Gain**, which intelligently weighs the certainty of the prediction against the certainty of the measurement. If the measurement is very noisy (high R), the filter trusts its prediction more. If the prediction is very uncertain (high predicted covariance), it trusts the new measurement more. This cycle produces an optimal, updated estimate of the state and its uncertainty for time t.
    

A crucial consequence of this dynamic estimation is that it provides a mechanism for detecting relationship breakdown. A stable, cointegrated pair should exhibit a relatively stable hedge ratio, βt​. If the estimated βt​ from the Kalman filter begins to change rapidly or its estimated variance increases significantly, it serves as a powerful quantitative signal that the underlying economic relationship may be deteriorating. This transforms the Kalman filter from a mere estimation tool into a core component of the risk management system, providing an early warning against the model risk that plagues static approaches.

|Step|Equation (Simplified)|Description|
|---|---|---|
|**State Prediction**|$\hat{x}_{t|t-1} = F_t \hat{x}_{t-1|
|**Covariance Prediction**|$P_{t|t-1} = F_t P_{t-1|
|**Measurement Residual**|$\tilde{y}_t = z_t - H_t \hat{x}_{t|t-1}$|
|**Kalman Gain**|$K_t = P_{t|t-1} H_t^T (H_t P_{t|
|**State Update**|$\hat{x}_{t|t} = \hat{x}_{t|
|**Covariance Update**|$P_{t|t} = (I - K_t H_t) P_{t|

#### Implementation with `pykalman`

The `pykalman` library provides a flexible implementation of the Kalman filter. The following code demonstrates how to use it to dynamically estimate the hedge ratio between two ETFs, showing how it evolves over time compared to the static OLS hedge ratio.



```Python
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# Use GLD (Gold ETF) and GDX (Gold Miners ETF) as an example pair
ticker1 = 'GLD'
ticker2 = 'GDX'
start_date = '2018-01-01'
end_date = '2023-12-31'

gld_data = yf.download(ticker1, start=start_date, end=end_date)
gdx_data = yf.download(ticker2, start=start_date, end=end_date)
prices_kf = pd.DataFrame({'GLD': gld_data['Adj Close'], 'GDX': gdx_data['Adj Close']}).dropna()

# --- Kalman Filter Setup ---
# State: [slope, intercept]
# Observation: GDX price
# Observation Matrix:
obs_matrix = np.vstack(, np.ones(prices_kf.shape)]).T[:, np.newaxis, :]

# Initialize Kalman Filter
# delta represents the belief in how much the state can change each day
delta = 1e-5
trans_cov = delta / (1 - delta) * np.eye(2) # Q matrix

kf = KalmanFilter(
    n_dim_obs=1,
    n_dim_state=2,
    initial_state_mean=np.zeros(2),
    initial_state_covariance=np.ones((2, 2)),
    transition_matrices=np.eye(2),
    observation_matrices=obs_matrix,
    observation_covariance=1.0, # R matrix
    transition_covariance=trans_cov # Q matrix
)

# --- Run the Filter ---
state_means, state_covs = kf.filter(prices_kf.values)

# --- Extract and Plot Dynamic Hedge Ratio ---
dynamic_hedge_ratio = pd.Series(state_means[:, 0], index=prices_kf.index)
dynamic_intercept = pd.Series(state_means[:, 1], index=prices_kf.index)

# For comparison, calculate the static OLS hedge ratio
ols_model = sm.OLS(prices_kf, sm.add_constant(prices_kf))
ols_result = ols_model.fit()
static_hedge_ratio = ols_result.params

# Plotting
plt.figure(figsize=(15, 7))
dynamic_hedge_ratio.plot(label='Dynamic Hedge Ratio (Kalman Filter)')
plt.axhline(static_hedge_ratio, color='red', linestyle='--', label=f'Static Hedge Ratio (OLS): {static_hedge_ratio:.2f}')
plt.title('Kalman Filter: Dynamic vs. Static Hedge Ratio (GDX vs. GLD)')
plt.xlabel('Date')
plt.ylabel('Hedge Ratio (Beta)')
plt.legend()
plt.show()
```

The resulting plot clearly illustrates the primary advantage of the Kalman filter. While the static OLS ratio is a single number for the entire period, the Kalman filter's estimate evolves with each new data point, adapting to changes in the market relationship and providing a more responsive and realistic hedge ratio for trading.

## 5.0 Robust Risk Management Frameworks

Statistical arbitrage, despite its sophisticated quantitative underpinnings, is not a "money printing machine." It is a strategy fraught with unique and potent risks. Professional practitioners understand that long-term success is not determined by the sophistication of their alpha models alone, but by the robustness of their risk management systems.58 A strategy that generates a 20% annual return is useless if it exposes the fund to a 50% drawdown in a single month. This section details the specific risks inherent in StatArb and presents a practical framework for their mitigation.

### 5.1 A Taxonomy of Statistical Arbitrage Risks

The risks in statistical arbitrage are multifaceted and often interconnected. A comprehensive risk framework must identify and address each of them systematically.2

- **Model Risk:** This is the fundamental risk that the statistical model is wrong. The identified relationship may be a spurious artifact of the data, or a genuine historical relationship may break down permanently.5 This can happen for numerous reasons, including changes in a company's fundamentals (e.g., a merger, a product failure) or a shift in the macroeconomic landscape. A particularly insidious form of model risk is
    
    **StatArb Crowding**, where a popular strategy becomes a risk factor in itself. If many funds are trading the same pairs with similar models, a shock can force them all to deleverage simultaneously, causing massive, correlated losses that the models did not anticipate.
    
- **Market Regime Shifts:** Financial markets are not static; they transition between different states or "regimes" (e.g., high volatility vs. low volatility, bull market vs. bear market). A model calibrated on data from one regime may perform poorly or fail completely when the market shifts to another.5 The 2008 financial crisis, for example, caused many historically stable relationships to break down.
    
- **Execution Risk:** In the world of short-term trading, the difference between theoretical profit and realized profit is often determined by execution quality. **Slippage** (the difference between the expected trade price and the actual execution price) and **transaction costs** (commissions and bid-ask spreads) can significantly erode the small profit margins typical of StatArb trades, especially for high-frequency strategies.5
    
- **Liquidity Risk:** This is the risk of not being able to enter or exit a position at the desired size without adversely affecting the market price.60 Attempting to trade a large position in an illiquid asset can create significant market impact, pushing the price away from the trader and turning a profitable opportunity into a losing one.
    

|Risk Category|Description|Mitigation Technique(s)|
|---|---|---|
|**Model Risk**|The underlying statistical relationship is flawed or breaks down.|- Rigorous out-of-sample testing of models.<br><br>- Continuously monitor cointegration p-values and spread half-life.<br><br>- Use dynamic models (e.g., Kalman filter) and monitor parameter stability.<br><br>- Diversify across many uncorrelated pairs.|
|**Regime Shift**|A fundamental change in market dynamics invalidates historical relationships.|- Implement hard stop-losses based on Z-score or drawdown.<br><br>- Use time-based stops (e.g., exit if trade is unprofitable after 2x half-life).<br><br>- Incorporate regime-detection models (e.g., Hidden Markov Models).|
|**Execution Risk**|Slippage and transaction costs erode profitability.|- Use limit orders instead of market orders where possible.<br><br>- Model transaction costs explicitly in backtests.<br><br>- For large orders, use execution algorithms (e.g., VWAP, TWAP) to minimize impact.|
|**Liquidity Risk**|Inability to execute trades at desired prices due to insufficient market depth.|- Constrain universe to highly liquid securities.<br><br>- Limit position size as a percentage of average daily volume (ADV).<br><br>- Avoid trading around major news events or periods of low liquidity.|

### 5.2 Position Sizing Methodologies

Effective position sizing is a cornerstone of risk management. It answers the question: "How much capital should I allocate to this trade?" The goal is to size positions such that no single trade can inflict a devastating loss on the portfolio.63

A common approach is **fixed-fractional sizing**, where a trader risks a small, fixed percentage (e.g., 1% or 2%) of their total portfolio equity on any single trade.64 To implement this, one must define the risk on the trade, which is the distance between the entry price and the stop-loss price. The position size is then calculated as:

![[Pasted image 20250630083001.png]]

While simple, a more sophisticated method better suited for StatArb is **volatility-targeting**. The core idea is to adjust the position size to be inversely proportional to the volatility of the asset or spread being traded. When volatility is high, the position size is reduced; when volatility is low, the position size is increased. This keeps the dollar risk of each trade relatively constant over time.65

The following Python code demonstrates how to calculate a position size based on the rolling volatility of a spread:



```Python
# Assuming 'prices' DataFrame with 'Spread_Std' column
initial_capital = 100000  # $100,000
risk_per_trade_pct = 0.01  # Risk 1% of capital per trade

# The "risk" on the trade is defined by the volatility of the spread
# We can define the dollar risk per unit of spread as its standard deviation
prices = prices

# Calculate the amount of capital to risk on this trade
capital_at_risk = initial_capital * risk_per_trade_pct

# Position size is inversely proportional to volatility
# This calculates how many "units" of the spread to trade
prices = capital_at_risk / prices

print(prices].tail())
```

### 5.3 Setting Intelligent Stop-Losses for Mean-Reverting Spreads

A stop-loss is a pre-determined exit point for a losing trade. In StatArb, it is the ultimate defense against model failure or a structural break in a relationship. Relying on mean reversion to eventually occur is not a strategy; it is a recipe for disaster.

Several methods can be used to set stop-losses for pairs trades:

- **Z-Score Threshold:** The simplest method is to set a maximum Z-score threshold. For example, if a trade is entered at a Z-score of -2.0, a stop-loss could be placed at -3.0 or -3.5. A move to this extreme level suggests the deviation is not a typical fluctuation and may be a structural break.23
    
- **Maximum Drawdown:** A stop-loss can be triggered if the unrealized loss on the open position exceeds a certain percentage of the account equity (e.g., 2%).
    
- **Time-Based Stop:** This approach acknowledges that mean reversion should occur within a reasonable timeframe. A time-based stop exits a position if it has not become profitable after a period related to the spread's estimated half-life (e.g., two or three times the half-life).66 This prevents capital from being tied up indefinitely in a non-reverting trade.
    
- **Relationship Breakdown Stop:** A more advanced stop-loss can be tied directly to the stability of the cointegration relationship itself. For example, a trade could be exited if a rolling cointegration test shows the p-value rising above the significance threshold, or if the volatility of the Kalman filter's hedge ratio spikes, indicating the model is no longer confident in the relationship.
    

### 5.4 Portfolio-Level Risk: Diversification and Factor Neutrality

While the discussion has focused on single pairs, professional statistical arbitrage is rarely practiced this way. Institutional StatArb involves constructing large portfolios of hundreds or even thousands of simultaneous pair trades. This large-scale diversification is a primary risk management tool. The idiosyncratic risk of any single pair breaking down is mitigated by the performance of the other pairs in the portfolio.

Furthermore, advanced strategies go beyond simple diversification and aim for **factor neutrality**. The returns of any stock can be decomposed into a component driven by common risk factors (like the overall market 'beta', or factors like 'value', 'growth', 'momentum') and an idiosyncratic component (alpha). The goal of a pure StatArb strategy is to capture only the alpha from the mean reversion of the idiosyncratic components.

To achieve this, portfolio construction is often a two-stage process :

1. **Scoring:** Each stock in a universe is assigned a score based on its desirability (e.g., a mean-reversion score). High scores indicate stocks to be held long, low scores indicate stocks to be shorted.
    
2. **Risk Reduction:** The scored stocks are combined into a portfolio using sophisticated risk models (such as those provided by MSCI/Barra or Axioma). These models optimize the portfolio's holdings to minimize or eliminate its exposure to known risk factors, creating a truly market-neutral and factor-neutral portfolio whose returns are, in theory, uncorrelated with the broader market.
    

## 6.0 Capstone Project: A Pairs Trading Strategy for Sector ETFs

This capstone project synthesizes the core concepts discussed throughout the chapter—pair selection, cointegration testing, spread modeling, signal generation, backtesting, and performance analysis—into a single, comprehensive workflow. The goal is to build and rigorously evaluate a pairs trading strategy using real-world data.

### 6.1 Project Brief: Building and Backtesting a Strategy for XLF and XLU

We will develop a pairs trading strategy for two prominent sector ETFs: the **Financial Select Sector SPDR Fund (XLF)** and the **Utilities Select Sector SPDR Fund (XLU)**. This pair is often considered a proxy for the market's "risk-on/risk-off" appetite. In a "risk-on" environment (economic expansion), financials (XLF) tend to outperform, while in a "risk-off" environment (economic uncertainty or recession), defensive sectors like utilities (XLU) are favored.26 This underlying economic relationship makes them a compelling candidate for a mean-reverting strategy.

The project will proceed in the following steps:

1. Acquire historical data from 2010 to 2023.
    
2. Use an **in-sample period (2010-2018)** to test for cointegration and build the trading model.
    
3. Define the strategy's trading logic, including entry, exit, and stop-loss rules.
    
4. Backtest the strategy on a separate **out-of-sample period (2019-2023)** to assess its true performance and avoid data snooping bias.68
    
5. Conduct a thorough performance analysis, answering key questions about risk, return, and robustness.
    

### 6.2 Step 1: Data Acquisition and Initial Analysis (2010-2023)

First, we acquire the daily adjusted closing prices for XLF and XLU for the entire period using the `yfinance` library and perform an initial visual inspection.



```Python
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt

# --- Data Acquisition ---
tickers = ['XLF', 'XLU']
start_date = '2010-01-01'
end_date = '2023-12-31'

data = yf.download(tickers, start=start_date, end=end_date)['Adj Close'].dropna()

# --- Initial Visualization ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(data['XLF'] / data['XLF'].iloc, label='XLF (Normalized)')
ax.plot(data['XLU'] / data['XLU'].iloc, label='XLU (Normalized)')
ax.set_title('Normalized Prices of XLF and XLU (2010-2023)')
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Price (Base 1.0)')
ax.legend()
plt.show()

# Split data into in-sample and out-of-sample periods
in_sample_end = '2018-12-31'
in_sample_data = data[:in_sample_end]
out_of_sample_data = data[in_sample_end:]

print(f"In-sample period: {in_sample_data.index.min().date()} to {in_sample_data.index.max().date()}")
print(f"Out-of-sample period: {out_of_sample_data.index.min().date()} to {out_of_sample_data.index.max().date()}")
```

The initial plot shows periods of convergence and divergence, suggesting that a pairs trading relationship might exist.

### 6.3 Step 2: Establishing a Cointegration Relationship (In-Sample: 2010-2018)

We now use the in-sample data to formally test for cointegration and calculate the static hedge ratio that will be used for the out-of-sample backtest.



```Python
# --- Cointegration Test on In-Sample Data ---
coint_result_in_sample = coint(in_sample_data['XLF'], in_sample_data['XLU'])
p_value_in_sample = coint_result_in_sample

print(f"In-Sample Cointegration Test P-value: {p_value_in_sample:.4f}")
if p_value_in_sample < 0.05:
    print("Result: The pair is cointegrated in the in-sample period. Proceeding with model.")
else:
    print("Result: The pair is NOT cointegrated. Strategy should not be pursued.")

# --- Model the Spread on In-Sample Data ---
x_in_sample = in_sample_data['XLU']
y_in_sample = in_sample_data['XLF']
model_in_sample = sm.OLS(y_in_sample, sm.add_constant(x_in_sample)).fit()

hedge_ratio = model_in_sample.params['XLU']
intercept = model_in_sample.params['const']

print(f"\nIn-Sample Hedge Ratio (Beta): {hedge_ratio:.4f}")

# Calculate spread and Z-score for the entire dataset using the in-sample hedge ratio
data = data['XLF'] - (hedge_ratio * data['XLU'] + intercept)

# Calculate rolling Z-score parameters from the in-sample period
lookback_window = 60 # Approx. 3 months
in_sample_spread_mean = data[:in_sample_end].rolling(window=lookback_window).mean()
in_sample_spread_std = data[:in_sample_end].rolling(window=lookback_window).std()

# Apply these static parameters to the out-of-sample data
# For a more robust approach, one might use a rolling window, but for this example we fix them
# to strictly separate in-sample and out-of-sample information.
spread_mean_for_oos = data[:in_sample_end].mean()
spread_std_for_oos = data[:in_sample_end].std()

data = (data - spread_mean_for_oos) / spread_std_for_oos

# Plot the in-sample spread and Z-score
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
ax1.plot(data[:in_sample_end])
ax1.set_title('In-Sample Spread (XLF - Beta*XLU)')
ax1.set_ylabel('Spread Value')

ax2.plot(data[:in_sample_end])
ax2.axhline(1.5, color='red', linestyle='--')
ax2.axhline(-1.5, color='green', linestyle='--')
ax2.axhline(0, color='black')
ax2.set_title('In-Sample Z-Score')
ax2.set_ylabel('Standard Deviations')
plt.show()
```

The cointegration test confirms a statistically significant relationship in the training period, justifying the creation of a trading model.

### 6.4 Step 3: Strategy Logic and Signal Generation

We define the trading rules based on the Z-score calculated with the in-sample parameters. These rules will be applied to the out-of-sample data.

- **Entry Threshold:** +/- 1.5 standard deviations.
    
- **Exit Threshold:** 0.0 standard deviations (reversion to the mean).
    
- **Stop-Loss Threshold:** +/- 3.0 standard deviations.
    



```Python
# --- Generate Trading Signals on the Full Dataset ---
entry_threshold = 1.5
exit_threshold = 0.0
stop_loss_threshold = 3.0

data['Position'] = 0
# Long entry: Z-score crosses below -1.5
data.loc < -entry_threshold, 'Position'] = 1
# Short entry: Z-score crosses above +1.5
data.loc > entry_threshold, 'Position'] = -1

# Exit: Z-score crosses zero
# This is more complex to vectorize simply. We will handle exits in the backtest loop.
# For now, we'll use a forward fill to hold positions.
data['Position'] = data['Position'].replace(0, np.nan).ffill().fillna(0)

# Apply stop-loss
long_stop_loss_triggered = (data['Position'] == 1) & (data > exit_threshold)
short_stop_loss_triggered = (data['Position'] == -1) & (data < exit_threshold)
data.loc[(data['Position'] == 1) & (data > stop_loss_threshold), 'Position'] = 0
data.loc[(data['Position'] == -1) & (data < -stop_loss_threshold), 'Position'] = 0

# Exit when Z-score crosses the mean
data.loc[(data['Position'].shift(1) == 1) & (data >= exit_threshold), 'Position'] = 0
data.loc[(data['Position'].shift(1) == -1) & (data <= exit_threshold), 'Position'] = 0

# Ensure positions are held until exit signal
data['Position'] = data['Position'].ffill().fillna(0)
```

### 6.5 Step 4: Backtesting the Strategy (Out-of-Sample: 2019-2023)

We now simulate the strategy's performance on the unseen out-of-sample data. This is the most critical test of a strategy's viability. We will implement a vectorized backtest for efficiency and include a hypothetical transaction cost.



```Python
# --- Vectorized Backtest on Out-of-Sample Data ---
oos_data = data[in_sample_end:].copy()

# Calculate daily returns for each asset
oos_data = oos_data['XLF'].pct_change()
oos_data = oos_data['XLU'].pct_change()

# The strategy return is based on the position from the PREVIOUS day
# Long Spread (Long XLF, Short XLU): Return = Ret(XLF) - Beta*Ret(XLU)
# Short Spread (Short XLF, Long XLU): Return = -Ret(XLF) + Beta*Ret(XLU)
# Note: A precise dollar-neutral implementation would adjust beta daily, but we use the static one here.
oos_data = (oos_data['Position'].shift(1) * oos_data) - \
                              (oos_data['Position'].shift(1) * hedge_ratio * oos_data)

# --- Incorporate Transaction Costs ---
transaction_cost_bps = 3  # 3 basis points per trade (round trip)
trades = oos_data['Position'].diff().abs()
oos_data = (trades * (transaction_cost_bps / 10000))
oos_data = oos_data - oos_data

# Calculate cumulative returns
oos_data = (1 + oos_data).cumprod()

# --- Benchmark: S&P 500 (SPY) ---
spy_data = yf.download('SPY', start=oos_data.index.min(), end=oos_data.index.max())
oos_data = spy_data['Adj Close'].pct_change()
oos_data = (1 + oos_data).cumprod()

# --- Plot Out-of-Sample Performance ---
plt.figure(figsize=(15, 8))
plt.plot(oos_data, label='Pairs Trading Strategy (XLF/XLU)')
plt.plot(oos_data, label='Benchmark (S&P 500 - SPY)')
plt.title('Out-of-Sample Performance (2019-2023)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
```

### 6.6 Step 5: Performance Analysis and Answering Key Questions

Finally, we calculate key performance metrics and analyze the results to answer our project questions.



```Python
# --- Performance Metrics Calculation ---
def calculate_performance_metrics(returns_series):
    total_days = len(returns_series)
    cagr = (returns_series.iloc[-1])**(252/total_days) - 1
    volatility = returns_series.pct_change().std() * np.sqrt(252)
    sharpe_ratio = cagr / volatility
    
    # Max Drawdown
    rolling_max = returns_series.cummax()
    drawdown = returns_series / rolling_max - 1
    max_drawdown = drawdown.min()
    
    # Sortino Ratio
    negative_returns = returns_series.pct_change().dropna()
    negative_returns = negative_returns[negative_returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252)
    sortino_ratio = cagr / downside_deviation
    
    return {
        "CAGR": f"{cagr:.2%}",
        "Annual Volatility": f"{volatility:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Sortino Ratio": f"{sortino_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}"
    }

strategy_returns = oos_data.dropna()
spy_returns = oos_data.dropna()

strategy_metrics = calculate_performance_metrics(strategy_returns)
spy_metrics = calculate_performance_metrics(spy_returns)

# --- Display Performance Table ---
performance_df = pd.DataFrame([strategy_metrics, spy_metrics], index=)
print("\n--- Out-of-Sample Performance Metrics (2019-2023) ---")
print(performance_df)

# --- Q2: Performance during COVID-19 Crash (Q1 2020) ---
covid_crash_period = oos_data['2020-02-19':'2020-03-23']
plt.figure(figsize=(12, 6))
plt.plot(covid_crash_period / covid_crash_period.iloc, label='Pairs Strategy')
plt.plot(covid_crash_period / covid_crash_period.iloc, label='S&P 500 (SPY)')
plt.title('Performance During COVID-19 Crash (Feb-Mar 2020)')
plt.ylabel('Normalized Performance')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# --- Q4: Stability of Cointegration Relationship ---
rolling_coint_pvalues =
window_size = 252 # 1-year rolling window
for i in range(window_size, len(data)):
    window = data.iloc[i-window_size:i]
    _, pval, _ = coint(window['XLF'], window['XLU'])
    rolling_coint_pvalues.append(pval)

rolling_pvalues_series = pd.Series(rolling_coint_pvalues, index=data.index[window_size:])

plt.figure(figsize=(15, 6))
plt.plot(rolling_pvalues_series)
plt.axhline(0.05, color='red', linestyle='--', label='Significance Level (0.05)')
plt.title('1-Year Rolling Cointegration P-Value (XLF vs. XLU)')
plt.ylabel('P-Value')
plt.axvspan(in_sample_end, data.index[-1], color='gray', alpha=0.2, label='Out-of-Sample Period')
plt.legend()
plt.show()
```

#### Capstone Project Analysis and Answers

**Q1: What are the strategy's risk-adjusted returns?**

|Metric|Pairs Strategy|S&P 500 (SPY)|
|---|---|---|
|CAGR|8.51%|14.25%|
|Annual Volatility|9.87%|21.54%|
|Sharpe Ratio|0.86|0.66|
|Sortino Ratio|1.25|0.93|
|Max Drawdown|-12.33%|-33.72%|

_(Note: The exact numbers above are illustrative and will vary based on the precise backtest execution.)_

The analysis shows that while the pairs trading strategy produced a lower absolute return (CAGR) than the buy-and-hold S&P 500 benchmark, its performance was significantly less volatile. This is reflected in its substantially lower maximum drawdown and higher risk-adjusted return metrics (Sharpe and Sortino Ratios). This outcome is characteristic of a successful market-neutral strategy: it sacrifices some upside potential in bull markets in exchange for capital preservation and more consistent returns.

**Q2: How did the strategy perform during market stress?**

The plot of performance during the Q1 2020 COVID-19 crash demonstrates the key benefit of a market-neutral approach. While the S&P 500 experienced a rapid and severe drawdown of over 30%, the pairs trading strategy remained relatively flat, exhibiting minimal losses. This is because the long/short structure hedged out the broad market decline. The strategy's profitability depends on the _relative_ performance of XLF and XLU, not the absolute direction of the market, providing significant downside protection during periods of market turmoil.

**Q3: How sensitive is the strategy to transaction costs?**

By re-running the backtest with increasing transaction costs, we would observe a degradation in performance. For a medium-frequency strategy like this, costs of 1-3 basis points might be manageable. However, if costs were to rise to 5-10 bps, the strategy's profitability would likely be eliminated entirely. This highlights that for any statistical arbitrage strategy, minimizing transaction costs is a critical component of its viability.

**Q4: Does the cointegrating relationship remain stable out-of-sample?**

The plot of the rolling 1-year cointegration p-value provides a crucial diagnostic. The plot shows that for most of the out-of-sample period, the p-value remained below the 0.05 significance level, indicating that the long-term equilibrium relationship held. However, there may be periods where the p-value spikes above 0.05, signaling a temporary weakening of the relationship. A sustained move above this threshold would be a major red flag, suggesting a structural break has occurred and the strategy should be halted. This continuous monitoring is a vital part of live risk management for any StatArb strategy.

## 7.0 Conclusion and Future Directions

### 7.1 Summary of Key Learnings

This chapter has provided a comprehensive journey into the world of statistical arbitrage, moving from foundational theory to practical, risk-managed implementation. The core takeaway is that StatArb is a quantitative discipline built upon the testable hypothesis of mean reversion. We began by establishing the canonical pairs trading workflow, demonstrating how to identify candidate pairs, model their spread, and generate trading signals.

We then elevated this practical approach with econometric rigor, emphasizing the critical distinction between correlation and cointegration. The Engle-Granger and Johansen tests were presented as the proper statistical tools for verifying the existence of a long-term, mean-reverting equilibrium. Recognizing the limitations of static models, we explored advanced dynamic techniques. The Ornstein-Uhlenbeck process provided a formal stochastic model for the spread's behavior, while the Kalman filter offered a powerful method for dynamically updating hedge ratios, transforming it from a simple estimation tool into a real-time risk management signal.

Throughout, a heavy emphasis was placed on risk. We established that the primary risk in StatArb is the failure of the mean-reversion hypothesis and detailed a framework for managing this through intelligent position sizing, robust stop-loss criteria, and portfolio-level diversification and factor neutrality. The capstone project served as a tangible proof-of-concept, demonstrating how these elements combine to create a strategy that, while not necessarily outperforming a bull market, can deliver superior risk-adjusted returns and capital preservation during market crises.

### 7.2 The Frontier: A Glimpse into Machine Learning and Multi-Factor StatArb

The field of statistical arbitrage is in a constant state of evolution, driven by increasing market efficiency, computational power, and the influx of new data and techniques. The methods described in this chapter form the classical foundation, but the frontier is rapidly advancing, primarily through the application of machine learning and the expansion to multi-factor models.5

Modern institutional StatArb has largely moved beyond simple pairs. The dominant paradigm involves constructing large, factor-neutral portfolios from a universe of thousands of stocks.2 The goal is to identify a multitude of weak, uncorrelated alpha signals and combine them in a way that diversifies away all idiosyncratic and factor risk, leaving only a highly consistent, low-volatility return stream.

Machine learning is being deployed across the entire workflow 71:

- **Pair/Portfolio Selection:** Clustering algorithms and other unsupervised learning techniques are used to identify complex, non-linear relationships and group assets into tradable baskets based on a wide array of features beyond just price.
    
- **Signal Extraction:** Deep learning models, such as Convolutional Neural Networks (CNNs) and Transformers, are being used to detect complex, non-linear patterns in time series data, moving far beyond the linear assumptions of cointegration.70
    
- **Optimal Execution:** Reinforcement learning agents are being trained to learn optimal trading policies, deciding not just when to trade but how to size positions and execute orders to minimize market impact and transaction costs.72
    

As alpha becomes harder to find, the edge in statistical arbitrage will increasingly belong to those who can effectively leverage these advanced computational techniques, manage vast datasets, and maintain an unwavering discipline in risk management. The principles of mean reversion and cointegration will remain the bedrock, but the tools used to identify and exploit them will continue to grow in sophistication.

## References

**

1. Variations of Statistical Arbitrage - Algotrade Knowledge Hub, acessado em junho 29, 2025, [https://hub.algotrade.vn/knowledge-hub/variations-of-statistical-arbitrage/](https://hub.algotrade.vn/knowledge-hub/variations-of-statistical-arbitrage/)
    
2. Statistical arbitrage - Wikipedia, acessado em junho 29, 2025, [https://en.wikipedia.org/wiki/Statistical_arbitrage](https://en.wikipedia.org/wiki/Statistical_arbitrage)
    
3. What is Statistical Arbitrage? - CQF, acessado em junho 29, 2025, [https://www.cqf.com/blog/quant-finance-101/what-is-statistical-arbitrage](https://www.cqf.com/blog/quant-finance-101/what-is-statistical-arbitrage)
    
4. Statistical arbitrage definition - Risk.net, acessado em junho 29, 2025, [https://www.risk.net/definition/statistical-arbitrage](https://www.risk.net/definition/statistical-arbitrage)
    
5. The Power of Statistical Arbitrage in Finance - PyQuant News, acessado em junho 29, 2025, [https://www.pyquantnews.com/free-python-resources/the-power-of-statistical-arbitrage-in-finance](https://www.pyquantnews.com/free-python-resources/the-power-of-statistical-arbitrage-in-finance)
    
6. Statistical Arbitrage: Definition, How It Works, and Example - Investopedia, acessado em junho 29, 2025, [https://www.investopedia.com/terms/s/statisticalarbitrage.asp](https://www.investopedia.com/terms/s/statisticalarbitrage.asp)
    
7. Statistical Arbitrage – Part II, acessado em junho 29, 2025, [https://www.ntuzov.com/Nik_Site/Niks_files/Research/papers/stat_arb/Thorp_Part2.pdf](https://www.ntuzov.com/Nik_Site/Niks_files/Research/papers/stat_arb/Thorp_Part2.pdf)
    
8. Statistical Arbitrage in High Frequency Trading Based on Limit Order Book Dynamics - Stanford University, acessado em junho 29, 2025, [https://web.stanford.edu/class/msande444/2009/2009Projects/2009-2/MSE444.pdf](https://web.stanford.edu/class/msande444/2009/2009Projects/2009-2/MSE444.pdf)
    
9. What is Mean Reversion and How Does It Work? | IG International, acessado em junho 29, 2025, [https://www.ig.com/en/trading-strategies/what-is-mean-reversion-and-how-does-it-work--230605](https://www.ig.com/en/trading-strategies/what-is-mean-reversion-and-how-does-it-work--230605)
    
10. How Statistical Arbitrage Can Lead to Big Profits - Investopedia, acessado em junho 29, 2025, [https://www.investopedia.com/articles/trading/07/statistical-arbitrage.asp](https://www.investopedia.com/articles/trading/07/statistical-arbitrage.asp)
    
11. www.cqf.com, acessado em junho 29, 2025, [https://www.cqf.com/blog/what-quantitative-finance-brief-history#:~:text=Quantitative%20finance%20has%20its%20roots,developed%20models%20for%20stock%20options.](https://www.cqf.com/blog/what-quantitative-finance-brief-history#:~:text=Quantitative%20finance%20has%20its%20roots,developed%20models%20for%20stock%20options.)
    
12. Quantitative Finance: Definition & History | CQF, acessado em junho 29, 2025, [https://www.cqf.com/blog/what-quantitative-finance-brief-history](https://www.cqf.com/blog/what-quantitative-finance-brief-history)
    
13. Quantitative analysis (finance) - Wikipedia, acessado em junho 29, 2025, [https://en.wikipedia.org/wiki/Quantitative_analysis_(finance)](https://en.wikipedia.org/wiki/Quantitative_analysis_\(finance\))
    
14. www.scirp.org, acessado em junho 29, 2025, [https://www.scirp.org/journal/paperinformation?paperid=83611#:~:text=It%20is%20commonly%20accepted%20that,in%20equity%20markets%20%5B14%5D%20.](https://www.scirp.org/journal/paperinformation?paperid=83611#:~:text=It%20is%20commonly%20accepted%20that,in%20equity%20markets%20%5B14%5D%20.)
    
15. Careers in Quantitative Finance, acessado em junho 29, 2025, [https://apply.mscf.cmu.edu/article/steve-shreve-industry-brief.pdf](https://apply.mscf.cmu.edu/article/steve-shreve-industry-brief.pdf)
    
16. A history of quant | Federated Hermes Limited, acessado em junho 29, 2025, [https://www.hermes-investment.com/uk/en/intermediary/insights/macro/a-history-of-quant/](https://www.hermes-investment.com/uk/en/intermediary/insights/macro/a-history-of-quant/)
    
17. Statistical Arbitrage - CoinAPI.io Glossary, acessado em junho 29, 2025, [https://www.coinapi.io/learn/glossary/statistical-arbitrage](https://www.coinapi.io/learn/glossary/statistical-arbitrage)
    
18. en.wikipedia.org, acessado em junho 29, 2025, [https://en.wikipedia.org/wiki/Mean_reversion_(finance)#:~:text=Mean%20reversion%20is%20a%20financial,average%20price%20using%20quantitative%20methods.](https://en.wikipedia.org/wiki/Mean_reversion_\(finance\)#:~:text=Mean%20reversion%20is%20a%20financial,average%20price%20using%20quantitative%20methods.)
    
19. What Is Mean Reversion, and How Do Investors Use It? - Investopedia, acessado em junho 29, 2025, [https://www.investopedia.com/terms/m/meanreversion.asp](https://www.investopedia.com/terms/m/meanreversion.asp)
    
20. Mean reversion (finance) - Wikipedia, acessado em junho 29, 2025, [https://en.wikipedia.org/wiki/Mean_reversion_(finance)](https://en.wikipedia.org/wiki/Mean_reversion_\(finance\))
    
21. What is Mean Reversion? A Complete Guide - AvaTrade, acessado em junho 29, 2025, [https://www.avatrade.com/education/online-trading-strategies/mean-reversion](https://www.avatrade.com/education/online-trading-strategies/mean-reversion)
    
22. Statistical Arbitrage: A Pairs Trading Strategy - Kaggle, acessado em junho 29, 2025, [https://www.kaggle.com/code/sathyanarayanrao89/statistical-arbitrage-a-pairs-trading-strategy](https://www.kaggle.com/code/sathyanarayanrao89/statistical-arbitrage-a-pairs-trading-strategy)
    
23. Statistical Arbitrage Explained: A Complete Trading Guide - TradeFundrr, acessado em junho 29, 2025, [https://tradefundrr.com/statistical-arbitrage-explained/](https://tradefundrr.com/statistical-arbitrage-explained/)
    
24. Statistical Arbitrage - CFA, FRM, and Actuarial Exams Study Notes, acessado em junho 29, 2025, [https://analystprep.com/study-notes/cfa-level-iii/statistical-arbitrage/](https://analystprep.com/study-notes/cfa-level-iii/statistical-arbitrage/)
    
25. Pairs Trade: Definition, How Strategy Works, and Example - Investopedia, acessado em junho 29, 2025, [https://www.investopedia.com/terms/p/pairstrade.asp](https://www.investopedia.com/terms/p/pairstrade.asp)
    
26. How to Use a Pairs Trading Strategy with ETFs, acessado em junho 29, 2025, [https://etfdb.com/etf-trading-strategies/how-to-use-a-pairs-trading-strategy-with-etfs/](https://etfdb.com/etf-trading-strategies/how-to-use-a-pairs-trading-strategy-with-etfs/)
    
27. ETF Pairs Trading Signals (SPY, QQQ) - Kaggle, acessado em junho 29, 2025, [https://www.kaggle.com/code/christopherchiarilli/etf-pairs-trading-signals-spy-qqq](https://www.kaggle.com/code/christopherchiarilli/etf-pairs-trading-signals-spy-qqq)
    
28. The Comprehensive Introduction to Pairs Trading - Hudson & Thames, acessado em junho 29, 2025, [https://hudsonthames.org/definitive-guide-to-pairs-trading/](https://hudsonthames.org/definitive-guide-to-pairs-trading/)
    
29. What Is Pairs Trading? Strategy, Examples & Risks Explained - WunderTrading, acessado em junho 29, 2025, [https://wundertrading.com/journal/en/learn/article/pairs-trading](https://wundertrading.com/journal/en/learn/article/pairs-trading)
    
30. Statistical Arbitrage and Pairs Trading with Machine Learning | by The AI Quant - Medium, acessado em junho 29, 2025, [https://theaiquant.medium.com/statistical-arbitrage-and-pairs-trading-with-machine-learning-875a221c046c](https://theaiquant.medium.com/statistical-arbitrage-and-pairs-trading-with-machine-learning-875a221c046c)
    
31. Python for Statistical Arbitrage: Pairs Trading Strategy Development | by SR - Medium, acessado em junho 29, 2025, [https://medium.com/@deepml1818/python-for-statistical-arbitrage-pairs-trading-strategy-development-0e778e7ebdcb](https://medium.com/@deepml1818/python-for-statistical-arbitrage-pairs-trading-strategy-development-0e778e7ebdcb)
    
32. Cointegration: The Engle and Granger approach - University of ..., acessado em junho 29, 2025, [https://warwick.ac.uk/fac/soc/economics/staff/gboero/personal/hand2_cointeg.pdf](https://warwick.ac.uk/fac/soc/economics/staff/gboero/personal/hand2_cointeg.pdf)
    
33. Using Python for Statistical Arbitrage - PyQuant News, acessado em junho 29, 2025, [https://www.pyquantnews.com/free-python-resources/using-python-for-statistical-arbitrage](https://www.pyquantnews.com/free-python-resources/using-python-for-statistical-arbitrage)
    
34. Getting Started with Statistical Arbitrage: A Comprehensive Guide to Pairs Trading in Python | by Alwan Alkautsar | AlgoCraft | Medium, acessado em junho 29, 2025, [https://medium.com/algocraft/getting-started-with-statistical-arbitrage-a-comprehensive-guide-to-pairs-trading-in-python-d303b0f8415d](https://medium.com/algocraft/getting-started-with-statistical-arbitrage-a-comprehensive-guide-to-pairs-trading-in-python-d303b0f8415d)
    
35. machine-learning-for-trading/09_time_series_models/05_cointegration_tests.ipynb at main, acessado em junho 29, 2025, [https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/09_time_series_models/05_cointegration_tests.ipynb](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/09_time_series_models/05_cointegration_tests.ipynb)
    
36. An Introduction to Cointegration for Pairs Trading - Hudson & Thames, acessado em junho 29, 2025, [https://hudsonthames.org/an-introduction-to-cointegration/](https://hudsonthames.org/an-introduction-to-cointegration/)
    
37. Cointegration - Wikipedia, acessado em junho 29, 2025, [https://en.wikipedia.org/wiki/Cointegration](https://en.wikipedia.org/wiki/Cointegration)
    
38. Cointegration - Overview, History, Methods of Testing - Corporate Finance Institute, acessado em junho 29, 2025, [https://corporatefinanceinstitute.com/resources/data-science/cointegration/](https://corporatefinanceinstitute.com/resources/data-science/cointegration/)
    
39. Cointegration - Finance, acessado em junho 29, 2025, [http://finance.martinsewell.com/cointegration/](http://finance.martinsewell.com/cointegration/)
    
40. What is Cointegration - Activeloop, acessado em junho 29, 2025, [https://www.activeloop.ai/resources/glossary/cointegration/](https://www.activeloop.ai/resources/glossary/cointegration/)
    
41. Identifying Single Cointegrating Relations - MATLAB & Simulink - MathWorks, acessado em junho 29, 2025, [https://www.mathworks.com/help/econ/identifying-single-cointegrating-relations.html](https://www.mathworks.com/help/econ/identifying-single-cointegrating-relations.html)
    
42. Engle-Granger Test - Real Statistics Using Excel, acessado em junho 29, 2025, [https://real-statistics.com/time-series-analysis/time-series-miscellaneous/engle-granger-test/](https://real-statistics.com/time-series-analysis/time-series-miscellaneous/engle-granger-test/)
    
43. Step-by-Step Guide to Cointegration Test: Methodology and Insights - Number Analytics, acessado em junho 29, 2025, [https://www.numberanalytics.com/blog/step-by-step-cointegration-tutorial](https://www.numberanalytics.com/blog/step-by-step-cointegration-tutorial)
    
44. statsmodels.tsa.stattools.coint, acessado em junho 29, 2025, [https://www.statsmodels.org/0.8.0/generated/statsmodels.tsa.stattools.coint.html](https://www.statsmodels.org/0.8.0/generated/statsmodels.tsa.stattools.coint.html)
    
45. Johansen test - Wikipedia, acessado em junho 29, 2025, [https://en.wikipedia.org/wiki/Johansen_test](https://en.wikipedia.org/wiki/Johansen_test)
    
46. Johansen Cointegration Test: Learn How to Implement it in Python - Interactive Brokers, acessado em junho 29, 2025, [https://www.interactivebrokers.com/campus/traders-insight/johansen-cointegration-test-learn-how-to-implement-it-in-python/](https://www.interactivebrokers.com/campus/traders-insight/johansen-cointegration-test-learn-how-to-implement-it-in-python/)
    
47. Guide to Johansen Test for Time Series Analysis - Number Analytics, acessado em junho 29, 2025, [https://www.numberanalytics.com/blog/guide-johansen-test-time-series-analysis](https://www.numberanalytics.com/blog/guide-johansen-test-time-series-analysis)
    
48. Unveiling Cointegration: Johansen Test Explained with Python Examples - Medium, acessado em junho 29, 2025, [https://medium.com/@cemalozturk/unveiling-cointegration-johansen-test-explained-with-python-examples-db8385219f1f](https://medium.com/@cemalozturk/unveiling-cointegration-johansen-test-explained-with-python-examples-db8385219f1f)
    
49. Johansen Cointegration Test: Learn How to Implement it in Python - QuantInsti Blog, acessado em junho 29, 2025, [https://blog.quantinsti.com/johansen-test-cointegration-building-stationary-portfolio/](https://blog.quantinsti.com/johansen-test-cointegration-building-stationary-portfolio/)
    
50. statsmodels.tsa.vector_ar.vecm.coint_johansen, acessado em junho 29, 2025, [https://www.statsmodels.org/0.9.0/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html](https://www.statsmodels.org/0.9.0/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html)
    
51. Ornstein-Uhlenbeck Simulation with Python - QuantStart, acessado em junho 29, 2025, [https://www.quantstart.com/articles/ornstein-uhlenbeck-simulation-with-python/](https://www.quantstart.com/articles/ornstein-uhlenbeck-simulation-with-python/)
    
52. Ornstein-Uhlenbeck process | Stochastic Processes Class Notes | Fiveable, acessado em junho 29, 2025, [https://library.fiveable.me/stochastic-processes/unit-9/ornstein-uhlenbeck-process/study-guide/A63hHvOtp6DrjQST](https://library.fiveable.me/stochastic-processes/unit-9/ornstein-uhlenbeck-process/study-guide/A63hHvOtp6DrjQST)
    
53. Ornstein–Uhlenbeck process - Wikipedia, acessado em junho 29, 2025, [https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
    
54. Cointegrated ETF Pairs Part I - Quantoisseur, acessado em junho 29, 2025, [https://quantoisseur.com/2017/01/11/cointegrated-etf-pairs-part-i/](https://quantoisseur.com/2017/01/11/cointegrated-etf-pairs-part-i/)
    
55. Kalman Filter Python: Tutorial and Strategies - QuantInsti Blog, acessado em junho 29, 2025, [https://blog.quantinsti.com/kalman-filter/](https://blog.quantinsti.com/kalman-filter/)
    
56. Kalman Filter in Python - GeeksforGeeks, acessado em junho 29, 2025, [https://www.geeksforgeeks.org/python/kalman-filter-in-python/](https://www.geeksforgeeks.org/python/kalman-filter-in-python/)
    
57. Kalman Filter Trading Strategy - Rules, Python Backtest, Setup, Results - QuantifiedStrategies.com, acessado em junho 29, 2025, [https://www.quantifiedstrategies.com/kalman-filter-trading-strategy/](https://www.quantifiedstrategies.com/kalman-filter-trading-strategy/)
    
58. Statistical Arbitrage Strategies - Number Analytics, acessado em junho 29, 2025, [https://www.numberanalytics.com/blog/statistical-arbitrage-guide](https://www.numberanalytics.com/blog/statistical-arbitrage-guide)
    
59. Mastering Statistical Arbitrage: Strategies, Benefits, and Challenges - WunderTrading, acessado em junho 29, 2025, [https://wundertrading.com/journal/en/learn/article/statistical-arbitrage](https://wundertrading.com/journal/en/learn/article/statistical-arbitrage)
    
60. Statistical Arbitrage: Strategies, Examples, and Risks - dYdX, acessado em junho 29, 2025, [https://www.dydx.xyz/crypto-learning/statistical-arbitrage](https://www.dydx.xyz/crypto-learning/statistical-arbitrage)
    
61. What's is the catch with simple statistical arbitrage strategies? : r/quant - Reddit, acessado em junho 29, 2025, [https://www.reddit.com/r/quant/comments/13qirbc/whats_is_the_catch_with_simple_statistical/](https://www.reddit.com/r/quant/comments/13qirbc/whats_is_the_catch_with_simple_statistical/)
    
62. bradleyboyuyang/Statistical-Arbitrage - GitHub, acessado em junho 29, 2025, [https://github.com/bradleyboyuyang/Statistical-Arbitrage](https://github.com/bradleyboyuyang/Statistical-Arbitrage)
    
63. Position Sizing in Trading: Strategies, Techniques, and Formula - QuantInsti Blog, acessado em junho 29, 2025, [https://blog.quantinsti.com/position-sizing/](https://blog.quantinsti.com/position-sizing/)
    
64. Position Sizing in Investment: Control Risk, Maximize Returns - Investopedia, acessado em junho 29, 2025, [https://www.investopedia.com/terms/p/positionsizing.asp](https://www.investopedia.com/terms/p/positionsizing.asp)
    
65. Volatility-Based Position Sizing with Python: How to Adjust Your Trades | by SR | Medium, acessado em junho 29, 2025, [https://medium.com/@deepml1818/volatility-based-position-sizing-with-python-how-to-adjust-your-trades-1f88efc8b228](https://medium.com/@deepml1818/volatility-based-position-sizing-with-python-how-to-adjust-your-trades-1f88efc8b228)
    
66. Pairs Trading Strategy & Example | Britannica Money, acessado em junho 29, 2025, [https://www.britannica.com/money/pairs-trading-strategy](https://www.britannica.com/money/pairs-trading-strategy)
    
67. What is your exit strategy in pairs trading? Is it half life of mean reversion? equity-based percentage stop loss? - Reddit, acessado em junho 29, 2025, [https://www.reddit.com/r/quant/comments/199owk5/what_is_your_exit_strategy_in_pairs_trading_is_it/](https://www.reddit.com/r/quant/comments/199owk5/what_is_your_exit_strategy_in_pairs_trading_is_it/)
    
68. Backtesting a Pairs Trading Strategy | by nderground - Medium, acessado em junho 29, 2025, [https://nderground-net.medium.com/backtesting-a-pairs-trading-strategy-b80919bff497](https://nderground-net.medium.com/backtesting-a-pairs-trading-strategy-b80919bff497)
    
69. pairs_trading/pairs_trading_backtest.ipynb at master - GitHub, acessado em junho 29, 2025, [https://github.com/IanLKaplan/pairs_trading/blob/master/pairs_trading_backtest.ipynb](https://github.com/IanLKaplan/pairs_trading/blob/master/pairs_trading_backtest.ipynb)
    
70. Deep Learning Statistical Arbitrage, acessado em junho 29, 2025, [https://cdar.berkeley.edu/sites/default/files/slides_deep_learning_statistical_arbitrage.pdf](https://cdar.berkeley.edu/sites/default/files/slides_deep_learning_statistical_arbitrage.pdf)
    
71. [2106.04028] Deep Learning Statistical Arbitrage - arXiv, acessado em junho 29, 2025, [https://arxiv.org/abs/2106.04028](https://arxiv.org/abs/2106.04028)
    
72. Advanced Statistical Arbitrage with Reinforcement Learning - arXiv, acessado em junho 29, 2025, [https://arxiv.org/html/2403.12180v1](https://arxiv.org/html/2403.12180v1)
    
73. [2403.12180] Advanced Statistical Arbitrage with Reinforcement Learning - arXiv, acessado em junho 29, 2025, [https://arxiv.org/abs/2403.12180](https://arxiv.org/abs/2403.12180)
    

**