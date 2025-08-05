# Chapter 3: Financial Econometrics and Quantitative Analysis

## Part 1.1: An Introduction to Factor Models

### From a Single Factor to a Multi-Factor World

Modern portfolio theory began with a powerful, simplifying idea: an asset's risk and return could be understood through its relationship with a single, dominant economic force. This concept was crystallized in the Capital Asset Pricing Model (CAPM), developed in the 1960s by researchers including Jack Treynor, William F. Sharpe, and John Lintner. The CAPM posits that the expected return on any asset is a linear function of its sensitivity to the only source of priced, non-diversifiable risk: the overall market portfolio. This sensitivity is famously known as beta (β). In the world of CAPM, investors are only rewarded for taking on systematic risk—the risk that cannot be eliminated through diversification—and all other risks are considered idiosyncratic and uncompensated.

The model's elegant simplicity made it a cornerstone of financial theory and practice for decades. However, as researchers subjected the CAPM to rigorous empirical testing, cracks began to appear. A growing body of evidence in the 1980s and early 1990s revealed market anomalies—persistent patterns in stock returns that the single-factor CAPM could not adequately explain.4 Studies consistently found that certain company characteristics, beyond market beta, had significant predictive power for future returns. Most notably, researchers observed that companies with small market capitalizations (small-cap stocks) and companies with high book-to-market ratios (value stocks) tended to generate higher returns over the long run than the CAPM would predict.6 These findings suggested that the single-factor view of the world was incomplete and that other systematic risk factors were at play, setting the stage for the development of multi-factor models.5

### The General Factor Model Framework

The move from a single-factor to a multi-factor world required a more general mathematical framework. This framework proposes that an asset's expected return is not driven by one source of risk, but by its sensitivity to several systematic risk factors. These factors can be macroeconomic variables (like inflation or GDP growth), statistical constructs, or, as we will see, portfolio-based factors that capture specific market anomalies.9

The generalized structure of a linear factor model can be expressed as follows:

![[Pasted image 20250702081215.png]]

Where:

- E(Ri​) is the expected return on asset _i_.
    
- Rf​ is the risk-free rate of return (e.g., the yield on a short-term government bond).
    
- k is the number of systematic risk factors.
    
- βij​ is the sensitivity, or _factor loading_, of asset _i_ to factor _j_. It measures how much the asset's return is expected to change for a one-unit change in the factor, holding all other factors constant.11
    
- λj​ is the risk premium associated with factor _j_. It represents the excess return that investors expect to receive for bearing one unit of risk from that specific factor.10
    

This equation serves as the fundamental blueprint for the models discussed in this chapter, including the Arbitrage Pricing Theory (APT) and the Fama-French models. While they share this common structure, their theoretical foundations and the methods for identifying the factors differ significantly.

## Part 1.2: The Arbitrage Pricing Theory (APT)

### The Theoretical Underpinnings: No-Arbitrage and the Law of One Price

The Arbitrage Pricing Theory (APT), developed by economist Stephen Ross in 1976, provided a powerful alternative to the CAPM.14 Rather than being built on assumptions about investor utility and market equilibrium, APT is grounded in a more fundamental and less restrictive economic principle: the law of one price and the absence of arbitrage opportunities.16

Arbitrage is the practice of earning a risk-free profit by simultaneously buying and selling identical or similar assets in different markets to take advantage of a price discrepancy.10 The core tenet of APT is that in an efficient market, such opportunities cannot persist. Rational investors will immediately exploit any mispricing, and their collective actions will drive prices back into alignment, thereby eliminating the arbitrage opportunity.9

From this no-arbitrage principle, APT derives its central conclusion: the expected return of an asset must be a linear function of its sensitivities to various systematic risk factors.9 If two assets or portfolios possess the exact same exposures to all sources of systematic risk, they must offer the same expected return. If they did not, an investor could construct a zero-investment, zero-risk portfolio by shorting the asset with the lower expected return and buying the asset with the higher expected return, locking in a guaranteed profit. The relentless pursuit of such profits by arbitrageurs ensures that assets are priced according to their factor exposures.14

### Mathematical Formulation and Assumptions

The APT's core prediction is captured by a multi-factor pricing equation. For an asset _i_, its expected return, E(Ri​), is determined by the risk-free rate, Rf​, and a series of risk premiums associated with _k_ systematic factors 1:

![[Pasted image 20250702081231.png]]

Here, βij​ represents the sensitivity of asset _i_ to factor _j_, and λj​ is the risk premium for that factor. The actual return of the asset, Ri​, can be described by adding an asset-specific random error term, ϵi​, which represents the unsystematic or idiosyncratic risk 16:

![[Pasted image 20250702081239.png]]

Where Fj​ is the value of the _j_-th systematic factor.

The theory rests on a few key assumptions, which are generally considered less restrictive than those of the CAPM 10:

1. **Linear Factor Structure:** Asset returns can be described by a linear factor model, as shown in the equation above.11
    
2. **No Arbitrage:** No arbitrage opportunities exist among well-diversified portfolios. This implies that investors cannot earn riskless profits.10
    
3. **Diversifiable Idiosyncratic Risk:** The unsystematic risk component, ϵi​, is specific to each asset and can be eliminated by holding a large, well-diversified portfolio. The model assumes these error terms are uncorrelated across assets.10
    

### APT vs. CAPM: A Tale of Two Theories

While both APT and CAPM provide a linear model for expected returns, their origins and implications are fundamentally different.1 The CAPM is an equilibrium model derived from the premise that all investors are mean-variance optimizers and hold portfolios from the same efficient frontier. This leads to the conclusion that the single, all-encompassing "market portfolio" is the only priced risk factor.3 In this sense, CAPM is often described as a "demand-side" model, as its results stem from the optimization problems of individual investors.17

In contrast, APT is derived from the no-arbitrage condition, which is a much weaker assumption. It does not require market equilibrium or make strong assumptions about investor preferences or the normality of returns.15 APT allows for multiple sources of systematic risk and acknowledges that each investor will hold a unique portfolio tailored to their own risk preferences, rather than the single market portfolio prescribed by CAPM.16 Because its factor sensitivities reflect how underlying economic forces affect asset returns, APT is often considered a "supply-side" model.16

This distinction creates a fundamental trade-off. CAPM's assumptions, while often criticized as unrealistic (e.g., all investors can borrow and lend at the risk-free rate, there are no taxes or transaction costs), provide a clear, prescriptive model with a single, easily identifiable factor: the market excess return.3 APT, on the other hand, offers greater theoretical flexibility and realism but at a significant practical cost: the theory itself does not specify what the factors are, or even how many exist.1 This ambiguity makes APT theoretically robust but practically challenging to implement directly.

|Feature|Capital Asset Pricing Model (CAPM)|Arbitrage Pricing Theory (APT)|
|---|---|---|
|**Core Principle**|Market Equilibrium, Mean-Variance Optimization 1|No-Arbitrage, Law of One Price 16|
|**Number of Factors**|One (The Market Portfolio) 1|Multiple (Unspecified) 9|
|**Nature of Factors**|Prescribed: Market Excess Return|Unspecified: Macroeconomic, Statistical, or Fundamental 9|
|**Key Assumptions**|Homogeneous expectations, all investors hold the market portfolio, can borrow/lend at risk-free rate 15|Linear factor structure, no-arbitrage, diversifiable idiosyncratic risk 10|
|**Portfolio Held by Investors**|The single "Market Portfolio" 16|Unique, individual portfolios based on risk preferences 12|
|**Model Type**|Demand-Side (derived from investor optimization) 17|Supply-Side (derived from asset return generating process) 16|
|**Practicality**|Simple to implement but often empirically inaccurate 20|More complex and flexible, but requires factor identification 1|
|**Table 1: Comparative Analysis of CAPM and APT.** This table provides a clear, structured summary of the differences between the two cornerstone models.|||

### The Challenge of APT: Unspecified Factors

The primary practical limitation of APT is that its factors are not predefined.1 To apply the model, a quantitative analyst must first identify the relevant systematic factors that drive asset returns. This can be approached in two main ways:

1. **Economic Intuition:** Identify key macroeconomic variables that are believed to systematically affect all asset prices. Common candidates include unexpected changes in inflation, GDP growth, industrial production, corporate bond spreads, and shifts in the yield curve.9
    
2. **Statistical Analysis:** Use statistical techniques like Principal Component Analysis (PCA) or factor analysis on a large cross-section of asset returns. These methods extract the underlying statistical factors that explain the common variance in the data, without needing to assign them an economic name.9
    

This ambiguity created a significant challenge for practitioners. A model with unknown inputs is difficult to use for pricing assets or managing risk. This very challenge paved the way for the work of Eugene Fama and Kenneth French. Instead of deriving factors from abstract theory or purely statistical methods, they took an empirical approach. They asked a simple question: what observable, portfolio-based characteristics have historically explained the cross-section of stock returns? Their answer led to the development of the Fama-French models, which can be viewed as a practical, empirically-grounded specification of the general multifactor framework proposed by APT.

### Python in Practice: Identifying Statistical Factors with PCA

To make the concept of unspecified factors concrete, we can use Python to extract statistical factors from a basket of stocks using Principal Component Analysis (PCA). PCA is a dimensionality-reduction technique that transforms a set of correlated variables into a set of linearly uncorrelated variables called principal components. In finance, these components can be interpreted as statistical risk factors.

First, we'll gather monthly return data for a diverse set of 15 large US companies from different sectors using the `yfinance` library.



```Python
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Define a diverse list of tickers
tickers =

# Download historical data
data = yf.download(tickers, start='2014-01-01', end='2024-01-01', interval='1mo')

# Calculate monthly returns
monthly_returns = data['Adj Close'].pct_change().dropna()

print("Sample of Monthly Returns:")
print(monthly_returns.head())

# Standardize the data (important for PCA)
scaler = StandardScaler()
scaled_returns = scaler.fit_transform(monthly_returns)

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(scaled_returns)

# Create a DataFrame with the principal components (our statistical factors)
pc_df = pd.DataFrame(data=principal_components, 
                     columns=[f'PC_{i+1}' for i in range(len(tickers))],
                     index=monthly_returns.index)

print("\nPrincipal Components (Statistical Factors):")
print(pc_df.head())

# Analyze the explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

print("\nExplained Variance by Each Principal Component:")
print(explained_variance_ratio)

# Plot the explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center',
        label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.title('Explained Variance by Principal Components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
```

**Interpretation:** The output of this code shows the principal components, which are our statistically derived factors. The explained variance plot is particularly revealing. We typically find that the first principal component (PC_1) explains a large portion of the total variance (often 40-60%). When we correlate this first component with the overall market return (e.g., from the S&P 500), we usually find a very high correlation. This confirms that market risk is indeed the dominant systematic factor.

The subsequent components (PC_2, PC_3, etc.) represent other sources of common, systematic risk that are orthogonal to (uncorrelated with) the first component. These could be interpreted as statistical proxies for factors related to industry, size, value, or momentum, just as APT predicts, but without being explicitly defined beforehand. This exercise provides a tangible example of how one might begin to apply APT's logic in practice.

## Part 1.3: The Fama-French Models: An Empirical Revolution

### The Three-Factor Model (FF3): Size and Value Matter

The Fama-French Three-Factor Model, introduced by Eugene Fama and Kenneth French in their seminal 1992 and 1993 papers, represented a watershed moment in asset pricing.6 It was a direct and powerful empirical response to the shortcomings of the CAPM.5 By analyzing decades of US stock market data, Fama and French identified two systematic patterns that CAPM could not explain: the

**size effect** (small-cap stocks tend to outperform large-cap stocks) and the **value effect** (stocks with high book-to-market ratios tend to outperform those with low ratios).6

Instead of treating these as mere anomalies, they proposed that these effects represented compensation for bearing systematic risk and incorporated them as factors in a new, three-factor model.

**Defining the Factors:** The genius of the Fama-French approach lies in the construction of its factors. They are not abstract macroeconomic variables but are themselves portfolios of stocks, making them transparent, replicable, and directly investable. The three factors are 4:

1. **Mkt-RF (Market Risk Premium):** This is the excess return of a broad, value-weighted market portfolio over the risk-free rate. It is identical in concept to the single factor in the CAPM.
    
2. **SMB (Small Minus Big):** This factor is designed to capture the size premium. It is the return of a portfolio of small-cap stocks minus the return of a portfolio of large-cap stocks. It is a long-short portfolio: long small caps and short large caps.
    
3. **HML (High Minus Low):** This factor is designed to capture the value premium. It is the return of a portfolio of high book-to-market stocks (value stocks) minus the return of a portfolio of low book-to-market stocks (growth stocks). It is a long-short portfolio: long value stocks and short growth stocks.
    

The precise construction of these factors involves a two-by-three sort of stocks based on size (market capitalization) and value (book-to-market ratio), as detailed in the table below.

|Factor|Description|Portfolio Construction|
|---|---|---|
|**Mkt-RF**|Market Risk Premium|The return of the total value-weighted stock market portfolio minus the one-month Treasury bill rate.|
|**SMB**|Small Minus Big (Size Factor)|The average return on the three small-stock portfolios (Small Value, Small Neutral, Small Growth) minus the average return on the three big-stock portfolios (Big Value, Big Neutral, Big Growth).23|
|**HML**|High Minus Low (Value Factor)|The average return on the two value portfolios (Small Value, Big Value) minus the average return on the two growth portfolios (Small Growth, Big Growth).|
|**RMW**|Robust Minus Weak (Profitability Factor)|The average return on the two robust operating profitability portfolios minus the average return on the two weak operating profitability portfolios.21|
|**CMA**|Conservative Minus Aggressive (Investment Factor)|The average return on the two conservative investment portfolios minus the average return on the two aggressive investment portfolios.21|
|**Table 2: Construction of the Fama-French Factors.** This table demystifies the factors by showing they are simply returns from long-short portfolios based on firm characteristics.|||

**The FF3 Model Equation:** The three-factor model extends the CAPM by adding the SMB and HML factors to explain the excess return of an asset or portfolio (Ri​−Rf​) 6:

![[Pasted image 20250702081416.png]]

In this equation, the betas (βi,Mkt​, βi,SMB​, βi,HML​) represent the asset's sensitivity to each of the three factors. The αi​ (alpha) term is of particular interest: it represents the portion of the asset's return that is _not_ explained by its exposure to these three common risk factors. A statistically significant positive alpha suggests superior performance (or underpricing), while a negative alpha suggests underperformance.4 Fama and French found that this model could explain over 90% of the variation in diversified portfolio returns, a dramatic improvement over the CAPM's roughly 70%.6

### The Great Debate: Are Factor Premia Risk or Mispricing?

The empirical success of the Fama-French model sparked one of the most enduring debates in modern finance: why do the size and value premia exist? The answer cuts to the core of how we understand market efficiency. There are two primary schools of thought.6

**The Risk-Based Argument (Efficient Markets):** This is the view championed by Fama and French themselves. It aligns with the Efficient Market Hypothesis (EMH) and posits that the size and value premia are not free lunches but are fair compensation for bearing additional, non-diversifiable systematic risk.6 According to this view, small-cap and value firms are inherently riskier. They may be more vulnerable to economic downturns, have less stable earnings, face higher costs of capital, or be closer to financial distress.24 Investors, being rational and risk-averse, demand a higher expected return—a risk premium—to compensate for holding these riskier assets.8

**The Behavioral Argument (Inefficient Markets):** This alternative explanation, rooted in behavioral finance, argues that the premia arise from systematic and predictable errors in judgment made by investors.27 Proponents of this view suggest that investors are not always rational. They may suffer from cognitive biases, such as extrapolating past growth rates too far into the future. This leads them to become overly optimistic about "glamorous" growth stocks, bidding their prices up to unsustainable levels, while being overly pessimistic about "boring" value stocks, depressing their prices below their intrinsic value.24 The value premium is then earned when the high expectations for growth stocks are not met and the low expectations for value stocks are exceeded, causing prices to revert toward their fundamental values.29 Similarly, the size premium could be driven by a preference for "lottery-like" investments, where investors overpay for the small chance of a huge payoff often associated with small, speculative stocks.24

This debate is far from academic. If the premia are risk-based, they should be persistent over time as a permanent feature of the market's risk-reward trade-off. Factor investing, in this case, is a strategic allocation to specific risk profiles. However, if the premia are due to mispricing, they could theoretically be arbitraged away as more investors become aware of them and exploit the inefficiency.28 For a quantitative analyst, understanding this distinction is critical for assessing the long-term viability of factor-based strategies.

### The Five-Factor Model (FF5): Adding Profitability and Investment

For over two decades, the three-factor model was the workhorse of empirical asset pricing. However, further research uncovered other anomalies that FF3 struggled to explain. In response, Fama and French proposed an updated five-factor model in 2015.21 This model incorporates two new factors related to a company's internal operations and investment policies:

1. **RMW (Robust Minus Weak):** This is a profitability factor. It represents the excess return of firms with robust (high) operating profitability over those with weak (low) profitability.
    
2. **CMA (Conservative Minus Aggressive):** This is an investment factor. It represents the excess return of firms that invest conservatively (low asset growth) over those that invest aggressively (high asset growth).
    

The inclusion of these factors was motivated by valuation theory, which suggests that, all else equal, companies with higher profitability and more conservative investment policies should have higher expected returns.8

A striking and profound finding from the five-factor model research was the diminished role of the original value factor, HML. Fama and French found that with the addition of the RMW and CMA factors, the HML factor often became redundant in explaining the cross-section of average returns.21 This suggests that "value," as traditionally measured by the book-to-market ratio, might not be a fundamental risk dimension in itself. Instead, it may largely be a proxy for deeper characteristics related to profitability and investment. A company often appears "cheap" (high B/M) precisely

_because_ it has low profitability and/or has engaged in aggressive, often value-destroying, investment. The five-factor model helps to disentangle these effects, providing a much sharper and more economically intuitive picture of what drives stock returns.

### Python in Practice: Running a Fama-French Regression

Let's put the theory into practice by performing a full Fama-French five-factor regression analysis on a single stock: Apple Inc. (AAPL).34 This will serve as a template for analyzing any asset or portfolio.

Step 1: Acquiring Data

We need two sets of data: the historical returns for AAPL and the historical Fama-French factor returns. We can get AAPL data from yfinance and the factor data directly from Kenneth French's Data Library using the pandas-datareader package.35

Step 2: Data Preparation

This is a crucial step that involves aligning the two datasets. We will convert AAPL's daily prices to monthly returns, calculate the excess return for AAPL (AAPL return - Risk-Free Rate), and merge it with the factor data, which is already provided in monthly percentage terms.

Step 3: Regression with statsmodels

We will use the statsmodels library, a powerful Python package for statistical modeling, to perform an Ordinary Least Squares (OLS) regression.36

Step 4: In-depth Interpretation

We will analyze the model.summary() output in detail, using the guide in Table 3 to understand each component.



```Python
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm

# Step 1: Acquire Data
# Download Fama-French 5 Factors (monthly)
# The data is downloaded and parsed automatically by pandas_datareader
ff_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2014-01-01', end='2024-01-01')
# The factor values are in percentages, so we divide by 100
ff_factors = ff_factors / 100
ff_factors.rename(columns={'Mkt-RF': 'Mkt_RF'}, inplace=True) # Rename for formula compatibility
print("Fama-French 5 Factors (first 5 rows):")
print(ff_factors.head())

# Download AAPL stock data
aapl_daily = yf.download('AAPL', start='2014-01-01', end='2024-01-01')

# Step 2: Data Preparation
# Convert daily prices to monthly returns
aapl_monthly = aapl_daily['Adj Close'].resample('M').ffill().pct_change().dropna()
aapl_monthly.name = 'AAPL_Ret'

# Merge the datasets
# Convert monthly returns index to PeriodIndex to match factors
aapl_monthly.index = aapl_monthly.index.to_period('M')
merged_data = pd.merge(aapl_monthly, ff_factors, left_index=True, right_index=True)

# Calculate AAPL's excess return
merged_data = merged_data - merged_data

print("\nMerged Data (first 5 rows):")
print(merged_data.head())

# Step 3: Regression with statsmodels
# Define the model using the formula interface
# We are regressing AAPL's excess return on the five factors
model = sm.OLS.from_formula('AAPL_Excess_Ret ~ Mkt_RF + SMB + HML + RMW + CMA', data=merged_data)
results = model.fit()

# Step 4: In-depth Interpretation
print("\nFama-French 5-Factor Regression Results for AAPL:")
print(results.summary())
```

|Statistic|Description|What to Look For (in FF context)|
|---|---|---|
|**R-squared / Adj. R-squared**|The proportion of the asset's return variance that is explained by the factors. Adjusted R-squared accounts for the number of factors.|A high value (e.g., > 0.70) indicates the model is a good fit and that the asset's returns are well-explained by common systematic factors.|
|**Intercept (Alpha)**|The asset's average return after accounting for its exposure to the factors. It measures manager skill or asset-specific performance.|A statistically significant positive alpha (p-value < 0.05) suggests the asset has outperformed its benchmark. A non-significant alpha suggests performance is in line with risk exposures.|
|**Coef (for each factor)**|The estimated beta or sensitivity of the asset to that factor. It shows how much the asset's return changes for a 1% change in the factor return.|Mkt_RF: >1 indicates higher market risk than average. <1 indicates lower.<br><br>SMB: Positive indicates a small-cap tilt; negative indicates a large-cap tilt.<br><br>HML: Positive indicates a value tilt; negative indicates a growth tilt.<br><br>RMW: Positive indicates a high-profitability tilt; negative indicates a low-profitability tilt.<br><br>CMA: Positive indicates a conservative-investment tilt; negative indicates an aggressive-investment tilt.|
|**Std. Err**|The standard error of the coefficient estimate. It measures the precision of the beta estimate.|A smaller standard error relative to the coefficient suggests a more precise estimate.|
|**t-statistic**|The coefficient divided by its standard error. It tests the hypothesis that the true coefficient is zero.|An absolute value > 2 (approximately) is generally considered statistically significant.|
|**P>\|t\| (p-value)**|The probability of observing the estimated coefficient if the true coefficient were zero.|A p-value < 0.05 is the standard threshold for statistical significance, meaning we can be confident the factor has a real effect on the asset's return.|
|**Table 3: A Guide to Interpreting `statsmodels` OLS Regression Results.** This table serves as a pedagogical reference for students, explaining every key part of the regression summary table.|||

From the regression summary for AAPL, we can draw several conclusions. The **Adj. R-squared** will likely be high, showing that the five factors explain a large part of Apple's monthly return variation. The **Mkt_RF** coefficient (market beta) will be positive and significant, likely close to 1.2, indicating it's slightly more volatile than the market. The **SMB** coefficient will be significantly negative, confirming Apple's status as a large-cap stock. The **HML** coefficient will also be significantly negative, reflecting its strong identity as a growth stock. The **RMW** coefficient is expected to be positive and significant, as Apple is a highly profitable company. Finally, the **Intercept (alpha)** will tell us if, after accounting for all these risk exposures, Apple has generated excess returns. A small and statistically insignificant alpha would suggest that the Fama-French model does an excellent job of explaining Apple's performance.

## Part 1.4: Dynamic Factor Analysis with Rolling Regressions

### The Myth of Stable Betas

A standard OLS regression, like the one performed above, provides a single set of beta coefficients for the entire analysis period. This implicitly assumes that an asset's sensitivities to risk factors are constant over time. In reality, this is rarely true.40 A company's business strategy, financial leverage, competitive landscape, and product mix all evolve. As they do, the company's exposure to systematic risks will also change. For example, a young, rapidly growing tech company might have a very different risk profile than it does 20 years later as a mature, dividend-paying blue-chip stock.

To capture this dynamic nature, we can employ a rolling regression. This technique involves running the same regression repeatedly on a moving window of data (e.g., the last 60 months). This produces a time series of beta coefficients, allowing us to visualize how an asset's risk exposures have changed over time.

### Python in Practice: Implementing a Rolling Fama-French Regression

The `statsmodels` library provides a convenient `RollingOLS` class for this exact purpose.41 We will apply it to our merged AAPL and factor dataset to calculate 60-month rolling betas for the three original Fama-French factors.



```Python
from statsmodels.regression.rolling import RollingOLS

# Ensure data is sorted by date
merged_data.sort_index(inplace=True)

# Define the dependent and independent variables
y = merged_data
X = sm.add_constant(merged_data]) # Using FF3 for clearer visualization

# Set the rolling window size (e.g., 60 months = 5 years)
window_size = 60

# Fit the rolling OLS model
rolling_model = RollingOLS(y, X, window=window_size)
rolling_results = rolling_model.fit()

# Extract the rolling parameters (betas and alpha)
rolling_params = rolling_results.params.dropna()

print("\nSample of Rolling Betas (last 5 periods):")
print(rolling_params.tail())

# Plot the rolling coefficients
fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
fig.suptitle('AAPL: 60-Month Rolling Fama-French 3-Factor Betas', fontsize=16)

# Plot Rolling Alpha (Intercept)
axes.plot(rolling_params.index.to_timestamp(), rolling_params['const'])
axes.set_title('Rolling Alpha (Intercept)')
axes.grid(True)

# Plot Rolling Market Beta
axes.plot(rolling_params.index.to_timestamp(), rolling_params)
axes.set_title('Rolling Market Beta (Mkt_RF)')
axes.grid(True)

# Plot Rolling Size Beta
axes.plot(rolling_params.index.to_timestamp(), rolling_params)
axes.set_title('Rolling Size Beta (SMB)')
axes.grid(True)

# Plot Rolling Value Beta
axes.plot(rolling_params.index.to_timestamp(), rolling_params['HML'])
axes.set_title('Rolling Value Beta (HML)')
axes.grid(True)

plt.xlabel('Date')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
```

**Analysis:** The plots generated by this code tell a dynamic story about Apple's risk profile. We might observe its market beta (Mkt_RF) fluctuating around market crises, perhaps increasing during downturns. The SMB beta would likely remain consistently negative, confirming its large-cap nature. The HML beta is particularly interesting; we might see it become more negative during periods of intense product innovation and growth (like the iPhone era) and perhaps drift closer to zero as the company matured and began returning significant capital to shareholders, taking on more "value" characteristics. This dynamic analysis provides a far richer understanding than a single, static regression.

## Part 1.5: Capstone Project I: Factor Analysis of an Exchange-Traded Fund (ETF)

**Objective:** This project applies the concepts we've learned to conduct a comprehensive factor analysis of a real-world financial product: the Invesco QQQ Trust (QQQ), an ETF that tracks the NASDAQ-100 Index. This index is heavily weighted towards large-cap technology and growth companies. Our goal is to use the Fama-French five-factor model to look "under the hood" and quantify QQQ's true risk drivers.36

### Questions & Responses

#### 1. Data Acquisition

The first step is to gather the necessary data: 10 years of monthly price data for QQQ and the corresponding Fama-French 5-factor data.

**Response (Code):**



```Python
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm

# Define the time period
start_date = '2014-01-01'
end_date = '2024-01-01'

# Download Fama-French 5 Factors
ff_factors_capstone = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=start_date, end=end_date)
ff_factors_capstone = ff_factors_capstone / 100
ff_factors_capstone.rename(columns={'Mkt-RF': 'Mkt_RF'}, inplace=True)

# Download QQQ data
qqq_daily = yf.download('QQQ', start=start_date, end=end_date)
qqq_monthly = qqq_daily['Adj Close'].resample('M').ffill().pct_change().dropna()
qqq_monthly.name = 'QQQ_Ret'

# Prepare data for regression
qqq_monthly.index = qqq_monthly.index.to_period('M')
qqq_data = pd.merge(qqq_monthly, ff_factors_capstone, left_index=True, right_index=True)
qqq_data = qqq_data - qqq_data

print("Capstone Project Data Ready. Sample:")
print(qqq_data.head())
```

#### 2. Static Regression Analysis

**Question:** Run a Fama-French five-factor regression for QQQ over the entire 10-year period. What are its sensitivities to market, size, value, profitability, and investment? Is its alpha statistically significant?

**Response (Code and Interpretation):**



```Python
# Run the static 5-factor regression for QQQ
static_model_qqq = sm.OLS.from_formula(
    'QQQ_Excess_Ret ~ Mkt_RF + SMB + HML + RMW + CMA', 
    data=qqq_data
)
static_results_qqq = static_model_qqq.fit()

print("\n--- Static Fama-French 5-Factor Regression for QQQ ---")
print(static_results_qqq.summary())
```

**Interpretation:** The results from the `statsmodels` summary table will reveal the fundamental characteristics of the QQQ portfolio.

- **Market Sensitivity (Mkt_RF):** We expect a beta significantly greater than 1.0 (around 1.1-1.2), confirming that QQQ is more volatile than the overall market.
    
- **Size Sensitivity (SMB):** The beta for SMB should be negative and statistically significant, indicating a strong tilt towards large-cap stocks, as the NASDAQ-100 comprises the largest non-financial companies on the exchange.
    
- **Value Sensitivity (HML):** The HML beta will be strongly negative and significant. This is the model's way of quantifying QQQ's pronounced "growth" characteristic, as it is dominated by companies with low book-to-market ratios.
    
- **Profitability Sensitivity (RMW):** The RMW beta is likely to be positive and significant, reflecting that many of the large tech companies in the NASDAQ-100 are highly profitable.
    
- **Investment Sensitivity (CMA):** The CMA beta might be negative, suggesting a tilt towards companies that invest aggressively, which is common for technology firms.
    
- **Alpha (Intercept):** The alpha term is crucial. A small and statistically insignificant alpha would imply that the five-factor model fully explains QQQ's returns. A significant positive alpha would suggest that the collection of stocks in the NASDAQ-100 has delivered returns _above and beyond_ what would be expected from their risk factor exposures, perhaps due to innovation or other unmodeled characteristics.
    

|Factor|Coefficient|Std. Error|t-Statistic|P-value|
|---|---|---|---|---|
|**Intercept (Alpha)**|**|**|**|**|
|**Mkt_RF**|**|**|**|**|
|**SMB**|**|**|**|**|
|**HML**|**|**|***|**|
|**RMW**|**|**|**|**|
|**CMA**|**|**|**|**|
|**R-squared**|**|**Adj. R-squared**|**|**Obs.**|
|**Table 4: Summary of Fama-French 5-Factor Regression for QQQ (10-Year Period).** This table presents the final results of the static analysis in a clean, professional format. _(Note: Results are placeholders to be filled by running the code.)_|||||

#### 3. Dynamic Analysis with Rolling Regression

**Question:** Perform a 36-month rolling regression on QQQ's excess returns. How have its betas to the Mkt-RF and HML factors evolved over time? What might explain these changes?

**Response (Code and Interpretation):**



```Python
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt

# Define variables for rolling regression
y_qqq = qqq_data
X_qqq = sm.add_constant(qqq_data])

# Fit the rolling OLS model with a 36-month window
rolling_model_qqq = RollingOLS(y_qqq, X_qqq, window=36)
rolling_results_qqq = rolling_model_qqq.fit()
rolling_params_qqq = rolling_results_qqq.params.dropna()

# Plot the rolling Market and Value betas
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle('QQQ: 36-Month Rolling Betas', fontsize=16)

ax1.plot(rolling_params_qqq.index.to_timestamp(), rolling_params_qqq)
ax1.set_title('Rolling Market Beta (Mkt_RF)')
ax1.axhline(y=1.0, color='r', linestyle='--', label='Market Beta = 1')
ax1.legend()
ax1.grid(True)

ax2.plot(rolling_params_qqq.index.to_timestamp(), rolling_params_qqq['HML'])
ax2.set_title('Rolling Value Beta (HML)')
ax2.axhline(y=0, color='r', linestyle='--', label='Neutral Value/Growth')
ax2.legend()
ax2.grid(True)

plt.xlabel('Date')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()
```

**Interpretation:** The plots will provide a dynamic narrative of QQQ's risk profile.

- **Rolling Market Beta:** We might see the market beta spike during periods of high market volatility, such as the COVID-19 crash in March 2020, as correlations tend to increase during crises.
    
- **Rolling Value Beta (HML):** This plot is likely to be very revealing. We would expect the HML beta to become increasingly negative during the tech boom of the late 2010s and the post-pandemic rally, indicating that QQQ's portfolio became even more tilted towards "growth" stocks. If the beta were to drift towards zero in recent years, it might suggest a shift in the characteristics of the underlying companies or a rotation in the market. This dynamic view is far more insightful for risk management than a single static beta.
    

## Part 1.6: Capstone Project II: Building and Evaluating a "Smart Beta" Value Portfolio

**Objective:** This final project transitions from analysis to active construction. We will guide you through the process of building a simple "smart beta" portfolio that systematically tilts towards the value factor, as defined by Fama and French. We will then backtest this strategy and use the five-factor model to evaluate its performance and confirm its factor exposures.43

### Methodology & Code

#### 1. Universe Selection and Data Sourcing

We will use the current constituents of the S&P 500 as our investment universe. It is important to acknowledge that this introduces **survivorship bias**, as we are not including companies that were in the index historically but have since been removed due to poor performance or acquisition. For a pedagogical example, this is a necessary simplification. For a production-grade backtest, one would need a historical constituents database. We will use `yfinance` to approximate the Book-to-Market (B/M) ratio by pulling the `bookValue` and `marketCap` for each stock.46

#### 2. Portfolio Construction and Backtesting

The strategy is as follows:

- At the beginning of each year, calculate the B/M ratio for all stocks in our universe.
    
- Rank the stocks by their B/M ratio.
    
- Form an equal-weighted portfolio of the top 20% (quintile) of stocks—these are our "value" stocks.
    
- Hold this portfolio for one year.
    
- Rebalance at the start of the next year.
    

We will simulate this process over a 10-year period.

**Response (Code):**



```Python
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import time

# Note: This is a simplified, pedagogical example.
# It uses current S&P 500 constituents (survivorship bias) and yfinance for fundamentals,
# which can be inconsistent. A professional implementation would use point-in-time data.

# Get S&P 500 tickers
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies').tolist()

# Define backtest period
backtest_start = '2014-01-01'
backtest_end = '2024-01-01'
years = range(pd.to_datetime(backtest_start).year, pd.to_datetime(backtest_end).year)

portfolio_returns =

for year in years:
    print(f"Processing year: {year}...")
    # Get fundamentals at the start of the year
    bm_ratios = {}
    for ticker in sp500_tickers:
        try:
            stock_info = yf.Ticker(ticker).info
            book_value = stock_info.get('bookValue')
            market_cap = stock_info.get('marketCap')
            if book_value and market_cap and book_value > 0 and market_cap > 0:
                bm_ratios[ticker] = book_value / (market_cap / 1_000_000) # Assuming market cap is in full
        except Exception as e:
            # print(f"Could not get info for {ticker}: {e}")
            pass
        time.sleep(0.1) # Be respectful to the API

    if not bm_ratios:
        print(f"No fundamental data for year {year}. Skipping.")
        continue

    # Form value portfolio (top quintile by B/M)
    sorted_stocks = sorted(bm_ratios.items(), key=lambda item: item, reverse=True)
    quintile_size = len(sorted_stocks) // 5
    value_portfolio_tickers = [ticker for ticker, bm in sorted_stocks[:quintile_size]]

    if not value_portfolio_tickers:
        print(f"Value portfolio is empty for year {year}. Skipping.")
        continue
        
    # Get price data for the year
    start_of_year = f'{year}-01-01'
    end_of_year = f'{year}-12-31'
    prices = yf.download(value_portfolio_tickers, start=start_of_year, end=end_of_year, progress=False)['Adj Close']
    
    if prices.empty:
        print(f"No price data for year {year}. Skipping.")
        continue

    # Calculate daily returns for the equal-weighted portfolio
    daily_returns = prices.pct_change().dropna(how='all')
    portfolio_daily_return = daily_returns.mean(axis=1)
    portfolio_returns.append(portfolio_daily_return)

# Combine all yearly returns into one series
backtest_results = pd.concat(portfolio_returns)
backtest_results.name = 'Value_Portfolio_Ret'

# Download SPY for benchmark
spy = yf.download('SPY', start=backtest_start, end=backtest_end)['Adj Close'].pct_change().dropna()
spy.name = 'SPY_Ret'

# Combine and analyze
performance = pd.concat([backtest_results, spy], axis=1).dropna()

# Calculate summary stats
annualized_return = (1 + performance).prod()**(252 / len(performance)) - 1
annualized_volatility = performance.std() * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_volatility

print("\n--- Backtest Performance Summary ---")
print("Value Portfolio:")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

print("\nSPY Benchmark:")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

```

#### 3. Performance Attribution

Now, we run a Fama-French 5-factor regression on our backtested portfolio's returns to see if we successfully captured the value premium.

**Response (Code and Analysis):**



```Python
# Prepare data for regression
# Resample our daily portfolio returns to monthly
value_monthly = (1 + backtest_results).resample('M').prod() - 1
value_monthly.index = value_monthly.index.to_period('M')

# Merge with factors
value_factor_data = pd.merge(value_monthly, ff_factors_capstone, left_index=True, right_index=True)
value_factor_data = value_factor_data - value_factor_data

# Run the 5-factor regression
value_model = sm.OLS.from_formula(
    'Value_Excess_Ret ~ Mkt_RF + SMB + HML + RMW + CMA', 
    data=value_factor_data
)
value_results = value_model.fit()

print("\n--- Fama-French 5-Factor Attribution for Value Portfolio ---")
print(value_results.summary())
```

**Analysis:** The regression output is the ultimate test of our strategy.

- **HML Beta:** The most important coefficient is HML. We expect it to be positive and statistically significant, which would confirm that our portfolio successfully captured the value factor. A high HML beta (e.g., > 0.5) would show a strong value tilt.
    
- **Other Betas:** We can also examine the other factor exposures. For instance, value stocks are often smaller, so we might see a positive SMB beta.
    
- **Alpha:** The alpha will tell us if our specific implementation of a value strategy generated returns beyond its factor exposures. A non-significant alpha would mean the portfolio's performance is fully explained by its tilt towards value, size, and other factors.
    

### Conclusion & Discussion

This capstone project demonstrates the full cycle of quantitative finance: from theory (the value premium) to construction (building a factor-tilted portfolio) to evaluation (backtesting and attribution analysis). The results will likely show that while our simple strategy did capture a value tilt (positive HML beta), its performance relative to the market benchmark (SPY) can vary dramatically over different time periods. The value premium, for instance, significantly underperformed growth during much of the 2010s before showing signs of a comeback.

This highlights several practical challenges for quants:

- **Factor Cyclicality:** Factor premia are not guaranteed and can underperform for long stretches.
    
- **Data Quality:** Sourcing clean, point-in-time fundamental data is a major challenge and is critical for avoiding lookahead bias.
    
- **Implementation Costs:** Real-world portfolios incur transaction costs from rebalancing, which can erode returns.
    
- **Model Risk:** The factors themselves and their definitions can change, as evidenced by the evolution from the FF3 to the FF5 model.
    

Despite these challenges, factor models provide an indispensable framework for understanding the drivers of risk and return, enabling the design and evaluation of sophisticated, evidence-based investment strategies.

|Metric|Value Portfolio|S&P 500 (SPY)|
|---|---|---|
|**Annualized Return**|**|**|
|**Annualized Volatility**|**|**|
|**Sharpe Ratio**|**|**|
|**Max Drawdown**|**|**|
|**Fama-French 5-Factor Alpha**|**|**|
|**Table 5: Backtest Performance Summary: Value Portfolio vs. S&P 500 (e.g., 2014-2024).** This table provides a clear quantitative comparison of the constructed strategy against the benchmark. _(Note: Results are placeholders to be filled by running the code and would require a more robust backtesting framework for Max Drawdown calculation.)_|||
## References
**

1. CAPM vs. Arbitrage Pricing Theory: What's the Difference? - Investopedia, acessado em julho 2, 2025, [https://www.investopedia.com/articles/markets/080916/capm-vs-arbitrage-pricing-theory-how-they-differ.asp](https://www.investopedia.com/articles/markets/080916/capm-vs-arbitrage-pricing-theory-how-they-differ.asp)
    
2. The capital-asset-pricing model and arbitrage pricing theory: A unification - PMC, acessado em julho 2, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC20614/](https://pmc.ncbi.nlm.nih.gov/articles/PMC20614/)
    
3. Arbitrage Price Theory vs. Capital Asset Pricing - Economics Online, acessado em julho 2, 2025, [https://www.economicsonline.co.uk/competitive_markets/arbitrage-price-theory-vs-capital-asset-pricing.html/](https://www.economicsonline.co.uk/competitive_markets/arbitrage-price-theory-vs-capital-asset-pricing.html/)
    
4. Fama-French three-factor model | Financial Mathematics Class ..., acessado em julho 2, 2025, [https://library.fiveable.me/financial-mathematics/unit-10/fama-french-three-factor-model/study-guide/lPO1ENtdqIZlPEXH](https://library.fiveable.me/financial-mathematics/unit-10/fama-french-three-factor-model/study-guide/lPO1ENtdqIZlPEXH)
    
5. Testing The Three Factor Model Of Fama And French: Evidence From An Emerging Market - CORE, acessado em julho 2, 2025, [https://core.ac.uk/download/pdf/236412809.pdf](https://core.ac.uk/download/pdf/236412809.pdf)
    
6. Fama and French Three Factor Model Definition: Formula and Interpretation - Investopedia, acessado em julho 2, 2025, [https://www.investopedia.com/terms/f/famaandfrenchthreefactormodel.asp](https://www.investopedia.com/terms/f/famaandfrenchthreefactormodel.asp)
    
7. How Does the Fama French 3 Factor Model Work? - SmartAsset, acessado em julho 2, 2025, [https://smartasset.com/investing/fama-french-3-factor-model](https://smartasset.com/investing/fama-french-3-factor-model)
    
8. Intuition behind Fama-French factors - Quantitative Finance Stack Exchange, acessado em julho 2, 2025, [https://quant.stackexchange.com/questions/23250/intuition-behind-fama-french-factors](https://quant.stackexchange.com/questions/23250/intuition-behind-fama-french-factors)
    
9. What is Arbitrage Pricing Theory? | CQF, acessado em julho 2, 2025, [https://www.cqf.com/blog/quant-finance-101/what-is-arbitrage-pricing-theory](https://www.cqf.com/blog/quant-finance-101/what-is-arbitrage-pricing-theory)
    
10. Arbitrage Pricing Theory (APT), Its Assumptions and Relation to Multifactor Models - CFA, FRM, and Actuarial Exams Study Notes - AnalystPrep, acessado em julho 2, 2025, [https://analystprep.com/study-notes/cfa-level-2/arbitrage-pricing-theory-apt-its-assumptions-and-relation-to-multifactor-models/](https://analystprep.com/study-notes/cfa-level-2/arbitrage-pricing-theory-apt-its-assumptions-and-relation-to-multifactor-models/)
    
11. Arbitrage Pricing Theory (APT) | Meaning, Applications, Criticisms - Finance Strategists, acessado em julho 2, 2025, [https://www.financestrategists.com/wealth-management/valuation/arbitrage-pricing-theory-apt/](https://www.financestrategists.com/wealth-management/valuation/arbitrage-pricing-theory-apt/)
    
12. Arbitrage Pricing Theory - Defintion, Formula, Example - Corporate Finance Institute, acessado em julho 2, 2025, [https://corporatefinanceinstitute.com/resources/wealth-management/arbitrage-pricing-theory-apt/](https://corporatefinanceinstitute.com/resources/wealth-management/arbitrage-pricing-theory-apt/)
    
13. Arbitrage Pricing Theory (APT) - Formula with Example | Hero Vired, acessado em julho 2, 2025, [https://herovired.com/learning-hub/blogs/arbitrage-pricing-theory/](https://herovired.com/learning-hub/blogs/arbitrage-pricing-theory/)
    
14. Arbitrage Pricing Theory (APT): Formula and How It's Used - Investopedia, acessado em julho 2, 2025, [https://www.investopedia.com/terms/a/apt.asp](https://www.investopedia.com/terms/a/apt.asp)
    
15. Arbitrage Pricing Theory: It's Not Just Fancy Math - Investopedia, acessado em julho 2, 2025, [https://www.investopedia.com/articles/active-trading/082415/arbitrage-pricing-theory-its-not-just-fancy-math.asp](https://www.investopedia.com/articles/active-trading/082415/arbitrage-pricing-theory-its-not-just-fancy-math.asp)
    
16. Arbitrage pricing theory - Wikipedia, acessado em julho 2, 2025, [https://en.wikipedia.org/wiki/Arbitrage_pricing_theory](https://en.wikipedia.org/wiki/Arbitrage_pricing_theory)
    
17. Arbitrage Pricing Theory - Definition, Formula, Excel Download - Financial Edge Training, acessado em julho 2, 2025, [https://www.fe.training/free-resources/portfolio-management/arbitrage-pricing-theory/](https://www.fe.training/free-resources/portfolio-management/arbitrage-pricing-theory/)
    
18. Understanding the Arbitrage Pricing Theory: A Comprehensive Guide - Morpher, acessado em julho 2, 2025, [https://www.morpher.com/blog/arbitrage-pricing-theory](https://www.morpher.com/blog/arbitrage-pricing-theory)
    
19. Arbitrage in Math Economics A Pricing Theory Guide - Number Analytics, acessado em julho 2, 2025, [https://www.numberanalytics.com/blog/arbitrage-math-economics-pricing-guide](https://www.numberanalytics.com/blog/arbitrage-math-economics-pricing-guide)
    
20. CAPM vs APT. Which One Is Right for You? - Kubicle, acessado em julho 2, 2025, [https://kubicle.com/capm-vs-apt-which-one-is-right-for-you/](https://kubicle.com/capm-vs-apt-which-one-is-right-for-you/)
    
21. Fama–French three-factor model - Wikipedia, acessado em julho 2, 2025, [https://en.wikipedia.org/wiki/Fama%E2%80%93French_three-factor_model](https://en.wikipedia.org/wiki/Fama%E2%80%93French_three-factor_model)
    
22. www.investopedia.com, acessado em julho 2, 2025, [https://www.investopedia.com/terms/f/famaandfrenchthreefactormodel.asp#:~:text=of%20the%20Model%3F-,The%20Fama%20and%20French%20model%20has%20three%20factors%3A%20the%20size,risk%2Dfree%20rate%20of%20return.](https://www.investopedia.com/terms/f/famaandfrenchthreefactormodel.asp#:~:text=of%20the%20Model%3F-,The%20Fama%20and%20French%20model%20has%20three%20factors%3A%20the%20size,risk%2Dfree%20rate%20of%20return.)
    
23. Kenneth R. French - Description of Fama/French Factors, acessado em julho 2, 2025, [https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html)
    
24. Size vs value: what to consider when factor investing - ebi Portfolios, acessado em julho 2, 2025, [https://ebi.co.uk/ft-adviser-size-vs-value-what-to-consider-when-factor-investing/](https://ebi.co.uk/ft-adviser-size-vs-value-what-to-consider-when-factor-investing/)
    
25. The Value Premium: Risk or Mispricing? - - Alpha Architect, acessado em julho 2, 2025, [https://alphaarchitect.com/value-premium-risk-mispricing/](https://alphaarchitect.com/value-premium-risk-mispricing/)
    
26. Fama–French three-factor model: Explained | TIOmarkets, acessado em julho 2, 2025, [https://tiomarkets.com/en/article/fama-french-three-factor-model-guide](https://tiomarkets.com/en/article/fama-french-three-factor-model-guide)
    
27. Behavioral Finance: Biases, Mean– Variance Returns, and Risk Premiums - CiteSeerX, acessado em julho 2, 2025, [https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=6dfdfe7f353829587ced44f59a3f6a66e1003b5a](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=6dfdfe7f353829587ced44f59a3f6a66e1003b5a)
    
28. Behavioral Finance and Its Impact on Asset Valuation - Fairvalue Calculator, acessado em julho 2, 2025, [https://www.fairvalue-calculator.com/en/behavioral-finance-and-its-impact-on-asset-valuation/](https://www.fairvalue-calculator.com/en/behavioral-finance-and-its-impact-on-asset-valuation/)
    
29. Are value premiums driven by behavioral factors?, acessado em julho 2, 2025, [https://www.sec.or.th/TH/Documents/SECWorkingPapersForum/wpf-research-151167-03.pdf](https://www.sec.or.th/TH/Documents/SECWorkingPapersForum/wpf-research-151167-03.pdf)
    
30. www.longtermtrends.net, acessado em julho 2, 2025, [https://www.longtermtrends.net/fama-and-french-5-factor-model/#:~:text=Developed%20by%20Eugene%20Fama%20and,potentially%20outperforming%20large%2Dcap%20stocks.](https://www.longtermtrends.net/fama-and-french-5-factor-model/#:~:text=Developed%20by%20Eugene%20Fama%20and,potentially%20outperforming%20large%2Dcap%20stocks.)
    
31. Fama-French 5-factor model: Why more is not always better | Robeco Switzerland, acessado em julho 2, 2025, [https://www.robeco.com/en-ch/insights/2024/10/fama-french-5-factor-model-why-more-is-not-always-better](https://www.robeco.com/en-ch/insights/2024/10/fama-french-5-factor-model-why-more-is-not-always-better)
    
32. THE FAMA-FRENCH FIVE-FACTOR MODEL: EVIDENCE FROM THE JSE - WIReDSpace, acessado em julho 2, 2025, [https://wiredspace.wits.ac.za/bitstreams/4e4ebd54-40a3-4ab2-a980-e5b94bd6bef1/download](https://wiredspace.wits.ac.za/bitstreams/4e4ebd54-40a3-4ab2-a980-e5b94bd6bef1/download)
    
33. Celebrating Groundbreaking Research with Giants of Finance: Fama and French, acessado em julho 2, 2025, [https://www.ifa.com/articles/celebrating_groundbreaking_research_with_giants_finance_fama_french](https://www.ifa.com/articles/celebrating_groundbreaking_research_with_giants_finance_fama_french)
    
34. View of Application of Fama-French Three-Factor Model in, acessado em julho 2, 2025, [https://bcpublication.org/index.php/BM/article/view/3853/3760](https://bcpublication.org/index.php/BM/article/view/3853/3760)
    
35. Kenneth R. French - Data Library, acessado em julho 2, 2025, [https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
    
36. Fama-French Factor Model in Python - SEC-API.io, acessado em julho 2, 2025, [https://sec-api.io/resources/fama-french-factor-model](https://sec-api.io/resources/fama-french-factor-model)
    
37. Get Stock Data from Yahoo Finance in Python - YouTube, acessado em julho 2, 2025, [https://www.youtube.com/watch?v=nfJ1Ou0vonE](https://www.youtube.com/watch?v=nfJ1Ou0vonE)
    
38. The Fama French 3-factor model | Python, acessado em julho 2, 2025, [https://campus.datacamp.com/courses/introduction-to-portfolio-risk-management-in-python/factor-investing?ex=7](https://campus.datacamp.com/courses/introduction-to-portfolio-risk-management-in-python/factor-investing?ex=7)
    
39. Python Tutorial. Fama-French Three Factors Model - YouTube, acessado em julho 2, 2025, [https://www.youtube.com/watch?v=9mhA6idc0Ys](https://www.youtube.com/watch?v=9mhA6idc0Ys)
    
40. omartinsky/FamaFrench: Implementation of 5-factor Fama French Model - GitHub, acessado em julho 2, 2025, [https://github.com/omartinsky/FamaFrench](https://github.com/omartinsky/FamaFrench)
    
41. Rolling Regression - statsmodels 0.15.0 (+661), acessado em julho 2, 2025, [https://www.statsmodels.org/dev/examples/notebooks/generated/rolling_ls.html](https://www.statsmodels.org/dev/examples/notebooks/generated/rolling_ls.html)
    
42. Rolling Regression - Statsmodels, acessado em julho 2, 2025, [https://www.statsmodels.org/v0.12.2/examples/notebooks/generated/rolling_ls.html](https://www.statsmodels.org/v0.12.2/examples/notebooks/generated/rolling_ls.html)
    
43. Fama-French Three-Factor Model and Extensions | Intro to Investments Class Notes, acessado em julho 2, 2025, [https://library.fiveable.me/introduction-investments/unit-11/fama-french-three-factor-model-extensions/study-guide/anpESwxH97OulTOg](https://library.fiveable.me/introduction-investments/unit-11/fama-french-three-factor-model-extensions/study-guide/anpESwxH97OulTOg)
    
44. Guide to Hedging Against Fama-French Factors - PyQuant News, acessado em julho 2, 2025, [https://www.pyquantnews.com/free-python-resources/guide-to-hedging-against-fama-french-factors](https://www.pyquantnews.com/free-python-resources/guide-to-hedging-against-fama-french-factors)
    
45. 3: The Fama French 3-Factor Model - Factor Investing - WordPress.com, acessado em julho 2, 2025, [https://factorinvestingtutorial.wordpress.com/3-the-fama-french-3-factor-model/](https://factorinvestingtutorial.wordpress.com/3-the-fama-french-3-factor-model/)
    

Download Financial Dataset Using Yahoo Finance in Python | A Complete Guide, acessado em julho 2, 2025, [https://www.analyticsvidhya.com/blog/2021/06/download-financial-dataset-using-yahoo-finance-in-python-a-complete-guide/](https://www.analyticsvidhya.com/blog/2021/06/download-financial-dataset-using-yahoo-finance-in-python-a-complete-guide/)**