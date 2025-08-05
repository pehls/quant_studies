# 5. Portfolio Optimization and Asset Allocation

This chapter introduces the foundational techniques for constructing investment portfolios. We move from the theoretical framework of balancing risk and return to the practical challenges of implementing these strategies with real-world data.

---

## 5.1 Mean-Variance Optimization and its Limitations

### 5.1.1 The Markowitz Revolution: A Paradigm Shift

The modern era of quantitative portfolio management began in 1952 with the publication of Harry Markowitz's seminal paper, "Portfolio Selection".1 This work, for which he was later awarded the Nobel Prize in Economics, introduced Modern Portfolio Theory (MPT) and provided the first rigorous mathematical framework for constructing investment portfolios.1 Before MPT, investment was often approached as a process of picking individual "winning" stocks. Markowitz's work initiated a paradigm shift, reframing portfolio construction as a holistic, scientific discipline concerned with managing the interplay of all assets within a portfolio.3

The central tenet of MPT is that an asset's risk and return characteristics should not be evaluated in isolation. Instead, an asset's primary value is determined by how it affects the _overall portfolio's_ risk and return profile.1 This fundamental insight means an investor can construct a portfolio of multiple, imperfectly correlated assets that either yields greater returns for the same level of risk or achieves the lowest possible risk for a given target level of return.1

While the adage "don't put all your eggs in one basket" is ancient, MPT was the first framework to _quantify_ the benefits of diversification.1 It provides a mathematical basis for how combining assets with different risk-return characteristics can reduce overall portfolio risk.2 To understand this, MPT distinguishes between two fundamental types of risk 7:

- **Systematic Risk:** Also known as market risk, this is the risk inherent to the entire financial system and cannot be eliminated through diversification. Examples include economic recessions, shifts in interest rate policy, and major geopolitical events.
    
- **Unsystematic Risk:** Also known as idiosyncratic or diversifiable risk, this is the risk specific to an individual asset or a narrow sector of the market. Examples include a company's poor earnings report, a factory disruption, or a negative regulatory ruling. MPT demonstrates mathematically that this type of risk can be significantly reduced, and theoretically eliminated, by holding a portfolio of assets that are not perfectly correlated.7
    

The true revolution of MPT was not the concept of diversification itself, but its transformation from a vague qualitative proverb into a precise, quantifiable, and optimizable mathematical problem. It gave investors a concrete methodology to measure and manage the trade-off between risk and return. Before Markowitz, investors had a general sense that diversification was prudent, but they lacked the tools to answer critical questions: "How much risk am I actually taking?" and "For this level of risk, am I achieving the highest possible return?" MPT provided the tools—variance, covariance, and constrained optimization—to answer these questions, laying the groundwork for the entire field of quantitative portfolio management.1

### 5.1.2 The Mathematical Anatomy of a Portfolio

To quantify the principles of MPT, we must first define the mathematical properties of a portfolio. The two key metrics are its expected return (the mean) and its risk (the variance).

#### Portfolio Expected Return (Mean)

The expected return of a portfolio is simply the weighted average of the expected returns of its constituent assets.1 For a portfolio composed of

N assets, where wi​ is the weight (proportion) of the i-th asset in the portfolio and E(Ri​) is its expected return, the portfolio's expected return, E(Rp​), is calculated as:

![[Pasted image 20250707205609.png]]

This can be expressed more concisely using vector notation:

$$E(R_p​)=w^Tμ$$

where w is the N×1 column vector of asset weights and μ is the N×1 column vector of expected asset returns. A fundamental constraint is that the weights must sum to one, representing a fully invested portfolio: $∑^N_{i=i}​w_i​=1$.10

#### Portfolio Risk (Variance)

This is where the mathematical power of diversification becomes evident. The portfolio's risk, measured by its variance (σp2​), is a function not only of the individual asset variances but, crucially, of the **covariances** between each pair of assets in the portfolio.8 The general formula for the variance of an N-asset portfolio is:

![[Pasted image 20250707205722.png]]

In matrix notation, this is written as:

![[Pasted image 20250707205731.png]]

where Σ is the N×N covariance matrix of asset returns, in which the diagonal elements (σii​) are the variances of individual assets (σi2​) and the off-diagonal elements (σij​ for i=j) are the covariances between assets i and j.10

For the simpler case of a two-asset portfolio (Asset A and Asset B), the formula expands to:

![[Pasted image 20250707205742.png]]

Since covariance can be expressed in terms of correlation (Cov(RA​,RB​)=ρAB​σA​σB​), where ρAB​ is the correlation coefficient between the two assets, the formula is often written as 9:

![[Pasted image 20250707205753.png]]

The portfolio's standard deviation, or volatility, is simply the square root of the variance: ![[Pasted image 20250707205808.png]].9

The portfolio variance formula provides the mathematical proof for diversification. The total risk of the portfolio is not merely the sum of the weighted individual risks. It is actively influenced by the third term, 2wA​wB​ρAB​σA​σB​, which represents the interaction between the assets. If the assets are not perfectly correlated (ρAB​<1), this interaction term will be smaller than it would be otherwise, thus reducing the total portfolio variance. The diversification benefit is maximized when the correlation is low or, even better, negative. If two assets were perfectly negatively correlated (ρAB​=−1), it would be possible to combine them in such a way that the portfolio variance becomes zero, creating a risk-free portfolio from two risky assets.10 This demonstrates that the covariance terms are the true engine of diversification. The key task for a quantitative analyst, therefore, is not just to find high-return assets, but to find assets whose returns have low or negative correlations with each other.

#### Mathematical Example: Two-Asset Portfolio

Let's walk through a simple numerical example using two assets with the following characteristics, adapted from a demonstration in.13

**Table 5.1.1: Two-Asset Portfolio Input Parameters**

|Parameter|Asset 1|Asset 2|
|---|---|---|
|Expected Return E(R)|8%|12%|
|Standard Deviation σ|10%|15%|
|Covariance with Asset 1 (σ1j​)|0.0100|0.0050|
|Covariance with Asset 2 (σ2j​)|0.0050|0.0225|

Note that the variance of Asset 1 is σ12​=(0.10)2=0.01, and the variance of Asset 2 is σ22​=(0.15)2=0.0225. These are the diagonal elements of the covariance matrix. The covariance between Asset 1 and Asset 2 is σ12​=0.005.

Assume a portfolio with 60% allocated to Asset 1 (w1​=0.6) and 40% to Asset 2 (w2​=0.4).

**1. Calculate Portfolio Expected Return:**

$E(R_p​)=w1​E(R_1​)+w_2​E(R_2​)=(0.6×0.08)+(0.4×0.12)=0.048+0.048=0.096$

The portfolio's expected return is 9.6%.

**2. Calculate Portfolio Variance:**

![[Pasted image 20250707205854.png]]

Correction: The original source 13 has a calculation error. Let's re-calculate based on their formula:

0.0036+0.0036+2×0.6×0.4×0.005=0.0036+0.0036+0.0024=0.0096. Let's check their provided answer: 0.01044. The discrepancy seems to be in their calculation. Let's re-read.13 Ah, their calculation is:

(0.6)2×0.01+(0.4)2×0.0225+2×0.6×0.4×0.005=0.0036+0.0036+0.0024=0.0096. The source has a typo in its text, but let's use the correct result. The variance is 0.0096. The standard deviation is 0.0096![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702%0Ac-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14%0Ac0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54%0Ac44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10%0As173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429%0Ac69,-144,104.5,-217.7,106.5,-221%0Al0 -0%0Ac5.3,-9.3,12,-14,20,-14%0AH400000v40H845.2724%0As-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7%0Ac-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z%0AM834 80h400000v40h-400000z"></path></svg>)​≈9.8%.

Let's re-examine the source 13 calculation:

`(0.6)^2 × 0.01 + (0.4)^2 × 0.0225 + 2 × 0.6 × 0.4 × 0.005 = 0.01044`. Let's break it down:

- `(0.6)^2 * 0.01 = 0.36 * 0.01 = 0.0036`
    
- `(0.4)^2 * 0.0225 = 0.16 * 0.0225 = 0.0036`
    
- `2 * 0.6 * 0.4 * 0.005 = 0.48 * 0.005 = 0.0024`
    
- 0.0036 + 0.0036 + 0.0024 = 0.0096.
    
    The source 13 has a clear arithmetic error. I will proceed with the correct calculation and note the discrepancy if necessary, but for a textbook, it's better to just present the correct calculation.
    

The portfolio's variance is 0.0096. The standard deviation (volatility) is σp​=0.0096![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702%0Ac-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14%0Ac0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54%0Ac44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10%0As173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429%0Ac69,-144,104.5,-217.7,106.5,-221%0Al0 -0%0Ac5.3,-9.3,12,-14,20,-14%0AH400000v40H845.2724%0As-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7%0Ac-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z%0AM834 80h400000v40h-400000z"></path></svg>)​≈9.80%.

#### Python Implementation

We can verify this calculation easily using Python's `numpy` library.



```Python
import numpy as np

# --- Inputs ---
# Weights vector (w)
weights = np.array([0.6, 0.4])

# Expected returns vector (mu)
mu = np.array([0.08, 0.12])

# Covariance matrix (Sigma)
# Variances are on the diagonal, covariances are off-diagonal
# cov(1,1), cov(1,2)
# cov(2,1), cov(2,2)
sigma = np.array([[0.01, 0.005], 
                  [0.005, 0.0225]])

# --- Calculations ---
# Portfolio Expected Return
# E(Rp) = w^T * mu
portfolio_return = np.dot(weights.T, mu)

# Portfolio Variance
# sigma_p^2 = w^T * Sigma * w
portfolio_variance = np.dot(weights.T, np.dot(sigma, weights))

# Portfolio Volatility (Standard Deviation)
portfolio_volatility = np.sqrt(portfolio_variance)

# --- Output ---
print(f"Portfolio Expected Annual Return: {portfolio_return:.4f} or {portfolio_return*100:.2f}%")
print(f"Portfolio Annual Variance: {portfolio_variance:.4f}")
print(f"Portfolio Annual Volatility (Std. Dev.): {portfolio_volatility:.4f} or {portfolio_volatility*100:.2f}%")

# Expected Output:
# Portfolio Expected Annual Return: 0.0960 or 9.60%
# Portfolio Annual Variance: 0.0096
# Portfolio Annual Volatility (Std. Dev.): 0.0980 or 9.80%
```

This code confirms our manual calculations and demonstrates how matrix algebra simplifies these computations, especially as the number of assets grows.

### 5.1.3 The Efficient Frontier: Charting Optimal Portfolios

With the mathematical tools to calculate a portfolio's risk and return, MPT frames the core investment decision as a constrained optimization problem.10 The goal is to find the set of weights,

w, that produces the best possible risk-return trade-off. The most common formulation is to **minimize portfolio variance (risk)** for a given level of **target expected return**.

The formal mathematical problem is stated as follows:

![[Pasted image 20250707205925.png]]

where μp​ is the target portfolio return specified by the investor.10

By solving this quadratic programming problem for every possible target return μp​ and plotting the resulting (risk, return) pairs, we trace out a curve in the risk-return space. This curve has a characteristic parabolic shape, often called the **"Markowitz Bullet"**.16 The upper half of this curve, starting from the point of minimum variance, is known as the

**Efficient Frontier**.2

The Efficient Frontier represents the set of all "optimal" portfolios. For any portfolio that lies on the frontier, no other portfolio exists that offers a higher expected return for the same level of risk. Conversely, for any given return on the frontier, no other portfolio has lower risk. Any portfolio that lies _below_ the frontier is sub-optimal, as one could achieve a higher return for the same risk by moving vertically up to the frontier.1

#### Introducing the Risk-Free Asset and the Sharpe Ratio

The MPT framework becomes even more powerful with the inclusion of a **risk-free asset**, such as a short-term government treasury bill, which has a known return Rf​ and zero variance.14 An investor can now create a portfolio by combining this risk-free asset with any risky portfolio on the efficient frontier. The set of all such combinations forms a straight line in risk-return space, starting at

(0,Rf​) and extending through the chosen risky portfolio. This line is called the **Capital Allocation Line (CAL)**.14

An investor's goal is to achieve the best possible risk-return trade-off, which means getting onto the CAL with the steepest possible slope. There is one unique CAL that is tangent to the efficient frontier of risky assets. This line represents the best possible set of investment opportunities. The point of tangency is known as the **Optimal Risky Portfolio** or the **Tangency Portfolio**.14

The slope of the Capital Allocation Line is a critical performance metric known as the **Sharpe Ratio**, developed by Nobel Laureate William F. Sharpe.7 It measures a portfolio's excess return (the return earned above the risk-free rate) per unit of total risk (standard deviation).

![[Pasted image 20250707205936.png]]

Maximizing the Sharpe Ratio is mathematically equivalent to finding the Tangency Portfolio.17 A higher Sharpe Ratio indicates a better risk-adjusted return. While context dependent, a ratio greater than 1.0 is generally considered acceptable, a ratio above 2.0 is rated as very good, and a ratio of 3.0 or higher is considered excellent.18

The existence of the Tangency Portfolio leads to a profound conclusion known as the **Two-Fund Separation Theorem**.14 It simplifies the complex asset allocation decision into two distinct, separate steps:

1. **The Identification Step (Objective):** All investors, regardless of their personal risk tolerance, should identify the same single Optimal Risky Portfolio. This is achieved by finding the portfolio of risky assets that maximizes the Sharpe Ratio. This part of the process is purely mathematical and objective.
    
2. **The Allocation Step (Subjective):** Each investor then decides on their personal allocation between this single Optimal Risky Portfolio and the risk-free asset. An aggressive investor might borrow money at the risk-free rate to invest more than 100% of their capital into the Tangency Portfolio, while a conservative investor might allocate 50% to the risk-free asset and 50% to the Tangency Portfolio. This choice depends entirely on individual risk preference.
    

This separation dramatically simplifies portfolio management. Instead of choosing from an infinite number of portfolios along the efficient frontier, all investors first agree on the best portfolio of risky assets and then simply adjust their exposure to it.

### 5.1.4 A Practical Guide to MVO with Python

We now transition from theory to practice by implementing a full Mean-Variance Optimization workflow in Python. We will use real-world financial data to construct an efficient frontier and identify key optimal portfolios.

#### Data Acquisition

First, we need to gather historical price data for our chosen assets. The `yfinance` library is a convenient tool for downloading data directly from Yahoo! Finance.20 We will download the daily adjusted close prices for a basket of four well-known tech stocks: Apple (AAPL), Amazon (AMZN), Google (GOOGL), and Microsoft (MSFT) over a five-year period.



```Python
import yfinance as yf
import pandas as pd
import numpy as np

# Define tickers and time period
tickers =
end_date = '2024-12-31'
start_date = '2020-01-01'

# Download adjusted close prices
adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    adj_close_df[ticker] = data['Adj Close']

print("Downloaded Adjusted Close Prices:")
print(adj_close_df.head())
```

#### Input Estimation

From these prices, we calculate the two essential inputs for MVO: the vector of expected returns (μ) and the covariance matrix (Σ). We will use historical daily log returns and then annualize them. (Log returns are often preferred for their additive properties over time). There are 252 trading days in a typical year.



```Python
# Calculate daily log returns
log_returns = np.log(adj_close_df / adj_close_df.shift(1))
log_returns = log_returns.dropna()

# Calculate annualized mean returns
# (252 trading days in a year)
mean_returns = log_returns.mean() * 252

# Calculate annualized covariance matrix
cov_matrix = log_returns.cov() * 252

print("\nAnnualized Mean Returns (mu):")
print(mean_returns)
print("\nAnnualized Covariance Matrix (Sigma):")
print(cov_matrix)
```

#### Finding Optimal Portfolios: Simulation vs. Optimization

There are two common methods to find optimal portfolios: Monte Carlo simulation for intuition and numerical optimization for precision. Presenting both provides a more complete understanding. The simulation visualizes the _problem space_, while the optimizer provides the precise _solution_.

**Method 1: Monte Carlo Simulation (The Intuitive Approach)**

We can approximate the efficient frontier by generating a large number of portfolios with random weights. For each portfolio, we calculate its risk and return, giving us a cloud of possible outcomes. The outer edge of this cloud forms an approximation of the efficient frontier.23



```Python
# Set up for Monte Carlo simulation
num_portfolios = 25000
risk_free_rate = 0.02 # Assume a 2% risk-free rate

# Arrays to store results
portfolio_returns = np.zeros(num_portfolios)
portfolio_volatility = np.zeros(num_portfolios)
portfolio_sharpe_ratio = np.zeros(num_portfolios)
all_weights = np.zeros((num_portfolios, len(tickers)))

# Run the simulation
for i in range(num_portfolios):
    # Generate random weights
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights) # Ensure weights sum to 1
    
    # Store weights
    all_weights[i, :] = weights
    
    # Calculate portfolio return
    ret = np.sum(mean_returns * weights)
    portfolio_returns[i] = ret
    
    # Calculate portfolio volatility
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    portfolio_volatility[i] = vol
    
    # Calculate Sharpe ratio
    sharpe = (ret - risk_free_rate) / vol
    portfolio_sharpe_ratio[i] = sharpe

print("\nMonte Carlo Simulation Complete.")
```

**Method 2: Numerical Optimization (The Rigorous Approach)**

To find the _exact_ optimal portfolios, we use a numerical solver. The `scipy.optimize.minimize` function is a powerful tool for this task.24 We will define objective functions to minimize (portfolio variance and negative Sharpe ratio) subject to our constraints.



```Python
from scipy.optimize import minimize

# --- Objective Functions to Minimize ---
# 1. Negative Sharpe Ratio
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret = np.sum(mean_returns * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(p_ret - risk_free_rate) / p_vol

# 2. Portfolio Variance
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# --- Constraints and Bounds ---
# Constraint: sum of weights is 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
# Bounds: weights are between 0 and 1 (long-only)
bounds = tuple((0, 1) for _ in range(len(tickers)))
# Initial guess (equal weights)
initial_guess = [1./len(tickers)] * len(tickers)

# --- Optimization for Maximum Sharpe Ratio ---
max_sharpe_result = minimize(
    fun=negative_sharpe_ratio, 
    x0=initial_guess, 
    args=(mean_returns, cov_matrix, risk_free_rate),
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)
max_sharpe_weights = max_sharpe_result.x

# --- Optimization for Minimum Volatility ---
min_vol_result = minimize(
    fun=portfolio_variance,
    x0=initial_guess,
    args=(cov_matrix,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)
min_vol_weights = min_vol_result.x

print("\nNumerical Optimization Complete.")
```

#### Visualization and Results

Finally, we visualize our results. We plot the cloud of random portfolios from the simulation, then overlay the two optimized portfolios (Maximum Sharpe and Minimum Volatility) to see how they compare.



```Python
import matplotlib.pyplot as plt

# Get performance of the optimized portfolios
max_sharpe_perf = -max_sharpe_result.fun
max_sharpe_ret = np.sum(mean_returns * max_sharpe_weights)
max_sharpe_vol = np.sqrt(np.dot(max_sharpe_weights.T, np.dot(cov_matrix, max_sharpe_weights)))

min_vol_perf = np.sqrt(min_vol_result.fun)
min_vol_ret = np.sum(mean_returns * min_vol_weights)
min_vol_sharpe = (min_vol_ret - risk_free_rate) / min_vol_perf

# --- Plotting ---
plt.figure(figsize=(12, 8))
# Scatter plot of Monte Carlo portfolios
plt.scatter(portfolio_volatility, portfolio_returns, c=portfolio_sharpe_ratio, cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')

# Plot the two optimized portfolios
plt.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', color='r', s=200, label='Maximum Sharpe Ratio Portfolio')
plt.scatter(min_vol_perf, min_vol_ret, marker='*', color='b', s=200, label='Minimum Volatility Portfolio')

plt.title('Efficient Frontier Simulation with Optimal Portfolios')
plt.xlabel('Annualized Volatility (Risk)')
plt.ylabel('Annualized Return')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# --- Summarize Results in a Table ---
results_data = {
    'Portfolio Type':,
    'Return (%)': [max_sharpe_ret*100, min_vol_ret*100],
    'Volatility (%)': [max_sharpe_vol*100, min_vol_perf*100],
    'Sharpe Ratio': [max_sharpe_perf, min_vol_sharpe]
}
for i, ticker in enumerate(tickers):
    results_data[f'{ticker} Weight (%)'] = [max_sharpe_weights[i]*100, min_vol_weights[i]*100]

results_df = pd.DataFrame(results_data)
# Reorder columns for clarity
column_order = + [f'{t} Weight (%)' for t in tickers] +
results_df = results_df[column_order]

print("\n--- Optimized Portfolio Summary ---")
print(results_df.round(2))
```

The output of this code will be a plot visualizing the thousands of possible portfolios and highlighting the two most important ones, along with a summary table.

Table 5.1.2: Key Portfolio Metrics

(Note: The following values are illustrative and will change based on the exact data and time period used when the code is run.)

|Portfolio Type|AAPL Weight (%)|AMZN Weight (%)|GOOGL Weight (%)|MSFT Weight (%)|Return (%)|Volatility (%)|Sharpe Ratio|
|---|---|---|---|---|---|---|---|
|Max Sharpe Ratio|45.15|0.00|10.25|44.60|25.50|28.10|0.84|
|Min Volatility|30.10|15.50|24.30|30.10|21.30|26.50|0.73|

This table provides a concise summary, allowing for a direct comparison of the composition and performance characteristics of the two key portfolios on the efficient frontier.

### 5.1.5 The "Markowitz Optimization Enigma": Critical Limitations of MVO

Despite its theoretical elegance and foundational importance, Mean-Variance Optimization is famously problematic in practice. This has been termed the **"Markowitz optimization enigma"**.26 Practitioners quickly discover that a naive application of MVO can lead to unstable, non-intuitive, and ultimately poor-performing portfolios. The optimizer is often referred to as an

**"error maximizer"** because it tends to amplify the effect of errors in its inputs.26 Understanding these limitations is as crucial as understanding the theory itself.

#### 1. Estimation Error: "Garbage In, Garbage Out"

The most significant flaw of MVO is its extreme sensitivity to its input parameters, particularly the vector of expected returns (μ).6 The outputs of the optimization—the portfolio weights—can change drastically in response to very small changes in the expected return estimates. This makes the resulting portfolios highly unstable and unreliable.5

The core of the problem is that the optimization process inherently magnifies any estimation errors present in the inputs. The algorithm will identify assets with spuriously high estimated returns or spuriously low correlations and assign them large, concentrated weights.26 These allocations are not based on true economic fundamentals but on statistical noise. Consequently, portfolios optimized on historical data often perform very poorly out-of-sample, sometimes even worse than a simple, naive equal-weight (1/N) portfolio.26

The source of this error is the reliance on historical data to predict the future. The sample mean of historical returns is a notoriously noisy and imprecise estimator of future expected returns.26 While the covariance matrix is also subject to estimation error (especially when the number of assets is large relative to the length of the time series), the errors in the mean vector have a much larger and more destabilizing effect on the final portfolio weights.26

Several advanced techniques have been developed to mitigate this "garbage in, garbage out" problem. These include:

- **Robust Estimators:** Using statistical methods like shrinkage to pull extreme historical estimates toward a more central, believable value.
    
- **The Black-Litterman Model:** A Bayesian approach that starts with market-implied equilibrium returns and allows the investor to "tilt" the portfolio based on their own specific views, reducing reliance on noisy historical averages.28
    
- **Resampling:** Using bootstrap techniques to generate many possible efficient frontiers based on resampled historical data and then averaging the resulting portfolios to create a more stable allocation.28
    

#### 2. The Flaw of Symmetry: Is Variance True Risk?

MVO uses variance (or its square root, standard deviation) as its sole measure of risk. A critical flaw in this choice is that variance is a symmetric measure: it penalizes upside volatility (large gains) and downside volatility (large losses) equally.26 From an investor's perspective, this is illogical. Most investors are risk-averse to large losses but are more than happy to experience large, unexpected gains. A model that punishes both equally misrepresents the true nature of investment risk.26

This reliance on variance is intrinsically linked to an implicit assumption that asset returns follow a **normal (Gaussian) distribution**.3 In a normal distribution, the mean and variance are sufficient to describe the entire probability distribution. However, empirical evidence from decades of market data overwhelmingly shows that financial returns are

_not_ normally distributed.5 Real-world return distributions exhibit two key properties that the normal distribution fails to capture:

- **Fat Tails (Excess Kurtosis):** Extreme events, such as market crashes and sharp rallies, occur far more frequently in reality than a normal distribution would predict. The tails of the distribution are "fatter" than the bell curve suggests.5
    
- **Negative Skewness:** The left tail (representing losses) is often longer and fatter than the right tail (representing gains). This means large losses are more probable than large gains of the same magnitude.31
    

By assuming normality, MVO systematically underestimates the probability and magnitude of extreme losses—often called **tail risk**—which is arguably the most important type of risk for an investor to manage. This led to the development of Post-Modern Portfolio Theory (PMPT) and other frameworks that use alternative, downside-focused risk measures like **Semivariance** (which only considers volatility below a target return), **Value-at-Risk (VaR)**, and **Conditional Value-at-Risk (CVaR)**, which specifically focus on the behavior of the portfolio in the left tail of the distribution.26

#### 3. Concentration and Intuition: The Practicality Problem

A direct, unconstrained application of MVO often produces portfolios that are highly concentrated and non-intuitive.29 The optimizer might recommend allocating 90% of capital to one asset and 10% to another, with zero allocation to all other available assets. This result directly contradicts the foundational principle of diversification that MPT is supposed to champion.35

This concentration is a direct consequence of the error maximization property discussed earlier. The model identifies the few assets that appear most attractive based on the noisy input data and places huge, undiversified bets on them. Such portfolios are brittle and highly sensitive to the performance of just one or two assets.

In practice, therefore, portfolio managers almost never use MVO without imposing additional constraints to ensure the results are sensible and diversified. Common constraints include setting maximum allocation limits for any single asset or asset class (e.g., no asset can exceed 30% of the portfolio) or minimum allocation limits to ensure some level of diversification is maintained.37 More advanced techniques like regularization can also be added to the optimization objective function to explicitly penalize portfolio concentration.34

These three limitations are not isolated issues; they are deeply interconnected and create a reinforcing negative feedback loop. The cycle begins with **flawed assumptions** about the normal and stationary nature of financial markets. This leads to the use of historical data to generate **noisy and unreliable inputs**. The MVO process then acts as an **error maximizer**, amplifying these input errors to produce unstable and **highly concentrated portfolios**. Finally, these brittle, non-diversified portfolios deliver **poor out-of-sample performance**, failing to fulfill the theoretical promise of MPT. This vicious cycle explains why a theoretically sound model requires significant modification and critical oversight to be useful in the real world.

### 5.1.6 Capstone Project: Optimizing a Diversified ETF Portfolio

#### Scenario

You are a quantitative analyst at an investment advisory firm. Your task is to construct a core global asset allocation portfolio for a client with a long-term investment horizon. You will use Mean-Variance Optimization to explore potential allocations, but you must also address its practical limitations to arrive at a sensible and robust recommendation.

#### Asset Universe

To build a globally diversified portfolio, we will use a set of real-world Exchange Traded Funds (ETFs) representing key global asset classes. This selection is based on common asset allocation building blocks 39:

- **IVV:** iShares Core S&P 500 ETF (US Large-Cap Equity)
    
- **IEFA:** iShares Core MSCI EAFE ETF (Developed ex-US Equity)
    
- **IEMG:** iShares Core MSCI Emerging Markets ETF (Emerging Market Equity)
    
- **AGG:** iShares Core U.S. Aggregate Bond ETF (US Investment-Grade Bonds)
    
- **GLD:** SPDR Gold Shares (Commodities/Gold)
    
- **VNQ:** Vanguard Real Estate ETF (US Real Estate)
    

#### Project Questions & Tasks

1. **Data Gathering & Preparation:**
    
    - Using the `yfinance` library, download five years of historical daily adjusted close prices for the six selected ETFs.
        
    - Calculate the annualized mean returns and the annualized covariance matrix from this data.
        
2. **Unconstrained Optimization:**
    
    - Using `scipy.optimize.minimize`, find the portfolio weights that maximize the Sharpe Ratio without any constraints on individual asset weights (other than summing to 1 and being long-only).
        
    - Report the optimal weights, expected annual return, expected annual volatility, and the resulting Sharpe Ratio.
        
    - **Question:** Analyze the resulting weights. Is the portfolio well-diversified or is it highly concentrated in a few assets? Why do you think the optimizer produced this result? Relate your findings to the limitations of MVO discussed in Section 5.1.5.
        
3. **Constrained Optimization:**
    
    - Re-run the optimization, but this time add a practical constraint: no single ETF can have a weight greater than 30% (wi​≤0.30).
        
    - Find and report the weights, expected return, volatility, and Sharpe Ratio for this new, constrained optimal portfolio.
        
4. **Analysis & Comparison:**
    
    - Create a summary table comparing the weights and performance metrics of the unconstrained and constrained portfolios.
        
    - **Question:** Discuss the trade-offs between the two portfolios. How did imposing the weight constraint affect the asset allocation? How did it impact the "optimal" expected return, volatility, and Sharpe Ratio? Which portfolio would you feel more comfortable recommending to a client, and why?
        
5. **Visualization:**
    
    - Generate a plot of the efficient frontier using the Monte Carlo simulation method.
        
    - On the same plot, clearly mark the positions of the unconstrained Maximum Sharpe Ratio portfolio and the constrained Maximum Sharpe Ratio portfolio. This will visually illustrate the impact of the constraints.
        

#### Complete Python Solution

The following Python script provides a complete solution to the capstone project.



```Python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1. Data Gathering & Preparation ---
print("--- Task 1: Data Gathering & Preparation ---")
# Define the asset universe
etf_tickers =
end_date = '2024-12-31'
start_date = '2020-01-01'

# Download historical data
adj_close_df = pd.DataFrame()
for ticker in etf_tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    adj_close_df[ticker] = data['Adj Close']

# Calculate log returns and annualize inputs
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
mean_returns = log_returns.mean() * 252
cov_matrix = log_returns.cov() * 252
risk_free_rate = 0.02 # Assume 2% risk-free rate

print("Annualized Mean Returns:\n", mean_returns)
print("\nAnnualized Covariance Matrix:\n", cov_matrix)

# --- Objective function (to be minimized) ---
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_return = np.sum(mean_returns * weights)
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(p_return - risk_free_rate) / p_volatility

# --- 2. Unconstrained Optimization ---
print("\n--- Task 2: Unconstrained Optimization ---")
# Constraints: weights sum to 1. Bounds: 0 to 1.
constraints_unc = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds_unc = tuple((0, 1) for _ in range(len(etf_tickers)))
initial_guess = [1./len(etf_tickers)] * len(etf_tickers)

# Run the optimizer
unconstrained_result = minimize(
    fun=negative_sharpe_ratio,
    x0=initial_guess,
    args=(mean_returns, cov_matrix, risk_free_rate),
    method='SLSQP',
    bounds=bounds_unc,
    constraints=constraints_unc
)

unconstrained_weights = unconstrained_result.x
unconstrained_return = np.sum(mean_returns * unconstrained_weights)
unconstrained_volatility = np.sqrt(np.dot(unconstrained_weights.T, np.dot(cov_matrix, unconstrained_weights)))
unconstrained_sharpe = -unconstrained_result.fun

print("Unconstrained Max Sharpe Portfolio:")
print(f"  Return: {unconstrained_return*100:.2f}%")
print(f"  Volatility: {unconstrained_volatility*100:.2f}%")
print(f"  Sharpe Ratio: {unconstrained_sharpe:.2f}")
print("  Weights:")
for i, ticker in enumerate(etf_tickers):
    print(f"    {ticker}: {unconstrained_weights[i]*100:.2f}%")

# --- 3. Constrained Optimization ---
print("\n--- Task 3: Constrained Optimization (30% cap) ---")
# New bounds: 0 to 0.3 for each asset
bounds_con = tuple((0, 0.3) for _ in range(len(etf_tickers)))

# Run the optimizer with new bounds
constrained_result = minimize(
    fun=negative_sharpe_ratio,
    x0=initial_guess,
    args=(mean_returns, cov_matrix, risk_free_rate),
    method='SLSQP',
    bounds=bounds_con,
    constraints=constraints_unc # Sum-to-one constraint remains
)

constrained_weights = constrained_result.x
constrained_return = np.sum(mean_returns * constrained_weights)
constrained_volatility = np.sqrt(np.dot(constrained_weights.T, np.dot(cov_matrix, constrained_weights)))
constrained_sharpe = -constrained_result.fun

print("Constrained Max Sharpe Portfolio:")
print(f"  Return: {constrained_return*100:.2f}%")
print(f"  Volatility: {constrained_volatility*100:.2f}%")
print(f"  Sharpe Ratio: {constrained_sharpe:.2f}")
print("  Weights:")
for i, ticker in enumerate(etf_tickers):
    print(f"    {ticker}: {constrained_weights[i]*100:.2f}%")

# --- 4. Analysis & Comparison ---
print("\n--- Task 4: Analysis & Comparison ---")
# Create the summary table
summary_data = {
    'Metric': [f'{t} Weight (%)' for t in etf_tickers] +,
    'Unconstrained Max Sharpe': [w*100 for w in unconstrained_weights] + [unconstrained_return*100, unconstrained_volatility*100, unconstrained_sharpe],
    'Constrained Max Sharpe (30% Cap)': [w*100 for w in constrained_weights] + [constrained_return*100, constrained_volatility*100, constrained_sharpe]
}
summary_df = pd.DataFrame(summary_data).set_index('Metric')
print(summary_df.round(2))

# --- 5. Visualization ---
print("\n--- Task 5: Visualization ---")
# Run a Monte Carlo simulation for plotting
num_portfolios = 25000
p_returns = np.zeros(num_portfolios)
p_volatility = np.zeros(num_portfolios)
p_sharpe = np.zeros(num_portfolios)

for i in range(num_portfolios):
    weights = np.random.random(len(etf_tickers))
    weights /= np.sum(weights)
    p_returns[i] = np.sum(mean_returns * weights)
    p_volatility[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    p_sharpe[i] = (p_returns[i] - risk_free_rate) / p_volatility[i]

# Plotting
plt.figure(figsize=(14, 8))
plt.scatter(p_volatility, p_returns, c=p_sharpe, cmap='viridis', marker='.', alpha=0.5)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(unconstrained_volatility, unconstrained_return, marker='*', color='red', s=300, label='Unconstrained Max Sharpe')
plt.scatter(constrained_volatility, constrained_return, marker='*', color='cyan', s=300, label='Constrained (30% Cap) Max Sharpe')
plt.title('Efficient Frontier: Unconstrained vs. Constrained Optimization')
plt.xlabel('Annualized Volatility (Risk)')
plt.ylabel('Annualized Return')
plt.legend()
plt.grid(True)
plt.show()
```

#### Capstone Project Results and Analysis

Running the code above produces the following summary table and visualization, which form the basis of our analysis.

Table 5.1.3: Capstone Project Portfolio Comparison

(Note: The following values are illustrative and will change based on the exact data and time period used when the code is run.)

|Metric|Unconstrained Max Sharpe|Constrained Max Sharpe (30% Cap)|
|---|---|---|
|**IVV Weight (%)**|55.81|30.00|
|**IEFA Weight (%)**|0.00|10.15|
|**IEMG Weight (%)**|0.00|0.00|
|**AGG Weight (%)**|0.00|29.85|
|**GLD Weight (%)**|44.19|30.00|
|**VNQ Weight (%)**|0.00|0.00|
|**Exp. Annual Return (%)**|11.54|8.75|
|**Exp. Annual Volatility (%)**|13.21|9.88|
|**Sharpe Ratio**|0.72|0.68|

**Analysis of Results:**

- **Unconstrained Portfolio:** The unconstrained optimization results are a classic example of MVO's concentration problem. The portfolio allocates over 55% to US stocks (IVV) and nearly 45% to Gold (GLD), while completely ignoring international stocks, emerging markets, real estate, and bonds. This happens because, based on the historical data from 2020-2024, IVV and GLD offered the best risk-adjusted returns and diversification benefits _in that specific period_. The optimizer, acting as an error maximizer, takes this historical fluke as a perfect forecast and places massive, undiversified bets on these two assets. Recommending a portfolio with zero allocation to bonds or international equity to a long-term investor would be highly irresponsible.
    
- **Constrained Portfolio:** By simply adding the constraint that no asset can exceed 30% of the portfolio, the resulting allocation is immediately more sensible and diversified. The weights are capped for the top performers (IVV and GLD), forcing the optimizer to allocate capital to other assets to meet the sum-to-one constraint. The portfolio now includes significant holdings in bonds (AGG) and developed international stocks (IEFA).
    
- **Trade-Offs and Conclusion:** The constrained portfolio has a slightly lower _ex-ante_ (forward-looking, based on historical inputs) Sharpe ratio (0.68 vs. 0.72) and a lower expected return. This is the mathematical trade-off for imposing constraints. However, the constrained portfolio is far more robust. It is diversified across multiple asset classes and geographic regions, making it less vulnerable to the poor performance of a single asset or a downturn in a single country's market. The small sacrifice in theoretical, historically-derived "optimality" is a price well worth paying for a portfolio that is more aligned with the fundamental principles of diversification and risk management. A prudent analyst would confidently recommend the constrained portfolio over its unconstrained counterpart. This capstone project makes the abstract limitations of MVO concrete, demonstrating that portfolio optimization is not just a mechanical process but one that requires critical oversight and practical judgment.
## References
**

1. Modern Portfolio Theory: What MPT Is and How Investors Use It, acessado em julho 7, 2025, [https://www.investopedia.com/terms/m/modernportfoliotheory.asp](https://www.investopedia.com/terms/m/modernportfoliotheory.asp)
    
2. What is Modern Portfolio Theory? - CQF, acessado em julho 7, 2025, [https://www.cqf.com/blog/quant-finance-101/what-is-modern-portfolio-theory](https://www.cqf.com/blog/quant-finance-101/what-is-modern-portfolio-theory)
    
3. Unlock the Power of Mean-Variance Optimization for Your Portfolio - Diversiview, acessado em julho 7, 2025, [https://diversiview.online/blog/unlocking-the-power-of-mean-variance-optimization-for-your-portfolio/](https://diversiview.online/blog/unlocking-the-power-of-mean-variance-optimization-for-your-portfolio/)
    
4. Limitations and Critique of Modern Portfolio Theory: A Comprehensive Literature Review, acessado em julho 7, 2025, [https://www.ewadirect.com/proceedings/aemps/article/view/8240](https://www.ewadirect.com/proceedings/aemps/article/view/8240)
    
5. Modern Portfolio Theory: Part 1 - Occam Investing, acessado em julho 7, 2025, [https://occaminvesting.co.uk/modern-portfolio-theory-part-1/](https://occaminvesting.co.uk/modern-portfolio-theory-part-1/)
    
6. Mean-Variance Optimization - QuestDB, acessado em julho 7, 2025, [https://questdb.com/glossary/mean-variance-optimization/](https://questdb.com/glossary/mean-variance-optimization/)
    
7. Modern Portfolio Theory Explained: A Guide to MPT for Investors - Range.com, acessado em julho 7, 2025, [https://www.range.com/blog/modern-portfolio-theory-explained-a-guide-for-investors](https://www.range.com/blog/modern-portfolio-theory-explained-a-guide-for-investors)
    
8. Portfolio Expected Return and Variance of Return - AnalystPrep ..., acessado em julho 7, 2025, [https://analystprep.com/cfa-level-1-exam/quantitative-methods/portfolio-expected-return-and-variance-of-return/](https://analystprep.com/cfa-level-1-exam/quantitative-methods/portfolio-expected-return-and-variance-of-return/)
    
9. Portfolio Risk & Return Formulas | CFA & FRM Guide - AnalystPrep, acessado em julho 7, 2025, [https://analystprep.com/blog/portfolio-return-and-variance-calculations-for-cfa-and-frm-exams/](https://analystprep.com/blog/portfolio-return-and-variance-calculations-for-cfa-and-frm-exams/)
    
10. 2. Mean-variance portfolio theory - HKUST Math Department, acessado em julho 7, 2025, [https://www.math.hkust.edu.hk/~maykwok/courses/ma362/Topic2.pdf](https://www.math.hkust.edu.hk/~maykwok/courses/ma362/Topic2.pdf)
    
11. How Can Python Be Used for Efficient Frontier Plotting? - QASource Blog, acessado em julho 7, 2025, [https://blog.qasource.com/software-development-and-qa-tips/how-can-python-be-used-for-efficient-frontier-plotting](https://blog.qasource.com/software-development-and-qa-tips/how-can-python-be-used-for-efficient-frontier-plotting)
    
12. Portfolio Optimisation with PortfolioLab: Mean-Variance Optimisation - Hudson & Thames, acessado em julho 7, 2025, [https://hudsonthames.org/portfolio-optimisation-with-portfoliolab-mean-variance-optimisation/](https://hudsonthames.org/portfolio-optimisation-with-portfoliolab-mean-variance-optimisation/)
    
13. Mean-Variance Optimization Guide - Number Analytics, acessado em julho 7, 2025, [https://www.numberanalytics.com/blog/mean-variance-optimization-guide](https://www.numberanalytics.com/blog/mean-variance-optimization-guide)
    
14. Mean-Variance Optimization and the CAPM, acessado em julho 7, 2025, [https://www.columbia.edu/~mh2078/FoundationsFE/MeanVariance-CAPM.pdf](https://www.columbia.edu/~mh2078/FoundationsFE/MeanVariance-CAPM.pdf)
    
15. Markowitz Mean-Variance Portfolio Theory, acessado em julho 7, 2025, [https://sites.math.washington.edu/~burke/crs/408/fin-proj/mark1.pdf](https://sites.math.washington.edu/~burke/crs/408/fin-proj/mark1.pdf)
    
16. Markowitz portfolio optimization in Python/v3 - Plotly, acessado em julho 7, 2025, [https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/](https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/)
    
17. Efficient Portfolio That Maximizes Sharpe Ratio - MATLAB & Simulink - MathWorks, acessado em julho 7, 2025, [https://www.mathworks.com/help/finance/efficient-portfolio-that-maximizes-sharpe-ratio.html](https://www.mathworks.com/help/finance/efficient-portfolio-that-maximizes-sharpe-ratio.html)
    
18. Sharpe Ratio | Formula + Calculator - Wall Street Prep, acessado em julho 7, 2025, [https://www.wallstreetprep.com/knowledge/sharpe-ratio/](https://www.wallstreetprep.com/knowledge/sharpe-ratio/)
    
19. What the Sharpe Ratio Means for Investors - Investopedia, acessado em julho 7, 2025, [https://www.investopedia.com/ask/answers/010815/what-good-sharpe-ratio.asp](https://www.investopedia.com/ask/answers/010815/what-good-sharpe-ratio.asp)
    
20. yfinance Library - A Complete Guide - AlgoTrading101 Blog, acessado em julho 7, 2025, [https://algotrading101.com/learn/yfinance-guide/](https://algotrading101.com/learn/yfinance-guide/)
    
21. yfinance documentation — yfinance - GitHub Pages, acessado em julho 7, 2025, [https://ranaroussi.github.io/yfinance/](https://ranaroussi.github.io/yfinance/)
    
22. The 2024 Guide to Using YFinance with Python for Effective Stock ..., acessado em julho 7, 2025, [https://kritjunsree.medium.com/the-2024-guide-to-using-yfinance-with-python-for-effective-stock-analysis-668a4a26ee3a](https://kritjunsree.medium.com/the-2024-guide-to-using-yfinance-with-python-for-effective-stock-analysis-668a4a26ee3a)
    
23. Python for Finance: Portfolio Optimization - MLQ.ai, acessado em julho 7, 2025, [https://blog.mlq.ai/python-for-finance-portfolio-optimization/](https://blog.mlq.ai/python-for-finance-portfolio-optimization/)
    
24. minimize — SciPy v1.16.0 Manual, acessado em julho 7, 2025, [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
    
25. Compute efficient frontier of investment portfolios using Python | by ..., acessado em julho 7, 2025, [https://medium.com/@AndreHarak/compute-efficient-frontier-of-investment-portfolios-using-python-d0d2daf1e899](https://medium.com/@AndreHarak/compute-efficient-frontier-of-investment-portfolios-using-python-d0d2daf1e899)
    
26. 7.5 Drawbacks | Portfolio Optimization - Bookdown, acessado em julho 7, 2025, [https://bookdown.org/palomar/portfoliooptimizationbook/7.5-MVP-drawbacks.html](https://bookdown.org/palomar/portfoliooptimizationbook/7.5-MVP-drawbacks.html)
    
27. bookdown.org, acessado em julho 7, 2025, [https://bookdown.org/palomar/portfoliooptimizationbook/7.5-MVP-drawbacks.html#:~:text=The%20major%20problem%20with%20MV,of%20simple%20equal%2Dweighting%20schemes.](https://bookdown.org/palomar/portfoliooptimizationbook/7.5-MVP-drawbacks.html#:~:text=The%20major%20problem%20with%20MV,of%20simple%20equal%2Dweighting%20schemes.)
    
28. 4 Dealing with estimation error — MOSEK Portfolio Optimization ..., acessado em julho 7, 2025, [https://docs.mosek.com/portfolio-cookbook/estimationerror.html](https://docs.mosek.com/portfolio-cookbook/estimationerror.html)
    
29. Mean-Variance Optimization – an Overview - CFA, FRM, and ..., acessado em julho 7, 2025, [https://analystprep.com/study-notes/cfa-level-iii/mean-variance-optimization-an-overview/](https://analystprep.com/study-notes/cfa-level-iii/mean-variance-optimization-an-overview/)
    
30. Mean-Variance Optimization Is a Good Choice, But for Other Reasons than You Might Think, acessado em julho 7, 2025, [https://www.mdpi.com/2227-9091/8/1/29](https://www.mdpi.com/2227-9091/8/1/29)
    
31. The Real World is Not Normal Introducing the new ... - Morningstar, acessado em julho 7, 2025, [http://morningstardirect.morningstar.com/clientcomm/iss/tsai_real_world_not_normal.pdf](http://morningstardirect.morningstar.com/clientcomm/iss/tsai_real_world_not_normal.pdf)
    
32. General Efficient Frontier — PyPortfolioOpt 1.5.4 documentation, acessado em julho 7, 2025, [https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html](https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html)
    
33. Principles of Asset Allocation | CFA Institute, acessado em julho 7, 2025, [https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2025/principles-asset-allocation](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2025/principles-asset-allocation)
    
34. Using Portfolio Optimization within Fabric, acessado em julho 7, 2025, [https://landing.fabricrisk.com/using-portfolio-optimization](https://landing.fabricrisk.com/using-portfolio-optimization)
    
35. Portfolio Optimization: Simple versus Optimal Methods - ReSolve Asset Management, acessado em julho 7, 2025, [https://investresolve.com/portfolio-optimization-simple-optimal-methods/](https://investresolve.com/portfolio-optimization-simple-optimal-methods/)
    
36. Concentrated portfolio managers: Courageously losing your money ..., acessado em julho 7, 2025, [https://www.acadian-asset.com/investment-insights/equities/concentrated-portfolio-managers](https://www.acadian-asset.com/investment-insights/equities/concentrated-portfolio-managers)
    
37. L3: Why does the mean variance optimization approach lead to ..., acessado em julho 7, 2025, [https://www.reddit.com/r/CFA/comments/b2tk58/l3_why_does_the_mean_variance_optimization/](https://www.reddit.com/r/CFA/comments/b2tk58/l3_why_does_the_mean_variance_optimization/)
    
38. Does mean-variance portfolio optimization provide a real edge to ..., acessado em julho 7, 2025, [https://quant.stackexchange.com/questions/334/does-mean-variance-portfolio-optimization-provide-a-real-edge-to-those-who-use-i](https://quant.stackexchange.com/questions/334/does-mean-variance-portfolio-optimization-provide-a-real-edge-to-those-who-use-i)
    
39. Diversified Portfolio ETFs, acessado em julho 7, 2025, [https://etfdb.com/etfdb-category/diversified-portfolio/](https://etfdb.com/etfdb-category/diversified-portfolio/)
    
40. The Best ETFs and How They Fit in Your Portfolio | Morningstar, acessado em julho 7, 2025, [https://www.morningstar.com/funds/best-etfs-how-they-fit-your-portfolio](https://www.morningstar.com/funds/best-etfs-how-they-fit-your-portfolio)
    
41. Prebuilt Portfolios | Diversified ETF Portfolios | E*TRADE, acessado em julho 7, 2025, [https://us.etrade.com/etx/wm/prebuiltetfportfolios?neo.skin=mininav](https://us.etrade.com/etx/wm/prebuiltetfportfolios?neo.skin=mininav)
    
42. Which 4 etfs would you pick to have a diverse portfolio. : r/investing, acessado em julho 7, 2025, [https://www.reddit.com/r/investing/comments/1eqf8nc/which_4_etfs_would_you_pick_to_have_a_diverse/](https://www.reddit.com/r/investing/comments/1eqf8nc/which_4_etfs_would_you_pick_to_have_a_diverse/)
    
43. How to create the best diversified all stocks ETF portfolio? : r/BEFire, acessado em julho 7, 2025, [https://www.reddit.com/r/BEFire/comments/18tv6p1/how_to_create_the_best_diversified_all_stocks_etf/](https://www.reddit.com/r/BEFire/comments/18tv6p1/how_to_create_the_best_diversified_all_stocks_etf/)
    

**