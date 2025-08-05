### Introduction: The Proactive Response to Market Uncertainty

The bedrock of traditional investment management is asset allocation, the process of deciding how to distribute a portfolio's capital across different asset classes like equities, bonds, and commodities.1 The most common approach is

**Static Asset Allocation**, a disciplined, long-term strategy where an investor sets a fixed target mix—for instance, a 60% allocation to equities and 40% to bonds—and largely adheres to it through market cycles.2 The primary mechanism for adjustment in a static strategy is periodic rebalancing, which serves to bring the portfolio back to its original target weights after market movements have caused it to drift.1 This strategy is built on the principles of long-term investment and the belief in eventual market corrections, relying on the inherent risk distribution of the chosen asset mix to deliver returns over time.2

However, the principal weakness of the static paradigm lies in its passivity. By design, it does not react to short-term market volatility or changing economic conditions.1 Consequently, a static portfolio can experience sharp and significant drawdowns during volatile periods, as it lacks a mechanism to proactively mitigate emerging risks or capitalize on fleeting opportunities.2

This limitation gives rise to **Dynamic Asset Allocation (DAA)**. DAA is an active investment strategy that involves the frequent and flexible adjustment of a portfolio's asset weights based on an assessment of current market conditions, economic indicators, or the performance of specific securities.2 The fundamental goal is to actively manage the portfolio's risk-return profile—increasing exposure to assets with strong upward momentum during bullish phases and reducing exposure to declining assets or shifting toward safer havens during bearish phases.2 Unlike static allocation, which is bound by a predetermined mix, dynamic strategies often operate without a fixed target, granting the portfolio manager significant flexibility to navigate the investment landscape.4

The world of dynamic allocation is vast, encompassing a spectrum of strategies, each with a distinct philosophy for reacting to market changes. This chapter will explore three primary families of these rules-based, quantitative strategies:

1. **Insurance-Based Strategies (e.g., CPPI):** These strategies aim to provide a capital guarantee or a "floor" on the portfolio's value, creating a payoff profile similar to a protective option.
    
2. **Risk-Based Strategies (e.g., Target Volatility, Risk Parity):** These strategies shift the focus from allocating capital to allocating and managing _risk_. They adjust exposures to maintain a constant level of portfolio risk or to ensure each asset contributes equally to that risk.
    
3. **Trend-Based Strategies (e.g., Time-Series Momentum):** These strategies are built on the empirical observation that price trends persist. They systematically take long positions in assets with positive trends and short positions (or move to cash) in assets with negative trends.
    

For the quantitative analyst, moving from a conceptual understanding of these strategies to their rigorous, automated implementation is the critical challenge. This chapter provides the necessary mathematical foundations, practical Python implementations, and critical evaluation frameworks to master these powerful techniques.

### Section 1: The Toolkit for Dynamic Strategy Evaluation

Before delving into specific dynamic strategies, it is essential to establish a common toolkit for their analysis and evaluation. This involves understanding their fundamental rebalancing philosophies, the real-world costs they incur, and the quantitative metrics used to measure their performance.

#### Core Principles: Pro-cyclical vs. Counter-cyclical Rebalancing

At their core, dynamic rebalancing rules can be categorized into two opposing philosophies: counter-cyclical and pro-cyclical. This distinction dictates how a strategy behaves in response to market movements and is the primary determinant of its performance characteristics in different market regimes.

- **Counter-cyclical (Concave Strategies):** This philosophy embodies the principle of "selling winners and buying losers." The classic example is a **Constant-Mix** strategy, which seeks to maintain a fixed percentage allocation, such as the traditional 60/40 stock/bond portfolio.7 When stocks outperform bonds, their weight in the portfolio increases beyond the 60% target. A constant-mix strategy would then sell a portion of the outperforming stocks and use the proceeds to buy underperforming bonds, thereby returning to the 60/40 mix. This approach is inherently mean-reverting. It performs well in choppy, oscillating markets where trends frequently reverse, as it systematically profits from these reversals. However, it tends to underperform in strongly trending markets, as it continuously sells the asset that is leading the gains.7 The payoff profile of a concave strategy is curved downwards, reflecting diminishing returns as the underlying asset's value increases significantly.
    
- **Pro-cyclical (Convex Strategies):** This philosophy follows the principle of "buying winners and selling losers." It is the defining characteristic of momentum and portfolio insurance strategies.7 When an asset's price is rising, a pro-cyclical strategy increases its exposure to that asset to capitalize on continued upward movement. Conversely, as an asset's price falls, the strategy reduces its exposure to mitigate further losses. This approach thrives in markets with strong, persistent trends. Its primary weakness is its vulnerability to being "whipsawed" in volatile, directionless markets, where it may repeatedly buy at a local peak only to sell at a local trough, incurring losses from the reversals.7 The payoff profile of a convex strategy is curved upwards, reflecting accelerating returns as the underlying asset's value increases.
    

#### Implementation Costs: The Drag of High Turnover

A theoretical backtest is frictionless, but real-world implementation is not. Dynamic strategies, by their very nature of frequent rebalancing, incur higher transaction costs than their static counterparts.4 This cost is quantified by the

**portfolio turnover rate**, which measures how frequently assets within a portfolio are bought and sold.9

The portfolio turnover ratio is typically calculated over a one-year period using the following formula 9:

![[Pasted image 20250708131924.png]]

A high turnover ratio implies significant trading activity, which introduces several costs that can erode performance:

- **Direct Costs:** These are the most obvious costs and include brokerage commissions and bid-ask spreads paid on every transaction.4
    
- **Indirect Costs:** For larger funds, frequent trading can have a market impact, where the act of buying or selling an asset moves its price unfavorably. Furthermore, high turnover can lead to tax inefficiency, as frequent selling may realize short-term capital gains, which are often taxed at higher rates than long-term gains.11
    

Understanding and measuring turnover is therefore critical for assessing the practical viability of any dynamic strategy. A strategy that looks promising on paper may prove unprofitable after accounting for the drag of its implementation costs.

##### Python Implementation: Portfolio Turnover Calculation

The following Python function calculates the portfolio turnover based on a time series of portfolio weights. It measures the total change in weights from one period to the next, which serves as a proxy for the amount of trading required to rebalance the portfolio.



```Python
import numpy as np
import pandas as pd

def calculate_portfolio_turnover(weights_df):
    """
    Calculates the annualized portfolio turnover.

    Args:
        weights_df (pd.DataFrame): DataFrame with dates as index and asset weights as columns.

    Returns:
        float: Annualized portfolio turnover rate.
    """
    # Calculate the absolute difference in weights from one period to the next
    turnover = np.abs(weights_df.shift(1) - weights_df).sum(axis=1)
    
    # The sum of absolute changes is twice the one-way turnover, so we divide by 2
    turnover = turnover / 2.0
    
    # Assuming the weights frequency allows for a simple mean to be representative
    # For daily data, we annualize by the number of trading days
    # This is a simplified measure. A more precise measure would use asset values.
    # For this book, we define turnover as the average daily one-way trade volume.
    avg_daily_turnover = turnover.mean()
    
    # Annualize the turnover
    # Assuming daily rebalancing, there are approx. 252 trading days in a year.
    # The interpretation is the percentage of the portfolio traded in a year.
    annualized_turnover = avg_daily_turnover * 252
    
    return annualized_turnover

# --- Example Usage ---
# Create a sample weights DataFrame (e.g., from a backtest)
dates = pd.date_range(start='2023-01-01', periods=100)
weights_data = {
    'AssetA': np.linspace(0.5, 0.3, 100),
    'AssetB': np.linspace(0.5, 0.7, 100)
}
sample_weights = pd.DataFrame(weights_data, index=dates)

turnover_rate = calculate_portfolio_turnover(sample_weights)
print(f"Annualized Portfolio Turnover: {turnover_rate:.2%}")
```

#### Performance and Risk Metrics: A Quant's Scorecard

To move beyond a simple comparison of returns, a robust scorecard of performance and risk metrics is required. The choice of which metric to emphasize is not neutral; it implicitly reflects an investor's preferences and tolerance for different types of risk. An investor who is highly averse to losses, for example, will care more about metrics that penalize downside volatility and large drawdowns than one who is purely focused on maximizing return per unit of total volatility.

The essential metrics for evaluating dynamic strategies include:

- **Annualized Return and Volatility:** These are the foundational measures of performance and risk, representing the geometric average return per year and the standard deviation of those returns.12
    
- **Sharpe Ratio:** Developed by William F. Sharpe, this is the most common measure of risk-adjusted return. It quantifies the excess return earned per unit of total risk (volatility).13 A higher Sharpe ratio is generally better.
    
    ![[Pasted image 20250708131942.png]]
    
    where E is the expected portfolio return, Rf​ is the risk-free rate, and σp​ is the standard deviation of the portfolio's excess returns.
    
- **Sortino Ratio:** This ratio is a modification of the Sharpe ratio that differentiates between "good" (upside) and "bad" (downside) volatility. It measures the excess return per unit of _downside_ risk, using the standard deviation of only negative returns in the denominator. This makes it particularly useful for evaluating strategies designed to protect against losses, as it does not penalize them for upside volatility.13
    
    ![[Pasted image 20250708131950.png]]
    
    where σd​ is the standard deviation of negative asset returns (the downside deviation).
    
- **Maximum Drawdown (MDD) and Calmar Ratio:** MDD measures the largest peak-to-trough decline in the value of a portfolio, serving as a crucial indicator of tail risk.16 It answers the question: "What is the most an investor could have lost?". The Calmar Ratio then relates performance to this worst-case loss by dividing the annualized return by the absolute value of the MDD.
    

##### Python Implementation: Performance Metrics Calculator

The following Python class takes a series of portfolio returns and calculates the key performance metrics discussed above. This provides a reusable and standardized tool for the backtests in this chapter.



```Python
import numpy as np
import pandas as pd

class PerformanceAnalytics:
    def __init__(self, returns_series, risk_free_rate=0.02):
        """
        Initializes the PerformanceAnalytics class.

        Args:
            returns_series (pd.Series): A pandas Series of portfolio returns (daily, monthly, etc.).
            risk_free_rate (float): The annual risk-free rate.
        """
        if not isinstance(returns_series, pd.Series):
            raise TypeError("returns_series must be a pandas Series.")
        self.returns = returns_series
        self.risk_free_rate = risk_free_rate
        # Infer frequency for annualization factor
        self.freq = pd.infer_freq(self.returns.index)
        if self.freq in:
            self.annualization_factor = 252
        elif self.freq in:
            self.annualization_factor = 52
        elif self.freq in:
            self.annualization_factor = 12
        else:
            # Default to daily if frequency can't be inferred
            self.annualization_factor = 252
            print("Warning: Could not infer frequency. Assuming 252 periods per year.")

    def annualized_return(self):
        """Calculates the annualized geometric return."""
        total_return = (1 + self.returns).prod()
        num_years = len(self.returns) / self.annualization_factor
        return total_return ** (1 / num_years) - 1

    def annualized_volatility(self):
        """Calculates the annualized volatility."""
        return self.returns.std() * np.sqrt(self.annualization_factor)

    def sharpe_ratio(self):
        """Calculates the annualized Sharpe ratio."""
        ann_return = self.annualized_return()
        ann_volatility = self.annualized_volatility()
        if ann_volatility == 0:
            return np.nan
        return (ann_return - self.risk_free_rate) / ann_volatility

    def sortino_ratio(self):
        """Calculates the annualized Sortino ratio."""
        ann_return = self.annualized_return()
        downside_returns = self.returns[self.returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        downside_std = downside_returns.std() * np.sqrt(self.annualization_factor)
        if downside_std == 0:
            return np.nan
        return (ann_return - self.risk_free_rate) / downside_std

    def max_drawdown(self):
        """Calculates the maximum drawdown."""
        cumulative_returns = (1 + self.returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    def calmar_ratio(self):
        """Calculates the Calmar ratio."""
        ann_return = self.annualized_return()
        mdd = self.max_drawdown()
        if mdd == 0:
            return np.nan
        return ann_return / abs(mdd)

    def summary(self):
        """Returns a dictionary of all performance metrics."""
        stats = {
            "Annualized Return": self.annualized_return(),
            "Annualized Volatility": self.annualized_volatility(),
            "Sharpe Ratio": self.sharpe_ratio(),
            "Sortino Ratio": self.sortino_ratio(),
            "Max Drawdown": self.max_drawdown(),
            "Calmar Ratio": self.calmar_ratio()
        }
        return stats

# --- Example Usage ---
# Create a sample returns series
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=504, freq='B')
sample_returns = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)

# Analyze performance
analyzer = PerformanceAnalytics(sample_returns, risk_free_rate=0.02)
performance_summary = analyzer.summary()

# Print summary
for metric, value in performance_summary.items():
    print(f"{metric}: {value:.4f}")
```

### Section 2: Insurance-Based Strategies: Constant Proportion Portfolio Insurance (CPPI)

Constant Proportion Portfolio Insurance (CPPI) is a foundational dynamic strategy designed to achieve an asymmetric return profile: it provides protection against downside risk while allowing for participation in market upside.17 Its appeal lies in its simple, rules-based logic that creates a payoff structure similar to holding a call option on a risky asset, but does so synthetically by dynamically trading the underlying asset and a safe asset, without ever using options contracts.8 This makes it a classic example of a "convex" strategy.7

#### Conceptual Framework: A Synthetic Call Option

The implementation of a CPPI strategy revolves around three core components that dictate its behavior 8:

1. **The Floor:** This is the minimum value below which the investor does not want the portfolio to fall. It acts as the capital guarantee level and is the bedrock of the strategy's risk management. The floor is typically set as a percentage of the initial investment (e.g., 80% or 90%) and can be held constant or grow at the risk-free rate over time.
    
2. **The Cushion:** This is the engine of the strategy's risk-taking. It is calculated as the difference between the current total portfolio value (Vt​) and the current value of the floor (Ft​). The cushion, Ct​=Vt​−Ft​, represents the amount of capital the portfolio can afford to lose without breaching the floor.
    
3. **The Multiplier (m):** This is a constant, chosen by the investor, that is greater than 1. It determines the leverage applied to the cushion to calculate the total exposure to the risky asset. A higher multiplier implies a more aggressive strategy.
    

It is crucial to understand that CPPI is not a form of "insurance" in the traditional, contractual sense. There is no third-party insurer underwriting the floor value.19 The "protection" is purely algorithmic, generated by the dynamic trading rule. This distinction is paramount, as the guarantee is not absolute and is subject to specific market risks, which will be discussed later.

#### Mathematical Formulation and Dynamics

The core of the CPPI strategy is its simple yet powerful allocation rule. At any given time t, the dollar exposure to the risky asset (e.g., equities) is determined by the multiplier and the cushion 17:

![[Pasted image 20250708132018.png]]

The remainder of the portfolio is allocated to a safe asset (e.g., cash or Treasury bills):

![[Pasted image 20250708132027.png]]

The dynamic nature of the strategy is evident from this formula.

- **In a rising market:** As the portfolio value Vt​ increases, the cushion expands. The strategy dictates buying more of the risky asset, increasing the portfolio's equity exposure. This is a pro-cyclical "buy high" behavior.
    
- **In a falling market:** As Vt​ decreases, the cushion shrinks. The strategy dictates selling the risky asset to reduce exposure. This is a pro-cyclical "sell low" behavior.7
    

If the market falls continuously, the cushion will approach zero. At this point, the exposure to the risky asset also becomes zero, and the entire portfolio is allocated to the safe asset. The portfolio is then said to be "cashed out" or "deleveraged".17

A Numerical Example:

Consider a portfolio with an initial value (V0​) of $100. The investor sets a floor (F0​) at $80 and chooses a multiplier (m) of 3.

- **Initial Allocation:**
    
    - Cushion = V0​−F0​=100−80=20
        
    - Risky Asset Exposure = m×Cushion=3×20=60
        
    - Safe Asset Allocation = 100−60=40
        
    - The initial portfolio is 60% in the risky asset and 40% in the safe asset.
        
- **Scenario 1: Risky asset rises by 10%**
    
    - The risky portion grows to 60×1.10=66.
        
    - The safe portion remains at $40 (assuming a zero risk-free rate for simplicity).
        
    - New Portfolio Value V1​=66+40=106.
        
    - New Cushion = 106−80=26.
        
    - **Rebalancing:** New Risky Exposure = 3×26=78. The strategy sells 78−66=12 from the safe asset to buy more of the risky asset.
        
- **Scenario 2: Risky asset falls by 10% from the initial state**
    
    - The risky portion falls to 60×0.90=54.
        
    - New Portfolio Value V1​=54+40=94.
        
    - New Cushion = 94−80=14.
        
    - **Rebalancing:** New Risky Exposure = 3×14=42. The strategy sells 54−42=12 of the risky asset and moves it to the safe asset.
        

Under continuous time and assuming the risky asset follows a geometric Brownian motion, the dynamics of the cushion can be derived. The change in the cushion, dCt​, is driven by the leveraged exposure to the risky asset and the growth of the floor. This leads to the following stochastic differential equation for the cushion 21:

![[Pasted image 20250708132042.png]]

This equation shows that the cushion itself behaves like a leveraged asset, with its volatility amplified by the multiplier m.

#### Python Implementation: A Step-by-Step Backtest

The following Python class provides a complete framework for backtesting a CPPI strategy. It takes historical price data for a risky and a safe asset, along with the core CPPI parameters, and simulates the strategy's performance over time.



```Python
import pandas as pd
import numpy as np

class CPPIBacktester:
    def __init__(self, risky_asset_returns, safe_asset_returns, initial_capital=1000, floor_pct=0.8, multiplier=3, rebalance_freq='M'):
        """
        Initializes the CPPI backtester.

        Args:
            risky_asset_returns (pd.Series): Returns of the risky asset.
            safe_asset_returns (pd.Series): Returns of the safe asset.
            initial_capital (float): The starting value of the portfolio.
            floor_pct (float): The floor as a percentage of initial capital.
            multiplier (float): The CPPI multiplier (m).
            rebalance_freq (str): Rebalancing frequency ('D' for daily, 'W' for weekly, 'M' for monthly).
        """
        self.risky_returns = risky_asset_returns
        self.safe_returns = safe_asset_returns
        self.initial_capital = initial_capital
        self.floor_pct = floor_pct
        self.m = multiplier
        self.rebalance_freq = rebalance_freq

        # Align data
        self.data = pd.DataFrame({
            'risky': self.risky_returns,
            'safe': self.safe_returns
        }).dropna()
        
        self.results = {}

    def run_backtest(self):
        """Runs the CPPI backtest simulation."""
        # Setup results DataFrame
        n_steps = len(self.data)
        account_history = pd.Series(index=self.data.index, dtype=float)
        risky_weight_history = pd.Series(index=self.data.index, dtype=float)
        
        # Initial values
        account_value = self.initial_capital
        floor_value = self.initial_capital * self.floor_pct
        
        # Set initial allocation based on the first day's logic
        cushion = account_value - floor_value
        risky_exposure = max(0, self.m * cushion)
        risky_exposure = min(risky_exposure, account_value) # Cannot be more than 100% of portfolio
        safe_exposure = account_value - risky_exposure
        
        # Rebalancing dates
        rebalancer = self.data.resample(self.rebalance_freq).first().index

        for i, (date, row) in enumerate(self.data.iterrows()):
            # Store current state
            account_history.iloc[i] = account_value
            risky_weight_history.iloc[i] = risky_exposure / account_value
            
            # Rebalance if it's a rebalancing day
            if date in rebalancer:
                cushion = account_value - floor_value
                risky_exposure = max(0, self.m * cushion)
                risky_exposure = min(risky_exposure, account_value)
                safe_exposure = account_value - risky_exposure

            # Calculate portfolio value for the next period
            risky_return = row['risky']
            safe_return = row['safe']
            
            risky_exposure *= (1 + risky_return)
            safe_exposure *= (1 + safe_return)
            
            account_value = risky_exposure + safe_exposure
            
            # Update floor value (grows at the risk-free rate)
            floor_value *= (1 + safe_return)

        self.results['account_history'] = account_history
        self.results['risky_weight_history'] = risky_weight_history
        self.results['portfolio_returns'] = account_history.pct_change().dropna()
        
        return self.results

# --- Example Usage ---
# Generate sample data
np.random.seed(42)
dates = pd.date_range(start='2010-01-01', periods=120, freq='M')
risky_rets = pd.Series(np.random.normal(0.01, 0.05, len(dates)), index=dates)
safe_rets = pd.Series(np.random.normal(0.002, 0.001, len(dates)), index=dates)

# Initialize and run backtest
cppi_test = CPPIBacktester(risky_rets, safe_rets, multiplier=4, floor_pct=0.80)
results = cppi_test.run_backtest()

# Analyze performance
analyzer = PerformanceAnalytics(results['portfolio_returns'], risk_free_rate=0.02)
print("CPPI Performance Summary:")
print(analyzer.summary())

# Plot results
import matplotlib.pyplot as plt
results['account_history'].plot(title='CPPI Portfolio Value', figsize=(12, 6))
plt.ylabel('Portfolio Value')
plt.show()

results['risky_weight_history'].plot(title='Risky Asset Allocation (%)', figsize=(12, 6))
plt.ylabel('Weight in Risky Asset')
plt.show()
```

#### Analysis and Limitations: Gap Risk and Cash-Lock

While elegant in theory, the CPPI strategy is fraught with practical challenges and limitations that a quantitative analyst must understand and model.

- **Gap Risk:** This is the most significant flaw in the CPPI framework.22 The strategy's capital guarantee is predicated on the assumption of continuous trading or, at a minimum, the ability to rebalance before losses become too severe. In real markets, which trade discretely, a sudden and large price drop—a "gap"—can occur between rebalancing periods. If this drop is larger than the cushion can absorb, the portfolio's value can crash through the floor before the manager has a chance to sell the risky asset and de-risk.7 The magnitude of this risk is directly related to the multiplier,
    
    m. The maximum single-period loss the portfolio can withstand before the floor is breached is approximately 1/m. Therefore, a higher, more aggressive multiplier significantly increases the portfolio's vulnerability to gap risk.7
    
- **Cash-Lock (Deleveraging):** This occurs when a series of negative returns erodes the cushion to zero or near-zero. According to the CPPI rule, the allocation to the risky asset must then be reduced to zero. The portfolio becomes fully invested in the safe asset and is "locked" in cash.17 The critical issue is that once cashed-out, the strategy has no mechanism to re-enter the risky asset market. Even if the market subsequently stages a strong recovery, the CPPI portfolio cannot participate, and the investor is left to simply earn the risk-free rate until maturity. This makes the strategy particularly ineffective in volatile, choppy markets that lack a clear, persistent trend.
    
- **Transaction Costs:** The pro-cyclical "buy high, sell low" nature of CPPI can lead to high portfolio turnover, especially in oscillating markets. This frequent trading incurs transaction costs that act as a persistent drag on performance, potentially negating the strategy's benefits.7
    

### Section 3: Risk-Based Strategies: Target Volatility & Risk Parity

A distinct class of dynamic strategies shifts the primary focus of portfolio construction away from allocating capital to allocating and managing _risk_. This paradigm is built on a crucial empirical observation: while asset returns are notoriously difficult to predict, risk, often measured as volatility, exhibits predictable patterns like clustering and mean reversion.24 This relative predictability of risk makes it a more reliable anchor for a dynamic allocation strategy than forecasting returns.

#### Part A: The Target Volatility Strategy

The Target Volatility strategy, also known as a volatility-targeting or vol-targeting strategy, is a dynamic approach with a straightforward objective: to maintain a constant level of portfolio volatility over time.24 It achieves this by actively adjusting the portfolio's exposure to risky assets. When market volatility is forecasted to be high, the strategy reduces its exposure (de-leverages), moving capital into a safe asset like cash. Conversely, when market volatility is forecasted to be low, the strategy increases its exposure, often using leverage to do so.25 The primary goal is not necessarily to maximize returns, but rather to provide a "smoother ride" for the investor by stabilizing the portfolio's risk profile and mitigating the severity of drawdowns during turbulent periods.26

##### Volatility Forecasting with GARCH

A robust target volatility strategy depends on an accurate forecast of future volatility. While simple rolling historical standard deviation can be used, it is often slow to react to changing market regimes. A more sophisticated and widely used approach is the **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** model. GARCH models are exceptionally well-suited for financial time series because they are specifically designed to capture two of their most prominent empirical features 27:

1. **Volatility Clustering:** The tendency for periods of high volatility to be followed by more high volatility, and for periods of low volatility to be followed by more low volatility.
    
2. **Mean Reversion:** The tendency for volatility to revert to a long-run average over time.
    

The most common variant is the **GARCH(1,1)** model, which models the conditional variance at time t, denoted σt2​, as a function of three components 27:

![[Pasted image 20250708132105.png]]

Where:

- ω (omega) is a constant term, which helps anchor the long-run average variance.
    
- $ϵ^2_{t−1}$​ is the squared residual (shock or surprise) from the previous period. The coefficient α (alpha) governs the reaction to this shock. This is the **ARCH term**.
    
- $σ^2_{t−1}​$ is the forecasted variance from the previous period. The coefficient β (beta) governs the persistence of volatility. This is the **GARCH term**.
    

For the model to be stable, the condition α+β<1 must hold, which ensures that volatility is mean-reverting.31

##### Python Implementation: GARCH(1,1) Volatility Forecasting

The `arch` library in Python provides a powerful and convenient way to implement GARCH models. The following code demonstrates how to fit a GARCH(1,1) model to a series of asset returns and generate rolling one-step-ahead volatility forecasts.



```Python
# Ensure the arch library is installed: pip install arch
from arch import arch_model
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Fetch and prepare data
ticker = 'SPY'
start_date = '2010-01-01'
end_date = '2023-12-31'
spy_data = yf.download(ticker, start=start_date, end=end_date)
returns = spy_data['Adj Close'].pct_change().dropna() * 100 # GARCH works better with returns in %

# 2. Specify and fit the GARCH(1,1) model
# We use a constant mean and assume normal distribution for residuals
garch_model = arch_model(returns, vol='Garch', p=1, q=1,
                         mean='Constant', dist='Normal')
garch_result = garch_model.fit(disp='off')

# 3. Interpret the model summary
print(garch_result.summary())

# 4. Generate volatility forecasts
# The conditional_volatility attribute contains the fitted volatility
forecasted_vol = garch_result.conditional_volatility

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(returns.index, returns, label='SPY Daily Returns (%)', color='grey', alpha=0.6)
plt.plot(forecasted_vol.index, forecasted_vol, label='GARCH(1,1) Forecasted Volatility', color='red')
plt.title('SPY Returns and GARCH Volatility Forecast')
plt.legend()
plt.show()
```

##### Mathematical Formulation of Target Volatility

Once a forecast for the next period's volatility (σforecast,t​) is obtained, the allocation to the risky asset is determined by a simple scaling rule 26:

![[Pasted image 20250708132214.png]]

Here, σtarget​ is the desired annualized volatility level for the portfolio (e.g., 10% or 12%). The weight for the safe asset is simply 1−WeightRisky,t​. Note that if σforecast,t​<σtarget​, the weight will be greater than 1, implying the use of leverage. If σforecast,t​>σtarget​, the weight will be less than 1, implying a partial allocation to cash.26

##### Python Implementation: Target Volatility Backtest

The following code integrates the GARCH forecasting into a full backtest. It re-estimates the GARCH model and rebalances the portfolio on a rolling basis.



```Python
def run_target_vol_backtest(returns, target_vol, lookback_window=252):
    """
    Runs a backtest for a target volatility strategy using a rolling GARCH(1,1) model.

    Args:
        returns (pd.Series): Daily returns of the risky asset (in %).
        target_vol (float): The target annualized volatility (in %).
        lookback_window (int): The size of the rolling window for GARCH estimation.

    Returns:
        pd.Series: The returns of the target volatility strategy.
    """
    n_steps = len(returns)
    weights = pd.Series(index=returns.index, dtype=float)
    
    # Use tqdm for progress bar if available: from tqdm import tqdm
    # for i in tqdm(range(lookback_window, n_steps)):
    for i in range(lookback_window, n_steps):
        # Define the rolling window for estimation
        window_returns = returns.iloc[i - lookback_window : i]
        
        # Fit GARCH model on the window
        # In a real scenario, handle convergence errors
        try:
            model = arch_model(window_returns, vol='Garch', p=1, q=1).fit(disp='off')
            # Forecast one step ahead
            forecast = model.forecast(horizon=1)
            predicted_vol_daily = np.sqrt(forecast.variance.iloc[-1, 0])
        except:
            # Fallback to historical vol if GARCH fails
            predicted_vol_daily = window_returns.std()

        # Annualize the forecasted volatility
        predicted_vol_annual = predicted_vol_daily * np.sqrt(252)
        
        # Calculate the target weight
        # Cap leverage at a reasonable level, e.g., 200%
        weights.iloc[i] = min(2.0, target_vol / predicted_vol_annual)

    # Shift weights to avoid lookahead bias (trade on next day's open)
    strategy_returns = (weights.shift(1) * (returns / 100)).dropna()
    return strategy_returns

# --- Example Usage ---
# Using the SPY returns from the previous example
target_annual_vol = 12.0 # Target 12% annualized volatility
tv_strategy_returns = run_target_vol_backtest(returns, target_annual_vol)

# Analyze performance
tv_analyzer = PerformanceAnalytics(tv_strategy_returns, risk_free_rate=0.02)
print("\nTarget Volatility Strategy Performance:")
print(tv_analyzer.summary())

# Plot cumulative returns
(1 + tv_strategy_returns).cumprod().plot(title='Target Volatility Strategy Cumulative Returns', figsize=(12,6))
plt.show()
```

#### Part B: The Risk Parity Strategy

Risk Parity represents a more sophisticated evolution of risk-based allocation. Its genesis lies in the critique of traditional capital allocation, such as the 60/40 portfolio. In a 60/40 portfolio, while capital is split 60% to 40%, the _risk_ is not. Because equities are significantly more volatile than bonds, the equity portion can contribute over 90% of the total portfolio risk, making the portfolio far less diversified from a risk perspective than it appears.35

The core objective of the Risk Parity strategy is to construct a portfolio where each asset class contributes equally to the total portfolio risk.35 This forces true diversification by preventing any single asset class from dominating the portfolio's risk profile.

##### Defining and Formulating Risk Contribution

To equalize risk contributions, one must first define them mathematically. The total portfolio volatility (standard deviation) is given by ![[Pasted image 20250708132256.png]]​, where w is the vector of asset weights and Σ is the covariance matrix of asset returns. The marginal contribution of asset i to the portfolio's volatility is its weight multiplied by the partial derivative of the portfolio volatility with respect to that weight. The total **risk contribution (RC)** of asset i is:

![[Pasted image 20250708132232.png]]

The sum of the risk contributions of all assets equals the total portfolio volatility ($∑_{i=1}^N​RC_i​=σ_p$​). The goal of a Risk Parity portfolio is to find the weight vector w such that the risk contributions are equal for all assets:

![[Pasted image 20250708132307.png]]

This is equivalent to stating that for any two assets i and j:

$w_i​(Σw)_i​=w_ j​(Σw)_ j​$

This system of equations does not have a simple closed-form solution and must be solved using numerical optimization.39 A breakthrough by Spinu (2013) showed that this problem can be transformed into a convex optimization problem, which guarantees a unique and efficiently computable solution. The formulation is as follows 39:

![[Pasted image 20250708132419.png]]

Here, bi​ is the desired risk budget for asset i. For a standard Risk Parity portfolio, all budgets are equal, so bi​=1/N for all i. After solving for the optimal vector x, the final portfolio weights w are found by normalizing x so that they sum to 1: w=x/∑xi​.

A key feature of risk-based strategies like Risk Parity is that they do not use expected return forecasts as an input.41 This is a significant departure from Modern Portfolio Theory and is motivated by the fact that return forecasts are notoriously unstable and prone to large estimation errors, whereas covariance matrices (the sole input for risk parity) are more stable and predictable.42 By focusing on the more predictable component (risk) and ignoring the less predictable one (return), the strategy aims to build a more robust and structurally sound portfolio. This makes them "all-weather" strategies, designed to perform reasonably well across different economic environments, rather than tactical strategies trying to time market movements. While they may underperform a 100% equity portfolio in a strong bull market, their value is demonstrated over a full cycle through superior drawdown protection, which can lead to higher long-term risk-adjusted returns.26

##### Python Implementation: Risk Parity Optimization

Solving for Risk Parity weights can be done using general-purpose optimizers in `scipy`, but specialized libraries like `riskfolio-lib` make the process much more straightforward.



```Python
# Ensure riskfolio-lib is installed: pip install riskfolio-lib
import riskfolio as rp
import yfinance as yf
import pandas as pd

# 1. Fetch data for a multi-asset universe
assets = # Equities, Bonds, Gold, Commodities
start_date = '2010-01-01'
end_date = '2023-12-31'

asset_data = yf.download(assets, start=start_date, end=end_date)['Adj Close']
returns = asset_data.pct_change().dropna()

# 2. Use riskfolio-lib to calculate Risk Parity weights
# Create a Portfolio object
port = rp.Portfolio(returns=returns)

# Method to estimate asset returns and covariance
# We use historical estimates, but others are available
port.assets_stats(method_mu='hist', method_cov='hist')

# Optimization model
# model='Classic' means we use historical data
# rm='MV' stands for Mean-Variance, but for RP it just uses the covariance matrix
# obj='Sharpe' is a placeholder, as RP doesn't optimize for Sharpe
w_rp = port.rp_optimization(model='Classic', rm='MV', rf=0.02, b=None) # b=None means equal risk contribution

print("Risk Parity Weights:")
print(w_rp.T)

# 3. Visualize the risk contribution
ax = rp.plot_risk_con(w=w_rp,
                      cov=port.cov,
                      returns=port.returns,
                      rm='MV',
                      rf=0,
                      alpha=0.05,
                      color="tab:blue",
                      height=6,
                      width=10,
                      ax=None)
plt.show()
```

### Section 4: Trend-Based Strategies: Time-Series Momentum (TSMOM)

Trend-based strategies operate on one of the most persistent and well-documented anomalies in financial markets: momentum. This is the empirical tendency for assets that have performed well in the recent past to continue performing well, and for assets that have performed poorly to continue performing poorly.44

#### Conceptual Framework: Riding the Wave

It is critical to distinguish between the two main families of momentum strategies, as they have fundamentally different portfolio construction rules and risk characteristics 16:

1. **Cross-Sectional (or Relative) Momentum:** This is the classic academic version of momentum. It involves ranking a universe of assets (e.g., all stocks in the S&P 500) based on their past returns (e.g., over the last 12 months). The strategy then takes long positions in the top-performing decile or quintile ("winners") and short positions in the bottom-performing decile or quintile ("losers"). By construction, a cross-sectional momentum portfolio is always market-neutral.
    
2. **Time-Series Momentum (TSMOM) or Trend-Following:** This strategy, which is the focus of this section, looks at each asset in isolation. It evaluates an asset based on its _own_ past performance, not relative to other assets. If an asset's recent return is positive (i.e., it is in an uptrend), the strategy takes a long position. If its recent return is negative (a downtrend), the strategy takes a short position or moves to a risk-free asset like cash. This means a TSMOM portfolio can be net long the entire market (during a broad bull market), net short the entire market (during a broad crash), or somewhere in between. This ability to be dynamically long or short makes it a powerful tool for navigating different market regimes.48
    

The true power of TSMOM lies not just in its ability to generate positive returns on its own, but in its unique "crisis alpha" property. Most traditional assets and strategies suffer during prolonged market crises. A TSMOM strategy, however, is designed to adapt. As a sustained downtrend develops in an asset class like equities, the TSMOM signal will eventually flip from positive to negative, causing the strategy to exit its long position and enter a short position. This allows it to be profitable during the very periods when traditional long-only portfolios are experiencing their largest losses.49 This characteristic—a positive skew and low-to-negative correlation with other asset classes during tail events—makes TSMOM an exceptionally powerful diversifier.

#### Signal Construction and Mathematical Formulation

The core of any trend-following strategy is the rule used to define the trend. While various methods exist, such as moving average crossovers, a simple and robust approach is the **TSMOM rule**, based on the sign of an asset's cumulative return over a specified look-back period.51

The TSMOM signal for asset i at time t, using a look-back period of L months, is formally defined as:

![[Pasted image 20250708132439.png]]
The signal is +1 if the cumulative return over the past L periods is positive (uptrend) and -1 if it is negative (downtrend). A common look-back period is 12 months.

A naive implementation would allocate equal capital to each long or short position. However, a more sophisticated approach, borrowing from risk-based strategies, is to use **volatility scaling**. The size of the position taken in each asset is made inversely proportional to that asset's volatility. This ensures that each position contributes a similar amount of risk to the overall portfolio, preventing a single high-volatility asset from dominating the strategy's risk profile.49

The position size for asset i is calculated as:

![[Pasted image 20250708132448.png]]

where σtarget, asset​ is a target volatility for each individual position and σi,t​ is the forecasted volatility of asset i. The final weight of asset i in the portfolio before overall portfolio-level scaling is:

![[Pasted image 20250708132457.png]]

Finally, the entire portfolio of positions is often scaled to a portfolio-level volatility target (e.g., 10% or 15%) to maintain a consistent risk profile over time.49

#### Python Implementation: A Multi-Asset TSMOM Backtest

The following Python code implements a TSMOM strategy across a diversified set of liquid futures contracts, represented by ETFs. The backtest uses monthly data, a 12-month look-back for the signal, and a 36-month rolling window for volatility estimation.



```Python
import yfinance as yf
import pandas as pd
import numpy as np

def run_tsmom_backtest(returns_df, lookback_period=12, vol_lookback=36, portfolio_vol_target=0.15):
    """
    Runs a backtest for a Time-Series Momentum (TSMOM) strategy.

    Args:
        returns_df (pd.DataFrame): Monthly returns for multiple assets.
        lookback_period (int): Look-back period for momentum signal (in months).
        vol_lookback (int): Look-back period for volatility calculation (in months).
        portfolio_vol_target (float): Target annualized volatility for the portfolio.

    Returns:
        pd.Series: The returns of the TSMOM strategy.
    """
    signals = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
    
    # Calculate momentum signal (sign of past 12-month return)
    # Using rolling sum of log returns is more robust than product of simple returns
    log_returns = np.log(1 + returns_df)
    mom_signal = np.sign(log_returns.rolling(window=lookback_period).sum())
    
    # Calculate position sizes based on inverse volatility
    # Use a rolling window for historical volatility
    volatility = returns_df.rolling(window=vol_lookback).std() * np.sqrt(12) # Annualized
    
    # Set a target volatility for each asset position
    # This is a simplification; often a constant vol target is used for all assets
    asset_vol_target = 0.40 # 40% annualized vol target per position
    position_size = asset_vol_target / volatility
    
    # Combine signal and size to get weights
    # Cap individual asset weights to avoid extreme positions
    weights = (mom_signal * position_size).clip(-2, 2) # Example cap
    
    # Scale the overall portfolio to the target volatility
    # Ex-ante portfolio volatility forecast
    # For simplicity, we use an expanding covariance matrix
    portfolio_weights = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float).fillna(0)
    for t in range(vol_lookback, len(returns_df)):
        current_weights = weights.iloc[t]
        if current_weights.abs().sum() == 0:
            continue
            
        # Forecast covariance matrix using data up to t-1
        cov_matrix = returns_df.iloc[t-vol_lookback:t].cov() * 12
        
        # Calculate ex-ante portfolio volatility
        ex_ante_vol = np.sqrt(current_weights.T @ cov_matrix @ current_weights)
        
        # Scale weights to meet portfolio volatility target
        if ex_ante_vol > 0:
            scale_factor = portfolio_vol_target / ex_ante_vol
            portfolio_weights.iloc[t] = current_weights * scale_factor

    # Shift weights to avoid lookahead bias and calculate returns
    strategy_returns = (portfolio_weights.shift(1) * returns_df).sum(axis=1)
    
    return strategy_returns.dropna()

# --- Example Usage ---
# Fetch monthly data for a diversified set of assets
assets =
start_date = '2007-01-01'
end_date = '2023-12-31'
monthly_data = yf.download(assets, start=start_date, end=end_date, interval='1mo')['Adj Close']
monthly_returns = monthly_data.pct_change().dropna()

# Run TSMOM backtest
tsmom_returns = run_tsmom_backtest(monthly_returns)

# Analyze performance
tsmom_analyzer = PerformanceAnalytics(tsmom_returns, risk_free_rate=0.02)
print("\nTime-Series Momentum Strategy Performance:")
print(tsmom_analyzer.summary())

# Plot cumulative returns
(1 + tsmom_returns).cumprod().plot(title='TSMOM Strategy Cumulative Returns', figsize=(12,6))
plt.show()
```

### Section 5: Capstone Project I: A Comparative Backtest in Historical Crises

This project aims to provide a rigorous, empirical comparison of the dynamic strategies discussed in this chapter against a static benchmark. The focus is not only on overall performance but specifically on how these strategies behave during periods of significant market stress, which is often their primary value proposition.

#### Project Setup

- **Objective:** To backtest and compare the performance of a static 60/40 portfolio, CPPI, Target Volatility, Risk Parity, and Time-Series Momentum strategies, with a special focus on the 2008 Global Financial Crisis (GFC) and the 2020 COVID-19 crash.
    
- **Asset Universe:** We will use highly liquid ETFs to represent the core asset classes:
    
    - **SPY:** SPDR S&P 500 ETF Trust (US Equities)
        
    - **TLT:** iShares 20+ Year Treasury Bond ETF (Long-Term US Treasuries)
        
    - **GLD:** SPDR Gold Shares (Gold)
        
    - **DBC:** Invesco DB Commodity Index Tracking Fund (Commodities)
        
- **Time Period:** January 2007 - December 2023. This period is chosen specifically because it includes multiple distinct market regimes: the lead-up to the GFC, the crisis itself, a long-running bull market, the sharp COVID-19 crash and rapid recovery, and the inflationary period of 2022 where both stocks and bonds fell simultaneously.50
    
- **Strategies for Backtesting:**
    
    1. **Static 60/40:** 60% SPY, 40% TLT. Rebalanced monthly. This is our benchmark.
        
    2. **CPPI (m=2 & m=4):** On SPY as the risky asset, with TLT as the safe asset. Floor set at 80% of the initial capital. We will test two multipliers to see the impact of aggressiveness. Rebalanced monthly.
        
    3. **Target Volatility:** On SPY, targeting an annualized volatility of 12%. Rebalanced daily based on a rolling GARCH(1,1) model.
        
    4. **Risk Parity:** Across SPY, TLT, and GLD. Rebalanced monthly.
        
    5. **Time-Series Momentum:** Across all four assets (SPY, TLT, GLD, DBC). Rebalanced monthly based on a 12-month signal.
        

#### Python Implementation: Integrated Backtesting Script

The following script brings together the components developed throughout the chapter to run all five backtests and generate a comparative performance summary.



```Python
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import riskfolio as rp
import matplotlib.pyplot as plt

# --- Data Loading and Preparation ---
assets =
start_date = '2007-01-01'
end_date = '2023-12-31'

# Daily data for TV
daily_data = yf.download(assets, start=start_date, end=end_date)['Adj Close']
daily_returns = daily_data.pct_change().dropna()

# Monthly data for other strategies
monthly_data = daily_data.resample('M').last()
monthly_returns = monthly_data.pct_change().dropna()

# --- Performance Analytics Class (from Section 1) ---
class PerformanceAnalytics:
    def __init__(self, returns_series, risk_free_rate=0.02):
        self.returns = returns_series
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = 252 if pd.infer_freq(self.returns.index) in else 12

    def annualized_return(self):
        total_return = (1 + self.returns).prod()
        num_years = len(self.returns) / self.annualization_factor
        return total_return ** (1 / num_years) - 1 if num_years > 0 else 0

    def annualized_volatility(self):
        return self.returns.std() * np.sqrt(self.annualization_factor)

    def sharpe_ratio(self):
        ann_return = self.annualized_return()
        ann_volatility = self.annualized_volatility()
        return (ann_return - self.risk_free_rate) / ann_volatility if ann_volatility!= 0 else np.nan

    def sortino_ratio(self):
        ann_return = self.annualized_return()
        downside_returns = self.returns[self.returns < 0]
        if len(downside_returns) == 0: return np.inf
        downside_std = downside_returns.std() * np.sqrt(self.annualization_factor)
        return (ann_return - self.risk_free_rate) / downside_std if downside_std!= 0 else np.nan

    def max_drawdown(self):
        cumulative_returns = (1 + self.returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    def get_summary(self, crisis_periods):
        summary = {
            "Annualized Return": self.annualized_return(),
            "Annualized Volatility": self.annualized_volatility(),
            "Sharpe Ratio": self.sharpe_ratio(),
            "Sortino Ratio": self.sortino_ratio(),
            "Max Drawdown (Overall)": self.max_drawdown(),
        }
        for name, (start, end) in crisis_periods.items():
            crisis_returns = self.returns[start:end]
            summary = PerformanceAnalytics(crisis_returns).max_drawdown()
        return summary

# --- Strategy Implementations ---

# 1. Static 60/40
def run_static_60_40(returns):
    weights = {'SPY': 0.6, 'TLT': 0.4}
    portfolio_returns = (returns] * pd.Series(weights)).sum(axis=1)
    return portfolio_returns

# 2. CPPI (from Section 2, simplified for monthly)
def run_cppi(risky_rets, safe_rets, m, floor_pct):
    #... (Using the class from Section 2 would be cleaner, but for brevity here...)
    account_value = 1000
    floor_value = 1000 * floor_pct
    n_steps = len(risky_rets)
    account_history = pd.Series(index=risky_rets.index, dtype=float)
    
    for i in range(n_steps):
        cushion = account_value - floor_value
        risky_w = min(1, max(0, m * cushion / account_value))
        safe_w = 1 - risky_w
        
        portfolio_return = risky_w * risky_rets.iloc[i] + safe_w * safe_rets.iloc[i]
        account_value *= (1 + portfolio_return)
        floor_value *= (1 + safe_rets.iloc[i])
        account_history.iloc[i] = account_value
        
    return account_history.pct_change().dropna()

# 3. Target Volatility (from Section 3, simplified for demonstration)
def run_target_vol(returns, target_vol=0.12):
    # Simplified version using rolling vol instead of GARCH for speed
    vol = returns.rolling(window=60).std() * np.sqrt(252)
    weights = (target_vol / vol).shift(1).dropna()
    weights = weights.clip(0, 2) # Cap leverage
    strategy_returns = (weights * returns).dropna()
    return strategy_returns

# 4. Risk Parity
def run_risk_parity(returns):
    weights_history = pd.DataFrame(index=returns.index, columns=returns.columns)
    for i in range(60, len(returns)):
        window = returns.iloc[i-60:i]
        port = rp.Portfolio(returns=window)
        port.assets_stats(method_mu='hist', method_cov='hist')
        w = port.rp_optimization(model='Classic', rm='MV', rf=0, b=None)
        weights_history.iloc[i] = w.values.flatten()
    
    weights_history = weights_history.ffill().dropna()
    strategy_returns = (weights_history.shift(1) * returns).sum(axis=1).dropna()
    return strategy_returns

# 5. TSMOM (from Section 4)
def run_tsmom(returns, lookback=12):
    signals = np.sign(returns.rolling(window=lookback).mean())
    strategy_returns = (signals.shift(1) * returns).mean(axis=1).dropna()
    return strategy_returns

# --- Execution and Analysis ---
print("Running backtests... This may take a few minutes.")

# Define crisis periods
crisis_periods = {
    "GFC": ('2007-10-01', '2009-03-31'),
    "COVID": ('2020-02-01', '2020-03-31'),
    "2022 Bear": ('2022-01-01', '2022-12-31')
}

# Run strategies
returns_60_40 = run_static_60_40(monthly_returns)
returns_cppi2 = run_cppi(monthly_returns, monthly_returns, m=2, floor_pct=0.8)
returns_cppi4 = run_cppi(monthly_returns, monthly_returns, m=4, floor_pct=0.8)
returns_tv = run_target_vol(daily_returns)
returns_rp = run_risk_parity(monthly_returns])
returns_tsmom = run_tsmom(monthly_returns)

# Consolidate results
strategies = {
    "Static 60/40": returns_60_40,
    "CPPI (m=2)": returns_cppi2,
    "CPPI (m=4)": returns_cppi4,
    "Target Vol (12%)": returns_tv,
    "Risk Parity": returns_rp,
    "TSMOM": returns_tsmom
}

summary_table = pd.DataFrame()
for name, rets in strategies.items():
    analyzer = PerformanceAnalytics(rets)
    summary_table[name] = pd.Series(analyzer.get_summary(crisis_periods))

# --- Turnover Calculation (simplified) ---
# Note: A full turnover calculation requires dollar values of trades.
# This is a proxy based on weight changes.
def calculate_turnover_proxy(weights_df):
    return np.abs(weights_df.diff()).sum(axis=1).mean() * 12

# Placeholder for turnover calculation
summary_table.loc = "N/A" 

print("\n--- Capstone Project Performance Summary ---")
print(summary_table.round(4))

# Plot cumulative returns
plt.figure(figsize=(15, 8))
for name, rets in strategies.items():
    (1 + rets).cumprod().plot(label=name)
plt.title('Comparative Performance of Dynamic Strategies')
plt.ylabel('Cumulative Growth of $1')
plt.legend()
plt.grid(True)
plt.show()
```

#### Analysis: Questions & Responses

**Q1: Overall Performance: Which strategy yields the best risk-adjusted returns (Sharpe & Sortino Ratios) over the full period?**

_Response Guidance:_ An analysis of the full-period performance metrics reveals that the dynamic strategies generally outperform the static 60/40 benchmark on a risk-adjusted basis. While the benchmark might show a respectable annualized return, its volatility and drawdowns, particularly during crises, weigh down its Sharpe and Sortino ratios.

Strategies like Risk Parity and TSMOM are expected to exhibit superior Sharpe and Sortino ratios. Risk Parity achieves this by creating a more balanced portfolio from a risk perspective, preventing the high volatility of equities from dominating performance.35 TSMOM achieves its high risk-adjusted return through its unique ability to profit from both uptrends and downtrends, effectively hedging against prolonged bear markets and generating "crisis alpha".49 The Target Volatility strategy will likely show a lower volatility figure than the benchmark, leading to a smoother return path and a competitive Sharpe ratio, even if its absolute return is not the highest.26 CPPI's performance is highly path-dependent; a strong, persistent bull market would favor the higher multiplier, but the volatility of the period may have led to a cash-lock event, hampering its long-term return.

**Q2: Crisis Alpha: How did each strategy perform during the 2008 GFC and the COVID-19 crash? Analyze maximum drawdown, recovery period, and total returns during these specific sub-periods.**

_Response Guidance:_ The behavior of the strategies during crises highlights their core philosophies.

- **Static 60/40:** This benchmark will serve as a baseline for crisis losses. During the GFC, the equity portion suffered immense losses, only partially cushioned by the bond allocation, resulting in a severe maximum drawdown.55 The 2022 bear market was particularly punishing for the 60/40, as rising interest rates caused both stocks and bonds to fall in tandem, defeating its diversification premise.58
    
- **CPPI:** The performance of CPPI during a crash is binary. If the initial market drop is sharp and gaps down (a "gap risk" event), the portfolio could breach its floor.7 If the decline is more gradual, the strategy will systematically de-risk, selling equities as the cushion shrinks. This protects capital but can lead to a "cash-lock," where the portfolio is entirely in safe assets and unable to participate in the subsequent recovery.17 This would be particularly evident in the rapid V-shaped recovery after the COVID-19 crash.
    
- **Target Volatility:** This strategy is explicitly designed to de-risk during crises. As volatility spiked in both 2008 and 2020, the strategy would have automatically reduced its exposure to SPY, resulting in a significantly smaller drawdown compared to the 60/40 benchmark.25 Its primary benefit is cushioning the portfolio from the worst of the crash.
    
- **Risk Parity:** During the GFC, a typical risk parity portfolio would have likely outperformed the 60/40 due to its lower allocation to equities and higher allocation to bonds, which rallied. However, during the 2022 crisis, when stock-bond correlation turned positive, risk parity strategies suffered significantly because their core diversification mechanism failed.58 This highlights the strategy's sensitivity to correlation regimes.
    
- **Time-Series Momentum:** This strategy is expected to be the star performer during prolonged crises like the GFC. As the downtrend in equities became established, the TSMOM signal would have flipped from long to short, allowing the strategy to profit from the market's decline.50 During the sharp but brief COVID-19 crash, its performance would depend on the rebalancing frequency; a monthly model might have been too slow to react perfectly, but it would have still de-risked from equities as the trend turned negative.
    

**Table 1: Capstone Project Performance Summary (Illustrative)**

|Metric|Static 60/40|CPPI (m=2)|CPPI (m=4)|Target Vol (12%)|Risk Parity|TSMOM|
|---|---|---|---|---|---|---|
|Annualized Return|0.0750|0.0450|0.0510|0.0820|0.0780|0.1050|
|Annualized Volatility|0.1450|0.0800|0.1200|0.1200|0.0950|0.1500|
|Sharpe Ratio|0.3793|0.3125|0.2583|0.5167|0.6105|0.5667|
|Sortino Ratio|0.5500|0.4800|0.4000|0.7500|0.9500|0.9800|
|Max Drawdown (Overall)|-0.3500|-0.1500|-0.2500|-0.1800|-0.1600|-0.2000|
|Max Drawdown (GFC)|-0.3400|-0.1000|-0.2200|-0.1500|-0.1200|0.1500|
|Max Drawdown (COVID)|-0.1500|-0.0500|-0.1000|-0.0800|-0.1000|-0.0700|
|Portfolio Turnover|0.0500|0.8500|1.2500|2.5000|0.4500|0.6000|

_Note: The values in this table are illustrative and will be populated by the actual results of the Python backtest script._

**Q3: Turnover & Efficiency: Compare the portfolio turnover rates calculated for each dynamic strategy. Which strategy is most cost-intensive to implement?**

_Response Guidance:_ The turnover analysis will likely show that the Target Volatility strategy, especially if rebalanced daily, has the highest turnover rate. This is because its allocation is a continuous function of forecasted volatility, which changes daily. TSMOM and CPPI will also have substantial turnover, as their allocations can shift dramatically based on trend signals or cushion values. Risk Parity, rebalanced monthly, may have lower turnover than the others but significantly more than the static benchmark. This highlights a critical trade-off: the strategies that offer the most dynamic risk management (Target Volatility, TSMOM) are also the most expensive to implement, and these costs must be factored into any net performance evaluation.4

**Q4: Parameter Sensitivity: For the CPPI strategy, demonstrate how varying the multiplier (e.g., m=2 vs. m=4) affects its risk and return profile.**

_Response Guidance:_ The comparative backtest of CPPI with m=2 and m=4 will illustrate a fundamental risk-return trade-off. The m=4 strategy will show higher returns during sustained bull market periods because it takes on more leverage. However, it will also exhibit a much larger drawdown during crashes and a higher probability of a gap risk event or a permanent cash-lock. The m=2 strategy will be more conservative, with lower returns but better capital preservation. This demonstrates that there is no "optimal" multiplier in isolation; the choice depends entirely on the investor's risk tolerance and their view on future market volatility and the likelihood of price gaps.7 A higher multiplier is a bet on smoother, trending markets, while a lower multiplier is a more defensive posture.

### Section 6: Capstone Project II: Real-World Application for a Retirement Fund

#### Scenario

Consider the role of a portfolio manager for a Target-Date 2040 retirement fund. These funds are designed to follow a "glide path," where the asset allocation automatically becomes more conservative over time. As the target retirement date of 2040 approaches, the fund's strategic allocation systematically shifts capital away from growth assets like equities and into income-producing, lower-risk assets like bonds.

The critical vulnerability of this structure is **sequence-of-returns risk**. A severe market crash occurring just a few years before the target date (e.g., in 2038) can be devastating for the plan participants. Unlike a younger investor, they do not have a long time horizon to recover from the losses, and a significant drawdown at this stage can permanently impair their retirement capital.19 A purely static glide path, while disciplined, is passive and offers no protection against such a scenario.

#### Task

The task is to design a dynamic overlay for the fund's equity allocation. This overlay should not replace the strategic glide path but rather augment it with a rules-based, tactical risk management system specifically designed to mitigate sequence-of-returns risk.

#### Proposed Solution: A Hybrid Glide-Path with a Target Volatility Overlay

A robust solution is to create a hybrid strategy that combines the long-term discipline of the strategic glide path with the adaptive risk management of a Target Volatility overlay.

- **Strategic Allocation (The Glide Path):** The fund continues to follow its long-term strategic asset allocation plan. For example, in 2035, the strategic target might be 50% equities and 50% bonds. In 2036, this might shift to 48% equities and 52% bonds. This provides the long-term, foundational asset mix.
    
- **Tactical Overlay (Target Volatility):** The _actual_ exposure to the equity portion of the portfolio is managed dynamically. The strategic weight serves as the baseline, but the final exposure is scaled based on market volatility. For instance, if the strategic allocation to equities is 50% and the fund employs a 12% target volatility overlay on its equity sleeve, the allocation would be adjusted as follows:
    
    - In a **high-volatility environment** (e.g., forecasted equity volatility is 24%), the tactical weight would be 24%12%​=0.5. The final equity exposure would be 50%×0.5=25%. The remaining 25% would be held in cash or short-term treasuries.
        
    - In a **low-volatility environment** (e.g., forecasted equity volatility is 8%), the tactical weight would be 8%12%​=1.5. The final equity exposure would be 50%×1.5=75%, using modest leverage to achieve the target risk level.
        

#### Implementation Sketch & Justification

A Python implementation would involve a function that calculates the final allocation at each rebalancing date.



```Python
def calculate_hybrid_allocation(strategic_equity_weight, forecasted_equity_vol, target_vol=0.12):
    """
    Calculates the final equity allocation for the hybrid strategy.

    Args:
        strategic_equity_weight (float): The current equity weight from the glide path.
        forecasted_equity_vol (float): The GARCH forecast for annualized equity volatility.
        target_vol (float): The target volatility for the equity sleeve.

    Returns:
        float: The final, dynamically adjusted weight for the equity allocation.
    """
    # Calculate the tactical scaling factor based on target volatility
    tactical_scale_factor = target_vol / forecasted_equity_vol
    
    # Apply constraints (e.g., no leverage or max leverage of 1.5x)
    tactical_scale_factor = min(1.5, max(0.2, tactical_scale_factor))
    
    # Calculate the final, dynamic equity weight
    dynamic_equity_weight = strategic_equity_weight * tactical_scale_factor
    
    return dynamic_equity_weight
```

**Justification:** This hybrid approach is superior to a purely static glide path for an investor nearing retirement for several key reasons. The primary problem facing this investor is not long-term growth, but the preservation of capital against a sudden, sharp drawdown from which they cannot recover. The Target Volatility overlay directly addresses this sequence-of-returns risk.60

Market crises are almost always accompanied by a dramatic spike in volatility.57 The Target Volatility mechanism is designed to react systematically to this exact signal. As volatility rises, indicating increasing market stress and a higher probability of large price swings, the strategy automatically reduces the portfolio's exposure to the riskiest asset class—equities.26 This action serves to cushion the portfolio from the most severe losses during a crash.

While this approach might mean forgoing some potential gains if a bull market accelerates just before retirement, that is a trade-off a risk-averse near-retiree should be willing to make. The primary objective has shifted from wealth accumulation to wealth preservation. This hybrid model elegantly combines the discipline and long-term perspective of a strategic glide path with the nimble, rules-based, and adaptive nature of a tactical risk management system. It makes the portfolio responsive to the immediate market environment at the most critical point in an investor's lifecycle, providing a much-needed layer of protection that a static model simply cannot offer.

## References
**

1. Static or dynamic asset allocation: What suits you best - Value Research, acessado em julho 8, 2025, [https://www.valueresearchonline.com/stories/224569/static-vs-dynamic-allocation/](https://www.valueresearchonline.com/stories/224569/static-vs-dynamic-allocation/)
    
2. Dynamic vs. Static Asset Allocation in Balanced Advantage Funds | Bajaj Finserv AMC, acessado em julho 8, 2025, [https://www.bajajamc.com/knowledge-centre/dynamic-vs-static-allocation-what-makes-balanced-advantage-funds-different](https://www.bajajamc.com/knowledge-centre/dynamic-vs-static-allocation-what-makes-balanced-advantage-funds-different)
    
3. Static vs Dynamic Asset Allocation: Which is Better? - Holistic Investment Planners, acessado em julho 8, 2025, [https://www.holisticinvestment.in/static-vs-dynamic-asset-allocation-investment-balance/](https://www.holisticinvestment.in/static-vs-dynamic-asset-allocation-investment-balance/)
    
4. Dynamic Asset Allocation - Overview, Advantages, Disadvantages, acessado em julho 8, 2025, [https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/dynamic-asset-allocation/](https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/dynamic-asset-allocation/)
    
5. Dynamic Asset Allocation: What it is, How it Works - Investopedia, acessado em julho 8, 2025, [https://www.investopedia.com/terms/d/dynamic-asset-allocation.asp](https://www.investopedia.com/terms/d/dynamic-asset-allocation.asp)
    
6. Dynamic Asset Allocation | Principles, Advantages, & Disadvantages - Study Finance, acessado em julho 8, 2025, [https://studyfinance.com/dynamic-asset-allocation/](https://studyfinance.com/dynamic-asset-allocation/)
    
7. Dynamic Strategies for Asset Allocation - CAIA, acessado em julho 8, 2025, [https://caia.org/sites/default/files/dynamic_strategies_for_asset_allocation.pdf](https://caia.org/sites/default/files/dynamic_strategies_for_asset_allocation.pdf)
    
8. Constant Proportion Portfolio Insurance (CPPI): Definition, Uses - Investopedia, acessado em julho 8, 2025, [https://www.investopedia.com/terms/c/cppi.asp](https://www.investopedia.com/terms/c/cppi.asp)
    
9. Portfolio Turnover Formula, Meaning, and Taxes - Investopedia, acessado em julho 8, 2025, [https://www.investopedia.com/terms/p/portfolioturnover.asp](https://www.investopedia.com/terms/p/portfolioturnover.asp)
    
10. Portfolio Turnover Ratio - Overview, Formula, How To Interpret - Corporate Finance Institute, acessado em julho 8, 2025, [https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/portfolio-turnover-ratio/](https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/portfolio-turnover-ratio/)
    
11. How Dynamic Asset Allocation Works - SmartAsset, acessado em julho 8, 2025, [https://smartasset.com/financial-advisor/dynamic-asset-allocation](https://smartasset.com/financial-advisor/dynamic-asset-allocation)
    
12. Volatility And Measures Of Risk-Adjusted Return With Python - Interactive Brokers, acessado em julho 8, 2025, [https://www.interactivebrokers.com/campus/ibkr-quant-news/volatility-and-measures-of-risk-adjusted-return-with-python/](https://www.interactivebrokers.com/campus/ibkr-quant-news/volatility-and-measures-of-risk-adjusted-return-with-python/)
    
13. Sharpe ratio and Sortino ratio | Python, acessado em julho 8, 2025, [https://campus.datacamp.com/courses/financial-trading-in-python/performance-evaluation-4?ex=8](https://campus.datacamp.com/courses/financial-trading-in-python/performance-evaluation-4?ex=8)
    
14. Sharpe Ratio Explained: Formula, Calculation in Excel & Python, and Examples, acessado em julho 8, 2025, [https://blog.quantinsti.com/sharpe-ratio-applications-algorithmic-trading/](https://blog.quantinsti.com/sharpe-ratio-applications-algorithmic-trading/)
    
15. Sortino ratio | Python, acessado em julho 8, 2025, [https://campus.datacamp.com/courses/introduction-to-portfolio-analysis-in-python/risk-and-return?ex=13](https://campus.datacamp.com/courses/introduction-to-portfolio-analysis-in-python/risk-and-return?ex=13)
    
16. (PDF) Trend Following and Momentum Strategies for Global REITs, acessado em julho 8, 2025, [https://www.researchgate.net/publication/342310198_Trend_Following_and_Momentum_Strategies_for_Global_REITs](https://www.researchgate.net/publication/342310198_Trend_Following_and_Momentum_Strategies_for_Global_REITs)
    
17. Constant proportion portfolio insurance - Wikipedia, acessado em julho 8, 2025, [https://en.wikipedia.org/wiki/Constant_proportion_portfolio_insurance](https://en.wikipedia.org/wiki/Constant_proportion_portfolio_insurance)
    
18. What Is Constant Proportion Portfolio Insurance (CPPI)? - SmartAsset, acessado em julho 8, 2025, [https://smartasset.com/investing/constant-proportion-portfolio-insurance](https://smartasset.com/investing/constant-proportion-portfolio-insurance)
    
19. CPPI Explained: Protect Your Portfolio Without Sacrificing Upside - Addis Hill, acessado em julho 8, 2025, [https://addishill.com/what-is-constant-proportion-portfolio-insurance/](https://addishill.com/what-is-constant-proportion-portfolio-insurance/)
    
20. Mastering CPPI: The Essential Guide in Math Econ - Number Analytics, acessado em julho 8, 2025, [https://www.numberanalytics.com/blog/ultimate-cppi-guide-math-econ](https://www.numberanalytics.com/blog/ultimate-cppi-guide-math-econ)
    
21. Constant Proportion Portfolio Insurance in presence of Jumps in Asset Prices, acessado em julho 8, 2025, [http://www.planchet.net/EXT/ISFA/1226.nsf/0/9034828ca6162f07c12577ae00246cb3/$FILE/cppi%20in%20presence%20of%20jumps%20in%20asset%20price.pdf](http://www.planchet.net/EXT/ISFA/1226.nsf/0/9034828ca6162f07c12577ae00246cb3/$FILE/cppi%20in%20presence%20of%20jumps%20in%20asset%20price.pdf)
    
22. Dynamic asset allocation - Wikipedia, acessado em julho 8, 2025, [https://en.wikipedia.org/wiki/Dynamic_asset_allocation](https://en.wikipedia.org/wiki/Dynamic_asset_allocation)
    
23. Hub article: 'CPPI structures return' - FVC, acessado em julho 8, 2025, [https://www.futurevc.co.uk/hubdisplay.cfm?contententryid=156](https://www.futurevc.co.uk/hubdisplay.cfm?contententryid=156)
    
24. An Introduction to Volatility Targeting - QuantPedia, acessado em julho 8, 2025, [https://quantpedia.com/an-introduction-to-volatility-targeting/](https://quantpedia.com/an-introduction-to-volatility-targeting/)
    
25. The Impact of Volatility Targeting | Man Group, acessado em julho 8, 2025, [https://www.man.com/insights/the-impact-of-volatility-targeting](https://www.man.com/insights/the-impact-of-volatility-targeting)
    
26. Harnessing Volatility Targeting in Multi-Asset Portfolios - Research Affiliates, acessado em julho 8, 2025, [https://www.researchaffiliates.com/content/dam/ra/publications/pdf/1014-harnessing-volatility-targeting.pdf](https://www.researchaffiliates.com/content/dam/ra/publications/pdf/1014-harnessing-volatility-targeting.pdf)
    
27. GARCH vs. GJR-GARCH Models in Python for Volatility Forecasting - QuantInsti Blog, acessado em julho 8, 2025, [https://blog.quantinsti.com/garch-gjr-garch-volatility-forecasting-python/](https://blog.quantinsti.com/garch-gjr-garch-volatility-forecasting-python/)
    
28. An Introduction to the Use of ARCH/GARCH models in Applied Econometrics - NYU Stern, acessado em julho 8, 2025, [https://www.stern.nyu.edu/rengle/GARCH101.PDF](https://www.stern.nyu.edu/rengle/GARCH101.PDF)
    
29. Mastering ARCH and GARCH Models in Modern Econ, acessado em julho 8, 2025, [https://www.numberanalytics.com/blog/mastering-arch-garch-models-modern-econ](https://www.numberanalytics.com/blog/mastering-arch-garch-models-modern-econ)
    
30. 4. Machine Learning-Based Volatility Prediction - Machine Learning for Financial Risk Management with Python [Book] - O'Reilly Media, acessado em julho 8, 2025, [https://www.oreilly.com/library/view/machine-learning-for/9781492085249/ch04.html](https://www.oreilly.com/library/view/machine-learning-for/9781492085249/ch04.html)
    
31. GARCH in Econometrics: A Quick, Clear Guide Today - Number Analytics, acessado em julho 8, 2025, [https://www.numberanalytics.com/blog/garch-econometrics-essential-guide](https://www.numberanalytics.com/blog/garch-econometrics-essential-guide)
    
32. MSCI World 12% Volatility Target Select Index Methodology, acessado em julho 8, 2025, [https://www.msci.com/documents/10199/2a6a8d3b-ca0d-af9c-02dc-2914b2b72608](https://www.msci.com/documents/10199/2a6a8d3b-ca0d-af9c-02dc-2914b2b72608)
    
33. Closed-End Formula for options linked to Target Volatility Strategies, acessado em julho 8, 2025, [https://arxiv.org/pdf/1902.08821](https://arxiv.org/pdf/1902.08821)
    
34. Demystifying Volatility-Controlled Indices | S&P Global, acessado em julho 8, 2025, [https://www.spglobal.com/spdji/en/documents/education/education-demystifying-volatility-controlled-indices.pdf](https://www.spglobal.com/spdji/en/documents/education/education-demystifying-volatility-controlled-indices.pdf)
    
35. Risk parity - Wikipedia, acessado em julho 8, 2025, [https://en.wikipedia.org/wiki/Risk_parity](https://en.wikipedia.org/wiki/Risk_parity)
    
36. Understanding Risk Parity | CME Group, acessado em julho 8, 2025, [https://www.cmegroup.com/education/files/understanding-risk-parity-2013-06.pdf](https://www.cmegroup.com/education/files/understanding-risk-parity-2013-06.pdf)
    
37. Risk Parity Portfolios: - PanAgora Asset Management, acessado em julho 8, 2025, [https://www.panagora.com/assets/PanAgora-Risk-Parity-Portfolios-Efficient-Portfolios-Through-True-Diversification.pdf](https://www.panagora.com/assets/PanAgora-Risk-Parity-Portfolios-Efficient-Portfolios-Through-True-Diversification.pdf)
    
38. What is risk parity? - SEI, acessado em julho 8, 2025, [https://www.seic.com/sites/default/files/2023-10/SEI-InvestmentFundamental-Risk-Parity.pdf](https://www.seic.com/sites/default/files/2023-10/SEI-InvestmentFundamental-Risk-Parity.pdf)
    
39. Risk Parity Portfolio - The Hong Kong University of Science and ..., acessado em julho 8, 2025, [https://palomar.home.ece.ust.hk/ELEC5470_lectures/slides_risk_parity_portfolio.pdf](https://palomar.home.ece.ust.hk/ELEC5470_lectures/slides_risk_parity_portfolio.pdf)
    
40. Chapter 8 Risk parity portfolio | Portfolio Construction - Bookdown, acessado em julho 8, 2025, [https://bookdown.org/shenjian0824/portr/risk-parity-portfolio.html](https://bookdown.org/shenjian0824/portr/risk-parity-portfolio.html)
    
41. Risk Budgeting Portfolio Optimization with Deep Reinforcement Learning, acessado em julho 8, 2025, [https://www.pm-research.com/content/iijjfds/5/4/86](https://www.pm-research.com/content/iijjfds/5/4/86)
    
42. Optimise portfolio with target volatility · Issue #116 · robertmartin8/PyPortfolioOpt - GitHub, acessado em julho 8, 2025, [https://github.com/robertmartin8/PyPortfolioOpt/issues/116](https://github.com/robertmartin8/PyPortfolioOpt/issues/116)
    
43. Portfolio Optimization and the Efficient Frontier - OMSCS Notes, acessado em julho 8, 2025, [https://www.omscs-notes.com/machine-learning-trading/portfolio-optimization-efficient-frontier/](https://www.omscs-notes.com/machine-learning-trading/portfolio-optimization-efficient-frontier/)
    
44. Momentum Trading: Types, Strategies, and More - QuantInsti Blog, acessado em julho 8, 2025, [https://blog.quantinsti.com/momentum-trading-strategies/](https://blog.quantinsti.com/momentum-trading-strategies/)
    
45. Introduction to Momentum Trading - Investopedia, acessado em julho 8, 2025, [https://www.investopedia.com/trading/introduction-to-momentum-trading/](https://www.investopedia.com/trading/introduction-to-momentum-trading/)
    
46. What's the Difference Between Momentum and Trend Following? - Venn by Two Sigma, acessado em julho 8, 2025, [https://www.venn.twosigma.com/insights/momentum-and-trend-following](https://www.venn.twosigma.com/insights/momentum-and-trend-following)
    
47. www.efmaefm.org, acessado em julho 8, 2025, [https://www.efmaefm.org/0efmameetings/efma%20annual%20meetings/2013-Reading/papers/EFMA2013_0130_fullpaper.pdf](https://www.efmaefm.org/0efmameetings/efma%20annual%20meetings/2013-Reading/papers/EFMA2013_0130_fullpaper.pdf)
    
48. Asset Class Trend-Following - Quantpedia, acessado em julho 8, 2025, [https://quantpedia.com/strategies/asset-class-trend-following](https://quantpedia.com/strategies/asset-class-trend-following)
    
49. Time Series Momentum (aka Trend-Following): A Good Time for a Refresh - - Alpha Architect, acessado em julho 8, 2025, [https://alphaarchitect.com/time-series-momentum-aka-trend-following-the-historical-evidence/](https://alphaarchitect.com/time-series-momentum-aka-trend-following-the-historical-evidence/)
    
50. The Best of Strategies for the Worst of Times: Can Portfolios Be Crisis Proofed? - Duke People, acessado em julho 8, 2025, [https://people.duke.edu/~charvey/Research/Published_Papers/P140_The_best_of.pdf](https://people.duke.edu/~charvey/Research/Published_Papers/P140_The_best_of.pdf)
    
51. Time Series Momentum Effect - Quantpedia, acessado em julho 8, 2025, [https://quantpedia.com/strategies/time-series-momentum-effect](https://quantpedia.com/strategies/time-series-momentum-effect)
    
52. Trend Filtering Methods for Momentum Strategies - Roncalli, Thierry, acessado em julho 8, 2025, [http://www.thierry-roncalli.com/download/lwp-tf.pdf](http://www.thierry-roncalli.com/download/lwp-tf.pdf)
    
53. (PDF) A New Approach To Trend-Following - ResearchGate, acessado em julho 8, 2025, [https://www.researchgate.net/publication/373267634_A_New_Approach_To_Trend-Following](https://www.researchgate.net/publication/373267634_A_New_Approach_To_Trend-Following)
    
54. Economic Commentaries: Hedge funds and the financial crisis of 2008, acessado em julho 8, 2025, [http://archive.riksbank.se/upload/Dokument_riksbank/Kat_publicerat/Ekonomiska%20kommentarer/2009/ek_kom_no3_eng.pdf](http://archive.riksbank.se/upload/Dokument_riksbank/Kat_publicerat/Ekonomiska%20kommentarer/2009/ek_kom_no3_eng.pdf)
    
55. The Impact of Diversification on Portfolio Performance: A Case Study of The 2008 Financial Crisis - Cowrywise, acessado em julho 8, 2025, [https://cowrywise.com/blog/case-study-of-2008-financial-crisis/](https://cowrywise.com/blog/case-study-of-2008-financial-crisis/)
    
56. The joint dynamics of investor beliefs and trading during the COVID-19 crash | PNAS, acessado em julho 8, 2025, [https://www.pnas.org/doi/10.1073/pnas.2010316118](https://www.pnas.org/doi/10.1073/pnas.2010316118)
    
57. 'Safe Assets' during COVID-19: A Portfolio Management Perspective - MDPI, acessado em julho 8, 2025, [https://www.mdpi.com/2813-2432/2/1/2](https://www.mdpi.com/2813-2432/2/1/2)
    
58. Risk Parity Not Performing? Blame the Weather. | Portfolio for the Future - CAIA Association, acessado em julho 8, 2025, [https://caia.org/blog/2024/01/02/risk-parity-not-performing-blame-weather](https://caia.org/blog/2024/01/02/risk-parity-not-performing-blame-weather)
    
59. The End of The Golden Era for Risk Parity - The Hedge Fund Journal, acessado em julho 8, 2025, [https://thehedgefundjournal.com/the-end-of-the-golden-era-for-risk-parity/](https://thehedgefundjournal.com/the-end-of-the-golden-era-for-risk-parity/)
    
60. Optimizing Pension Outcomes Using Target Volatility Investment Concept - Scholars @ Bentley, acessado em julho 8, 2025, [https://scholars.bentley.edu/cgi/viewcontent.cgi?article=1000&context=etd_2022](https://scholars.bentley.edu/cgi/viewcontent.cgi?article=1000&context=etd_2022)
    
61. The Fed - Banks' Backtesting Exceptions during the COVID-19 Crash: Causes and Consequences - Federal Reserve Board, acessado em julho 8, 2025, [https://www.federalreserve.gov/econres/notes/feds-notes/banks-backtesting-exceptions-during-the-covid-19-crash-causes-and-consequences-20210708.html](https://www.federalreserve.gov/econres/notes/feds-notes/banks-backtesting-exceptions-during-the-covid-19-crash-causes-and-consequences-20210708.html)
    

Volatility Targeting in Trading and Portfolio Construction - DayTrading.com, acessado em julho 8, 2025, [https://www.daytrading.com/volatility-targeting](https://www.daytrading.com/volatility-targeting)**