### Introduction: The Cost of Action

The classical framework of Markowitz portfolio optimization provides a powerful theoretical foundation for asset allocation. However, its direct application often yields strategies that are impractical and, ultimately, unprofitable. The primary reason for this discrepancy is the model's standard assumption of a frictionless market—a world where assets can be bought and sold in any quantity, at any time, without cost.1 In reality, every transaction incurs a cost, and ignoring these costs can lead to portfolio strategies that generate excessive trading, or "turnover," which systematically erodes investment returns.3

The core task of a portfolio manager is rarely to construct a portfolio from scratch. Instead, it is a continuous process of rebalancing an existing portfolio, `w_old`, to a new, more desirable allocation, `w_new`, in response to changing market conditions and forecasts.5 A cost-agnostic optimization model will recommend a new "optimal" portfolio every time its inputs—expected returns (

`μ`) and the covariance matrix (`Σ`)—change, no matter how slightly. This leads to a phenomenon known as "portfolio churn," where the strategy dictates frequent, small trades that chase statistical noise rather than meaningful economic signals.

The incorporation of transaction costs fundamentally alters the optimization problem and its solution. It introduces a form of profitable inertia, leading to the critical concept of a **no-trade region**.6 This is a region around the theoretical optimal portfolio where the marginal benefit of rebalancing is smaller than the cost required to execute the trades. Within this region, the optimal action is to do nothing. A trade is only justified when the current portfolio has drifted far enough from the ideal allocation that the expected improvement in the portfolio's risk/return profile outweighs the associated trading frictions.

Furthermore, modeling transaction costs provides a benefit that extends beyond simple accounting. There is a profound mathematical equivalence between penalizing portfolio turnover and making the optimization process more robust to estimation errors in its inputs.3 An optimizer with no concept of cost will aggressively reallocate assets based on a new

`μ` vector, even if the change is statistically insignificant. By introducing a cost penalty for trading, the model implicitly demands a higher degree of confidence before it acts. A trade will only be executed if the perceived advantage of the new allocation is substantial enough to overcome the cost hurdle. In this sense, transaction cost modeling serves a dual purpose: it accounts for real-world expenses while simultaneously acting as a powerful and economically intuitive form of **regularization**, preventing the portfolio from overfitting to noisy data and leading to more stable, practical, and ultimately more successful investment strategies.

## 1. A Taxonomy of Trading Frictions

To build robust portfolio models, it is essential to first establish a clear vocabulary for the different types of costs that arise during trading. These costs can be categorized along two primary axes: their visibility (explicit vs. implicit) and their relationship to trade size (fixed vs. variable).

### 1.1 Explicit vs. Implicit Costs

This classification distinguishes between direct, observable payments and indirect, opportunity-style costs.9

- **Explicit Costs** are the direct, out-of-pocket expenses for which a receipt is typically issued. They are transparent, easy to measure, and directly recorded in accounting statements.9 The most common examples in portfolio management are:
    
    - **Brokerage Commissions:** Fees paid to a broker for executing a trade.1
        
    - **Transfer Fees & Taxes:** Regulatory or exchange fees associated with the transaction.
        
- **Implicit Costs** are indirect and often unreceipted costs that arise from the mechanics of the trading process itself. They represent the opportunity cost of not being able to transact at a single, ideal price.11 Key sources of implicit costs include:
    
    - **Bid-Ask Spread:** This is the difference between the highest price a buyer is willing to pay for an asset (the bid) and the lowest price a seller is willing to accept (the ask). When executing a market order, an investor buys at the higher ask price and sells at the lower bid price, with the spread representing a cost captured by the market maker.11
        
    - **Market Impact (or Price Impact):** This is the adverse price movement caused by the trade itself. A large buy order consumes available liquidity at the best ask price and moves up to higher-priced offers, pushing the average execution price upward. Similarly, a large sell order pushes the price downward. This effect is particularly significant for institutional traders moving large blocks of assets.12
        
    - **Delay Costs (or Opportunity Costs):** This cost arises from the failure to execute a trade at the desired moment. If a buy order is delayed and the asset's price rises in the interim, that price difference is a delay cost. It also includes the unrealized profit from a trade that was considered but not executed.11
        

### 1.2 Fixed vs. Variable Costs

This classification, based on the relationship between the cost and the size of the transaction, is the most critical for mathematical modeling as it directly influences the complexity and solvability of the optimization problem.1

- **Fixed Costs** are independent of the transaction volume. A classic example is a flat brokerage commission, such as $5 per trade, which is incurred whether 10 shares or 10,000 shares are traded.1
    
- **Variable Costs** are dependent on the transaction volume. The bid-ask spread, for instance, is typically proportional to the number of shares traded. Market impact is also a variable cost, though its relationship with trade size is often non-linear.1
    

The distinction between these cost structures is paramount because it determines the mathematical properties of the optimization problem. Models incorporating only variable costs that are convex functions of trade size (like proportional or quadratic costs) can typically be solved efficiently using convex optimization techniques like Quadratic Programming (QP). However, the introduction of fixed costs creates a non-convex objective function, transforming the problem into a much harder combinatorial challenge that often requires specialized mixed-integer programming solvers.17

## 2. Proportional Transaction Costs: The Linear Model

The most fundamental and widely used transaction cost model assumes that costs are directly proportional to the value of the assets traded. This is a reasonable approximation for costs stemming from the bid-ask spread and percentage-based commissions.3

### 2.1 Mathematical Formulation

Let `w_old` be the vector of current portfolio weights and `w_new` be the vector of target weights. The change in holdings for asset _i_ is `Δw_i = w_i_new - w_i_old`. The proportional cost is modeled as a "V-shaped" function of this change. To allow for different costs for buying and selling (e.g., different commission rates or spreads), we define separate cost rates, `k_i^+` for buying and `k_i^-` for selling.1

The cost for trading asset _i_, denoted `C_i(Δw_i)`, is given by:
![[Pasted image 20250726155210.png]]

The total transaction cost is the sum over all assets: `C(Δw) = Σ C_i(Δw_i)`.

The absolute value function implicit in this formulation makes the objective function non-differentiable, which poses a challenge for standard optimizers. However, this can be elegantly resolved through a standard linearization technique. For each asset _i_, we introduce two new non-negative decision variables 1:

- `u_i`: the proportion of wealth used to **buy** asset _i_.
    
- `v_i`: the proportion of wealth generated from **selling** asset _i_.
    

These variables are linked to the change in weights by the following constraints for each asset _i_:

- `w_i_new - w_i_old = u_i - v_i`
    
- `u_i ≥ 0`
    
- `v_i ≥ 0`
    

With this substitution, the non-linear total cost term `Σ k_i |Δw_i|` is replaced by a perfectly linear expression:

![[Pasted image 20250726155230.png]]

An optimizer will naturally ensure that for any given asset, `u_i` and `v_i` are not simultaneously positive, as it is always suboptimal to incur costs by buying and selling the same asset at the same time.20

### 2.2 Integration into Mean-Variance Optimization

With the cost function linearized, we can now integrate it into the standard mean-variance framework. This involves modifying both the objective function and the budget constraint.

**Modified Objective Function:** The goal is no longer to maximize the gross risk-adjusted return, but the _net_ risk-adjusted return, after accounting for trading costs.19

![[Pasted image 20250726155242.png]]
Here, `γ` is the risk-aversion parameter.

**Modified Budget Constraint:** The portfolio must be self-financing, meaning the total cost of trading must be covered by the portfolio's capital, reducing the amount available for investment.17 The standard budget constraint

`1^T w_new = 1` is updated to:

![[Pasted image 20250726155255.png]]

The crucial outcome of this reformulation is that the problem remains convex. It is a **Quadratic Program (QP)**, which can be solved efficiently and to global optimality even for portfolios with thousands of assets using standard optimization software.17

### 2.3 Python Implementation Example

We can implement this model using popular Python libraries. We'll show two approaches: a high-level one with `PyPortfolioOpt` and a more fundamental one with `CVXPY`.

#### 2.3.1 High-Level Implementation with `PyPortfolioOpt`

`PyPortfolioOpt` abstracts away the underlying variable definitions, allowing for rapid implementation. The key is to add the transaction cost as a secondary objective to the main optimization problem.22



```Python
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

# Assume we have a pandas DataFrame 'df' of historical prices
# And a numpy array 'w_old' of previous weights
# Example data
tickers =
w_old = np.array([0.25, 0.25, 0.25, 0.25])
# In a real scenario, you would load your price data here.
# For demonstration, we'll create random data.
np.random.seed(0)
df = pd.DataFrame(100 + np.random.randn(500, 4).cumsum(axis=0), columns=tickers)


# 1. Calculate inputs
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# 2. Instantiate EfficientFrontier
ef = EfficientFrontier(mu, S)

# 3. Add the transaction cost objective
# k represents the proportional cost (e.g., 0.1% or 10 bps)
# Here we assume same cost for buying and selling
k = 0.001 
ef.add_objective(objective_functions.transaction_cost, w_prev=w_old, k=k)

# 4. Optimize for a primary objective, e.g., max Sharpe ratio
# The optimizer will now balance maximizing Sharpe with minimizing transaction costs
ef.max_sharpe()
w_new = ef.clean_weights()

print("Old Weights:", dict(zip(tickers, w_old)))
print("New Weights (with proportional cost):", w_new)
ef.portfolio_performance(verbose=True);
```

#### 2.3.2 Fundamental Implementation with `CVXPY`

Building the model from scratch with `CVXPY` provides a deeper understanding of the mechanics and connects directly to the mathematical formulation.



```Python
import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt import risk_models, expected_returns

# Use the same data as the previous example
tickers =
w_old = np.array([0.25, 0.25, 0.25, 0.25])
np.random.seed(0)
df = pd.DataFrame(100 + np.random.randn(500, 4).cumsum(axis=0), columns=tickers)

mu = expected_returns.mean_historical_return(df).values
S = risk_models.sample_cov(df).values
n = len(tickers)

# Define optimization variables
w_new = cp.Variable(n)
u = cp.Variable(n) # Proportions bought
v = cp.Variable(n) # Proportions sold

# Define parameters
gamma = cp.Parameter(nonneg=True, value=1) # Risk aversion
k_buy = cp.Parameter(n, value=np.full(n, 0.001)) # 10 bps buy cost
k_sell = cp.Parameter(n, value=np.full(n, 0.001)) # 10 bps sell cost

# Define the objective function
# Maximize: mu'w - gamma/2 * w'Sw - costs
risk_adjusted_return = mu.T @ w_new - gamma/2 * cp.quad_form(w_new, S)
transaction_costs = k_buy.T @ u + k_sell.T @ v
objective = cp.Maximize(risk_adjusted_return - transaction_costs)

# Define the constraints
constraints =

# Formulate and solve the problem
prob = cp.Problem(objective, constraints)
prob.solve()

print("\nCVXPY Implementation Results:")
print("Status:", prob.status)
print("New Weights (with proportional cost):", dict(zip(tickers, np.round(w_new.value, 4))))
print("Total Turnover (buy + sell):", f"{np.sum(u.value) + np.sum(v.value):.4f}")
```

## 3. Fixed Transaction Costs and the Integer Programming Challenge

While proportional costs are common, many trading scenarios involve fixed costs, such as a flat commission per trade. This seemingly small change has profound implications for the optimization problem, moving it from the realm of efficient convex optimization to a much more complex combinatorial world.

### 3.1 The Non-Convex Hurdle

A fixed cost introduces a discontinuity into the cost function. The cost to trade asset _i_ is zero if no trade occurs (`Δw_i = 0`), but jumps to a constant positive value, `F_i`, for any non-zero trade, regardless of its size.17 This "all-or-nothing" characteristic makes the cost function non-convex.17

Because the problem is no longer convex, standard QP solvers fail. Finding the globally optimal portfolio requires evaluating a vast number of discrete choices: for each of the `n` assets, the manager can choose to buy, sell, or do nothing. This leads to `3^n` possible trading scenarios, an exponential complexity that makes exhaustive search computationally infeasible for all but the smallest portfolios (e.g., `n > 15`).17 The problem is classified as

**NP-hard**, meaning there is no known algorithm that can solve it efficiently in all cases.19

### 3.2 Formulation via Mixed-Integer Programming

The standard approach to modeling problems with discrete choices is **Mixed-Integer Programming**. We introduce binary (0 or 1) indicator variables to represent the decision to trade.1

- `y_i_buy`: A binary variable that is 1 if we buy asset _i_, and 0 otherwise.
    
- `y_i_sell`: A binary variable that is 1 if we sell asset _i_, and 0 otherwise.
    

These binary variables are then linked to the continuous trade variables (`u_i` for buying, `v_i` for selling) using what is known as the "Big-M" method. We define a large constant `M` that serves as a loose upper bound on the size of any possible trade (e.g., `M=1` since weights cannot exceed 100%). The constraints are:

- `u_i ≤ M * y_i_buy`
    
- `v_i ≤ M * y_i_sell`
    

These constraints enforce the logic that a buy trade (`u_i > 0`) can only occur if the buy indicator is switched on (`y_i_buy = 1`), and similarly for selling.

The total cost function in the objective is then updated to include both the variable (proportional) and fixed components:

![[Pasted image 20250726155328.png]]

The resulting optimization problem is a **Mixed-Integer Quadratic Program (MIQP)**, which seeks to find the optimal values for both the continuous weight variables and the discrete binary trade indicators.

The behavioral consequence of introducing fixed costs is significant. With purely proportional costs, it is optimal to trade from outside the no-trade region precisely _to its boundary_.6 However, a fixed cost acts as a barrier to entry for trading. Once the decision to trade is made and the fixed cost is paid, it becomes a sunk cost for that transaction. Therefore, if the portfolio has drifted far enough for a trade to be worthwhile, it is often optimal to trade a larger amount to "get one's money's worth" from the fixed fee. This means trading not just to the boundary, but deeper into the

_interior_ of the no-trade region.6 This leads to a strategy characterized by

**fewer, but larger and more decisive, trades**.

### 3.3 Python Implementation with `CVXPY`

`CVXPY` can formulate MIQP problems by defining variables with the `boolean=True` attribute. Solving these problems requires a solver capable of handling mixed-integer programs, such as the open-source `GLPK_MI` or `CBC`, or more powerful commercial solvers like `MOSEK` or `GUROBI`.



```Python
import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt import risk_models, expected_returns

# Use the same data as before
tickers =
w_old = np.array([0.25, 0.25, 0.25, 0.25])
np.random.seed(0)
df = pd.DataFrame(100 + np.random.randn(500, 4).cumsum(axis=0), columns=tickers)

mu = expected_returns.mean_historical_return(df).values
S = risk_models.sample_cov(df).values
n = len(tickers)

# Define continuous and binary variables
w_new = cp.Variable(n)
u = cp.Variable(n)
v = cp.Variable(n)
y_buy = cp.Variable(n, boolean=True)
y_sell = cp.Variable(n, boolean=True)

# Define parameters
gamma = cp.Parameter(nonneg=True, value=1)
k_prop = cp.Parameter(n, value=np.full(n, 0.001)) # Proportional cost
F_fixed = cp.Parameter(n, value=np.full(n, 0.0001)) # Fixed cost (e.g., 0.01% of total portfolio value)
M = 1.0 # Big-M constant

# Define the objective function with fixed and proportional costs
risk_adjusted_return = mu.T @ w_new - gamma/2 * cp.quad_form(w_new, S)
prop_costs = k_prop.T @ u + k_prop.T @ v
fixed_costs = F_fixed.T @ y_buy + F_fixed.T @ y_sell
objective = cp.Maximize(risk_adjusted_return - prop_costs - fixed_costs)

# Define the constraints
constraints =

# Formulate and solve the MIQP problem
# Requires a MIQP-capable solver like GLPK_MI, CBC, MOSEK, or GUROBI
# Example: prob.solve(solver=cp.GLPK_MI)
prob = cp.Problem(objective, constraints)
# Note: You need to have a mixed-integer solver installed for this to run.
# For example, `pip install glpk` and then use `solver=cp.GLPK_MI`.
# prob.solve(solver=cp.GLPK_MI, verbose=True) 
# Since a solver might not be installed by default, we will not run solve() here.
print("\nMIQP problem formulated successfully.")
print("To solve, ensure a mixed-integer solver is installed and call prob.solve().")
```

## 4. Advanced Cost Models for Institutional Realism

For institutional investors or those seeking higher fidelity, the linear and fixed-cost models can be extended to capture more complex, real-world trading frictions.

### 4.1 Piecewise Linear Costs

In practice, transaction costs are not always purely linear. Brokers often provide volume discounts, where the commission rate decreases as the size of the trade increases. For example, a broker might charge 10 basis points (bps) on the first $1 million traded, 8 bps on the next $4 million, and so on. This results in a **piecewise linear convex cost function**.25

Although more complex, these models can still be formulated within a convex optimization framework. The approach involves decomposing the trade variable for each asset into multiple segments, one for each tier of the fee schedule, and adding corresponding linear constraints. The problem remains a QP (or can be formulated as a Second-Order Cone Program, SOCP) and is thus efficiently solvable.26

### 4.2 Market Impact: The Price of Size

For any trader executing large orders, market impact is often the most significant and challenging cost to model. It reflects the fact that the act of trading itself moves the market price adversely.12 Market impact is typically decomposed into two components 15:

1. **Temporary Impact:** The price pressure that exists only during the execution of the trade. It reflects the cost of demanding immediate liquidity. This impact dissipates after the trade is completed.
    
2. **Permanent Impact:** The lasting shift in the equilibrium price caused by the trade. This occurs because large trades can reveal information to the market (e.g., the presence of a large, informed seller), causing other participants to update their valuation of the asset.
    

The seminal **Almgren-Chriss framework** provides a dynamic model for optimal execution that explicitly balances the trade-off between market impact (which is minimized by trading slowly) and timing risk (the risk of adverse price movements during a long execution window, which is minimized by trading quickly).28 While a full dynamic model is beyond the scope of this chapter, we can incorporate static market impact models into our single-period optimization.

Mathematical Formulation:

The functional form of market impact has been a subject of extensive empirical research.

- **Quadratic Cost (Linear Impact):** An early and mathematically convenient model assumes that price impact is linear in the rate of trading. This results in a total transaction cost that is quadratic in the trade size: `C(Δw_i) ∝ (Δw_i)^2`.15 This keeps the overall optimization objective quadratic and convex.
    
- **Power-Law Cost (Square-Root Impact):** More recent empirical studies have shown that market impact is often a concave function of trade size. A widely accepted model is the "square-root law," which states that the price impact is proportional to the square root of the trade size relative to the average daily volume.3 This leads to a total cost function that grows with a power of 1.5:
    
    `C(Δw_i) ∝ |Δw_i|^1.5`.
    

This power-law cost function is still convex. It can be modeled in modern optimization frameworks using **power cones**. A term of the form `a|x|^β` can be included in a convex program by introducing an auxiliary variable `t` and adding the conic constraint `(t, 1, x) ∈ PowerCone(1/β, (β-1)/β)`. This requires an advanced solver like MOSEK but allows for a highly realistic, non-quadratic cost model to be solved efficiently.1

### Python Implementation with `cvxportfolio`

The `cvxportfolio` library is specifically designed to handle these sophisticated cost models with ease. Its `TransactionCost` object provides a state-of-the-art implementation that includes terms for proportional costs, market impact with a configurable exponent, and even allows for time-varying forecasts of volatility and volume.34



```Python
import cvxportfolio as cvx
import pandas as pd
import numpy as np

# Assume we have market_data (prices, volumes, etc.)
# and an initial portfolio value
# For demonstration, we'll create some dummy data
tickers =
n = len(tickers)
np.random.seed(0)
index = pd.to_datetime(pd.date_range('2020-01-01', '2022-12-31'))
prices = pd.DataFrame(100 + np.random.randn(len(index), n).cumsum(axis=0), 
                      columns=tickers, index=index)
volumes = pd.DataFrame(np.random.rand(len(index), n) * 1E8, 
                       columns=tickers, index=index)
returns = prices.pct_change().dropna()
market_data = cvx.UserProvidedMarketData(returns=returns, volumes=volumes, prices=prices)

# Define a sophisticated transaction cost model
# a = proportional cost (e.g., 5 bps bid-ask spread)
# b = market impact coefficient
# exponent = 1.5 for square-root impact model
tcost_model = cvx.TransactionCost(a=0.0005, b=1.0, exponent=1.5)

# Define a holding cost model (e.g., for shorting fees)
hcost_model = cvx.HoldingCost(short_fees=0.0001)

# Define the risk model
risk_model = cvx.FullCovariance()

# Define the objective for a single-period optimization policy
# The policy will maximize returns minus penalties for risk and costs
policy = cvx.SinglePeriodOptimization(
    objective = cvx.ReturnsForecast() - 0.5 * risk_model - tcost_model - hcost_model,
    constraints = [cvx.LongOnly(), cvx.LeverageLimit(1)]
)

# This policy can now be used in a market simulator for backtesting
simulator = cvx.MarketSimulator(market_data=market_data)
result = simulator.backtest(policy, start_time='2022-01-01', end_time='2022-12-31')

print("\ncvxportfolio Backtest Summary:")
print(result)
```

## 5. Capstone Project: A Realistic Multi-Asset Rebalancing Strategy

This project synthesizes the chapter's concepts by tackling a common and practical problem: rebalancing an existing ETF portfolio to a new target allocation while carefully managing various types of transaction costs.

### 5.1 Project Brief

An investor holds an existing portfolio composed of several Exchange-Traded Funds (ETFs). Based on a recent strategic review, a new target allocation has been determined. The objective is to find the optimal set of trades to move the portfolio from its current weights to the target, formulating the problem as an optimization that minimizes portfolio risk while explicitly accounting for the costs of rebalancing.35

### 5.2 Data and Parameterization

First, we acquire the necessary data and establish realistic parameters for our cost models.

**Data Acquisition:** We will use the `yfinance` library to download five years of daily historical price and volume data for a diversified set of liquid ETFs. From this data, we will estimate the expected returns vector (`μ`) and the covariance matrix (`Σ`) using standard methods from the `PyPortfolioOpt` library.37

**Asset Universe and Rebalancing Task:** The specific rebalancing problem is defined in the table below.

**Table 5.6.1: Asset Universe and Rebalancing Task**

|Ticker|Asset Class|Initial Weight (%)|Target Weight (%)|
|---|---|---|---|
|SPY|US Large-Cap Equity|40|30|
|QQQ|US Tech Equity|20|25|
|EFA|Developed ex-US Equity|20|20|
|AGG|US Aggregate Bonds|10|20|
|GLD|Gold|10|5|

**Transaction Cost Parameters:** We define a set of realistic cost parameters for each ETF, informed by market data and empirical studies.

**Table 5.6.2: Transaction Cost Parameter Assumptions**

|Ticker|Fixed Cost ($/trade)|Bid-Ask Spread (bps)|Market Impact Coeff. (`b`)|
|---|---|---|---|
|SPY|5.00|1.0|0.5|
|QQQ|5.00|1.5|0.6|
|EFA|5.00|3.0|0.8|
|AGG|5.00|2.0|0.7|
|GLD|5.00|2.5|0.75|

_Justification of Parameters:_

- **Fixed Cost:** A flat $5 fee per trade is a typical commission at many retail brokerages.16
    
- **Bid-Ask Spread:** Spreads for highly liquid ETFs like SPY are very tight, often around 1-2 bps.39 We assign slightly wider spreads to less liquid assets like EFA, reflecting higher transaction friction.14
    
- **Market Impact Coefficient:** Empirical literature suggests these coefficients are often in the range of 0.5 to 1.0.34 We assign smaller values to the most liquid assets (SPY, QQQ) and larger values to the less liquid ones.
    

### 5.3 Complete Solution and Analysis

We will now implement the full solution in Python, solving the rebalancing problem under three distinct scenarios to analyze the impact of different cost models. The objective will be to maximize the portfolio's quadratic utility, which balances expected return against risk and costs.



```Python
# Full Python Implementation for Capstone Project
import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
from pypfopt import expected_returns, risk_models

# --- 1. Data Acquisition and Parameter Estimation ---
tickers =
# Download 5 years of data
ohlc = yf.download(tickers, period="5y")
prices = ohlc['Adj Close'].dropna()
volumes = ohlc['Volume'].dropna()

# Estimate mu and Sigma
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)
n = len(tickers)

# --- 2. Problem Setup ---
w_old = np.array([0.40, 0.20, 0.20, 0.10, 0.10])
portfolio_value = 1_000_000 # Assume a $1M portfolio

# Cost parameters from Table 5.6.2
# Proportional cost = half of bid-ask spread
# 1 bp = 0.0001
prop_costs_bps = np.array([1.0, 1.5, 3.0, 2.0, 2.5])
k_prop = prop_costs_bps / 2 / 10000 

# Fixed cost per trade in dollar terms
fixed_cost_usd = 5.0
# Convert to proportion of portfolio value
F_fixed = fixed_cost_usd / portfolio_value

# Market impact parameters
b_impact = np.array([0.5, 0.6, 0.8, 0.7, 0.75])
# Average daily volume in dollars
avg_daily_volume = (volumes.mean() * prices.mean()).values

# --- 3. Optimization Function ---
def solve_rebalance(w_old, mu, S, gamma_val=1.0, use_fixed_costs=False, use_market_impact=False):
    """Solves the rebalancing problem with different cost models."""
    w_new = cp.Variable(n, name='w_new')
    u = cp.Variable(n, name='u') # buy
    v = cp.Variable(n, name='v') # sell

    # --- Objective ---
    gamma = cp.Parameter(nonneg=True, value=gamma_val)
    ret = mu.T @ w_new
    risk = cp.quad_form(w_new, S)
    
    # Proportional costs
    proportional_cost_expr = k_prop.T @ (u + v)
    
    total_cost = proportional_cost_expr
    
    # --- Constraints ---
    constraints = [
        w_new - w_old == u - v,
        w_new >= 0,
        u >= 0,
        v >= 0,
    ]

    # --- Add Fixed Costs (MIQP) ---
    if use_fixed_costs:
        y_trade = cp.Variable(n, boolean=True)
        fixed_cost_expr = F_fixed * cp.sum(y_trade)
        total_cost += fixed_cost_expr
        
        M = 1.0 # Big-M
        constraints.append(u + v <= M * y_trade)
        
    # --- Add Market Impact Costs ---
    if use_market_impact:
        # Market impact cost C(u) = b * sigma * (u / V)^0.5 * u
        # C(trade) = b * (trade_vol / ADV)^0.5 * trade_vol
        # We model cost as: b * |trade_weight|^1.5 / sqrt(ADV_weight)
        # where ADV_weight = ADV / portfolio_value
        adv_weight = avg_daily_volume / portfolio_value
        trade_abs = u + v
        # Use power cone: t >= x^(3/2) -> (t, 1, x) in Power3D(2/3)
        # We need to scale this by b / sqrt(adv_weight)
        # Let cost_i = scale_i * trade_abs_i^1.5
        # This is a convex term.
        impact_cost_expr = cp.sum(cp.multiply(b_impact / np.sqrt(adv_weight), cp.power(trade_abs, 1.5)))
        total_cost += impact_cost_expr

    objective = cp.Maximize(ret - gamma * risk - total_cost)
    constraints.append(cp.sum(w_new) + total_cost == 1)

    prob = cp.Problem(objective, constraints)
    try:
        solver_opts = {}
        if use_fixed_costs:
            solver_opts['solver'] = cp.GLPK_MI
            solver_opts['verbose'] = False
        prob.solve(**solver_opts)
        
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return w_new.value, u.value, v.value, total_cost.value
        else:
            return f"Solver failed: {prob.status}", None, None, None
    except Exception as e:
        return f"Error: {e}", None, None, None

# --- 4. Run Scenarios and Analyze ---
results = {}

# Scenario 1: Proportional Costs Only
w1, u1, v1, c1 = solve_rebalance(w_old, mu, S)
results['Proportional Only'] = {'w_new': w1, 'cost': c1, 'turnover': np.sum(u1+v1)}

# Scenario 2: Proportional + Fixed Costs
w2, u2, v2, c2 = solve_rebalance(w_old, mu, S, use_fixed_costs=True)
results['Proportional + Fixed'] = {'w_new': w2, 'cost': c2, 'turnover': np.sum(u2+v2) if u2 is not None else 'N/A'}

# Scenario 3: Proportional + Market Impact
w3, u3, v3, c3 = solve_rebalance(w_old, mu, S, use_market_impact=True)
results['Proportional + Impact'] = {'w_new': w3, 'cost': c3, 'turnover': np.sum(u3+v3)}

# --- 5. Display Results ---
summary_df = pd.DataFrame(index=tickers)
summary_df['Initial'] = w_old * 100

for scenario, data in results.items():
    if isinstance(data['w_new'], str):
        summary_df[scenario] = data['w_new']
    else:
        summary_df[scenario] = data['w_new'] * 100

print("\n--- Capstone Project Results: Optimal Portfolio Weights (%) ---")
print(summary_df.round(2))

print("\n--- Analysis of Scenarios ---")
print("\nQuestion 1: What is the optimal rebalancing trade with only proportional costs?")
print("Answer: The model makes several trades to fine-tune the portfolio. The trades are:")
trades1 = pd.Series(u1 - v1, index=tickers)
print(trades1[abs(trades1) > 1e-5].round(4))
print(f"Total turnover is {results['Proportional Only']['turnover']:.2%}, costing {results['Proportional Only']['cost']:.4f} of the portfolio value.")

print("\nQuestion 2: How does adding a fixed cost change the strategy?")
if isinstance(results['Proportional + Fixed']['w_new'], str):
    print(f"Answer: The MIQP solver could not be run. {results['Proportional + Fixed']['w_new']}")
else:
    print("Answer: The model executes fewer, more concentrated trades to overcome the fixed-cost hurdle. Some small, marginal trades are suppressed.")
    trades2 = pd.Series(u2 - v2, index=tickers)
    print(trades2[abs(trades2) > 1e-5].round(4))
    print(f"Total turnover is {results['Proportional + Fixed']['turnover']:.2%}, costing {results['Proportional + Fixed']['cost']:.4f} of the portfolio value.")
    print("Notice that the number of trades is smaller compared to the proportional-only case.")

print("\nQuestion 3: How does including market impact alter the plan?")
print("Answer: The model penalizes large trades, leading to a more conservative rebalancing. The largest required trades are moderated.")
trades3 = pd.Series(u3 - v3, index=tickers)
print(trades3[abs(trades3) > 1e-5].round(4))
print(f"Total turnover is {results['Proportional + Impact']['turnover']:.2%}, costing {results['Proportional + Impact']['cost']:.4f} of the portfolio value.")
print("The sale of SPY and purchase of AGG are likely smaller than in the proportional-only case to avoid high impact costs.")

print("\nQuestion 4: Final Comparison")
final_comparison = pd.DataFrame({
    'Net Return (%)': [mu @ res['w_new'] * 100 if not isinstance(res['w_new'], str) else 'N/A' for res in results.values()],
    'Volatility (%)': [np.sqrt(res['w_new'].T @ S @ res['w_new']) * 100 if not isinstance(res['w_new'], str) else 'N/A' for res in results.values()],
    'Turnover (%)': [res['turnover'] * 100 if res['turnover']!= 'N/A' else 'N/A' for res in results.values()],
    'Total Cost (bps)': [res['cost'] * 10000 if res['cost'] is not None else 'N/A' for res in results.values()]
}, index=results.keys())
print(final_comparison.round(2))
```

### 5.4 Conclusions

This chapter has demonstrated that transaction cost modeling is not an optional afterthought but a central component of practical portfolio optimization. By moving beyond the frictionless ideal, we arrive at strategies that are more robust, stable, and ultimately more likely to succeed in the real world.

The analysis reveals a clear hierarchy of modeling sophistication and its impact on strategy:

1. **Proportional Costs** introduce a baseline level of friction, creating a "no-trade region" and preventing the portfolio from chasing every minor fluctuation. The resulting optimization problem remains a convex QP, which is efficiently solvable.
    
2. **Fixed Costs** fundamentally change the problem's nature to a non-convex MIQP. This leads to a qualitatively different trading behavior characterized by fewer, larger, and more decisive rebalancing events, as the model will only act when the benefits clearly outweigh the fixed hurdle of trading.
    
3. **Market Impact Costs**, particularly non-linear models like the square-root law, are crucial for institutional-scale trading. They penalize large, rapid trades, forcing the optimizer to seek a delicate balance between achieving the target allocation and the cost of liquidity.
    

The capstone project provides a concrete illustration of these principles. As the cost model becomes more realistic—from simple proportional costs to including fixed and market impact components—the optimal rebalancing strategy becomes progressively more conservative. The optimizer intelligently reduces turnover, suppresses marginal trades, and moderates the size of large block trades. This demonstrates the dual role of transaction costs: they are both a direct drag on performance and an implicit, powerful regularizer that enforces discipline and robustness upon the investment process. For the quantitative practitioner, mastering these models is essential for bridging the gap between financial theory and profitable implementation.

## References
**

1. 6 Transaction costs — MOSEK Portfolio Optimization Cookbook 1.6.0, acessado em julho 26, 2025, [https://docs.mosek.com/portfolio-cookbook/transaction.html](https://docs.mosek.com/portfolio-cookbook/transaction.html)
    
2. THE MEAN-VARIANCE APPROACH TO PORTFOLIO OPTIMIZATION SUBJECT TO TRANSACTION COSTS 1. Introduction, acessado em julho 26, 2025, [https://orsj.org/wp-content/or-archives50/pdf/e_mag/Vol.39_01_099.pdf](https://orsj.org/wp-content/or-archives50/pdf/e_mag/Vol.39_01_099.pdf)
    
3. Technical Note—A Robust Perspective on Transaction Costs in Portfolio Optimization - LBS Research Online, acessado em julho 26, 2025, [https://lbsresearch.london.edu/920/1/VDM%20Robust%20Perspective%20on%20Transaction%20Costs.pdf](https://lbsresearch.london.edu/920/1/VDM%20Robust%20Perspective%20on%20Transaction%20Costs.pdf)
    
4. Dynamic Portfolio Choice with Linear Rebalancing Rules - Columbia Business School, acessado em julho 26, 2025, [https://business.columbia.edu/sites/default/files-efs/pubfiles/25471/Moallemi_linear.pdf](https://business.columbia.edu/sites/default/files-efs/pubfiles/25471/Moallemi_linear.pdf)
    
5. TRANSACTION COST AND RESAMPLING IN MEAN-VARIANCE PORTFOLIO OPTIMIZATION - CiteSeerX, acessado em julho 26, 2025, [https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7dc5ca3793e9209a938c04f9493ca22a1850daf2](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7dc5ca3793e9209a938c04f9493ca22a1850daf2)
    
6. Mean-Variance Portfolio Rebalancing with Transaction Costs - Philip H. Dybvig, acessado em julho 26, 2025, [https://dybfin.wustl.edu/research/papers/tcost190313.pdf](https://dybfin.wustl.edu/research/papers/tcost190313.pdf)
    
7. Optimal Consumption and Investment with Fixed and Proportional Transaction Costs, acessado em julho 26, 2025, [https://oar.princeton.edu/bitstream/88435/pr1vk4f/1/Optimal%20consumption%20and%20investment%20with%20fixed%20and%20proportional%20transaction%20costs.pdf](https://oar.princeton.edu/bitstream/88435/pr1vk4f/1/Optimal%20consumption%20and%20investment%20with%20fixed%20and%20proportional%20transaction%20costs.pdf)
    
8. Rebalancing with transaction costs: theory, simulations, and actual data, acessado em julho 26, 2025, [https://d-nb.info/1273162811/34](https://d-nb.info/1273162811/34)
    
9. Understanding the Difference Between Implicit and Explicit Costs - Royal Sundaram, acessado em julho 26, 2025, [https://www.royalsundaram.in/knowledge-centre/others/difference-between-implicit-and-explicit-costs](https://www.royalsundaram.in/knowledge-centre/others/difference-between-implicit-and-explicit-costs)
    
10. Implicit Cost vs Explicit Cost: Key Differences Explained - Profitjets, acessado em julho 26, 2025, [https://profitjets.com/blog/implicit-cost-vs-explicit-cost/](https://profitjets.com/blog/implicit-cost-vs-explicit-cost/)
    
11. Execution Costs - CFA, FRM, and Actuarial Exams Study Notes, acessado em julho 26, 2025, [https://analystprep.com/study-notes/cfa-level-2/explain-the-components-of-execution-costs-including-explicit-and-implicit-costs/](https://analystprep.com/study-notes/cfa-level-2/explain-the-components-of-execution-costs-including-explicit-and-implicit-costs/)
    
12. Implicit Commissions. In the institutional trading world… | by Daniel Aisen | Proof Reading | Medium, acessado em julho 26, 2025, [https://medium.com/prooftrading/implicit-commissions-7049a30ce7d4](https://medium.com/prooftrading/implicit-commissions-7049a30ce7d4)
    
13. Managing Inventory with Proportional Transaction Costs - EPFL, acessado em julho 26, 2025, [https://www.epfl.ch/schools/cdm/wp-content/uploads/2018/08/Passerini_amamef.pdf](https://www.epfl.ch/schools/cdm/wp-content/uploads/2018/08/Passerini_amamef.pdf)
    
14. What Is a Bid-Ask Spread, and How Does It Work in Trading? - Investopedia, acessado em julho 26, 2025, [https://www.investopedia.com/terms/b/bid-askspread.asp](https://www.investopedia.com/terms/b/bid-askspread.asp)
    
15. Optimal Execution of Portfolio Transactions∗, acessado em julho 26, 2025, [https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
    
16. What Are the Average Brokerage Fees? - BrokerChooser, acessado em julho 26, 2025, [https://brokerchooser.com/education/investing/brokerage-fee/what-are-average-brokerage-fees](https://brokerchooser.com/education/investing/brokerage-fee/what-are-average-brokerage-fees)
    
17. Portfolio optimization with linear and fixed transaction costs - Stanford University, acessado em julho 26, 2025, [https://web.stanford.edu/~boyd/papers/pdf/portfolio_submitted.pdf](https://web.stanford.edu/~boyd/papers/pdf/portfolio_submitted.pdf)
    
18. Portfolio optimization with linear and fixed transaction costs - IDEAS/RePEc, acessado em julho 26, 2025, [https://ideas.repec.org/a/spr/annopr/v152y2007i1p341-36510.1007-s10479-006-0145-1.html](https://ideas.repec.org/a/spr/annopr/v152y2007i1p341-36510.1007-s10479-006-0145-1.html)
    
19. A Note on Portfolio Optimization with Quadratic Transaction Costs - Amundi Research Center, acessado em julho 26, 2025, [https://research-center.amundi.com/files/nuxeo/dl/d860d631-4b46-4e95-bdd6-0bb78e70bd3f?inline=](https://research-center.amundi.com/files/nuxeo/dl/d860d631-4b46-4e95-bdd6-0bb78e70bd3f?inline)
    
20. Portfolio optimization subject to transaction costs - Quantitative Finance Stack Exchange, acessado em julho 26, 2025, [https://quant.stackexchange.com/questions/30939/portfolio-optimization-subject-to-transaction-costs](https://quant.stackexchange.com/questions/30939/portfolio-optimization-subject-to-transaction-costs)
    
21. Markowitz Portfolios under Transaction Costs - | Department of Economics | UZH, acessado em julho 26, 2025, [https://www.econ.uzh.ch/apps/workingpapers/wp/econwp420.pdf](https://www.econ.uzh.ch/apps/workingpapers/wp/econwp420.pdf)
    
22. PyPortfolioOpt/pypfopt/objective_functions.py at master - GitHub, acessado em julho 26, 2025, [https://github.com/robertmartin8/PyPortfolioOpt/blob/master/pypfopt/objective_functions.py](https://github.com/robertmartin8/PyPortfolioOpt/blob/master/pypfopt/objective_functions.py)
    
23. Mean-Variance Optimization — PyPortfolioOpt 1.5.4 documentation - Read the Docs, acessado em julho 26, 2025, [https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html](https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html)
    
24. Portfolio Optimization Under Fixed Transaction Costs, acessado em julho 26, 2025, [https://www.math.cmu.edu/~shaikhet/_docs/MScSeminar.pdf](https://www.math.cmu.edu/~shaikhet/_docs/MScSeminar.pdf)
    
25. TYPES OF TRANSACTION COST MODELS - Inside the Black Box: The Simple Truth About Quantitative Trading [Book], acessado em julho 26, 2025, [https://www.oreilly.com/library/view/inside-the-black/9780470432068/9780470432068_types_of_transaction_cost_models.html](https://www.oreilly.com/library/view/inside-the-black/9780470432068/9780470432068_types_of_transaction_cost_models.html)
    
26. (PDF) Large scale portfolio optimization with piecewise linear transaction costs, acessado em julho 26, 2025, [https://www.researchgate.net/publication/228747715_Large_scale_portfolio_optimization_with_piecewise_linear_transaction_costs](https://www.researchgate.net/publication/228747715_Large_scale_portfolio_optimization_with_piecewise_linear_transaction_costs)
    
27. Large Scale Portfolio Optimization with Piecewise Linear ..., acessado em julho 26, 2025, [https://www.math.uwaterloo.ca/~ltuncel/publications/corr2006-19.pdf](https://www.math.uwaterloo.ca/~ltuncel/publications/corr2006-19.pdf)
    
28. Optimal Execution Strategies - Almgren-Chriss Model - QuestDB, acessado em julho 26, 2025, [https://questdb.com/glossary/optimal-execution-strategies-almgren-chriss-model/](https://questdb.com/glossary/optimal-execution-strategies-almgren-chriss-model/)
    
29. Solving the Almgren Chris Model - Dean Markwick, acessado em julho 26, 2025, [https://dm13450.github.io/2024/06/06/Solving-the-Almgren-Chris-Model.html](https://dm13450.github.io/2024/06/06/Solving-the-Almgren-Chris-Model.html)
    
30. 295-2017-Trading-lightly-cross-impact-and-optimal-portfolio-execution.pdf - CFM, acessado em julho 26, 2025, [https://www.cfm.com/wp-content/uploads/2022/12/295-2017-Trading-lightly-cross-impact-and-optimal-portfolio-execution.pdf](https://www.cfm.com/wp-content/uploads/2022/12/295-2017-Trading-lightly-cross-impact-and-optimal-portfolio-execution.pdf)
    
31. Market impact models and optimal execution algorithms - Imperial College London, acessado em julho 26, 2025, [https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/cfm-imperial-institute-of-quantitative-finance/events/Lillo-Imperial-Lecture3.pdf](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/cfm-imperial-institute-of-quantitative-finance/events/Lillo-Imperial-Lecture3.pdf)
    
32. Multiperiod Portfolio Optimization with General Transaction Costs, acessado em julho 26, 2025, [https://optimization-online.org/wp-content/uploads/2013/07/3962.pdf](https://optimization-online.org/wp-content/uploads/2013/07/3962.pdf)
    
33. Three models of market impact - Baruch MFE Program, acessado em julho 26, 2025, [https://mfe.baruch.cuny.edu/wp-content/uploads/2012/09/Chicago2016OptimalExecution.pdf](https://mfe.baruch.cuny.edu/wp-content/uploads/2012/09/Chicago2016OptimalExecution.pdf)
    
34. Cost models — Cvxportfolio 1.5.0 documentation, acessado em julho 26, 2025, [https://www.cvxportfolio.com/en/1.5.0/costs.html](https://www.cvxportfolio.com/en/1.5.0/costs.html)
    
35. Portfolio Rebalancing - Stephen Diehl, acessado em julho 26, 2025, [https://www.stephendiehl.com/posts/portfolio_rebalance/](https://www.stephendiehl.com/posts/portfolio_rebalance/)
    
36. Rebalancing a multi-asset portfolio: A guide to the choices and trade-offs - Wellington Management, acessado em julho 26, 2025, [https://www.wellington.com/en/insights/rebalancing-a-multi-asset-portfolio](https://www.wellington.com/en/insights/rebalancing-a-multi-asset-portfolio)
    
37. yfinance Library - A Complete Guide - AlgoTrading101 Blog, acessado em julho 26, 2025, [https://algotrading101.com/learn/yfinance-guide/](https://algotrading101.com/learn/yfinance-guide/)
    
38. Download Financial Dataset Using Yahoo Finance in Python | A Complete Guide, acessado em julho 26, 2025, [https://www.analyticsvidhya.com/blog/2021/06/download-financial-dataset-using-yahoo-finance-in-python-a-complete-guide/](https://www.analyticsvidhya.com/blog/2021/06/download-financial-dataset-using-yahoo-finance-in-python-a-complete-guide/)
    
39. The ABCs of ETF liquidity - Vanguard Advisors, acessado em julho 26, 2025, [https://advisors.vanguard.com/insights/article/the-abcs-of-etf-liquidity](https://advisors.vanguard.com/insights/article/the-abcs-of-etf-liquidity)
    
40. ​Understanding bid ask spreads - RBC Global Asset Management, acessado em julho 26, 2025, [https://www.rbcgam.com/en/ca/learn-plan/types-of-investments/understanding-bid-ask-spreads/detail](https://www.rbcgam.com/en/ca/learn-plan/types-of-investments/understanding-bid-ask-spreads/detail)
    

DM Trading Cost Models - Deutsche Bank Autobahn, acessado em julho 26, 2025, [https://static.autobahn.db.com/microSite/docs/DBTradingCostModels-v1.1.pdf](https://static.autobahn.db.com/microSite/docs/DBTradingCostModels-v1.1.pdf)**