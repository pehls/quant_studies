# 4. Derivatives Pricing and Hedging: 6 - Calibration of Models to Market Data

## 1. The Principle of Calibration: Aligning Models with Markets

### 1.1 Introduction: From Theoretical Models to Market Prices

In quantitative finance, mathematical models are the instruments we use to understand, price, and hedge complex financial derivatives. Models such as Black-Scholes, Heston, or Hull-White provide a theoretical framework for the behavior of assets and interest rates. However, a theoretical model is of little practical use until it is anchored to the reality of the marketplace. **Model calibration** is the process of tuning a model's parameters to ensure its outputs—most commonly, the prices of simple, liquidly traded options—are consistent with the prices observed in the market.1 It is the critical procedure that bridges the gap between abstract financial theory and empirical market data.

A powerful way to conceptualize calibration is as the **inverse problem** to pricing.3 In a standard pricing problem, we are given a set of model parameters (e.g., volatility, interest rate) and we compute the theoretical price of a derivative. In the calibration problem, the inputs and outputs are reversed: we are given a set of market-observed prices for benchmark instruments, and we must deduce the set of model parameters that could have generated them. This inversion is rarely straightforward; as we will see, it is often an ill-posed problem, meaning a unique, stable solution may not exist without further mathematical and economic considerations.3

The ultimate goal of calibration is to select a parameter vector, θ, that forces the model to accurately reflect current market conditions. A well-calibrated model can then be used with greater confidence for its primary purposes: pricing more complex, illiquid derivatives for which no market price exists, calculating reliable hedge ratios (the Greeks), and performing robust risk management and portfolio optimization.2

### 1.2 Historical vs. Implied Parameters: The Rationale for Calibration

A foundational concept that must be grasped is the distinction between parameters estimated from historical data and those inferred from current market prices. This difference lies at the very heart of why calibration is not just useful, but essential for pricing and hedging.

- **Historical Parameters** are calculated from past time-series data of an asset's price. For example, historical volatility is the annualized standard deviation of the asset's daily log returns over a specific look-back period (e.g., 30, 60, or 90 days). These parameters are backward-looking; they are a statistical measurement of what _has happened_ in the past.6
    
- **Implied Parameters** are inferred from the current market prices of derivative contracts, most commonly vanilla options. For instance, the Black-Scholes implied volatility of an option is the value of the volatility parameter that, when plugged into the Black-Scholes formula, returns the option's observed market price. These parameters are forward-looking; they represent the market's collective, consensus expectation of what _will happen_ in the future, under a risk-neutral framework.6
    

The necessity of using implied parameters for pricing stems from the **arbitrage-free mandate**. A model used to price an exotic derivative must, as a prerequisite, correctly price the simple, liquid derivatives on the same underlying asset. Imagine a bank's pricing model that, using some set of parameters, calculates the price of a standard, at-the-money call option on Apple (AAPL) to be $5.00, while that same option is actively trading on the exchange for $5.20. The model is demonstrably wrong in a way that creates an immediate arbitrage opportunity. A trader could, in theory, buy the "cheap" option as priced by the model and sell the "expensive" one in the market to lock in a risk-free profit.

In practice, the model is used to price an exotic derivative that has no market price. To do this consistently, the model must first be forced to agree with the market on the prices of simple instruments. This "forcing" is the act of calibration. The parameters that emerge from this process—the implied parameters—are, by definition, the parameters that make the model's pricing of vanilla options consistent with the market. Using these implied parameters to then price an exotic derivative ensures that the exotic's price is consistent with the observable universe of related securities, thereby respecting the no-arbitrage condition. Calibration is therefore not merely a statistical "best-fit" exercise; it is a fundamental requirement for internal consistency in a pricing framework.

The following table summarizes this crucial distinction.

**Table 1: Comparison of Historical and Implied Parameters**

|Feature|Historical Parameters (e.g., Volatility)|Implied Parameters (e.g., Volatility)|
|---|---|---|
|**Source**|Past time-series of underlying asset prices.7|Current market prices of derivative contracts (options).7|
|**Nature**|Backward-looking, statistical, realized.8|Forward-looking, market expectation, risk-neutral.6|
|**Primary Use**|Statistical analysis, risk models (e.g., VaR based on historical simulation), backtesting.|Derivative pricing, hedging, inferring market sentiment.2|
|**Key Insight**|Describes what _did_ happen.|Reflects what the market _thinks will_ happen.7|
|**Relationship**|Serves as a baseline. Implied volatility often reverts to historical levels but can deviate significantly based on market events.7||

### 1.3 The Scope of Calibration

While derivatives pricing is the most prominent application, the importance of calibration extends across the quantitative finance landscape.

- **Derivatives Pricing:** This is the most direct application. A model is calibrated to liquid vanilla options to price more complex, illiquid exotic options in an arbitrage-free manner.
    
- **Risk Management:** A calibrated model provides more accurate and market-consistent risk sensitivities (the Greeks). For example, delta-hedging is the practice of immunizing a derivative's value against small changes in the underlying's price. The amount to hedge is determined by the option's delta. An uncalibrated or poorly calibrated model will produce an incorrect delta, leading to an over-hedged portfolio (which ties up unnecessary capital) or an under-hedged portfolio (which leaves the position exposed to risk).9
    
- **Portfolio Optimization:** Accurate models of underlying assets and their dependencies, calibrated to current market conditions, are essential for making informed investment and asset allocation decisions.2
    

## 2. The Calibration Engine: An Optimization Perspective

### 2.1 Formulating the Objective Function

At its heart, the process of calibration is a numerical optimization task.5 We define an

**objective function** (also known as a loss function or cost function) that measures the "distance" or "error" between the model's outputs and the market's observed data. The calibration then becomes a search for the set of model parameters, θ, that minimizes this objective function.

The most common formulation is a **least-squares problem**. Given a set of N benchmark instruments (e.g., options with different strikes Ki​ and maturities Ti​), the objective function L(θ) to be minimized is typically the sum of squared errors:

![[Pasted image 20250707225628.png]]

In this equation:

- $P_{model}​(K_i​,T_i​;θ)$ is the price of the i-th instrument as calculated by our financial model using the parameter set θ.
    
- $P_{market}​(K_i​,T_i​)$ is the observed market price of that same instrument.
    
- $wi​$ is a weighting factor for instrument i, allowing us to assign more importance to certain instruments in the calibration set.5
    

### 2.2 A Survey of Loss Functions

The choice of how to define the error and the weights is a critical modeling decision that reflects the practitioner's goals and beliefs about the market. There is no single "correct" loss function; the selection is an art that balances mathematical convenience with financial intuition.

Price Error vs. Implied Volatility Error

Instead of minimizing the error in dollar prices, it is often preferable to minimize the error in terms of implied volatilities. Traders and risk managers typically think and quote in terms of volatility, not price. Furthermore, implied volatility is a more stable and comparable unit across different strikes and maturities. The objective function becomes:

![[Pasted image 20250707225731.png]]

Here, σmodel​ is the implied volatility produced by the model with parameters θ, and σmarket​ is the market implied volatility.10 This approach requires an extra step—inverting a pricing formula (like Black-Scholes) to get volatilities from prices—but often leads to more stable and meaningful calibrations.10

Weighted Least Squares

Using uniform weights (wi​=1) for all instruments is a naive choice. Some options are more important, more liquid, and more sensitive to model parameters than others. A standard industry practice is to weight the errors by the inverse of the option's Vega (V), the sensitivity of the option price to a change in volatility.

![[Pasted image 20250707225746.png]]

This weighting scheme gives more importance to at-the-money (ATM) options, which have the highest Vega and are typically the most liquidly traded. It effectively tells the optimizer: "It is more important to correctly price the ATM options than the deep out-of-the-money options".12

Regularization and Ill-Posed Problems

As previously noted, calibration is often an ill-posed problem.3 The objective function may be non-convex, with many local minima, or it may have long, flat "valleys" where very different parameter sets produce almost identical pricing errors.10 This can lead to parameter instability: a tiny change in market data could cause the calibrated parameters to jump dramatically to a new, economically nonsensical value.

**Regularization** is a technique used to combat this by adding a penalty term, R(θ), to the objective function. This term penalizes undesirable parameter characteristics, such as extreme values or instability over time.

![[Pasted image 20250707225754.png]]
The regularization parameter, α, controls the trade-off between fitting the market data (the error term) and satisfying the penalty condition.3

- **Tikhonov Regularization:** This is one of the most common forms, where the penalty is on the squared norm of the parameter vector, R(θ)=∣∣θ∣∣2. It encourages solutions with smaller, more "reasonable" parameter values.3
    
- **Consistency Hints:** This is a more general and powerful concept. The penalty term can be designed to enforce known economic relationships or, crucially, to ensure parameter stability over time. For instance, one could add a penalty term like R(θ)=∣∣θt​−θt−1​∣∣2, which explicitly penalizes large day-over-day changes in the calibrated parameters.13 This transforms the problem from simple "curve fitting" into genuine "calibration," where the model must be consistent not only with today's market but also with itself over time.
    

This leads to a profound point about the art of defining the "best fit." The choice of a loss function is a tool for managing different types of model risk. A practitioner might deliberately accept a slightly larger pricing error on a given day (known as **Type 1 model risk**) in exchange for more stable and economically plausible parameters over time (reducing **Type 2 model risk**, or recalibration risk).14 By increasing the regularization parameter

α, the modeler explicitly prioritizes stability over a perfect but potentially fragile fit.

The following table summarizes some common loss functions used in practice.

**Table 2: Common Loss Functions in Model Calibration**
## Option Pricing Loss Functions Summary

| **Loss Function Type**        | **Mathematical Formulation**                                                                 | **Rationale & Use Case**                                                                                             |
| :---------------------------- | :------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| **Absolute Price Error**      | $$\sum w_i \|P_{\text{model}} - P_{\text{market}}\|$$                                        | Measures direct absolute price difference between model and market. Simple but sensitive to scale.                   |
| **Relative Price Error**      | $$\sum w_i \left( \frac{P_{\text{model}} - P_{\text{market}}}{P_{\text{market}}} \right)^2$$ | Normalizes errors — useful when prices have different magnitudes; ensures comparability across strikes/maturities.   |
| **Vega-Weighted Price Error** | $$\sum \frac{1}{\text{Vega}_i^2} (P_{\text{model}} - P_{\text{market}})^2$$                  | Weighs errors by the inverse squared Vega. Prioritizes ATM options where pricing is more sensitive to volatility.    |
| **Implied Volatility Error**  | $$\sum w_i (\sigma_{\text{model}} - \sigma_{\text{market}})^2$$                              | Aligns with how traders quote and interpret options markets. More stable across different moneyness and maturities.  |
| **Tikhonov Regularization**   | $$\text{Error Term} + \alpha \| \theta \|^2$$                                                | Adds a penalty on model parameters to prevent overfitting. Controls complexity via regularization strength $\alpha$. |

### 2.3 Numerical Optimization Algorithms

Once an objective function is defined, a numerical algorithm is required to find the parameter vector θ that minimizes it. The choice of optimizer involves a trade-off between speed, robustness, and complexity.

The Levenberg-Marquardt Algorithm

For non-linear least-squares problems, the Levenberg-Marquardt (LM) algorithm is the undisputed industry standard.16 It is a "trust-region" method that ingeniously interpolates between two other methods:

1. **Gradient Descent:** When the current parameter guess is far from the optimal solution, LM behaves like the gradient descent method. It takes small, cautious steps in the direction of the steepest descent of the objective function. This approach is slow but very robust and unlikely to diverge.18
    
2. **Gauss-Newton Method:** When the current guess is close to the minimum, LM behaves like the Gauss-Newton method, which uses a linear approximation of the model to converge quadratically (very quickly) to the solution.18
    

This adaptive, hybrid nature makes LM both efficient and reliable. It is the default optimizer in many financial libraries, including QuantLib, for calibration tasks.15 The core of the algorithm involves iteratively solving the linear system:

$$(J^TJ+λI)h=−J^Tr$$

where J is the Jacobian matrix of the residuals, r is the vector of residuals, h is the update step for the parameters, and λ is the crucial damping parameter. When λ is large, the algorithm resembles gradient descent; when λ is small, it resembles the Gauss-Newton method. The algorithm adaptively adjusts λ at each iteration based on whether the proposed step successfully reduced the error.17

Global and Stochastic Search Methods

A major drawback of gradient-based methods like LM is that they are local optimizers. If the objective function is non-convex, as is common in calibration, they can easily get stuck in a local minimum, failing to find the true global minimum.10

To address this, one can employ global or stochastic search methods:

- **Differential Evolution (DE)** and **Particle Swarm Optimization (PSO)** are examples of evolutionary or nature-inspired algorithms. They explore the parameter space more broadly and are less susceptible to getting trapped in local minima. However, this robustness comes at a significant computational cost, as they typically require many more function evaluations.11
    
- **Multi-start Optimization** offers a practical compromise. It involves running a fast local optimizer, like Levenberg-Marquardt, from a multitude of different random starting points. The best result across all runs is then chosen as the final solution. This significantly increases the probability of finding the global minimum without the extreme computational cost of a pure stochastic search.20
    

Modern Trends: Derivative-Free and AI-Based Methods

A practical challenge in calibrating complex models is that the gradient of the objective function (specifically, the Jacobian matrix J) can be difficult, slow, or inaccurate to compute. Calculating it analytically can be a monumental task, and approximating it with finite differences is often slow and introduces numerical noise, which can mislead the optimizer.21 This has spurred two important trends:

1. **Derivative-Free Optimization (DFO):** These are solvers designed for "black-box" objective functions where derivatives are unavailable or unreliable. For problems where the function evaluation is expensive and the gradients are noisy, a DFO solver can often find a better solution in fewer function calls than a gradient-based method struggling with poor derivative estimates.21
    
2. **Deep Calibration:** This is a cutting-edge approach that uses deep neural networks to approximate the model's pricing function itself. The slow, complex pricing model is replaced by a very fast, pre-trained neural network. Because neural networks are differentiable by design (via backpropagation), the gradient calculation becomes computationally trivial. This allows for extremely fast gradient-based optimization of the calibration problem. This is an active area of research, particularly for models where the pricing step is the primary bottleneck.20
    

## 3. Case Study 1: Calibrating the Heston Stochastic Volatility Model

We now apply these abstract concepts to the calibration of the Heston model, a cornerstone of stochastic volatility modeling.

### 3.1 Model Dynamics and Parameters

The Heston model, introduced by Steven Heston in 1993, addresses the primary limitation of the Black-Scholes model: the assumption of constant volatility. It models volatility as a random process that is correlated with the asset's price process.23 Under the risk-neutral measure Q, the model is defined by a system of two correlated stochastic differential equations (SDEs):

![[Pasted image 20250707234712.png]]

where the two Wiener processes are correlated such that $dWt^S_t​dW^v_t​=ρdt$. The model has five parameters, each with a clear financial interpretation 23:

- v0​: The initial variance of the asset price at time t=0.
    
- κ (kappa): The speed of mean reversion. This parameter governs how quickly the variance vt​ tends to revert to its long-term average. A high κ implies rapid reversion.
    
- θ (theta): The long-term mean variance. This is the level to which the variance process reverts over time.
    
- ξ (xi, often denoted σv​): The volatility of variance. This parameter controls the magnitude of the randomness in the variance process itself.
    
- ρ (rho): The correlation between the asset's random shock ($dW_t^S$​) and the variance's random shock ($dW_t^v​$). For equity markets, ρ is typically negative, capturing the well-documented **leverage effect**: when the stock price falls, its volatility tends to rise.
    

### 3.2 Heston Pricing via Characteristic Functions

While there is no simple closed-form solution for option prices like the Black-Scholes formula, Heston provided a semi-analytical solution based on Fourier analysis. The price of a European option can be calculated by numerically integrating its characteristic function.25 This is significantly faster than using Monte Carlo simulation, making calibration computationally feasible.

The price of a European call option is given by:

$$C(S_t, v_t, K, T) = S_t P_1 - K e^{-r(T-t)} P_2$$

where P1​ and P2​ are probabilities that are calculated via Fourier inversion:

$$P_j(x, v, t, T; \phi) = \frac{1}{2} + \frac{1}{\pi} \int_0^{\infty} \text{Re}\left[ \frac{e^{-i \phi \ln K} \, \varphi(\phi - i (j-1))}{i \phi} \right] d\phi
$$

Here, fj​ is the complex-valued characteristic function of the log-asset price, whose explicit (though lengthy) form depends on the Heston parameters.25 This integral must be computed numerically.

### 3.3 Python Implementation: Calibrating to a Volatility Surface

We will now walk through a complete Python example of calibrating the Heston model to a surface of market-observed implied volatilities using the `QuantLib` library. `QuantLib` is an open-source, production-quality library for quantitative finance.

**Steps:**

1. **Setup Environment:** Import necessary libraries and define global settings like the calculation date.
    
2. **Define Market Data:** Input the risk-free rate, dividend yield, and a matrix of implied volatilities corresponding to a grid of option expiration dates and strike prices.
    
3. **Construct Volatility Surface:** Use `QuantLib`'s `BlackVarianceSurface` to create an object that can interpolate the market data, providing a continuous volatility smile for any strike and maturity.
    
4. **Initialize Heston Model:** Create a `HestonProcess` with an initial guess for the five parameters (κ,θ,ξ,ρ,v0​). This process is then used to build a `HestonModel`.
    
5. **Create Calibration Helpers:** For a chosen maturity slice of the volatility surface, create a list of `HestonModelHelper` objects. Each helper represents a single market option and links its market-quoted volatility to the Heston model's pricing engine (`AnalyticHestonEngine`). The engine uses the characteristic function approach to price options.
    
6. **Perform Calibration:** Instantiate a `LevenbergMarquardt` optimizer and define the stopping criteria for the optimization. The `model.calibrate()` method is then called, which runs the LM algorithm to find the parameters that minimize the squared error between the model-implied volatilities and the market volatilities provided by the helpers.
    
7. **Analyze Results:** Once the calibration converges, extract the optimized parameters from the model and generate a report comparing the final model volatilities to the market volatilities to assess the goodness-of-fit.
    



```Python
import QuantLib as ql
import math
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Setup Environment ---
# Set the evaluation date for all calculations
calculation_date = ql.Date(9, 11, 2021)
ql.Settings.instance().evaluationDate = calculation_date
calendar = ql.UnitedStates(m=ql.UnitedStates.NYSE)
day_count = ql.Actual365Fixed()

# --- 2. Define Market Data ---
spot_price = 659.37
risk_free_rate = 0.01
dividend_rate = 0.00

# Create flat term structures for risk-free rate and dividend yield
flat_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, risk_free_rate, day_count)
)
dividend_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, dividend_rate, day_count)
)

# Market data: A matrix of implied volatilities
# Rows: Strikes, Columns: Expiries
expiration_dates =
strikes = [560.46, 593.43, 626.40, 659.37, 692.34, 725.31, 758.28]
data = [0.34177, 0.30394, 0.27832, 0.26453, 0.25916, 0.25941, 0.26127],
    [0.33472, 0.30209, 0.27887, 0.26511, 0.25883, 0.25791, 0.25868],
    [0.32773, 0.29822, 0.27663, 0.26388, 0.25754, 0.25583, 0.25594],
    [0.31682, 0.29235, 0.27361, 0.26173, 0.25515, 0.25231, 0.25159],
    [0.30745, 0.28632, 0.26931, 0.25854, 0.25224, 0.24883, 0.24773],
    [0.30021, 0.28114, 0.26573, 0.25589, 0.25007, 0.24649, 0.24523],
    [0.29841, 0.28203, 0.26888, 0.26042, 0.25531, 0.25243, 0.25141],
    [0.29753, 0.28212, 0.26998, 0.26211, 0.25745, 0.25489, 0.25411],
    [0.29613, 0.28111, 0.26945, 0.26194, 0.25752, 0.25519, 0.25455],
    [0.29481, 0.27993, 0.26867, 0.26144, 0.25721, 0.25501, 0.25447]

implied_vols = ql.Matrix(len(strikes), len(expiration_dates))
for i in range(implied_vols.rows()):
    for j in range(implied_vols.columns()):
        implied_vols[i][j] = data[j][i]

# --- 3. Construct Volatility Surface ---
black_var_surface = ql.BlackVarianceSurface(
    calculation_date, calendar, expiration_dates, strikes, implied_vols, day_count
)
black_var_surface.setInterpolation("bicubic")

# --- 4. Initialize Heston Model ---
# Initial guess for Heston parameters
v0 = 0.06; kappa = 11.0; theta = 0.13; rho = -0.35; sigma = 4.0
process = ql.HestonProcess(flat_ts, dividend_ts,
                           ql.QuoteHandle(ql.SimpleQuote(spot_price)),
                           v0, kappa, theta, sigma, rho)
model = ql.HestonModel(process)
engine = ql.AnalyticHestonEngine(model)

# --- 5. Create Calibration Helpers ---
# We will calibrate to the 1-year options slice
one_year_idx = 5 # Corresponds to ql.Date(9,12,2022)
date = expiration_dates[one_year_idx]
helpers =
for j, s in enumerate(strikes):
    t = (date - calculation_date)
    p = ql.Period(t, ql.Days)
    # Use the interpolated volatility from the surface
    vol = black_var_surface.blackVol(p.length() / 365.25, s)
    helper = ql.HestonModelHelper(p, calendar, spot_price, s,
                                  ql.QuoteHandle(ql.SimpleQuote(vol)),
                                  flat_ts, dividend_ts)
    helper.setPricingEngine(engine)
    helpers.append(helper)

# --- 6. Perform Calibration ---
# Set up the optimization algorithm
lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
end_criteria = ql.EndCriteria(500, 50, 1.0e-8, 1.0e-8, 1.0e-8)

# Calibrate the model
model.calibrate(helpers, lm, end_criteria)

# --- 7. Analyze Results ---
theta_cal, kappa_cal, sigma_cal, rho_cal, v0_cal = model.params()

print("--- Calibrated Heston Parameters ---")
print(f"Theta (long-term variance): {theta_cal:.6f}")
print(f"Kappa (mean-reversion speed): {kappa_cal:.6f}")
print(f"Sigma (vol of vol): {sigma_cal:.6f}")
print(f"Rho (correlation): {rho_cal:.6f}")
print(f"v0 (initial variance): {v0_cal:.6f}")
print("-" * 36)

print("\n--- Calibration Goodness-of-Fit ---")
print(f"{'Strike':>10} {'Market Vol':>15} {'Model Vol':>15} {'Rel Error (%)':>20}")
print("=" * 62)
avg_err = 0.0
for i, helper in enumerate(helpers):
    market_vol = helper.marketValue()
    model_vol = helper.impliedVolatility(helper.modelValue(), 1e-4, 1000, 1e-8, 1.0)
    rel_error = (model_vol / market_vol - 1.0) * 100
    avg_err += abs(rel_error)
    print(f"{strikes[i]:10.2f} {market_vol:15.5f} {model_vol:15.5f} {rel_error:20.5f}")

print("-" * 62)
print(f"Average Absolute Error: {avg_err / len(helpers):.5f}%")
```

This code provides a complete, runnable example of the calibration process. It demonstrates how to take raw market data, structure it within `QuantLib`, set up a sophisticated model like Heston, and use industry-standard optimization techniques to find the parameters that best align the model with market reality.15

## 4. Case Study 2: Calibrating the SABR Volatility Smile Model

The SABR model is a staple in interest rate and FX markets, celebrated for its ability to capture the volatility smile and skew observed in option prices with just a few intuitive parameters.

### 4.1 Model Dynamics and the Volatility Smile

SABR stands for "Stochastic Alpha, Beta, Rho." It models the joint evolution of a forward rate, Ft​, and its stochastic volatility, αt​, using the following SDEs 28:

$$dF_t = \alpha_t (F_t)^\beta dW_t^F$$

$$d\alpha_t = \nu \alpha_t dW_t^\alpha$$

where the Wiener processes are correlated such that dWtF​dWtα​=ρdt. The model's parameters are:

- α: The initial level of volatility.
    
- β: The exponent that governs the relationship between the forward rate and its volatility. It is often fixed exogenously based on market convention (e.g., β=1 for lognormal dynamics, β=0 for normal dynamics, β=0.5 for CIR-like dynamics).
    
- ρ: The correlation between the forward rate and its volatility. It controls the "skew" of the smile.
    
- ν: The volatility of volatility ("vol of vol"). It controls the "convexity" or "curvature" of the smile.
    

### 4.2 Hagan's Approximation Formula

The widespread adoption of the SABR model is largely due to the highly accurate analytical approximation for the Black-Scholes implied volatility derived by Hagan et al. (2002). This formula allows practitioners to compute the entire volatility smile almost instantaneously without resorting to slow numerical methods like Monte Carlo or PDE solvers. This analytical tractability makes calibration extremely fast and efficient.28

### 4.3 Python Implementation: Calibrating to an Interest Rate Smile

We will now demonstrate how to calibrate the SABR model using the specialized and user-friendly `pysabr` library. The process is remarkably concise.

**Steps:**

1. **Setup:** Import `numpy` for numerical operations and the `Hagan2002LognormalSABR` class from the `pysabr` library.
    
2. **Define Market Data:** Specify the current forward rate, the option's time to expiry, a fixed value for β, and the market data as two arrays: one for strike prices and one for their corresponding lognormal implied volatilities.
    
3. **Calibrate:** Instantiate the `Hagan2002LognormalSABR` object with the fixed parameters (F,T,β). Then, simply call the `.fit()` method, passing the strike and volatility arrays. The library handles the underlying optimization internally.
    
4. **Analyze Results:** The `.fit()` method returns the calibrated parameters (α,ρ,ν). We can then use the calibrated model to generate a smooth smile and plot it against the discrete market data points to visually assess the quality of the fit.
    



```Python
import numpy as np
import matplotlib.pyplot as plt
from pysabr import Hagan2002LognormalSABR

# --- 1. & 2. Setup and Market Data ---
# Market data for a 10-year swaption
f = 2.527 / 100  # Forward swap rate
t = 10.0         # Time to expiry (10 years)
beta = 0.5       # Fixed beta parameter

# Market smile data: strikes and corresponding lognormal implied vols
strikes = np.array([-0.4729, 0.5271, 1.0271, 1.5271, 1.7771, 2.0271, 2.2771, 2.4021,
                    2.5271, 2.6521, 2.7771, 3.0271, 3.2771, 3.5271, 4.0271, 4.5271,
                    5.5271]) / 100
vols = np.array([19.64, 15.78, 14.30, 13.07, 12.55, 12.08,
                 11.69, 11.51, 11.36, 11.21, 11.09, 10.89,
                 10.75, 10.66, 10.62, 10.71, 11.10]) / 100

# --- 3. Calibrate ---
# Instantiate the SABR model. Note we don't provide alpha, rho, nu yet.
sabr_model = Hagan2002LognormalSABR(f=f, t=t, beta=beta)

# Fit the model to the market data to find the optimal alpha, rho, and nu
# The.fit() method returns the calibrated parameters
alpha_cal, rho_cal, volvol_cal = sabr_model.fit(strikes, vols)

# --- 4. Analyze Results ---
print("--- Calibrated SABR Parameters ---")
print(f"Alpha: {alpha_cal:.6f}")
print(f"Rho: {rho_cal:.6f}")
print(f"Volvol (Nu): {volvol_cal:.6f}")
print("-" * 34)

# Generate a fine grid of strikes for plotting a smooth smile
fine_strikes = np.linspace(strikes, strikes[-1], 100)

# Create a new SABR model instance with the calibrated parameters
calibrated_sabr = Hagan2002LognormalSABR(f=f, t=t, beta=beta, 
                                         alpha=alpha_cal, rho=rho_cal, volvol=volvol_cal)

# Compute the model's implied volatility on the fine grid
model_vols = [calibrated_sabr.lognormal_vol(k) for k in fine_strikes]

# Plot the results
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(10, 6))
plt.plot(strikes * 100, vols * 100, 'ro', label='Market Vols')
plt.plot(fine_strikes * 100, np.array(model_vols) * 100, 'b-', label='Calibrated SABR Smile')
plt.xlabel('Strike (%)')
plt.ylabel('Lognormal Implied Volatility (%)')
plt.title(f'SABR Model Calibration (F={f*100:.2f}%, T={t}Y, beta={beta})')
plt.axvline(f * 100, color='gray', linestyle='--', label='Forward Rate')
plt.legend()
plt.grid(True)
plt.show()
```

This example showcases the elegance and power of the SABR model and the `pysabr` library. With just a few lines of code, we can calibrate the model to a full volatility smile and visually confirm the excellent quality of the fit, a testament to the effectiveness of Hagan's formula.28

## 5. Case Study 3: Calibrating the Hull-White Short-Rate Model

We now turn our attention to the fixed-income world, where models must be calibrated not only to derivative prices but also to the underlying term structure of interest rates itself. The Hull-White model is a workhorse for this purpose.

### 5.1 Model Dynamics and Term Structure Fitting

The Hull-White one-factor model extends the Vasicek model by allowing the mean-reversion level to be time-dependent. This crucial feature enables the model to perfectly fit the initial term structure of interest rates observed in the market. The SDE for the instantaneous short rate, rt​, is 19:

$$dr_t = \left(\theta(t) - a r_t\right) dt + \sigma dW_t$$

The parameters to be calibrated from market option prices (such as interest rate caps, floors, or swaptions) are:

- a: The speed of mean reversion.
    
- σ: The volatility of the short rate.
    

The function θ(t) is not a free parameter to be calibrated. Instead, it is determined analytically to ensure the model's bond prices match the market's initial yield curve. This is a form of pre-calibration that is a necessary first step. The function θ(t) is uniquely determined by the market's initial instantaneous forward rate curve, fM(0,t), as follows 33:

$$\theta(t) = \frac{\partial f^M(0, t)}{\partial t} + a f^M(0, t) + \frac{\sigma^2}{2a}\left(1 - e^{-2a t}\right)
$$

By construction, any bond priced with the Hull-White model will exactly match the price implied by the initial yield curve, preventing arbitrage between the model and the bond market.

### 5.2 Analytical Pricing of Bonds and Caplets

The Hull-White model is an **affine term structure model**, which means that zero-coupon bond prices can be expressed in an exponential-affine form:

$$P(t,T)=A(t,T)e^{−B(t,T)r_t}$$​

where A(t,T) and B(t,T) are deterministic functions of time. This analytical tractability extends to European interest rate options like caps and floors, which can be priced in closed form. This is essential for enabling efficient calibration to the prices of these liquid instruments.33

### 5.3 Python Implementation: Calibrating to Swaption Volatilities

The following Python script demonstrates how to calibrate the Hull-White parameters a and σ to a set of market-quoted European swaption volatilities using `QuantLib`.

**Steps:**

1. **Setup:** Define the evaluation date, construct the initial yield curve from market rates, and specify the relevant interest rate index (e.g., `Euribor1Y`).
    
2. **Market Data:** Define the calibration instruments—a list of swaptions identified by their start date and length (e.g., a "1y5y" swaption starts in 1 year and has a 5-year tenor), along with their market-quoted volatilities.
    
3. **Setup Hull-White Model:** Instantiate a `HullWhite` model, linking it to the yield curve handle. Create a `JamshidianSwaptionEngine` as the pricing engine, which is suitable for one-factor affine models like Hull-White.
    
4. **Create Calibration Helpers:** Loop through the market data to create `SwaptionHelper` objects. Each helper links a market swaption (defined by its terms and volatility) to the model's pricing engine.
    
5. **Calibrate:** Instantiate the `LevenbergMarquardt` optimizer and `EndCriteria`. Call the `model.calibrate()` method, passing the list of helpers.
    
6. **Analyze Results:** Print the calibrated parameters a and σ. Generate a detailed report comparing the model's prices and implied volatilities to their market counterparts to assess the fit and calculate the cumulative error.
    



```Python
import QuantLib as ql
import math
from collections import namedtuple

# --- 1. Setup ---
today = ql.Date(15, ql.February, 2002)
settlement = ql.Date(19, ql.February, 2002)
ql.Settings.instance().evaluationDate = today

# Construct a flat yield curve for simplicity
term_structure = ql.YieldTermStructureHandle(
    ql.FlatForward(settlement, 0.04875825, ql.Actual365Fixed())
)
index = ql.Euribor1Y(term_structure)

# --- 2. Market Data ---
# Swaption data: start (years), length (years), volatility
CalibrationData = namedtuple("CalibrationData", "start, length, volatility")
data =

# --- 3. & 4. Setup Model and Create Helpers ---
# We need a pricing engine for the helpers
model = ql.HullWhite(term_structure)
engine = ql.JamshidianSwaptionEngine(model)

# Function to create swaption helpers from market data
def create_swaption_helpers(data, index, term_structure, engine):
    swaptions =
    fixed_leg_tenor = ql.Period(1, ql.Years)
    fixed_leg_daycounter = ql.Actual360()
    floating_leg_daycounter = ql.Actual360()
    for d in data:
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(d.volatility))
        helper = ql.SwaptionHelper(
            ql.Period(d.start, ql.Years),
            ql.Period(d.length, ql.Years),
            vol_handle,
            index,
            fixed_leg_tenor,
            fixed_leg_daycounter,
            floating_leg_daycounter,
            term_structure
        )
        helper.setPricingEngine(engine)
        swaptions.append(helper)
    return swaptions

swaption_helpers = create_swaption_helpers(data, index, term_structure, engine)

# --- 5. Calibrate ---
optimization_method = ql.LevenbergMarquardt(1.0e-8, 1.0e-8, 1.0e-8)
end_criteria = ql.EndCriteria(10000, 100, 1e-6, 1e-8, 1e-8)

model.calibrate(swaption_helpers, optimization_method, end_criteria)

# --- 6. Analyze Results ---
a_cal, sigma_cal = model.params()

print("--- Calibrated Hull-White Parameters ---")
print(f"Mean Reversion (a): {a_cal:.6f}")
print(f"Volatility (sigma): {sigma_cal:.6f}")
print("-" * 40)

def calibration_report(swaptions, data):
    print("\n--- Calibration Goodness-of-Fit ---")
    header = f"{'Swaption':>10} {'Market Vol':>12} {'Model Vol':>12} {'Market Price':>14} {'Model Price':>14} {'Rel Error (%)':>15}"
    print(header)
    print("=" * len(header))
    
    cum_err = 0.0
    for i, s in enumerate(swaptions):
        swaption_term = f"{data[i].start}y{data[i].length}y"
        model_price = s.modelValue()
        market_vol = data[i].volatility
        black_price = s.blackPrice(market_vol)
        
        implied_vol = 0.0
        try:
            implied_vol = s.impliedVolatility(model_price, 1e-5, 50, 0.0, 0.50)
        except RuntimeError:
            pass # Could not imply vol
            
        rel_error = (model_price / black_price - 1.0) * 100.0
        cum_err += (implied_vol / market_vol - 1.0)**2
        
        print(f"{swaption_term:>10} {market_vol:11.4f}% {implied_vol:11.4f}% {black_price:14.5f} {model_price:14.5f} {rel_error:14.2f}%")
        
    print("-" * len(header))
    print(f"Cumulative Squared Vol Error: {math.sqrt(cum_err):.6f}")

calibration_report(swaption_helpers, data)
```

This comprehensive example demonstrates the full workflow for calibrating an interest rate model in `QuantLib`. It shows how the model is simultaneously anchored to the yield curve (via the θ(t) construction) and to the volatility market (via calibration to swaptions), a dual requirement that is fundamental to interest rate modeling.19

## 6. Real-World Challenges and Model Risk

The preceding case studies provide a clean, textbook view of calibration. In practice, the process is fraught with challenges that require careful judgment and a deep understanding of the data and the models. This section moves beyond the mechanics to discuss the critical, practical issues that define professional quantitative work.

### 6.1 Data Filtering and Preparation

Raw market data is invariably noisy and contains information that can be misleading or detrimental to the calibration process. A robust pre-calibration filtering step is not optional; it is a mandatory part of any production-grade calibration workflow. The goal is to select a clean, representative set of instruments that reflect a true market consensus.

Common data filters include:

- **Liquidity Filters:** Options with zero or very low trading volume and open interest should be discarded. Their prices are stale or unreliable and do not represent a consensus market view.
    
- **Bid-Ask Spread Filters:** Options with excessively wide bid-ask spreads introduce ambiguity about the "true" market price. A common rule is to discard options where the spread is more than a certain percentage of the mid-price.
    
- **Moneyness and Sensitivity Filters:** Deep in-the-money (ITM) and deep out-of-the-money (OTM) options should be excluded. These options have very low sensitivity to volatility (low Vega) and changes in the underlying (low Gamma). Consequently, their implied volatilities are notoriously unstable and provide little useful information for calibrating volatility models. A common practice is to filter based on the option's Delta, keeping only options with, for example, an absolute delta between 0.1 and 0.9.
    
- **Near-Expiry Filters:** Options that are very close to expiration (e.g., less than one week) can exhibit erratic price behavior due to transaction costs and market microstructure effects. They are often excluded from calibration sets for longer-term models.
    

Implementing these filters in Python is straightforward using the `pandas` library to manipulate the dataframes of option data.36 The key is not the complexity of the code, but the financial justification for each filter.

### 6.2 The Ill-Posed Nature of Calibration

As we have alluded to, calibration is frequently an **ill-posed inverse problem**.3 This mathematical property has profound practical consequences:

- **Non-Convexity and Local Minima:** The objective function surface is rarely a simple, convex bowl. Instead, it is often a complex landscape riddled with multiple local minima and long, flat "valleys" where vastly different parameter sets yield nearly identical (and very low) calibration errors.10 This means that gradient-based optimizers are highly sensitive to their starting point; different initial guesses can lead to convergence at different local minima, yielding different "optimal" parameter sets.
    
- **Parameter Instability:** A direct consequence of the ill-posed nature is that the calibrated parameters can be extremely unstable. A small, economically insignificant change in market prices from one day to the next can cause the optimizer to jump from one valley to another, resulting in a dramatic shift in the calibrated parameters. This lack of stability is a major concern for risk management, as hedging strategies are highly sensitive to these parameters.10
    

### 6.3 Model Risk: The Peril of Recalibration

This brings us to the most subtle and critical challenge in calibration: **model risk**. The way models are used in practice often directly contradicts their underlying assumptions, creating a hidden but significant source of risk.14

The Core Contradiction

Models like Heston or Hull-White are formulated with the assumption that their parameters (κ,θ,σ, etc.) are constants. However, in practice, financial institutions recalibrate these models daily to force them to match the latest market prices. This act of changing the "constant" parameters every day is a fundamental contradiction of the model's own framework.14 This leads to a nuanced view of model risk, which can be broken down into different types:

- **Type 1 Model Risk (Calibration Error):** This is the most obvious risk—the inability of a chosen model to perfectly fit the market prices of all calibration instruments at a single point in time. It is the residual error of our optimization, `min L(θ)`. For example, the Black-Scholes model inherently has high Type 1 risk because it cannot reproduce the volatility smile.14
    
- **Type 2 Model Risk (Recalibration Risk):** This is the more insidious risk that arises from the instability of parameters over time. It is the risk associated with the fact that the parameters, which are assumed to be fixed, are in fact changing from day to day due to recalibration.14
    

The Conservation of Model Risk

A central, non-obvious principle is that these types of risk are often traded against each other. There is no free lunch. Consider a bank that initially uses the simple Black-Scholes model. It suffers from high Type 1 risk because its single volatility parameter cannot capture the market's smile, resulting in large calibration errors. To fix this, the bank upgrades to the more complex Heston model. The daily calibration error plummets, and Type 1 risk appears to be solved.

However, the bank may soon notice that the calibrated Heston parameters are wildly unstable. The mean-reversion speed, κ, might be 1.5 on Monday, 10.2 on Tuesday, and 0.8 on Wednesday. While the model fits the market perfectly each day, the parameters that achieve this fit are erratic. This creates enormous Type 2 risk. A hedge ratio calculated with κ=10.2 is vastly different from one calculated with κ=0.8. The risk management and hedging systems become unreliable.

In this scenario, the bank has not eliminated model risk; it has merely **transformed it** from a visible, easily quantifiable calibration error (Type 1) into a more subtle but equally dangerous parameter instability (Type 2). The total, aggregate model risk may not have decreased at all. This suggests that the goal of a practitioner should not be to chase zero calibration error at all costs. A truly robust calibration procedure is one that finds a reasonable balance between goodness-of-fit and parameter stability, often through the judicious use of regularization or by consciously choosing a simpler, more stable model, even if it has a slightly higher calibration error.14

## 7. Capstone Project: Calibrating the Heston Model to S&P 500 (SPX) Option Data

This capstone project synthesizes all the concepts from the chapter into a complete, end-to-end workflow. We will perform a realistic calibration of the Heston model to a set of S&P 500 (SPX) index options, analyze the results, and critically evaluate the process.

### 7.1 Project Objective

The goal of this project is to calibrate the five parameters of the Heston stochastic volatility model (κ,θ,ξ,ρ,v0​) using a single day's market data for SPX options. We will then analyze the quality of the calibration, interpret the economic meaning of the parameters, and investigate the stability and model risk inherent in the results.

### 7.2 Data Acquisition and Preparation

Acquiring high-quality, historical options data is a significant challenge for individual researchers.

- **Data Sources:** Professional traders and institutions subscribe to data vendors like the **CBOE DataShop**, **AlgoSeek**, or **ORATS**, which provide clean, comprehensive historical data including trades, quotes, and greeks.38 Academic researchers often gain access through university subscriptions to services like
    
    **Wharton Research Data Services (WRDS)**, which hosts datasets like OptionMetrics.41 Free sources like the
    
    `yfinance` Python library are excellent for stock data but offer limited, often delayed, and potentially erroneous options data, making them unsuitable for rigorous calibration.43
    
- **Data for this Project:** For this project, we will work with a simulated but realistic data file named `spx_options_2023-10-27.csv`. This file mimics the structure of data from a professional source and contains the following columns for SPX options on October 27, 2023: `Date`, `Expiry`, `Strike`, `Type` (Call/Put), `ImpliedVolatility`, `Bid`, `Ask`, `Volume`, `OpenInterest`. We will also need the underlying SPX index level and the relevant risk-free rate for that day.
    

The first step is to acquire the necessary market parameters and then filter the raw options data.



```Python
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# --- Data Acquisition ---
# Define the date for our analysis
analysis_date_str = '2023-10-27'
analysis_date = datetime.strptime(analysis_date_str, '%Y-%m-%d')

# Get the SPX closing price for the analysis date
spx_ticker = yf.Ticker('^SPX')
spx_hist = spx_ticker.history(start=analysis_date_str, end='2023-10-28')
spot_price = spx_hist['Close'].iloc

# Get the risk-free rate (using 3-month T-bill as a proxy)
rf_ticker = yf.Ticker('^IRX')
rf_hist = rf_ticker.history(start=analysis_date_str, end='2023-10-28')
risk_free_rate = rf_hist['Close'].iloc / 100.0

print(f"Analysis Date: {analysis_date_str}")
print(f"SPX Spot Price: {spot_price:.2f}")
print(f"Risk-Free Rate: {risk_free_rate:.4f}")

# Load the simulated options data
# In a real scenario, this would come from a data vendor.
# For this project, assume 'spx_options_2023-10-27.csv' is in the working directory.
try:
    raw_options_df = pd.read_csv('spx_options_2023-10-27.csv')
    raw_options_df['Expiry'] = pd.to_datetime(raw_options_df['Expiry'])
    raw_options_df = pd.to_datetime(raw_options_df)
    print(f"\nLoaded {len(raw_options_df)} raw option contracts.")
except FileNotFoundError:
    print("\nError: 'spx_options_2023-10-27.csv' not found. Please create this file or download it.")
    # Create a dummy file for demonstration if it doesn't exist
    dummy_data = {
        'Date': ['2023-10-27']*5, 'Expiry': ['2023-11-17']*5, 'Strike': ,
        'Type': ['Call']*5, 'ImpliedVolatility': [0.20, 0.18, 0.16, 0.15, 0.14],
        'Bid': , 'Ask': ,
        'Volume': , 'OpenInterest': 
    }
    raw_options_df = pd.DataFrame(dummy_data)
    raw_options_df.to_csv('spx_options_2023-10-27.csv', index=False)
    print("Created a dummy 'spx_options_2023-10-27.csv' file.")


# --- Data Filtering ---
print("\nFiltering options data...")
filtered_df = raw_options_df.copy()

# 1. Liquidity Filter: Volume and Open Interest
filtered_df = filtered_df[filtered_df['Volume'] > 10]
filtered_df = filtered_df[filtered_df['OpenInterest'] > 10]
print(f"After liquidity filter: {len(filtered_df)} contracts remaining.")

# 2. Bid-Ask Spread Filter
filtered_df['MidPrice'] = (filtered_df + filtered_df['Ask']) / 2
filtered_df = filtered_df['Ask'] - filtered_df
# Remove options with zero bid or where spread is too wide
filtered_df = filtered_df > 0]
filtered_df = filtered_df / filtered_df['MidPrice'] < 0.5] # Spread < 50% of mid
print(f"After bid-ask spread filter: {len(filtered_df)} contracts remaining.")

# 3. Time to Expiry Filter
filtered_df = (filtered_df['Expiry'] - filtered_df).dt.days
filtered_df = filtered_df > 7] # More than 7 days to expiry
print(f"After expiry filter (>7 days): {len(filtered_df)} contracts remaining.")

# 4. Moneyness Filter (keep only OTM and near-ATM options)
# We will focus on OTM calls and OTM puts as they are most liquid for volatility trading
otm_calls = filtered_df == 'Call') & (filtered_df >= spot_price)]
otm_puts = filtered_df == 'Put') & (filtered_df <= spot_price)]
filtered_df = pd.concat([otm_calls, otm_puts])
print(f"After moneyness filter (OTM only): {len(filtered_df)} contracts remaining.")

# Final clean dataset
filtered_df = filtered_df.sort_values(by=).reset_index(drop=True)
print(f"\nFinal filtered dataset contains {len(filtered_df)} option contracts.")
print(filtered_df.head())
```

### 7.3 Calibration Implementation (Python)

With a clean dataset, we now proceed with the calibration using `QuantLib`. The script will select a single maturity from our filtered data to calibrate against the volatility smile for that expiry.



```Python
import QuantLib as ql

# --- Setup QuantLib Environment ---
ql_calculation_date = ql.Date(analysis_date.day, analysis_date.month, analysis_date.year)
ql.Settings.instance().evaluationDate = ql_calculation_date
ql_calendar = ql.UnitedStates(m=ql.UnitedStates.NYSE)
ql_day_count = ql.Actual365Fixed()

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
rate_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(ql_calculation_date, risk_free_rate, ql_day_count)
)
# Assuming no dividends for SPX index
div_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(ql_calculation_date, 0.0, ql_day_count)
)

# --- Select Calibration Instruments ---
# Choose a single expiry to calibrate to. Let's pick one around 30 days out.
target_expiry_date = filtered_df['Expiry'].unique() # Example: pick the 3rd available expiry
calibration_set = filtered_df[filtered_df['Expiry'] == target_expiry_date]
print(f"\nCalibrating to options expiring on: {pd.to_datetime(target_expiry_date).strftime('%Y-%m-%d')}")
print(f"Using {len(calibration_set)} options for calibration.")

# --- Setup Heston Model and Helpers ---
# Initial guess for Heston parameters
v0 = 0.1**2; kappa = 1.0; theta = 0.15**2; rho = -0.6; sigma = 0.3
process = ql.HestonProcess(rate_handle, div_handle, spot_handle, v0, kappa, theta, sigma, rho)
model = ql.HestonModel(process)
engine = ql.AnalyticHestonEngine(model)

helpers =
for index, row in calibration_set.iterrows():
    tte_days = row
    strike = row
    vol = row['ImpliedVolatility']
    
    period = ql.Period(tte_days, ql.Days)
    option_type = ql.Option.Call if row == 'Call' else ql.Option.Put
    
    helper = ql.HestonModelHelper(
        period,
        ql_calendar,
        spot_price,
        strike,
        ql.QuoteHandle(ql.SimpleQuote(vol)),
        rate_handle,
        div_handle
    )
    helper.setPricingEngine(engine)
    helpers.append(helper)

# --- Perform Calibration ---
print("\nStarting Heston model calibration...")
lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
end_criteria = ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8)
model.calibrate(helpers, lm, end_criteria)
print("Calibration finished.")

# --- Store and Display Results ---
theta_cal, kappa_cal, sigma_cal, rho_cal, v0_cal = model.params()
rmse_error = model.calibrationError()

# Final calibrated parameter table
calibrated_params = {
    'Parameter': ['$v_0$', '$\\kappa$', '$\\theta$', '$\\xi$', '$\\rho$'],
    'Calibrated Value': [v0_cal, kappa_cal, theta_cal, sigma_cal, rho_cal],
    'Interpretation':
}
params_df = pd.DataFrame(calibrated_params)

fit_metrics = {
    'Metric':,
    'Value': [f"{rmse_error:.6f}", ""],
    'Description':
}
metrics_df = pd.DataFrame(fit_metrics)

# Calculate model volatilities and errors
model_vols, market_vols, strikes_out =,,
for i, helper in enumerate(helpers):
    market_vols.append(calibration_set.iloc[i]['ImpliedVolatility'])
    strikes_out.append(calibration_set.iloc[i])
    model_price = helper.modelValue()
    try:
        model_vols.append(helper.impliedVolatility(model_price, 1e-5, 500, 1e-8, 2.0))
    except RuntimeError:
        model_vols.append(np.nan)

avg_abs_err = np.nanmean(np.abs(np.array(model_vols) - np.array(market_vols)) / np.array(market_vols)) * 100
metrics_df.loc[1, 'Value'] = f"{avg_abs_err:.4f}"

print("\n--- Calibrated Heston Parameters ---")
print(params_df.to_string(index=False))
print("\n--- Goodness-of-Fit ---")
print(metrics_df.to_string(index=False))

# Plot the result
plt.figure(figsize=(12, 7))
plt.plot(strikes_out, np.array(market_vols) * 100, 'o', label='Market Implied Vols')
plt.plot(strikes_out, np.array(model_vols) * 100, 'x-', label='Calibrated Heston Vols')
plt.title(f"Heston Calibration to SPX Smile for Expiry {pd.to_datetime(target_expiry_date).strftime('%Y-%m-%d')}")
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility (%)")
plt.legend()
plt.grid(True)
plt.show()
```

### 7.4 Analysis and Interpretation (Questions & Responses)

This final section guides the reader to think critically about the project's results, reinforcing the chapter's key lessons on model risk and interpretation.

**Q1: How well did the calibrated model fit the market data across different strikes and maturities? Where are the largest errors, and why might they occur there?**

**A:** The plot generated by the script visually demonstrates the goodness-of-fit for the selected maturity. Typically, a calibrated Heston model will fit the data well near the at-the-money strike but will show larger errors at the extreme ends of the smile—the deep out-of-the-money puts and calls. This is a known limitation of the Heston model. While it introduces stochastic volatility, its assumptions (e.g., constant correlation, single volatility factor) are not always sufficient to capture the very steep "skew" or "smirk" observed in equity index options markets, especially for short-dated options. The largest errors often occur in the "wings" of the smile because the market prices in a higher probability of extreme events (tail risk) than the Heston model can accommodate.

**Q2: What do the calibrated parameters (κ,θ,ξ,ρ,v0​) imply about the market's view on volatility dynamics on this specific day?**

**A:** The calibrated parameters provide a snapshot of the market's expectations.

- **ρ (Correlation):** A strongly negative value (e.g., -0.6 to -0.8) is expected for SPX options, confirming the market's pricing of the leverage effect: as the index falls, volatility is expected to rise.
    
- **κ (Mean Reversion Speed):** A high value of κ (e.g., > 5) suggests the market expects current volatility levels to revert to the long-term mean relatively quickly. A low value suggests persistence in the current volatility regime.
    
- **θ (Long-Term Variance):** The square root of θ gives the long-term implied volatility. This can be compared to the long-term average of a volatility index like the VIX. If θ![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702%0Ac-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14%0Ac0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54%0Ac44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10%0As173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429%0Ac69,-144,104.5,-217.7,106.5,-221%0Al0 -0%0Ac5.3,-9.3,12,-14,20,-14%0AH400000v40H845.2724%0As-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7%0Ac-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z%0AM834 80h400000v40h-400000z"></path></svg>)​ is significantly higher than the historical VIX average, it suggests the market is pricing in a structurally higher volatility regime for the future.
    
- **ξ (Vol of Vol):** A high value for ξ indicates that the market expects volatility itself to be very volatile, leading to a more convex or "smiley" shape in the volatility curve.
    

**Q3: How stable are the calibrated parameters if we slightly change the set of options used for calibration (e.g., by tightening the moneyness filter)? What does this tell us about model risk?**

**A:** This question probes the ill-posed nature of the problem. To answer it, one would re-run the calibration script using a slightly different `calibration_set`—for example, by excluding options with the lowest and highest strikes. It is highly likely that the new calibration would yield a noticeably different set of "optimal" parameters, particularly for κ and ξ, which are known to be less stable.

This demonstrates **Type 2 model risk** in action. The fact that a minor, justifiable change in the input data leads to a significant change in the inferred parameters shows that these parameters are not absolute economic truths. They are, to a large extent, artifacts of our specific calibration choices (the data set, the loss function, the optimizer). This instability is a critical risk for any hedging program that relies on these parameters, as the hedge ratios could change significantly based on which options were included in the calibration that day.

**Q4: Using the calibrated model, how would you price an exotic option (e.g., a barrier option) that was not used in the calibration set? How would you assess the reliability of this price?**

**A:** Once the Heston model is calibrated, its parameters (κ,θ,ξ,ρ,v0​) are fixed. An exotic derivative, such as a barrier option, would then be priced using these parameters, typically via a Monte Carlo simulation of the Heston SDEs.

However, the reliability of this price is highly questionable and must be assessed with caution. The price is only as good as the model and its calibration.

- **Assess Sensitivity:** The first step is to understand which features of the volatility surface the exotic option is most sensitive to. A barrier option, for instance, is highly sensitive to the volatility dynamics near the barrier level. If the barrier is placed deep out-of-the-money, in a region where our Heston calibration had large errors (as per Q1), the resulting price for the barrier option is likely to be unreliable.
    
- **Acknowledge Model Risk:** A robust approach would be to explicitly acknowledge the parameter uncertainty demonstrated in Q3. Instead of reporting a single price for the exotic, a practitioner should calculate a _range_ of plausible prices. This could be done by pricing the exotic using several different parameter sets obtained from slightly different but equally valid calibrations (e.g., using different data filters or different days' data). Presenting a price range, rather than a single deceptive number, is a more honest and robust way to communicate the price while accounting for the inherent model risk in the calibration process. This moves the practice from pure pricing to robust risk management.
## References
**

1. Deep Dive: The Ultimate Guide to Model Calibration, acessado em julho 7, 2025, [https://www.numberanalytics.com/blog/ultimate-model-calibration-guide](https://www.numberanalytics.com/blog/ultimate-model-calibration-guide)
    
2. Calibration in Computational Finance - Number Analytics, acessado em julho 7, 2025, [https://www.numberanalytics.com/blog/ultimate-guide-to-calibration-in-computational-finance](https://www.numberanalytics.com/blog/ultimate-guide-to-calibration-in-computational-finance)
    
3. (PDF) Model Calibration - ResearchGate, acessado em julho 7, 2025, [https://www.researchgate.net/publication/227559301_Model_Calibration](https://www.researchgate.net/publication/227559301_Model_Calibration)
    
4. Calibration and Hedging in Finance - DiVA portal, acessado em julho 7, 2025, [http://www.diva-portal.org/smash/get/diva2:764597/SUMMARY01.pdf](http://www.diva-portal.org/smash/get/diva2:764597/SUMMARY01.pdf)
    
5. Mastering Calibration Techniques - Number Analytics, acessado em julho 7, 2025, [https://www.numberanalytics.com/blog/mastering-calibration-techniques-in-computational-finance](https://www.numberanalytics.com/blog/mastering-calibration-techniques-in-computational-finance)
    
6. Market and historical volatility - Wholesale Banking - Société Générale, acessado em julho 7, 2025, [https://wholesale.banking.societegenerale.com/en/news-insights/glossary/market-and-historical-volatility/](https://wholesale.banking.societegenerale.com/en/news-insights/glossary/market-and-historical-volatility/)
    
7. Implied Volatility vs. Historical Volatility: What's the Difference?, acessado em julho 7, 2025, [https://www.investopedia.com/articles/investing-strategy/071616/implied-vs-historical-volatility-main-differences.asp](https://www.investopedia.com/articles/investing-strategy/071616/implied-vs-historical-volatility-main-differences.asp)
    
8. Implied Volatility vs Historical Volatility | Blog - Option Samurai, acessado em julho 7, 2025, [https://optionsamurai.com/blog/implied-volatility-vs-historical-volatility-in-options-trading-unveiling-patterns-and-insights/](https://optionsamurai.com/blog/implied-volatility-vs-historical-volatility-in-options-trading-unveiling-patterns-and-insights/)
    
9. Computational-Finance/6. Model Calibration.ipynb at main - GitHub, acessado em julho 7, 2025, [https://github.com/andreachello/Computational-Finance/blob/main/6.%20Model%20Calibration.ipynb](https://github.com/andreachello/Computational-Finance/blob/main/6.%20Model%20Calibration.ipynb)
    
10. The calibration conundrum - City Research Online, acessado em julho 7, 2025, [https://openaccess.city.ac.uk/id/eprint/33834/1/C5_Calib_Ballotta.pdf](https://openaccess.city.ac.uk/id/eprint/33834/1/C5_Calib_Ballotta.pdf)
    
11. Parameter calibration of stochastic volatility Heston's model: Constrained optimization vs. differential evolution - Dialnet, acessado em julho 7, 2025, [https://dialnet.unirioja.es/descarga/articulo/8387459.pdf](https://dialnet.unirioja.es/descarga/articulo/8387459.pdf)
    
12. Heston Model calibration : r/quant - Reddit, acessado em julho 7, 2025, [https://www.reddit.com/r/quant/comments/10q9r99/heston_model_calibration/](https://www.reddit.com/r/quant/comments/10q9r99/heston_model_calibration/)
    
13. FINANCIAL MODEL CALIBRATION USING CONSISTENCY HINTS - Yaser S. Abu-Mostafa, acessado em julho 7, 2025, [https://work.caltech.edu/paper/01consis.pdf](https://work.caltech.edu/paper/01consis.pdf)
    
14. Quantifying the Model Risk Inherent in the Calibration and Recalibration of Option Pricing Models - MDPI, acessado em julho 7, 2025, [https://www.mdpi.com/2227-9091/9/1/13](https://www.mdpi.com/2227-9091/9/1/13)
    
15. HESTON MODEL CALIBRATION USING QUANTLIB IN PYTHON | by Aaron De la Rosa, acessado em julho 7, 2025, [https://medium.com/@aaron_delarosa/heston-model-calibration-using-quantlib-in-python-0089516430ef](https://medium.com/@aaron_delarosa/heston-model-calibration-using-quantlib-in-python-0089516430ef)
    
16. A Brief Description of the Levenberg-Marquardt Algorithm Implemened by levmar, acessado em julho 7, 2025, [https://www.researchgate.net/publication/239328019_A_Brief_Description_of_the_Levenberg-Marquardt_Algorithm_Implemened_by_levmar](https://www.researchgate.net/publication/239328019_A_Brief_Description_of_the_Levenberg-Marquardt_Algorithm_Implemened_by_levmar)
    
17. Levenberg–Marquardt algorithm - Wikipedia, acessado em julho 7, 2025, [https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
    
18. The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems - Duke People, acessado em julho 7, 2025, [https://people.duke.edu/~hpgavin/lm.pdf](https://people.duke.edu/~hpgavin/lm.pdf)
    
19. Short Interest Rate Model Calibration in QuantLib Python - G B, acessado em julho 7, 2025, [http://gouthamanbalaraman.com/blog/short-interest-rate-model-calibration-quantlib.html](http://gouthamanbalaraman.com/blog/short-interest-rate-model-calibration-quantlib.html)
    
20. Calibrating the Heston Model with Deep Differential Networks - arXiv, acessado em julho 7, 2025, [https://arxiv.org/html/2407.15536v1](https://arxiv.org/html/2407.15536v1)
    
21. Calibrate the Heston Model Faster Using Derivative-Free Optimization Techniques - nAG, acessado em julho 7, 2025, [https://nag.com/insights/optcorner-calibrate-the-heston-model-faster-using-derivative-free-optimization-techniques/](https://nag.com/insights/optcorner-calibrate-the-heston-model-faster-using-derivative-free-optimization-techniques/)
    
22. A gradient-based calibration method for the Heston model - Taylor & Francis Online, acessado em julho 7, 2025, [https://www.tandfonline.com/doi/full/10.1080/00207160.2024.2353189](https://www.tandfonline.com/doi/full/10.1080/00207160.2024.2353189)
    
23. The Heston Model: Defined & Explained with Calculations - SoFi, acessado em julho 7, 2025, [https://www.sofi.com/learn/content/heston-model/](https://www.sofi.com/learn/content/heston-model/)
    
24. Heston model - Wikipedia, acessado em julho 7, 2025, [https://en.wikipedia.org/wiki/Heston_model](https://en.wikipedia.org/wiki/Heston_model)
    
25. Estimating Option Prices with Heston's Stochastic Volatility Model, acessado em julho 7, 2025, [https://www.valpo.edu/mathematics-statistics/files/2015/07/Estimating-Option-Prices-with-Heston%E2%80%99s-Stochastic-Volatility-Model.pdf](https://www.valpo.edu/mathematics-statistics/files/2015/07/Estimating-Option-Prices-with-Heston%E2%80%99s-Stochastic-Volatility-Model.pdf)
    
26. noa/docs/quant/heston_model.ipynb at master - GitHub, acessado em julho 7, 2025, [https://github.com/grinisrit/noa/blob/master/docs/quant/heston_model.ipynb](https://github.com/grinisrit/noa/blob/master/docs/quant/heston_model.ipynb)
    
27. Pricing Models - QuantLib-Python - Read the Docs, acessado em julho 7, 2025, [https://quantlib-python-docs.readthedocs.io/en/latest/pricing_models.html](https://quantlib-python-docs.readthedocs.io/en/latest/pricing_models.html)
    
28. ynouri/pysabr: SABR model Python implementation - GitHub, acessado em julho 7, 2025, [https://github.com/ynouri/pysabr](https://github.com/ynouri/pysabr)
    
29. Calibrate a SABR model? - Quantitative Finance Stack Exchange, acessado em julho 7, 2025, [https://quant.stackexchange.com/questions/43341/calibrate-a-sabr-model](https://quant.stackexchange.com/questions/43341/calibrate-a-sabr-model)
    
30. SABR-calibration/SABR_calibration.py at master - GitHub, acessado em julho 7, 2025, [https://github.com/W-J-Trenberth/SABR-calibration/blob/master/SABR_calibration.py](https://github.com/W-J-Trenberth/SABR-calibration/blob/master/SABR_calibration.py)
    
31. Deep calibration of SABR stochastic volatility - Erasmus University Thesis Repository, acessado em julho 7, 2025, [https://thesis.eur.nl/pub/59437/Hugo_Stuijt_433154_thesis_final_version.pdf](https://thesis.eur.nl/pub/59437/Hugo_Stuijt_433154_thesis_final_version.pdf)
    
32. pysabr/pysabr/black.py at master · ynouri/pysabr - GitHub, acessado em julho 7, 2025, [https://github.com/ynouri/pysabr/blob/master/pysabr/black.py](https://github.com/ynouri/pysabr/blob/master/pysabr/black.py)
    
33. MAFS525 – Computational Methods for Pricing Structured Prod ..., acessado em julho 7, 2025, [https://www.math.hkust.edu.hk/~maykwok/courses/MAFS525/Topic4_4.pdf](https://www.math.hkust.edu.hk/~maykwok/courses/MAFS525/Topic4_4.pdf)
    
34. Hull White model - QuantPie, acessado em julho 7, 2025, [https://www.quantpie.co.uk/srm/hull_white_sr.php](https://www.quantpie.co.uk/srm/hull_white_sr.php)
    
35. Hull White Term Structure Simulations with QuantLib Python - G B - Gouthaman Balaraman, acessado em julho 7, 2025, [http://gouthamanbalaraman.com/blog/hull-white-simulation-quantlib-python.html](http://gouthamanbalaraman.com/blog/hull-white-simulation-quantlib-python.html)
    
36. 3. Filtering Data — Basic Analytics in Python, acessado em julho 7, 2025, [https://www.sfu.ca/~mjbrydon/tutorials/BAinPy/04_filter.html](https://www.sfu.ca/~mjbrydon/tutorials/BAinPy/04_filter.html)
    
37. Mastering Data Filtration in Python | by Abhijeet Dwivedi - Medium, acessado em julho 7, 2025, [https://medium.com/@dwivedi.abhijeet1301/mastering-data-filtration-in-python-1c0cb099fe03](https://medium.com/@dwivedi.abhijeet1301/mastering-data-filtration-in-python-1c0cb099fe03)
    
38. Market at a Glance Data Integration | API Subscription - Cboe DataShop, acessado em julho 7, 2025, [https://datashop.cboe.com/market-data-api](https://datashop.cboe.com/market-data-api)
    
39. US Index Options - QuantConnect.com, acessado em julho 7, 2025, [https://www.quantconnect.com/data/algoseek-us-index-options](https://www.quantconnect.com/data/algoseek-us-index-options)
    
40. Historical Options Data - Near End-of-day (Since 2007) - ORATS, acessado em julho 7, 2025, [https://orats.com/near-eod-data](https://orats.com/near-eod-data)
    
41. Finding and using financial data at Princeton University: Options - Research Guides, acessado em julho 7, 2025, [https://libguides.princeton.edu/econ-finance/options](https://libguides.princeton.edu/econ-finance/options)
    
42. Wharton Research Data Services - University of Pennsylvania, acessado em julho 7, 2025, [https://wrds-www.wharton.upenn.edu/](https://wrds-www.wharton.upenn.edu/)
    
43. More In-Depth Option Data · ranaroussi yfinance · Discussion #2078 - GitHub, acessado em julho 7, 2025, [https://github.com/ranaroussi/yfinance/discussions/2078](https://github.com/ranaroussi/yfinance/discussions/2078)
    
44. Get Free Options Data with Python: Yahoo finance & Pandas Tutorial - CodeArmo, acessado em julho 7, 2025, [https://www.codearmo.com/python-tutorial/options-trading-getting-options-data-yahoo-finance](https://www.codearmo.com/python-tutorial/options-trading-getting-options-data-yahoo-finance)
    

**