### 1. Introduction: The Failure of Constant Volatility

#### 1.1 The Black-Scholes-Merton (BSM) World: A Flawed Masterpiece

The Black-Scholes-Merton (BSM) model represents a cornerstone of modern financial theory, providing an elegant and powerful framework for pricing European options. Its derivation, grounded in the principle of no-arbitrage and dynamic hedging, was a monumental achievement. The model rests on a set of simplifying assumptions about the market:

- The price of the underlying asset follows a geometric Brownian motion, implying that continuously compounded returns are normally distributed.2
    
- The risk-free interest rate is known and constant over the life of the option.4
    
- Trading is continuous, and there are no transaction costs or taxes.5
    
- Short selling of the underlying asset is permitted without restriction.4
    
- The underlying asset pays no dividends during the option's life (though this can be adjusted for).3
    

Crucially, the BSM model makes a pivotal assumption that the volatility of the underlying asset's returns, denoted by the parameter σ, is constant and known throughout the option's life.5 Among the model's inputs—spot price, strike price, time to maturity, risk-free rate, and volatility—volatility is the only parameter that cannot be directly observed from the market. This particular assumption, while mathematically convenient, proves to be the model's most significant departure from reality and is the primary source of the phenomena explored in this chapter.

#### 1.2 Implied Volatility: The Market's Verdict

Given that the BSM model's assumption of constant volatility does not hold in practice, market participants developed a way to reconcile the theoretical model with observed market prices. This reconciliation is achieved through the concept of **implied volatility (IV)**. Implied volatility is defined as the value of the volatility parameter, σ, that, when input into the BSM pricing formula, yields a theoretical price equal to the option's current market price.7 In essence, it is the market's consensus on the expected future volatility of the underlying asset, "implied" by the option's price. It serves as a "plug figure" that forces the model to match reality.7

For a European call option, the BSM price, C, is given by:

$$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$

where:

- $​d_1 = \frac{\ln(S_0 / K) + \left( r + \frac{\sigma^2}{2} \right) T}{\sigma \sqrt{T}}$
    
- $d_2 = d_1 - \sigma \sqrt{T}$​
    
- S0​ is the current stock price.
    
- K is the strike price.
    
- r is the risk-free interest rate.
    
- T is the time to maturity.
    
- σ is the volatility.
    
- N(⋅) is the cumulative distribution function of the standard normal distribution.
    

As this formula cannot be analytically inverted to solve for σ as a function of the other variables, its calculation requires numerical root-finding algorithms, such as the Newton-Raphson method or bisection search.9 This computational step is a fundamental routine for any options trader or quant, turning market prices into a standardized measure of expected volatility.

#### 1.3 The Crash of 1987 and the Birth of the Smile

If the BSM model and its assumptions were perfectly correct, the implied volatility for all options on the same underlying asset with the same expiration date would be identical, regardless of their strike price. A plot of implied volatility against strike price would yield a flat, horizontal line. For a significant period, particularly in American equity markets before 1987, this was largely the case; the volatility surface was observed to be relatively flat.8

This paradigm was shattered by the stock market crash of October 19, 1987, known as "Black Monday," when the Dow Jones Industrial Average plummeted by over 22% in a single day.12 This event fundamentally and permanently altered investor psychology and risk perception. Before the crash, the log-normal distribution assumed by BSM, which assigns very low probabilities to such extreme events, was more widely accepted. Afterward, traders recognized that catastrophic "tail events" were more probable than the model suggested.11

This led to a persistent and significant demand for downside protection, primarily through the purchase of out-of-the-money (OTM) put options. This heightened demand, driven by a collective "crash-o-phobia," drove up the prices of these OTM puts.7 When these higher market prices were fed back into the BSM formula to calculate implied volatility, they produced much higher IV values for low-strike options compared to at-the-money (ATM) or in-the-money (ITM) options. This phenomenon created the persistent, downward-sloping pattern in equity markets known as the

**volatility skew** or **smirk**.12

This establishes a clear causal link: a significant market event (the 1987 crash) triggered a durable shift in investor sentiment (fear of tail risk), which created a structural supply/demand imbalance for options (high demand for OTM puts). This imbalance directly influenced market prices, and the discrepancy between these prices and the BSM model's theoretical values manifested as a non-flat implied volatility structure.16

The following table summarizes the key contradictions between the BSM model's assumptions and the realities of the market, which necessitate the more advanced modeling techniques discussed later in this chapter.

|BSM Assumption|Market Reality|Key Implication|
|---|---|---|
|**Constant Volatility**|Volatility varies by strike and time, creating a **volatility smile/skew**.8|The BSM model systematically misprices options that are not at-the-money.|
|**Lognormal Returns**|Asset returns exhibit **"fat tails"** and **skewness**.5|The BSM model underestimates the probability of large price moves (tail risk).|
|**Continuous Price Path**|Asset prices can experience sudden, discontinuous **jumps**.4|Delta hedging, which relies on continuous price moves, can fail catastrophically during a market jump.|
|**Frictionless Markets**|Markets have **transaction costs** and **bid-ask spreads**.6|Continuous hedging is both practically impossible and prohibitively expensive.|

### 2. Characterizing and Visualizing the Volatility Surface

#### 2.1 Defining the Smile, Skew, and Smirk

The term "volatility smile" is often used as a general catch-all, but more specific terms describe the distinct patterns observed across different markets. These patterns are graphical representations of implied volatility plotted against strike price for options with the same expiration date.7

- **Volatility Smile:** This is a symmetric, U-shaped curve where implied volatility is lowest for at-the-money (ATM) options and increases as options move further in-the-money (ITM) or out-of-the-money (OTM).11 This shape suggests the market anticipates a large price move but is uncertain about the direction. It is most commonly observed in foreign exchange (FX) markets and for near-term equity options.7
    
- **Volatility Skew (or "Smirk"):** This is an asymmetric or lopsided curve. The shape of the skew provides insight into the market's directional bias.12
    
    - **Reverse Skew (Negative Skew):** This is the most common pattern in equity and equity index markets. Implied volatility is highest for low-strike options (OTM puts) and slopes downward as the strike price increases.8 This "smirk" reflects the high premium investors are willing to pay for downside protection (puts), indicating a greater fear of market crashes than unexpected rallies.16
        
    - **Forward Skew (Positive Skew):** In this case, implied volatility increases with the strike price, being higher for OTM calls than OTM puts. This pattern is often seen in commodity markets.15 It suggests that market participants are more concerned about sudden price spikes (e.g., due to a supply disruption or demand shock) than price drops.
        

#### 2.2 The Volatility Surface: A Multi-dimensional View

The 2D plot of a smile or skew only captures volatility for a single expiration date. A more complete representation is the **implied volatility surface**, a three-dimensional plot that visualizes implied volatility as a function of both strike price (or moneyness, defined as K/S0​) and time to maturity.8

The volatility surface provides a rich, consolidated view of the market's expectations for future price movements across all traded options on an underlying asset.18 Its shape contains valuable information for traders and risk managers:

- **The Strike Axis (Skew/Smile):** As discussed, the shape along the strike axis reveals the market's perception of tail risk and directional bias for a given maturity.15
    
- **The Time Axis (Term Structure):** The shape along the time-to-maturity axis shows how volatility expectations change over different time horizons. A downward-sloping term structure (higher short-term IV) might indicate immediate uncertainty, while an upward-sloping (contango) structure is more typical. Kinks or bumps in the term structure can often be linked to specific future events, such as a company's earnings announcement or a scheduled central bank policy meeting, as traders bid up the price of options that expire around those dates.18
    

Analyzing the dynamics of the entire surface—how it twists, shifts, and changes shape over time—offers far deeper insights into evolving market sentiment than observing a single implied volatility value.14

#### 2.3 Python Implementation: Building and Plotting the Volatility Surface

This section provides a practical guide to fetching options data, calculating implied volatility, and visualizing the volatility surface using Python.

**Step 1: Data Acquisition**

Obtaining high-quality, historical options data can be challenging. For academic and professional use, sources like the CBOE DataShop or commercial data vendors are standard.20 However, for illustrative purposes, we can use the

`yfinance` library to fetch current option chain data for a liquid, publicly traded asset like the SPDR S&P 500 ETF (SPY), which serves as a good proxy for the broader market.18 It is important to note that

`yfinance` has limitations, particularly for accessing deep historical data or data for cash-settled indices like SPX.24

**Step 2: Calculating Implied Volatility**

Once we have the market price for each option, we must use a numerical solver to find the implied volatility. The `py_vollib` library offers a fast and accurate implementation based on Peter Jäckel's "Let's Be Rational" algorithm, which is highly efficient.27 The

`py_vollib_vectorized` package provides an even faster way to perform this calculation on an entire dataset at once.28

**Step 3 & 4: Plotting the Smile and Surface**

With the implied volatilities calculated, we can use `matplotlib` to create our visualizations. For the 2D smile, we filter the data for a single expiration date and plot IV against the strike price. For the 3D surface, we use `matplotlib`'s 3D plotting toolkit to plot IV against both strike and time to maturity.

The following Python code demonstrates this entire process.



```Python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from datetime import datetime

# --- Step 1: Data Acquisition ---
# Fetch data for SPY
ticker_symbol = "SPY"
ticker = yf.Ticker(ticker_symbol)
expirations = ticker.options

# Get current stock price and risk-free rate (using 10-year Treasury yield as a proxy)
spot_price = ticker.history(period='1d')['Close'].iloc[-1]
rf_rate_ticker = yf.Ticker("^TNX")
risk_free_rate = rf_rate_ticker.history(period='1d')['Close'].iloc[-1] / 100

# Loop through expirations to build a full options dataframe
all_options =
for expiry in expirations:
    try:
        opt_chain = ticker.option_chain(expiry)
        calls = opt_chain.calls
        calls['type'] = 'c'
        puts = opt_chain.puts
        puts['type'] = 'p'
        
        df = pd.concat([calls, puts])
        df = pd.to_datetime(expiry)
        all_options.append(df)
    except Exception as e:
        print(f"Could not fetch data for {expiry}: {e}")

options_df = pd.concat(all_options)

# --- Data Cleaning and Preparation ---
# Calculate time to maturity in years
options_df = (options_df - datetime.now()).dt.days / 365.0

# Calculate mid-price
options_df['mid_price'] = (options_df['bid'] + options_df['ask']) / 2

# Filter for liquid options
options_df = options_df[(options_df['bid'] > 0) & (options_df['ask'] > 0) & (options_df['volume'] > 10) & (options_df > 0)]

# --- Step 2: Calculate Implied Volatility ---
# Define a function to calculate IV, handling potential errors
def calculate_iv(row):
    try:
        return implied_volatility(
            price=row['mid_price'],
            S=spot_price,
            K=row['strike'],
            t=row,
            r=risk_free_rate,
            flag=row['type']
        )
    except:
        return np.nan

# Apply the function to the DataFrame
options_df['implied_volatility'] = options_df.apply(calculate_iv, axis=1)

# Drop rows where IV calculation failed
options_df.dropna(subset=['implied_volatility'], inplace=True)
options_df = options_df[options_df['implied_volatility'] < 2.0] # Remove extreme values

# --- Step 3: Plot the Volatility Smile for a Single Expiration ---
# Select a near-term expiration for plotting
near_term_expiry = options_df.unique()
smile_data = options_df == near_term_expiry) & (options_df['type'] == 'c')]
smile_data = smile_data.sort_values(by='strike')

plt.figure(figsize=(12, 7))
plt.plot(smile_data['strike'], smile_data['implied_volatility'], 'o-', label=f'SPY Calls Expiring {near_term_expiry.date()}')
plt.axvline(x=spot_price, color='r', linestyle='--', label=f'Spot Price: ${spot_price:.2f}')
plt.title(f'Volatility Smile for {ticker_symbol}', fontsize=16)
plt.xlabel('Strike Price ($)', fontsize=12)
plt.ylabel('Implied Volatility', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

# --- Step 4: Plot the 3D Volatility Surface ---
# Filter for calls and a reasonable range of strikes and maturities
surface_data = options_df[(options_df['type'] == 'c') & 
                          (options_df['strike'] > spot_price * 0.8) & 
                          (options_df['strike'] < spot_price * 1.2) &
                          (options_df < 1.0)]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Create the plot
surf = ax.plot_trisurf(surface_data, surface_data['strike'], surface_data['implied_volatility'], cmap=cm.viridis, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_title(f'Implied Volatility Surface for {ticker_symbol}', fontsize=16)
ax.set_xlabel('Time to Maturity (Years)', fontsize=12)
ax.set_ylabel('Strike Price ($)', fontsize=12)
ax.set_zlabel('Implied Volatility', fontsize=12)
ax.view_init(30, -120) # Adjust viewing angle
plt.show()
```

### 3. Modeling the Smile: A Journey Through Quantitative Models

The existence of the volatility smile demonstrates that the single-parameter BSM model is insufficient. Quantitative finance has responded by developing more sophisticated models. The evolution of these models can be seen as a journey, starting with the most direct extension of BSM, identifying its flaws, and progressively building more realistic frameworks.

#### 3.1 Local Volatility Models: Dupire's Equation

The first major attempt to formally incorporate the volatility smile into a consistent pricing framework was the **local volatility model**, independently developed by Bruno Dupire (1994) and Emanuel Derman & Iraj Kani (1994).29 The central idea is to abandon the assumption of constant volatility and instead propose that volatility is a deterministic function of time (t) and the current asset price $(St​): σ_local​=σ(t,S_t​)$.30

This approach leads to a remarkable result known as the **Dupire equation**, which provides an explicit formula for the local variance function directly from the market prices of European call options. It establishes a unique, arbitrage-free diffusion process that is, by construction, consistent with the entire observed volatility surface at a single point in time.31 The formula is:

$$\sigma^2_{\text{local}}(K, T) =
\frac{
\frac{\partial C}{\partial T} + (r - q) K \frac{\partial C}{\partial K} + q C
}{
\frac{1}{2} K^2 \frac{\partial^2 C}{\partial K^2}
}$$

Here, C(K,T) is the price of a call with strike K and maturity T, r is the risk-free rate, and q is the dividend yield. The derivatives of the call price with respect to maturity ($\frac{\partial C}{\partial T}$​, or Theta) and strike ($\frac{\partial C}{\partial K}$ and $\frac{\partial^2 C}{\partial K^2}2$​) are used. The denominator, containing the second derivative with respect to strike, is particularly important. According to the Breeden-Litzenberger result, this term is proportional to the risk-neutral probability density function of the asset price being at strike K at maturity T.32

Pros and Cons:

The primary advantage of the local volatility model is its ability to perfectly fit the market smile at the time of calibration. It provides a single, unified process that can price any European-style exotic derivative consistently with the vanilla options market.33

However, the model has a critical flaw in its **dynamics**. It predicts that as the underlying asset price changes, the volatility smile shifts along with it (a "sticky strike" behavior). In reality, market smiles tend to be more "sticky" to the at-the-money point; the smile shifts, but not in the rigid way the local volatility model predicts.35 This discrepancy between the model's predicted dynamics and observed market behavior leads to unstable hedges and poor risk management performance.35 The model wins the static battle of fitting today's prices but loses the dynamic war of predicting tomorrow's smile and providing robust hedges.

#### 3.2 Stochastic Volatility Models: The Heston Model

To address the dynamic failings of local volatility, a new class of models was developed where volatility itself is a random process. The most famous of these is the **Heston model**, introduced by Steven Heston in 1993.36 It is a two-factor model, with one stochastic process for the asset price and another for its variance. This framework allows for a much richer and more realistic representation of market dynamics.

The Heston model is defined by a pair of correlated stochastic differential equations (SDEs) 36:

1. **Asset Price Process:** $dS_t = r S_t \, dt + \sqrt{v_t} \, S_t \, dW_1^t$
    
2. **Variance Process:** $dv_t = \kappa (\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW_2^t$
    

The two Wiener processes, dW1t​ and dW2t​, have a correlation ρ, such that dW1t​dW2t​=ρdt. The model has five parameters, each with a clear financial interpretation:

- v0​: The initial variance of the asset's returns at time t=0.
    
- θ: The long-run average variance. The process is mean-reverting, meaning vt​ will tend to be pulled back towards this level.39
    
- κ: The speed of mean reversion. A higher κ means that volatility reverts to its long-term mean θ more quickly.39
    
- ξ: The volatility of variance (often called "vol of vol"). This parameter governs the volatility of the variance process itself and is a key driver of the smile's convexity or curvature.37
    
- ρ: The correlation between the asset's returns and its volatility. For equity markets, ρ is typically negative, capturing the empirical observation that asset prices tend to fall when volatility rises (and vice versa). This negative correlation is the primary driver of the volatility skew.36
    

Unlike BSM, the Heston model does not have a simple closed-form solution for option prices, but a semi-analytical solution exists using Fourier transforms, which allows for efficient pricing.37 Calibrating the model involves finding the set of five parameters that minimizes the difference between model prices and market prices, typically using a least-squares optimization algorithm.41

The following Python code illustrates how to calibrate a Heston model to market data using the `QuantLib` library.43



```Python
import QuantLib as ql
import numpy as np

# This example assumes market_data (strikes, maturities, vols) is available
# For a full working example, see the Capstone Project section.

# --- 1. Setup Market Data and Environment ---
# (Assuming calculation_date, calendar, day_count, spot, risk_free_rate are defined)
# Example data for one maturity
# strikes = [K1, K2,...]
# market_vols = [IV1, IV2,...]
# maturity_date = ql.Date(...)

# --- 2. Define Heston Model and Engine ---
# Initial parameter guesses
v0 = 0.01; kappa = 2.0; theta = 0.02; rho = -0.5; sigma = 0.5

# Setup Heston process
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, 0.0, day_count))
process = ql.HestonProcess(flat_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)

# Create model and engine
model = ql.HestonModel(process)
engine = ql.AnalyticHestonEngine(model)

# --- 3. Create Heston Helpers ---
# HestonModelHelper objects link market quotes to the model for calibration
helpers =
for i, strike in enumerate(strikes):
    vol_quote = ql.QuoteHandle(ql.SimpleQuote(market_vols[i]))
    period = ql.Period(maturity_date - calculation_date, ql.Days)
    helper = ql.HestonModelHelper(period, calendar, spot, strike, vol_quote, flat_ts, dividend_ts)
    helper.setPricingEngine(engine)
    helpers.append(helper)

# --- 4. Calibrate the Model ---
# Choose optimization method (Levenberg-Marquardt)
lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)

# Set parameter constraints
model.setParams()

# Run calibration
model.calibrate(helpers, lm, ql.EndCriteria(500, 50, 1.0e-8, 1.0e-8, 1.0e-8))

# --- 5. Retrieve and Display Results ---
theta_cal, kappa_cal, sigma_cal, rho_cal, v0_cal = model.params()
print(f"Calibrated Heston Parameters:")
print(f"theta = {theta_cal:.4f}, kappa = {kappa_cal:.4f}, sigma = {sigma_cal:.4f}, rho = {rho_cal:.4f}, v0 = {v0_cal:.4f}")

# Evaluate the fit
total_error = 0
for i, helper in enumerate(helpers):
    model_price = helper.modelValue()
    market_price = helper.marketValue()
    error = (model_price / market_price) - 1.0
    total_error += error**2

rmse = np.sqrt(total_error / len(helpers))
print(f"\nRMSE of fit: {rmse:.6f}")
```

#### 3.3 The SABR Model: A Practitioner's Favorite

The **SABR (Stochastic Alpha, Beta, Rho) model** is another powerful stochastic volatility framework that has become an industry benchmark, particularly for interest rate derivatives like swaptions and caps/floors.44 Its popularity stems from its intuitive parameters and, crucially, the existence of a highly accurate analytical approximation for implied volatility developed by Hagan et al. (2002), which makes calibration extremely fast.44

The SABR model describes the evolution of a forward rate Ft​ and its stochastic volatility σt​:

1. $dF_t​=σ_t​(F_t​)^βdW_t​$
    
2. $dσ_t​=νσ_t​dZ_t​$
    

Here, Wt​ and Zt​ are Wiener processes with correlation ρ, such that dWt​dZt​=ρdt. The model has four key parameters that directly control the shape of the volatility smile 45:

- α (or σ0​): The initial level of volatility. It primarily controls the overall height of the smile, shifting it up or down.
    
- β: The exponent parameter (0≤β≤1), which determines the underlying process for the forward rate. If β=1, the model is log-normal (like BSM). If β=0, it is normal (like the Bachelier model). This parameter has a strong influence on the "backbone" or general slope of the smile.
    
- ρ: The correlation between the forward rate and its volatility. This parameter directly controls the skew or tilt of the smile. A negative ρ creates a downward-sloping smirk.
    
- ν (`volvol`): The volatility of volatility. This parameter directly controls the convexity or curvature of the smile. A higher ν leads to a more pronounced "smile" shape.
    

The key to SABR's success is Hagan's asymptotic expansion formula, which provides a direct mapping from the model parameters to the Black-Scholes implied volatility for any given strike.44 This bypasses the need for slower numerical methods like Monte Carlo or PDE solvers during calibration.

While the SABR parameters are not perfectly orthogonal, their roles are more distinct than those of the Heston model, making calibration more intuitive. However, since both β and ρ influence the skew, calibrating both simultaneously can lead to instability. A common market practice is to fix the value of β based on market convention or historical analysis (e.g., β=0.5 for interest rates) and then calibrate the remaining three parameters (α,ρ,ν) to the observed market smile.46

The following Python code demonstrates how to calibrate the SABR model using the `pysabr` library.50



```Python
import numpy as np
import matplotlib.pyplot as plt
from pysabr import Hagan2002LognormalSABR

# This example assumes market_data (strikes, vols) and forward rate are available
# For a full working example, see the Capstone Project section.

# --- 1. Setup Market Data ---
# f = forward_rate
# t = time_to_maturity
# strikes = np.array([...])
# vols = np.array([...]) # Market lognormal implied vols
# beta = 0.5 # Fix beta based on market convention

# --- 2. Instantiate and Calibrate SABR Model ---
# The Hagan2002LognormalSABR class takes the fixed parameters
sabr_model = Hagan2002LognormalSABR(f=f, t=t, beta=beta)

# The.fit() method finds the best alpha, rho, and volvol
# It takes the arrays of strikes and market volatilities
alpha, rho, volvol = sabr_model.fit(strikes, vols)

print("Calibrated SABR Parameters:")
print(f"alpha = {alpha:.4f}, rho = {rho:.4f}, volvol = {volvol:.4f}")

# --- 3. Generate and Plot the Calibrated Smile ---
# Create a finer grid of strikes for a smooth plot
fine_strikes = np.linspace(strikes.min(), strikes.max(), 100)

# Use the calibrated parameters to get the model's volatility smile
calibrated_vols = [sabr_model.lognormal_vol(k) for k in fine_strikes]

plt.figure(figsize=(12, 7))
plt.plot(strikes, vols, 'o', label='Market Vols')
plt.plot(fine_strikes, calibrated_vols, '-', lw=2, label='Calibrated SABR Smile')
plt.title('SABR Model Calibration', fontsize=16)
plt.xlabel('Strike', fontsize=12)
plt.ylabel('Lognormal Implied Volatility', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()
```

The table below provides a concise summary and comparison of these three modeling paradigms.

|Feature|Local Volatility (Dupire)|Stochastic Volatility (Heston)|Stochastic Volatility (SABR)|
|---|---|---|---|
|**Core Idea**|Volatility is a deterministic function of asset price and time: σ(S,t).|Volatility is a mean-reverting random process.|Volatility is a random process, often with a CEV-like backbone.|
|**Key Strength**|Perfect static fit to the vanilla option smile by construction.33|Captures realistic smile dynamics like mean reversion and correlation.37|Fast, analytical approximation for calibration; intuitive parameters.45|
|**Key Weakness**|Produces unrealistic smile dynamics ("sticky strike"), leading to poor hedges.35|Computationally intensive calibration; no simple closed-form price for vanilla options.41|Approximation can be inaccurate for extreme strikes or long maturities; can produce arbitrage.46|
|**Typical Use Case**|Pricing complex exotic derivatives consistently with the vanilla surface.|Equity and FX option pricing, risk management, and dynamic hedging.|Interest rate derivatives (swaptions, caps), FX options, and quick smile interpolation.|

### 4. Hedging Beyond Delta: Managing Smile Risk

The existence of the volatility smile has profound implications for hedging. A delta-neutral position in the BSM world is not truly risk-free when volatility is not constant.

#### 4.1 The Failure of BSM Delta Hedging

In a world with a volatility smile, the standard BSM delta is an incomplete and often incorrect measure of an option's sensitivity to the underlying price. The "true" delta must also account for the indirect effect that a change in the spot price has on the option's implied volatility. Using the chain rule, we can express the total derivative of the option price C with respect to the spot price S as 51:

$$\frac{dC}{dS} = \frac{\partial C_{BS}}{\partial S}
+ \frac{\partial C_{BS}}{\partial \sigma} \frac{\partial \sigma}{\partial S}
= \delta_{BS} + \text{Vega} \times \frac{\partial \sigma}{\partial S}$$

This equation reveals a critical point: the correct hedge ratio is the BSM delta plus an adjustment term. This term is the product of the option's Vega (sensitivity to volatility) and $\frac{\partial \sigma}{\partial S}$, which is the slope of the volatility skew.51

In equity markets, the skew is downward sloping, meaning $\frac{\partial \sigma}{\partial S}$​ is negative. Since Vega is always positive for a long option, the adjustment term is negative. This implies that for a long call option, the true delta is **less than** the BSM delta. A trader who uses the standard BSM delta to hedge a long call position will systematically sell too much of the underlying asset, leading to predictable hedging losses, especially during large market moves.51 This is not a random error; it is a structural bias caused by ignoring the smile.

#### 4.2 Introducing the Smile Greeks: Vanna and Volga

To properly manage the risks introduced by the volatility smile, traders must look beyond delta and gamma and incorporate higher-order Greeks that measure sensitivity to changes in the smile's shape.

- **Vanna:** This is the second-order sensitivity of the option price to a change in spot and a change in volatility. It can be defined in two equivalent ways: the sensitivity of delta to a change in volatility (∂σ∂δ​), or the sensitivity of vega to a change in spot price (∂S∂Vega​).53 Vanna measures the risk associated with the
    
    **tilt or skew** of the volatility smile. A portfolio with non-zero Vanna will see its delta hedge change as market volatility fluctuates.55
    
- **Volga (or Vomma):** This is the second derivative of the option price with respect to volatility, or the sensitivity of vega to a change in volatility (∂σ∂Vega​).53 Volga measures the
    
    **convexity** of the option price with respect to volatility. It captures the risk associated with a change in the **curvature** of the smile. A portfolio that is vega-neutral but has positive Volga will profit if volatility moves significantly in either direction (i.e., if the smile becomes more pronounced).
    

#### 4.3 The Vanna-Volga Approach: A Model-Free Hedging Framework

The **Vanna-Volga approach** is a pragmatic and popular technique, especially in FX markets, for pricing and hedging exotic options in a way that is consistent with the vanilla smile, without resorting to a complex stochastic volatility model.53 It is a powerful example of financial engineering that uses observable market prices to construct a robust hedge.

The core idea is to create a portfolio of liquid, standard options that replicates the key smile risks (Vanna and Volga) of a more complex, exotic option. This hedging portfolio is typically constructed using three instruments 53:

1. **At-the-Money (ATM) Straddle:** A long call and a long put at the ATM strike. This portfolio is delta-neutral and has a high, almost pure **Vega** exposure.
    
2. **Risk Reversal (RR):** A long OTM call and a short OTM put (typically 25-delta). This portfolio is primarily sensitive to the **skew** of the smile and thus serves as a clean **Vanna** instrument.
    
3. **Butterfly (BF):** A long position in the OTM call and put "wings" and a short position in the ATM straddle "body." This portfolio profits from an increase in the smile's curvature and is thus a clean **Volga** instrument.
    

The methodology is as follows 53:

1. Calculate the Vega, Vanna, and Volga of the exotic option using the BSM model with a flat (ATM) volatility.
    
2. Construct a portfolio of the three standard instruments (ATM, RR, BF) that has the exact same Vega, Vanna, and Volga as the exotic option by solving a system of linear equations for the required weights (wATM​,wRR​,wBF​).
    
3. Calculate the "smile cost" of this hedging portfolio. This is the difference between the actual market price of the portfolio (using the skewed market volatilities) and its theoretical BSM price (using the flat ATM volatility).
    
4. The Vanna-Volga adjusted price of the exotic is its BSM price plus this smile cost:
    

$$Price_{VV} = Price_{BS}
+ w_{RR} \times (Price^{Market}_{RR} - Price^{BS}_{RR})
+ w_{BF} \times (Price^{Market}_{BF} - Price^{BS}_{BF})$$

This approach effectively bypasses the need to choose and calibrate a specific smile model. Instead, it uses the market's own pricing of skew (via the Risk Reversal) and curvature (via the Butterfly) to determine the correct price adjustment for the exotic option's smile risk. It is a "trader's rule of thumb" that has been formalized into a robust and widely used technique.57

### 5. Capstone Project: Dissecting and Modeling the SPX Volatility Surface

This project synthesizes the chapter's concepts by applying them to the S&P 500 (SPX) options market, the deepest and most liquid equity index options market in the world.

#### Part 1: Data Acquisition and Surface Construction

- **Questions:**
    
    1. How can you obtain historical daily option chain data for SPX for a specific date?
        
    2. What data cleaning and filtering steps are necessary (e.g., removing options with zero volume, zero bid, or wide bid-ask spreads)?
        
    3. How do you calculate implied volatility for the entire cleaned dataset?
        
    4. How can you construct and visualize the 2D volatility skew for several key maturities (e.g., 1-month, 3-month, 1-year) and the full 3D volatility surface?
        
- **Response Walkthrough:**
    
    1. **Data Source:** Obtaining free, high-quality historical SPX options data is notoriously difficult.25 For this project, we will use a sample CSV file representing a snapshot of SPX options data for a single trading day (e.g., from a vendor like CBOE DataShop 20). This ensures the project is reproducible. The provided Python script will load this data into a pandas DataFrame.
        
    2. **Data Cleaning:** Before analysis, the raw data must be cleaned to ensure quality. This involves:
        
        - Removing entries with zero bid price or zero trading volume, as these are illiquid and their prices are unreliable.
            
        - Filtering out options with excessively wide bid-ask spreads, which can indicate stale quotes or lack of interest.
            
        - Excluding deep in-the-money or far out-of-the-money options that may have erratic pricing.
            
        - Calculating a mid-price `(bid + ask) / 2` to use as the option's market price.
            
    3. **IV Calculation:** We will use the `py_vollib_vectorized` library to efficiently calculate the implied volatility for every option in the cleaned DataFrame. This library is optimized for performance on large datasets.28
        
    4. **Visualization:** Using `matplotlib`, we will generate:
        
        - **2D Skew Plots:** We will filter the data for options expiring in approximately 1 month, 3 months, and 1 year. For each maturity, we will plot implied volatility against the strike price to observe the characteristic SPX smirk and how its shape evolves with time to maturity.
            
        - **3D Surface Plot:** We will create a 3D scatter plot or surface plot showing implied volatility as a function of both time to maturity and moneyness (K/S0​). This will provide a comprehensive visualization of the entire volatility landscape on that day.
            

_(Note: The full Python code for this section is omitted for brevity but follows the structure of the implementation provided in Section 2.3, adapted for loading a CSV file instead of using `yfinance`.)_

#### Part 2: Comparative Model Calibration

- **Questions:**
    
    1. Using the 1-year maturity options data, calibrate both the Heston and SABR models. What are the resulting parameters?
        
    2. How do you interpret the calibrated parameters for each model (e.g., what does the negative `rho` in Heston signify? What do the SABR parameters tell you about the smile's shape)?
        
    3. Generate the volatility smiles from the calibrated models and plot them against the market data.
        
    4. Which model provides a better fit to the data? How can you quantify this using a metric like Root Mean Squared Error (RMSE)?
        
- **Response Walkthrough:**
    
    1. **Calibration:** We will provide two Python scripts. The first uses `QuantLib` to calibrate the five Heston parameters to the 1-year SPX options data.41 The second uses
        
        `pysabr` to calibrate the three free SABR parameters (fixing β=0.5) to the same data slice.50
        
    2. **Parameter Interpretation:** The calibrated parameters are presented in Table 4.
        
        - **Heston:** The calibrated `rho` will be significantly negative, confirming the strong negative correlation between SPX returns and volatility changes, which is the primary driver of the index's pronounced skew. The `kappa` will indicate the speed of mean reversion, while `xi` (vol of vol) will quantify the smile's convexity.
            
        - **SABR:** The calibrated `rho` will also be negative, controlling the skew. The `nu` (volvol) will be positive, controlling the curvature. The `alpha` parameter will anchor the level of the smile near the at-the-money point.
            
    3. **Visualization:** A plot will be generated showing the discrete market volatility points for the 1-year options, overlaid with the smooth, continuous smile curves generated by the calibrated Heston and SABR models. This provides a clear visual comparison of the models' fits.
        
    4. **Goodness-of-Fit:** We will calculate the Root Mean Squared Error (RMSE) between the market implied volatilities and the model-generated volatilities for both Heston and SABR. The model with the lower RMSE is considered to have a better static fit to the data for this specific maturity. It is common for SABR to provide a slightly better and more stable fit for a single smile slice due to its direct analytical formulation, while Heston's strength lies more in its dynamic properties.
        

|Model|Parameter|Calibrated Value (Illustrative)|Interpretation|RMSE (Illustrative)|
|---|---|---|---|---|
|**Heston**|θ (long-run var)|0.04|Long-term average volatility of 20%|0.0085|
||κ (reversion speed)|2.5|Moderately fast mean reversion||
||ξ (vol of vol)|0.45|Significant convexity in the smile||
||ρ (correlation)|-0.70|Strong negative correlation between price and vol (leverage effect)||
||v0​ (initial var)|0.0324|Starting volatility of 18%||
|**SABR**|α (initial vol)|0.182|Anchors the ATM vol level|**0.0062**|
||β (exponent)|0.5 (fixed)|Assumes a mix of log-normal and normal dynamics||
||ρ (correlation)|-0.65|Strong negative skew||
||ν (vol of vol)|0.55|High degree of smile curvature||

#### Part 3: Advanced Hedging Simulation (Conceptual)

- **Questions:**
    
    1. Describe the portfolio setup for a delta-hedged long call option under the BSM model.
        
    2. How would you augment this hedge to also be Vanna and Volga neutral using a risk reversal and a butterfly? Describe the composition of this new hedging portfolio.
        
    3. Conceptually, if the SPX were to fall sharply and volatility were to spike (a common scenario), why would the delta-vanna-volga hedge be expected to outperform the simple BSM delta hedge?
        
- **Response Walkthrough:**
    
    1. **BSM Delta Hedge:** The portfolio consists of being long one SPX call option and short δBS​ units of the underlying index (e.g., via SPX futures). The hedge is rebalanced periodically (e.g., daily) to maintain a delta-neutral position according to the BSM formula.
        
    2. **Delta-Vanna-Volga Hedge:** This is a more sophisticated hedge designed to neutralize smile risk. The portfolio would consist of:
        
        - Long one SPX call option.
            
        - Short δtrue​ units of the underlying index, where δtrue​ is the smile-adjusted delta (![[Pasted image 20250708000430.png]]​).
            
        - A position in a **Risk Reversal (RR)** to neutralize the portfolio's Vanna. If the call has positive Vanna, the hedge would involve shorting a RR (selling a call and buying a put).
            
        - A position in a Butterfly (BF) to neutralize the portfolio's Volga. If the call has positive Volga, the hedge would involve shorting a BF (selling the wings and buying the body).
            
            The exact sizes of the RR and BF positions are calculated to make the total portfolio Vanna and Volga zero.
            
    3. **Performance in a Crash Scenario:** Let's analyze what happens when the SPX falls sharply and volatility spikes.
        
        - **BSM Hedge Performance:** The BSM hedge would perform poorly for two main reasons. First, as established, the initial δBS​ was too high, meaning the trader was too short the index. As the market falls, buying back parts of this hedge at lower prices results in losses. Second, and more importantly, the hedge has no protection against the spike in volatility. The long call option gains value from the vol spike (positive vega), but the hedging P&L is exposed to the fact that the realized volatility is much higher than the original implied volatility used for hedging, leading to significant path-dependent losses.61
            
        - **Delta-Vanna-Volga Hedge Performance:** This hedge is designed for exactly this scenario. The initial delta is more accurate, reducing the error from the spot move. The short butterfly position (which is negative Volga) profits as the smile's curvature increases dramatically during the vol spike. The short risk reversal position helps manage the change in the delta hedge as the skew steepens. By neutralizing the first- and second-order sensitivities to the smile's shape, the P&L of the delta-vanna-volga hedged portfolio would be significantly more stable, demonstrating the practical value of actively managing smile risk.52
## References
**

1. Black–Scholes model - Wikipedia, acessado em julho 7, 2025, [https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
    
2. Review of normal distribution N(µ, σ 2). • Black-Scholes Formula Assumptions - OSU Math, acessado em julho 7, 2025, [https://math.osu.edu/~ban.1/5632/black_scholes.pdf](https://math.osu.edu/~ban.1/5632/black_scholes.pdf)
    
3. Black-Scholes Model: What It Is, How It Works, and Options Formula - Investopedia, acessado em julho 7, 2025, [https://www.investopedia.com/terms/b/blackscholes.asp](https://www.investopedia.com/terms/b/blackscholes.asp)
    
4. Assumptions of the Black-Scholes-Merton Option Valuation Model - CFA, FRM, and Actuarial Exams Study Notes - AnalystPrep, acessado em julho 7, 2025, [https://analystprep.com/study-notes/cfa-level-2/identify-assumptions-of-the-black-scholes-merton-option-valuation-model/](https://analystprep.com/study-notes/cfa-level-2/identify-assumptions-of-the-black-scholes-merton-option-valuation-model/)
    
5. The Black Scholes Model And Implied Volatility - Trading Interview, acessado em julho 7, 2025, [https://www.tradinginterview.com/courses/derivatives-theory/lessons/the-black-scholes-model-and-implied-volatility/](https://www.tradinginterview.com/courses/derivatives-theory/lessons/the-black-scholes-model-and-implied-volatility/)
    
6. Black-Scholes Model Assumptions - Macroption, acessado em julho 7, 2025, [https://www.macroption.com/black-scholes-assumptions/](https://www.macroption.com/black-scholes-assumptions/)
    
7. Volatility Skew and Smile - CFA, FRM, and Actuarial Exams Study Notes - AnalystPrep, acessado em julho 7, 2025, [https://analystprep.com/study-notes/cfa-level-iii/volatility-skew-and-smile/](https://analystprep.com/study-notes/cfa-level-iii/volatility-skew-and-smile/)
    
8. Volatility smile - Wikipedia, acessado em julho 7, 2025, [https://en.wikipedia.org/wiki/Volatility_smile](https://en.wikipedia.org/wiki/Volatility_smile)
    
9. Implied Volatility in Python; Compute the Volatilities Implied by Option Prices Observed in the Market using the SciPy Library | by Roi Polanitzer | Medium, acessado em julho 7, 2025, [https://medium.com/@polanitzer/implied-volatility-in-python-compute-the-volatilities-implied-by-option-prices-observed-in-the-e2085c184270](https://medium.com/@polanitzer/implied-volatility-in-python-compute-the-volatilities-implied-by-option-prices-observed-in-the-e2085c184270)
    
10. Calculating Implied Volatility with Python for Options Traders - YouTube, acessado em julho 7, 2025, [https://www.youtube.com/watch?v=h6fMYoig2Pg](https://www.youtube.com/watch?v=h6fMYoig2Pg)
    
11. Volatility Smile - Overview, When It is Observed, and Limitations, acessado em julho 7, 2025, [https://corporatefinanceinstitute.com/resources/derivatives/volatility-smile/](https://corporatefinanceinstitute.com/resources/derivatives/volatility-smile/)
    
12. Volatility Skew and Options: An Overview, acessado em julho 7, 2025, [https://www.optionseducation.org/news/volatility-skew-and-options-an-overview-1](https://www.optionseducation.org/news/volatility-skew-and-options-an-overview-1)
    
13. What Is a Volatility Smile? - Moomoo, acessado em julho 7, 2025, [https://www.moomoo.com/us/learn/detail-what-is-a-volatility-smile-88700-221271042](https://www.moomoo.com/us/learn/detail-what-is-a-volatility-smile-88700-221271042)
    
14. What is Volatility Skew and How Can You Trade It? - SoFi, acessado em julho 7, 2025, [https://www.sofi.com/learn/content/volatility-skew-options-trading/](https://www.sofi.com/learn/content/volatility-skew-options-trading/)
    
15. Volatility Skew: How it Can Signal Market Sentiment - Investopedia, acessado em julho 7, 2025, [https://www.investopedia.com/terms/v/volatility-skew.asp](https://www.investopedia.com/terms/v/volatility-skew.asp)
    
16. Volatility Skew - Definition, Types, How it Works - Corporate Finance Institute, acessado em julho 7, 2025, [https://corporatefinanceinstitute.com/resources/derivatives/volatility-skew/](https://corporatefinanceinstitute.com/resources/derivatives/volatility-skew/)
    
17. Volatility Skew: What it is, shapes & why it matters? - Equirus Capital, acessado em julho 7, 2025, [https://www.equirus.com/glossary/volatility-skew](https://www.equirus.com/glossary/volatility-skew)
    
18. Build an implied volatility surface with Python - PyQuant News, acessado em julho 7, 2025, [https://www.pyquantnews.com/the-pyquant-newsletter/build-an-implied-volatility-surface-with-python](https://www.pyquantnews.com/the-pyquant-newsletter/build-an-implied-volatility-surface-with-python)
    
19. Volatility Skew: How to Uncover Market Sentiment Shifts - Amberdata Blog, acessado em julho 7, 2025, [https://blog.amberdata.io/volatility-skew-how-to-uncover-market-sentiment-shifts](https://blog.amberdata.io/volatility-skew-how-to-uncover-market-sentiment-shifts)
    
20. Historical Options Data Download - Cboe Global Markets, acessado em julho 7, 2025, [https://www.cboe.com/us/options/market_statistics/historical_data/](https://www.cboe.com/us/options/market_statistics/historical_data/)
    
21. Download Historical Intraday Data (20 Years Data), acessado em julho 7, 2025, [https://firstratedata.com/](https://firstratedata.com/)
    
22. Historical Options Prices | TickData, acessado em julho 7, 2025, [https://www.tickdata.com/product/historical-options-data/](https://www.tickdata.com/product/historical-options-data/)
    
23. How to Get Historical Market Data Through Python Stock API - QuantInsti Blog, acessado em julho 7, 2025, [https://blog.quantinsti.com/historical-market-data-python-api/](https://blog.quantinsti.com/historical-market-data-python-api/)
    
24. How to get Stock Options data with Python and yFinance - YouTube, acessado em julho 7, 2025, [https://m.youtube.com/watch?v=ZLbVsPy13QI&pp=0gcJCYUJAYcqIYzv](https://m.youtube.com/watch?v=ZLbVsPy13QI&pp=0gcJCYUJAYcqIYzv)
    
25. Exploring the Archives: Is There a Database of Historical Options Data? : r/quant - Reddit, acessado em julho 7, 2025, [https://www.reddit.com/r/quant/comments/13quuap/exploring_the_archives_is_there_a_database_of/](https://www.reddit.com/r/quant/comments/13quuap/exploring_the_archives_is_there_a_database_of/)
    
26. Candle open exact same as candle close on Forex data · Issue ..., acessado em julho 7, 2025, [https://github.com/ranaroussi/yfinance/issues/1075](https://github.com/ranaroussi/yfinance/issues/1075)
    
27. vollib/py_vollib - GitHub, acessado em julho 7, 2025, [https://github.com/vollib/py_vollib](https://github.com/vollib/py_vollib)
    
28. Implied Volatility — py_vollib_vectorized 0.1 documentation, acessado em julho 7, 2025, [https://py-vollib-vectorized.readthedocs.io/en/latest/pkg_ref/iv.html](https://py-vollib-vectorized.readthedocs.io/en/latest/pkg_ref/iv.html)
    
29. Dupire Local Vol Derivation | PDF | Black–Scholes Model | Volatility (Finance) - Scribd, acessado em julho 7, 2025, [https://www.scribd.com/document/687668409/Dupire-Local-Vol-Derivation](https://www.scribd.com/document/687668409/Dupire-Local-Vol-Derivation)
    
30. The local volatility surface - KTH, acessado em julho 7, 2025, [https://www.math.kth.se/matstat/gru/5b1575/Projects2016/VolatilitySurface.pdf](https://www.math.kth.se/matstat/gru/5b1575/Projects2016/VolatilitySurface.pdf)
    
31. Local Volatility, Stochastic Volatility and Jump-Diffusion Models, acessado em julho 7, 2025, [http://www.columbia.edu/~mh2078/ContinuousFE/LocalStochasticJumps.pdf](http://www.columbia.edu/~mh2078/ContinuousFE/LocalStochasticJumps.pdf)
    
32. Local Volatility and Dupire's Equation - World Scientific Publishing, acessado em julho 7, 2025, [https://www.worldscientific.com/doi/pdf/10.1142/9789811212772_0001](https://www.worldscientific.com/doi/pdf/10.1142/9789811212772_0001)
    
33. Local volatility - Dupire's formula : r/quant - Reddit, acessado em julho 7, 2025, [https://www.reddit.com/r/quant/comments/1isgr6b/local_volatility_dupires_formula/](https://www.reddit.com/r/quant/comments/1isgr6b/local_volatility_dupires_formula/)
    
34. Derivation of Local Volatility - Fabrice Rouah, acessado em julho 7, 2025, [http://frouah.com/finance%20notes/Dupire%20Local%20Volatility.pdf](http://frouah.com/finance%20notes/Dupire%20Local%20Volatility.pdf)
    
35. Properties of the SABR model - DiVA portal, acessado em julho 7, 2025, [http://www.diva-portal.org/smash/get/diva2:430537/FULLTEXT01.pdf](http://www.diva-portal.org/smash/get/diva2:430537/FULLTEXT01.pdf)
    
36. Heston Model: Meaning, Overview, Methodology - Investopedia, acessado em julho 7, 2025, [https://www.investopedia.com/terms/h/heston-model.asp](https://www.investopedia.com/terms/h/heston-model.asp)
    
37. 7 Heston's Model and the Smile - ResearchGate, acessado em julho 7, 2025, [https://www.researchgate.net/publication/265478570_7_Heston's_Model_and_the_Smile](https://www.researchgate.net/publication/265478570_7_Heston's_Model_and_the_Smile)
    
38. Heston Model Simulation with Python - CodeArmo, acessado em julho 7, 2025, [https://www.codearmo.com/python-tutorial/heston-model-simulation-python](https://www.codearmo.com/python-tutorial/heston-model-simulation-python)
    
39. Heston Model: Options Pricing, Python Implementation and Parameters - QuantInsti Blog, acessado em julho 7, 2025, [https://blog.quantinsti.com/heston-model/](https://blog.quantinsti.com/heston-model/)
    
40. Heston model - Wikipedia, acessado em julho 7, 2025, [https://en.wikipedia.org/wiki/Heston_model](https://en.wikipedia.org/wiki/Heston_model)
    
41. HESTON MODEL CALIBRATION USING QUANTLIB IN PYTHON | by Aaron De la Rosa, acessado em julho 7, 2025, [https://medium.com/@aaron_delarosa/heston-model-calibration-using-quantlib-in-python-0089516430ef](https://medium.com/@aaron_delarosa/heston-model-calibration-using-quantlib-in-python-0089516430ef)
    
42. Heston Model calibration : r/quant - Reddit, acessado em julho 7, 2025, [https://www.reddit.com/r/quant/comments/10q9r99/heston_model_calibration/](https://www.reddit.com/r/quant/comments/10q9r99/heston_model_calibration/)
    
43. Modeling Volatility Smile and Heston Model Calibration Using ..., acessado em julho 7, 2025, [http://gouthamanbalaraman.com/blog/volatility-smile-heston-model-calibration-quantlib-python.html](http://gouthamanbalaraman.com/blog/volatility-smile-heston-model-calibration-quantlib-python.html)
    
44. SABR volatility model - Wikipedia, acessado em julho 7, 2025, [https://en.wikipedia.org/wiki/SABR_volatility_model](https://en.wikipedia.org/wiki/SABR_volatility_model)
    
45. The SABR Model (Stochastic Alpha Beta Rho) - Genius Mathematics Consultants, acessado em julho 7, 2025, [https://mathematicsconsultants.com/2024/05/31/the-sabr-model-stochastic-alpha-beta-rho/](https://mathematicsconsultants.com/2024/05/31/the-sabr-model-stochastic-alpha-beta-rho/)
    
46. Stability of the SABR model - Deloitte, acessado em julho 7, 2025, [https://www2.deloitte.com/content/dam/Deloitte/global/Documents/be-aers-fsi-sabr-stability.pdf](https://www2.deloitte.com/content/dam/Deloitte/global/Documents/be-aers-fsi-sabr-stability.pdf)
    
47. How to understand the SABR model parameters | by Quant Prep - Medium, acessado em julho 7, 2025, [https://medium.com/@quant_prep/how-to-understand-the-sabr-model-parameters-fdccb7e57c0d](https://medium.com/@quant_prep/how-to-understand-the-sabr-model-parameters-fdccb7e57c0d)
    
48. What is the importance of alpha, beta, rho in the SABR volatility model?, acessado em julho 7, 2025, [https://quant.stackexchange.com/questions/39849/what-is-the-importance-of-alpha-beta-rho-in-the-sabr-volatility-model](https://quant.stackexchange.com/questions/39849/what-is-the-importance-of-alpha-beta-rho-in-the-sabr-volatility-model)
    
49. Calibrate a SABR model? - Quantitative Finance Stack Exchange, acessado em julho 7, 2025, [https://quant.stackexchange.com/questions/43341/calibrate-a-sabr-model](https://quant.stackexchange.com/questions/43341/calibrate-a-sabr-model)
    
50. ynouri/pysabr: SABR model Python implementation - GitHub, acessado em julho 7, 2025, [https://github.com/ynouri/pysabr](https://github.com/ynouri/pysabr)
    
51. delta hedging with the smile - ResearchGate, acessado em julho 7, 2025, [https://www.researchgate.net/profile/Sami-Vaehaemaa/publication/226498536_Delta_hedging_with_the_smile/links/00b4951908eb4c1c3e000000/Delta-hedging-with-the-smile.pdf](https://www.researchgate.net/profile/Sami-Vaehaemaa/publication/226498536_Delta_hedging_with_the_smile/links/00b4951908eb4c1c3e000000/Delta-hedging-with-the-smile.pdf)
    
52. Smile-Implied Hedging with Volatility Risk, acessado em julho 7, 2025, [https://biblos.hec.ca/biblio/libreacces/33730747.pdf](https://biblos.hec.ca/biblio/libreacces/33730747.pdf)
    
53. Vanna–Volga pricing - Wikipedia, acessado em julho 7, 2025, [https://en.wikipedia.org/wiki/Vanna%E2%80%93Volga_pricing](https://en.wikipedia.org/wiki/Vanna%E2%80%93Volga_pricing)
    
54. Vanna–Volga pricing - Wikipedia, acessado em julho 7, 2025, [https://en.wikipedia.org/wiki/Vanna-Volga_pricing](https://en.wikipedia.org/wiki/Vanna-Volga_pricing)
    
55. Chapter 5 The Greeks | The Derivatives Academy - Bookdown, acessado em julho 7, 2025, [https://bookdown.org/maxime_debellefroid/MyBook/the-greeks.html](https://bookdown.org/maxime_debellefroid/MyBook/the-greeks.html)
    
56. Delta Hedging Effect of Volatility - Menthor Q, acessado em julho 7, 2025, [https://menthorq.com/guide/delta-hedging-effect-of-volatility/](https://menthorq.com/guide/delta-hedging-effect-of-volatility/)
    
57. Vanna-volga pricing - EconStor, acessado em julho 7, 2025, [https://www.econstor.eu/bitstream/10419/40192/1/573774811.pdf](https://www.econstor.eu/bitstream/10419/40192/1/573774811.pdf)
    
58. The Vanna-Volga method for implied volatilities - Deriscope, acessado em julho 7, 2025, [https://www.deriscope.com/docs/The_Vanna_Volga_method_for_implied_volatilities_Castagna_Mercurio_2007.pdf](https://www.deriscope.com/docs/The_Vanna_Volga_method_for_implied_volatilities_Castagna_Mercurio_2007.pdf)
    
59. Vanna-Volga methods applied to FX derivatives: from theory to market practice - arXiv, acessado em julho 7, 2025, [https://arxiv.org/pdf/0904.1074](https://arxiv.org/pdf/0904.1074)
    
60. Calibrating volatility smiles with SABR - PyQuant News, acessado em julho 7, 2025, [https://www.pyquantnews.com/the-pyquant-newsletter/calibrating-volatility-smiles-with-sabr](https://www.pyquantnews.com/the-pyquant-newsletter/calibrating-volatility-smiles-with-sabr)
    
61. PnL of Continuously Delta Hedged Option : r/quant - Reddit, acessado em julho 7, 2025, [https://www.reddit.com/r/quant/comments/1igxipr/pnl_of_continuously_delta_hedged_option/](https://www.reddit.com/r/quant/comments/1igxipr/pnl_of_continuously_delta_hedged_option/)
    
62. delta-hedging is failing - Quantitative Finance Stack Exchange, acessado em julho 7, 2025, [https://quant.stackexchange.com/questions/33284/delta-hedging-is-failing](https://quant.stackexchange.com/questions/33284/delta-hedging-is-failing)
    

Vanna Volga and Smile-consistent Implied Volatility Surface of Equity Index Option Kun Huang, acessado em julho 7, 2025, [https://acfr.aut.ac.nz/__data/assets/pdf_file/0017/185300/172496-K-Huang-Vanna-Volga_Auckland.pdf](https://acfr.aut.ac.nz/__data/assets/pdf_file/0017/185300/172496-K-Huang-Vanna-Volga_Auckland.pdf)**