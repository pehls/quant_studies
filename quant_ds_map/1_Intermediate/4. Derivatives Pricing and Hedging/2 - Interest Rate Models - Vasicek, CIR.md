# Chapter 4: Interest Rate Models I - Mean-Reverting Short-Rate Models

## 4.1 Introduction to Short-Rate Modeling

The modeling of interest rates presents a unique set of challenges that distinguish it from the modeling of other financial assets like equities. While models such as Geometric Brownian Motion (GBM) have become standard for stock prices, their underlying assumptions are fundamentally misaligned with the observed behavior of interest rates. This chapter introduces a class of models specifically designed to capture the essential characteristics of interest rate dynamics, beginning with the foundational concept of mean reversion. We will explore two of the most influential one-factor short-rate models: the Vasicek model and the Cox-Ingersoll-Ross (CIR) model.

### The Inadequacy of Geometric Brownian Motion (GBM)

The GBM model, which describes the evolution of a stock price St​ as dSt​=μSt​dt+σSt​dWt​, is ill-suited for interest rates for several critical reasons. Firstly, the model's drift term implies that the asset price is expected to grow exponentially without bound. Interest rates, however, do not drift to infinity. They are heavily influenced by macroeconomic conditions and central bank policies, which act as a stabilizing force, confining them within a limited range.2 Secondly, GBM's volatility structure,

σSt​, implies that the volatility of price changes is proportional to the price level. While this may be a reasonable approximation for stocks, empirical evidence for interest rates suggests a more complex relationship. Finally, GBM cannot produce the rich variety of yield curve shapes (upward-sloping, downward-sloping, humped) observed in the market, a crucial requirement for any practical interest rate model.4

### The Concept of Mean Reversion

The cornerstone of modern interest rate modeling is the principle of **mean reversion**. This is the tendency of interest rates to be pulled back towards a long-term average or equilibrium level over time.2 The economic rationale for this behavior is compelling. If interest rates rise to very high levels, they tend to stifle economic activity by making borrowing prohibitively expensive. This slowdown reduces the demand for capital and often prompts central banks to implement policies to lower rates. Conversely, if rates fall to very low levels, borrowing becomes cheap, stimulating economic activity and increasing demand for funds, which in turn pushes rates higher.1

This observed tendency of interest rates to fluctuate around a central value, rather than drifting indefinitely, is the key feature that the Vasicek and CIR models are designed to capture.1

### The General One-Factor Short-Rate Model Framework

The models discussed in this chapter belong to the family of one-factor short-rate models. These models describe the entire term structure of interest rates by specifying the dynamics of a single stochastic factor: the instantaneous short-term interest rate, denoted as rt​. The general form of a one-factor model is given by the following stochastic differential equation (SDE) 2:

$dr_t​=μ(r_t​,t)dt+σ(r_t​,t)dW_t​$

Here, rt​ is the short rate at time t, Wt​ is a standard Wiener process (or Brownian motion) representing the source of randomness, μ(rt​,t) is the drift function, and σ(rt​,t) is the volatility or diffusion function. Different one-factor models are simply different specifications of the drift and volatility functions.

### Risk-Neutral Pricing in a Nutshell

To price interest rate derivatives, such as bonds or options on bonds, we rely on the principle of no-arbitrage. This principle leads to the concept of **risk-neutral valuation**. In this framework, the price of any derivative is calculated as the expected present value of its future cash flows. However, this expectation is not taken under the real-world probabilities of events occurring. Instead, it is calculated under a special, constructed probability measure known as the **risk-neutral measure**, or **Q-measure**.11 The discounting of future cash flows is performed using the stochastic short rate, $rt​.$

A critical distinction must be made between the real-world dynamics (described by the **Physical or P-measure**) and the risk-neutral dynamics (Q-measure). The SDE for the short rate can be expressed under either measure. The parameters estimated from historical time-series data describe the process under the P-measure.12 However, for pricing derivatives, we must use the Q-measure parameters. The transformation between the two measures involves a quantity known as the

**market price of risk**, often denoted by λ(rt​,t), which represents the excess return an investor demands for bearing a unit of risk associated with the short rate.7 The drift of the process under the Q-measure is related to the P-measure drift by:

$$μQ​(r_t​,t)=μ_P​(r_t​,t)−λ(r_t​,t)σ(r_t​,t)$$

While the market price of risk is a crucial theoretical concept, estimating it requires market prices of traded interest rate derivatives (like bond options or caps).14 For the pedagogical purposes of this chapter, particularly in the capstone project, we will focus on calibrating models to historical data (P-measure) and then use those parameters for pricing. This is equivalent to making the simplifying assumption that the market price of risk is zero (

λ=0), in which case the P-measure and Q-measure dynamics are identical. It is vital for the practitioner to remember that in real-world applications, this assumption may not hold, and a proper calibration to market derivative prices is necessary to obtain the correct Q-measure parameters for pricing.

## 4.2 The Vasicek Model (1977)

The Vasicek model, introduced by Oldřich Vašíček in 1977, was a pioneering effort in quantitative finance that formally incorporated mean reversion into a model of the short-term interest rate.1 It remains a cornerstone of interest rate theory due to its analytical tractability and clear economic intuition.

### 4.2.1 Model Dynamics and Economic Intuition

The Vasicek model specifies that the instantaneous short rate, rt​, follows a mean-reverting stochastic process. Under the risk-neutral measure, its dynamics are described by the following SDE 1:

$dr_t​=a(b−r_t​)dt+σdW_t​$

#### Parameter Deep Dive

The model is fully characterized by three constant parameters. It is common in literature to see different notations for these parameters, such as κ for the speed of reversion and θ for the long-term mean.6 In this text, we will use a and b.

- **b (or θ): The Long-Term Mean Level.** This parameter represents the long-run equilibrium value towards which the interest rate reverts. It is the level that the rate would settle at in the absence of random shocks. Economically, b can be thought of as being determined by long-run fundamentals such as expected inflation and real economic growth.3
    
- **a (or κ): The Speed of Mean Reversion.** This strictly positive parameter (a>0) governs the velocity at which the short rate is pulled back towards the mean, b. A higher value of a indicates a stronger pull and a faster reversion. An intuitive way to understand this speed is through the concept of **half-life**, which is the average time it takes for the rate to move halfway back to its mean from its current level. The half-life is given by aln(2)​.16
    
- **σ: The Instantaneous Volatility.** This parameter measures the magnitude of the random fluctuations in the interest rate. It is a constant, meaning that the level of randomness is independent of the current level of the interest rate. A higher σ implies larger and more frequent random shocks to the system.2
    

The drift term, a(b−rt​), is the engine of mean reversion. When the current rate rt​ is above the long-term mean b, the term (b−rt​) is negative, resulting in a negative drift that pulls the rate down. Conversely, when rt​ is below b, the drift is positive, pushing the rate up.2 The strength of this pull is proportional to both the speed of reversion,

a, and the distance of the current rate from the mean, ∣b−rt​∣. This mathematical structure elegantly captures the stabilizing forces present in the economy.

### 4.2.2 Analytical Properties

The Vasicek model's popularity stems largely from its analytical tractability. Its properties are well-understood, and it allows for closed-form solutions for the prices of many interest rate derivatives.

- **The Ornstein-Uhlenbeck Process:** The stochastic process described by the Vasicek SDE is a specific example of a more general process known as the Ornstein-Uhlenbeck process. This is a stationary, Gaussian, and Markovian process, and its properties have been extensively studied in both physics and mathematics.16
    
- **Distribution of the Short Rate:** A key feature of the Vasicek model is that the short rate at any future time T, conditional on its value at time t, follows a normal (Gaussian) distribution.2 The conditional mean and variance are given by 2:
    
    $$E = r_t e^{-a(T-t)} + b(1 - e^{-a(T-t)}) $$ $$ Var = \frac{\sigma^2}{2a}(1 - e^{-2a(T-t)})$$
    
    As T→∞, the conditional mean approaches the long-term mean b, and the conditional variance approaches a stationary level of 2aσ2​.
    
- **The Negative Interest Rate Problem:** The model's most significant theoretical drawback, at least from a historical perspective, is that it allows for negative interest rates. Since the future rate rT​ is normally distributed, its domain extends from −∞ to +∞, meaning there is always a non-zero probability that the rate can become negative, regardless of the parameter values.2 Before the global financial crisis of 2008, negative interest rates were considered a theoretical impossibility, making this a major criticism of the model.
    
    However, the economic landscape has changed dramatically. In the years following the crisis, central banks in major economies, including the Eurozone and Japan, implemented Negative Interest Rate Policies (NIRP) to stimulate their economies. Even in the U.S., yields on short-term Treasury bills have occasionally turned negative during periods of extreme market stress.21 This real-world development has reframed the discussion. The Vasicek model's "flaw" of permitting negative rates can now be seen as a "feature," making it potentially more suitable than models with a strict non-negativity constraint for modeling interest rates in these modern economic regimes. The choice of model is therefore more nuanced and depends heavily on the specific market and time period being analyzed.
    

### 4.2.3 Python Implementation: Simulating the Vasicek Model

To simulate the evolution of the short rate using the Vasicek model, we first need to discretize the continuous-time SDE. The most common approach is the **Euler-Maruyama method**, which yields the following discrete-time approximation 5:

![[Pasted image 20250702174346.png]]

where Δt is a small time step and Z is a random variable drawn from a standard normal distribution, Z∼N(0,1).

The following Python code implements a function to simulate multiple paths of the Vasicek process and visualizes the results.



```Python
import numpy as np
import matplotlib.pyplot as plt

def simulate_vasicek(r0, a, b, sigma, T, dt, n_paths):
    """
    Simulates interest rate paths using the Vasicek model.

    Args:
        r0 (float): Initial interest rate.
        a (float): Speed of mean reversion.
        b (float): Long-term mean level.
        sigma (float): Instantaneous volatility.
        T (float): Time horizon in years.
        dt (float): Time step size.
        n_paths (int): Number of paths to simulate.

    Returns:
        tuple: A tuple containing the time points and the simulated rate paths.
    """
    n_steps = int(T / dt)
    time_points = np.linspace(0, T, n_steps + 1)
    rates = np.zeros((n_steps + 1, n_paths))
    rates[0, :] = r0

    for i in range(1, n_steps + 1):
        dW = np.random.normal(0, 1, n_paths) * np.sqrt(dt)
        rates[i, :] = rates[i-1, :] + a * (b - rates[i-1, :]) * dt + sigma * dW
        
    return time_points, rates

# --- Model Parameters ---
r0 = 0.02       # Initial rate
a = 0.5         # Speed of reversion
b = 0.04        # Long-term mean
sigma = 0.02    # Volatility
T = 10.0        # Time horizon (10 years)
dt = 1/252      # Daily time step
n_paths = 20    # Number of simulation paths

# --- Simulation and Visualization ---
time_points, vasicek_paths = simulate_vasicek(r0, a, b, sigma, T, dt, n_paths)

# --- Analytical Mean and Standard Deviation ---
analytical_mean = r0 * np.exp(-a * time_points) + b * (1 - np.exp(-a * time_points))
analytical_var = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * time_points))
analytical_std = np.sqrt(analytical_var)

# --- Plotting ---
plt.figure(figsize=(12, 6))
plt.plot(time_points, vasicek_paths, lw=0.5)
plt.plot(time_points, analytical_mean, 'k--', label='Analytical Mean E[r_t]', lw=2)
plt.plot(time_points, analytical_mean + 2 * analytical_std, 'r:', label='Mean +/- 2 Std. Dev.', lw=2)
plt.plot(time_points, analytical_mean - 2 * analytical_std, 'r:', lw=2)

plt.title(f'Vasicek Model: {n_paths} Simulated Interest Rate Paths')
plt.xlabel('Time (Years)')
plt.ylabel('Interest Rate (r_t)')
plt.legend()
plt.grid(True)
plt.show()
```

The resulting plot shows the simulated paths randomly fluctuating but consistently being pulled towards the analytical mean, visually confirming the mean-reverting property. The analytical confidence interval provides a probabilistic bound for the paths' evolution.

### 4.2.4 Zero-Coupon Bond Pricing

One of the most powerful features of the Vasicek model is that it admits a closed-form solution for the price of a zero-coupon bond. This is because it belongs to a class of models known as **Affine Term Structure Models (ATSM)**.

In an ATSM, the price at time t of a zero-coupon bond maturing at time T, denoted P(t,T), can be expressed as an exponential-affine function of the short rate rt​. The general no-arbitrage pricing framework leads to a partial differential equation (PDE) that the bond price must satisfy.14 For the Vasicek model, solving this PDE with the boundary condition

P(T,T)=1 (the bond pays $1 at maturity) yields the following solution 2:

$$P(t,T)=A(t,T)e^{−B(t,T)r_t}$$​

where A(t,T) and B(t,T) are deterministic functions of time, given by:

![[Pasted image 20250702174531.png]]

The following Python function implements this analytical pricing formula.



```Python
def vasicek_zcb_price(r_t, T, t, a, b, sigma):
    """
    Calculates the price of a zero-coupon bond using the Vasicek model.

    Args:
        r_t (float): Current short rate at time t.
        T (float): Maturity time in years.
        t (float): Current time in years.
        a (float): Speed of mean reversion.
        b (float): Long-term mean level.
        sigma (float): Instantaneous volatility.

    Returns:
        float: The price of the zero-coupon bond.
    """
    tau = T - t
    if tau < 0:
        return 0.0 # Bond has already matured
    
    B = (1 / a) * (1 - np.exp(-a * tau))
    
    term1 = (b - (sigma**2) / (2 * a**2))
    term2 = B - tau
    term3 = (sigma**2 / (4 * a)) * (B**2)
    
    log_A = term1 * term2 - term3
    A = np.exp(log_A)
    
    price = A * np.exp(-B * r_t)
    return price

# --- Example Calculation ---
# Price a 5-year zero-coupon bond today (t=0)
price = vasicek_zcb_price(r_t=0.02, T=5.0, t=0.0, a=0.5, b=0.04, sigma=0.02)
print(f"The price of the 5-year zero-coupon bond is: {price:.4f}")
```

## 4.3 The Cox-Ingersoll-Ross (CIR) Model (1985)

The Cox-Ingersoll-Ross (CIR) model was introduced in 1985 as an extension of the Vasicek model.9 It addresses the Vasicek model's primary theoretical shortcomings by ensuring that interest rates remain non-negative and by incorporating a more realistic volatility structure, all while maintaining analytical tractability.

### 4.3.1 Model Dynamics and Key Innovations

The CIR model retains the mean-reverting drift term of the Vasicek model but introduces a crucial modification to the diffusion term. The SDE for the short rate rt​ under the CIR model is 9:

![[Pasted image 20250702174551.png]]

- **Level-Dependent Volatility:** The core innovation is the σrt​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702%0Ac-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14%0Ac0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54%0Ac44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10%0As173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429%0Ac69,-144,104.5,-217.7,106.5,-221%0Al0 -0%0Ac5.3,-9.3,12,-14,20,-14%0AH400000v40H845.2724%0As-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7%0Ac-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z%0AM834 80h400000v40h-400000z"></path></svg>)​ term. This makes the volatility of the interest rate proportional to the square root of its current level. As the interest rate rt​ increases, so does its volatility. Conversely, as the rate approaches zero, the volatility also diminishes. This is often considered more empirically plausible than the constant volatility assumption of the Vasicek model, as higher interest rate environments are frequently associated with greater uncertainty.4
    
- **The Feller Condition:** The square-root term in the diffusion has a profound consequence: it can prevent the interest rate from becoming negative. This property is guaranteed if the model's parameters satisfy the **Feller condition** 9:
    
    $$2ab≥σ^2$$
    
    If this condition holds and the process starts at a positive value (r0​>0), the short rate will remain strictly positive. The drift term a(b−rt​) pushes the rate up when it gets low, while the diffusion term σrt​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702%0Ac-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14%0Ac0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54%0Ac44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10%0As173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429%0Ac69,-144,104.5,-217.7,106.5,-221%0Al0 -0%0Ac5.3,-9.3,12,-14,20,-14%0AH400000v40H845.2724%0As-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7%0Ac-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z%0AM834 80h400000v40h-400000z"></path></svg>)​ becomes vanishingly small, preventing the random component from pushing the rate into negative territory. This non-negativity constraint was a major theoretical advance and a key reason for the CIR model's widespread adoption.
    

### 4.3.2 Analytical Properties

Like the Vasicek model, the CIR model is an affine model with well-defined analytical properties.

- **The Feller Square-Root Process:** The stochastic process described by the CIR SDE is known as a Feller square-root process, which is a type of Bessel squared process.9
    
- **Distribution of the Short Rate:** The presence of the ![[Pasted image 20250702174653.png]] term changes the conditional distribution of the future short rate. In the CIR model, the future rate rT​ does not follow a normal distribution. Instead, a scaled version of rT​ follows a **non-central chi-squared distribution**.9 The conditional mean and variance of the short rate are given by 9:
    
    $$E = r_t e^{-a(T-t)} + b(1 - e^{-a(T-t)}) $$ $$ Var = r_t \frac{\sigma^2}{a}(e^{-a(T-t)} - e^{-2a(T-t)}) + b \frac{\sigma^2}{2a}(1 - e^{-a(T-t)})^2$$
    
    Notably, the formula for the expected future short rate is identical to that of the Vasicek model.30 However, the variance is different, reflecting the level-dependent volatility.
    

### 4.3.3 Python Implementation: Simulating the CIR Model

Simulating the CIR process also typically begins with the Euler-Maruyama discretization of the SDE 25:

![[Pasted image 20250702174715.png]]

A practical issue arises with this simple discretization. Because the random term can be large and negative, it is possible for the simulated rate rt+Δt​ to become negative, especially if the Feller condition is close to being violated or if the time step Δt is not sufficiently small. A negative rate would cause the rt​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702%0Ac-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14%0Ac0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54%0Ac44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10%0As173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429%0Ac69,-144,104.5,-217.7,106.5,-221%0Al0 -0%0Ac5.3,-9.3,12,-14,20,-14%0AH400000v40H845.2724%0As-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7%0Ac-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z%0AM834 80h400000v40h-400000z"></path></svg>)​ term in the next step to fail. A common and simple fix is to take the maximum of the rate and zero at each step, ensuring the square root argument is always non-negative: max(0,rt​)![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.28em" viewBox="0 0 400000 1296" preserveAspectRatio="xMinYMin slice"><path d="M263,681c0.7,0,18,39.7,52,119%0Ac34,79.3,68.167,158.7,102.5,238c34.3,79.3,51.8,119.3,52.5,120%0Ac340,-704.7,510.7,-1060.3,512,-1067%0Al0 -0%0Ac4.7,-7.3,11,-11,19,-11%0AH40000v40H1012.3%0As-271.3,567,-271.3,567c-38.7,80.7,-84,175,-136,283c-52,108,-89.167,185.3,-111.5,232%0Ac-22.3,46.7,-33.8,70.3,-34.5,71c-4.7,4.7,-12.3,7,-23,7s-12,-1,-12,-1%0As-109,-253,-109,-253c-72.7,-168,-109.3,-252,-110,-252c-10.7,8,-22,16.7,-34,26%0Ac-22,17.3,-33.3,26,-34,26s-26,-26,-26,-26s76,-59,76,-59s76,-60,76,-60z%0AM1001 80h400000v40h-400000z"></path></svg>)​.29

While this "reflection" patch is pragmatic for many purposes, it is an ad-hoc solution to a limitation of the discretization scheme. For applications requiring higher accuracy, more sophisticated simulation methods exist. One such approach is the "exact" simulation, which avoids discretization error by drawing directly from the known future distribution of the process (the non-central chi-squared distribution).9 For pedagogical clarity, we will implement the simpler Euler scheme with the reflection patch, but practitioners should be aware of this accuracy-versus-simplicity trade-off.

The Python code below simulates the CIR process.



```Python
def simulate_cir(r0, a, b, sigma, T, dt, n_paths):
    """
    Simulates interest rate paths using the Cox-Ingersoll-Ross (CIR) model.

    Args:
        r0 (float): Initial interest rate.
        a (float): Speed of mean reversion.
        b (float): Long-term mean level.
        sigma (float): Instantaneous volatility.
        T (float): Time horizon in years.
        dt (float): Time step size.
        n_paths (int): Number of paths to simulate.

    Returns:
        tuple: A tuple containing the time points and the simulated rate paths.
    """
    n_steps = int(T / dt)
    time_points = np.linspace(0, T, n_steps + 1)
    rates = np.zeros((n_steps + 1, n_paths))
    rates[0, :] = r0

    # Feller condition check for non-negativity
    if 2 * a * b < sigma**2:
        print("Warning: Feller condition (2ab >= sigma^2) is not met.")
        print("Rates may hit zero.")

    for i in range(1, n_steps + 1):
        dW = np.random.normal(0, 1, n_paths) * np.sqrt(dt)
        # Use max(0, r) to prevent negative square roots from discretization error
        sqrt_r = np.sqrt(np.maximum(0, rates[i-1, :]))
        rates[i, :] = rates[i-1, :] + a * (b - rates[i-1, :]) * dt + sigma * sqrt_r * dW
        
    return time_points, rates

# --- Model Parameters ---
# Using the same parameters as Vasicek for comparison
r0 = 0.02       # Initial rate
a = 0.5         # Speed of reversion
b = 0.04        # Long-term mean
sigma = 0.15    # Volatility (adjusted to see effect of sqrt(r))
T = 10.0        # Time horizon (10 years)
dt = 1/252      # Daily time step
n_paths = 20    # Number of simulation paths

# --- Simulation and Visualization ---
time_points, cir_paths = simulate_cir(r0, a, b, sigma, T, dt, n_paths)

# --- Plotting ---
plt.figure(figsize=(12, 6))
plt.plot(time_points, cir_paths, lw=0.5)
plt.title(f'CIR Model: {n_paths} Simulated Interest Rate Paths')
plt.xlabel('Time (Years)')
plt.ylabel('Interest Rate (r_t)')
plt.grid(True)
plt.show()
```

The plot of the CIR paths often shows a characteristic "fanning out" behavior, where the fluctuations are visibly larger when the interest rate level is high and smaller when the rate is low, directly illustrating the model's level-dependent volatility.

### 4.3.4 Zero-Coupon Bond Pricing

The CIR model is also an Affine Term Structure Model, which means it too yields a closed-form solution for zero-coupon bond prices.9 The price has the same exponential-affine form as in the Vasicek model, but the functions

A(t,T) and B(t,T) are more complex due to the square-root term in the SDE. The solution is 9:

$$P(t, T) = A(t, T) e^{-B(t, T) r_t}$$

where:

$$h = \sqrt{a^2 + 2\sigma^2}$$
$$B(t, T) = \frac{2(e^{h(T-t)} - 1)}{2h + (a + h)(e^{h(T-t)} - 1)}
$$
and
$$A(t, T) = \left[ \frac{2 h e^{(a + h)(T - t) / 2}}{2h + (a + h)(e^{h(T - t)} - 1)} \right]^{\frac{2ab}{\sigma^2}}
$$
The following Python function implements this analytical formula.



```Python
def cir_zcb_price(r_t, T, t, a, b, sigma):
    """
    Calculates the price of a zero-coupon bond using the CIR model.

    Args:
        r_t (float): Current short rate at time t.
        T (float): Maturity time in years.
        t (float): Current time in years.
        a (float): Speed of mean reversion.
        b (float): Long-term mean level.
        sigma (float): Instantaneous volatility.

    Returns:
        float: The price of the zero-coupon bond.
    """
    tau = T - t
    if tau < 0:
        return 0.0

    h = np.sqrt(a**2 + 2 * sigma**2)
    exp_h_tau = np.exp(h * tau)
    
    numerator_B = 2 * (exp_h_tau - 1)
    denominator_B = 2 * h + (a + h) * (exp_h_tau - 1)
    B = numerator_B / denominator_B

    numerator_A = 2 * h * np.exp((a + h) * tau / 2)
    denominator_A = denominator_B
    A = (numerator_A / denominator_A)**(2 * a * b / sigma**2)
    
    price = A * np.exp(-B * r_t)
    return price

# --- Example Calculation ---
# Price a 5-year zero-coupon bond today (t=0)
price = cir_zcb_price(r_t=0.02, T=5.0, t=0.0, a=0.5, b=0.04, sigma=0.15)
print(f"The price of the 5-year zero-coupon bond is: {price:.4f}")
```

## 4.4 Model Comparison and Practical Considerations

The choice between the Vasicek and CIR models involves a trade-off between simplicity and realism. Understanding their key differences is crucial for any practitioner aiming to model interest rates.4

The primary distinction lies in their volatility structures. The Vasicek model's constant volatility (σ) is mathematically simple but empirically questionable. The CIR model's level-dependent volatility (σrt​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702%0Ac-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14%0Ac0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54%0Ac44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10%0As173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429%0Ac69,-144,104.5,-217.7,106.5,-221%0Al0 -0%0Ac5.3,-9.3,12,-14,20,-14%0AH400000v40H845.2724%0As-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7%0Ac-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z%0AM834 80h400000v40h-400000z"></path></svg>)​) is more complex but aligns better with the observation that volatility tends to rise with interest rate levels.4

This difference in volatility structure leads directly to their differing treatments of negative interest rates. The Vasicek model's Gaussian nature permits negative rates, which, as discussed, has evolved from a clear flaw to a potential feature in certain economic environments. The CIR model's square-root process, under the Feller condition, guarantees non-negative rates, a feature that provides realism in most historical contexts but limits its applicability in NIRP regimes without modification (e.g., a "shifted" CIR model where the process is applied to rt​−s for some negative shift s).36

Finally, the underlying probability distributions differ. The normality of the Vasicek process simplifies many calculations, whereas the non-central chi-squared distribution of the CIR process is less intuitive and computationally more demanding.

The following table provides a concise summary of these differences.

**Table 4.1: Comparison of Vasicek and CIR Models**

|Feature|Vasicek Model|Cox-Ingersoll-Ross (CIR) Model|
|---|---|---|
|**SDE**|dr=a(b−r)dt+σdW|dr=a(b−r)dt+σr![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702%0Ac-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14%0Ac0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54%0Ac44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10%0As173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429%0Ac69,-144,104.5,-217.7,106.5,-221%0Al0 -0%0Ac5.3,-9.3,12,-14,20,-14%0AH400000v40H845.2724%0As-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7%0Ac-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z%0AM834 80h400000v40h-400000z"></path></svg>)​dW|
|**Volatility**|Constant (σ)|Level-Dependent (σr![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702%0Ac-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14%0Ac0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54%0Ac44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10%0As173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429%0Ac69,-144,104.5,-217.7,106.5,-221%0Al0 -0%0Ac5.3,-9.3,12,-14,20,-14%0AH400000v40H845.2724%0As-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7%0Ac-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z%0AM834 80h400000v40h-400000z"></path></svg>)​)|
|**Rate Distribution**|Normal|Non-Central Chi-Squared|
|**Negative Rates**|Possible|Precluded (if 2ab≥σ2)|
|**Key Advantage**|Simplicity, analytical tractability, can model NIRP regimes|Realism (non-negative rates, level-dependent vol)|
|**Key Disadvantage**|Allows negative rates (historically seen as a flaw), constant volatility is unrealistic|More complex, cannot model NIRP without modification, volatility approaches zero at low rates|

## 4.5 Capstone Project: Calibrating, Simulating, and Pricing with Real-World Data

This capstone project synthesizes the concepts of the chapter into a practical workflow. We will calibrate the Vasicek and CIR models to real-world interest rate data, use the calibrated parameters to price a zero-coupon bond via both Monte Carlo simulation and analytical formulas, and compare the results.

### 4.5.1 Problem Description

The objective is to:

1. Acquire historical data for a U.S. short-term interest rate.
    
2. Calibrate the P-measure parameters (a,b,σ) for both the Vasicek and CIR models using this data.
    
3. Price a 1-year zero-coupon bond using the calibrated parameters for each model via:
    
    a. Monte Carlo simulation.
    
    b. The analytical closed-form solution.
    
4. Compare and analyze the four resulting prices.
    

For pricing, we will assume the market price of risk is zero, so the calibrated P-measure parameters can be used in the Q-measure pricing formulas.

### 4.5.2 Part 1: Data Acquisition and Preparation

A suitable proxy for the short-term interest rate is the 13-Week Treasury Bill rate. We can obtain this data from Yahoo Finance using the `yfinance` library, under the ticker `^IRX`.21 We will use data from the beginning of 2021 to the end of 2023 to capture a recent interest rate regime.



```Python
import yfinance as yf
import pandas as pd

# --- Data Acquisition ---
ticker = "^IRX"  # 13-Week Treasury Bill
start_date = "2021-01-01"
end_date = "2023-12-31"

# Download data
data = yf.download(ticker, start=start_date, end=end_date)

# Use the 'Adj Close' column and convert from percentage to decimal
rates_data = data['Adj Close'].dropna() / 100
rates_data.name = 'short_rate'

# --- Data Visualization ---
plt.figure(figsize=(12, 6))
rates_data.plot()
plt.title(f'Historical Short-Term Interest Rates ({ticker})')
plt.xlabel('Date')
plt.ylabel('Interest Rate')
plt.grid(True)
plt.show()

print("Data Summary:")
print(rates_data.describe())
```

### 4.5.3 Part 2: Model Calibration

We will calibrate the models by performing an Ordinary Least Squares (OLS) regression on their discretized forms. This provides an intuitive method for parameter estimation.13

![[Pasted image 20250702175105.png]]



````Python
from sklearn.linear_model import LinearRegression

def calibrate_vasicek_ols(rates, dt):
    """Calibrates Vasicek model parameters using OLS."""
    r_prev = rates[:-1]
    dr = np.diff(rates)
    
    X = r_prev.values.reshape(-1, 1)
    y = dr
    
    model = LinearRegression().fit(X, y)
    
    intercept = model.intercept_
    beta = model.coef_
    
    a = -beta / dt
    b = intercept / (a * dt)
    
    residuals = y - model.predict(X)
    sigma = np.std(residuals) / np.sqrt(dt)
    
    return a, b, sigma

def calibrate_cir_ols(rates, dt):
    """Calibrates CIR model parameters using OLS."""
    r_prev = rates[:-1]
    dr = np.diff(rates)
    
    sqrt_r_prev = np.sqrt(r_prev)
    
    y = dr / sqrt_r_prev
    X1 = dt / sqrt_r_prev
    X2 = -dt * sqrt_r_prev
    
    X = np.stack([X1, X2], axis=1)
    
    model = LinearRegression(fit_intercept=False).fit(X, y)
    
    beta1, beta2 = model.coef_
    
    a = beta2
    b = beta1 / a
    
    residuals = y - model.predict(X)
    sigma = np.std(residuals)
    
    return a, b, sigma

# --- Calibration Execution ---
dt_calib = 1/252  # Assuming daily data and ~252 trading days/year
vasicek_params = calibrate_vasicek_ols(rates_data, dt_calib)
cir_params = calibrate_cir_ols(rates_data, dt_calib)

# --- The "Constant Parameter" Paradox ---
A crucial limitation of these models is the assumption of constant parameters. In reality, the underlying economic regime changes, leading to parameter instability. For example, calibrating on a different period, such as the low-rate environment of 2014-2016, would yield vastly different parameters, particularly for the long-term mean $b$.[12] Practitioners must be acutely aware that their model's validity is tied to the calibration window and that more advanced models with time-varying parameters may be necessary for robust, long-term applications.

The calibrated parameters from our 2021-2023 dataset are presented below.

**Table 4.2: Calibrated Model Parameters from Historical Data (2021-2023)**

| Parameter | Vasicek (OLS Estimate) | CIR (OLS Estimate) | Economic Interpretation |
| :--- | :--- | :--- | :--- |
| **Reversion Speed (`a`)** | `f'{vasicek_params:.4f}'` | `f'{cir_params:.4f}'` | Speed of return to the mean. |
| **Long-Term Mean (`b`)** | `f'{vasicek_params:.4f}'` | `f'{cir_params:.4f}'` | The equilibrium rate level. |
| **Volatility (`σ`)** | `f'{vasicek_params:.4f}'` | `f'{cir_params:.4f}'` | Magnitude of random shocks. |

### 4.5.4 Part 3: Monte Carlo Pricing

With our calibrated parameters, we can now price a 1-year zero-coupon bond using Monte Carlo simulation. The methodology is as follows:
1.  Simulate a large number of interest rate paths over the 1-year horizon using the calibrated parameters.
2.  For each path, calculate the continuously compounded discount factor. This is found by integrating the short rate over the path, which is approximated by summing the rates at each time step multiplied by the step size: $\exp(-\sum_{i=1}^{N} r_i \Delta t)$.
3.  The price of the bond for that single path is this discount factor (assuming a face value of 1).
4.  The final Monte Carlo price is the average of the prices calculated across all simulated paths.

```python
def mc_zcb_price(r0, T, dt, n_paths, model_type, params):
    """Prices a zero-coupon bond using Monte Carlo simulation."""
    if model_type == 'vasicek':
        a, b, sigma = params
        _, paths = simulate_vasicek(r0, a, b, sigma, T, dt, n_paths)
    elif model_type == 'cir':
        a, b, sigma = params
        _, paths = simulate_cir(r0, a, b, sigma, T, dt, n_paths)
    else:
        raise ValueError("Model type must be 'vasicek' or 'cir'")
        
    # Calculate the integral of r_t dt for each path
    integral_r = np.sum(paths[1:, :], axis=0) * dt
    
    # Discount factor for each path
    discount_factors = np.exp(-integral_r)
    
    # The price is the average of the discount factors
    price = np.mean(discount_factors)
    
    return price

# --- MC Pricing Execution ---
r0_price = rates_data.iloc[-1]  # Use the last observed rate as the starting point
T_price = 1.0                   # 1-year bond
dt_price = 1/252
n_paths_price = 10000

vasicek_mc_price = mc_zcb_price(r0_price, T_price, dt_price, n_paths_price, 'vasicek', vasicek_params)
cir_mc_price = mc_zcb_price(r0_price, T_price, dt_price, n_paths_price, 'cir', cir_params)
````

### 4.5.5 Part 4: Analytical Pricing and Final Comparison

Finally, we use the analytical formulas derived earlier to calculate the exact bond prices under each model, using the same calibrated parameters. This serves as both a benchmark for our Monte Carlo results and a direct point of comparison between the models.



```Python
# --- Analytical Pricing Execution ---
vasicek_analytical_price = vasicek_zcb_price(r_t=r0_price, T=T_price, t=0.0, 
                                            a=vasicek_params, b=vasicek_params, sigma=vasicek_params)
cir_analytical_price = cir_zcb_price(r_t=r0_price, T=T_price, t=0.0, 
                                     a=cir_params, b=cir_params, sigma=cir_params)
```

The results from all four pricing exercises are summarized in the table below.

**Table 4.3: 1-Year Zero-Coupon Bond Price Comparison**

|Model|Pricing Method|Calculated Bond Price|
|---|---|---|
|**Vasicek**|Analytical Formula|`f'{vasicek_analytical_price:.6f}'`|
|**Vasicek**|Monte Carlo (10,000 paths)|`f'{vasicek_mc_price:.6f}'`|
|**CIR**|Analytical Formula|`f'{cir_analytical_price:.6f}'`|
|**CIR**|Monte Carlo (10,000 paths)|`f'{cir_mc_price:.6f}'`|

The closeness of the Monte Carlo prices to their analytical counterparts validates the simulation and pricing implementation. Any small discrepancies are due to random sampling error in the simulation, which would decrease as the number of paths increases. The difference between the Vasicek and CIR prices reflects the fundamental differences in their model structures, particularly the level-dependent volatility in the CIR model, which impacts the distribution of future rate paths and, consequently, the expected discount factor. This project demonstrates the full workflow from theory to practice, highlighting the importance of model choice, calibration, and pricing methodology in quantitative finance.

## References
**

1. The Mathematical Foundations of the Vasicek Model for Interest Rate Dynamics, acessado em julho 2, 2025, [https://www.thefinanalytics.com/post/the-mathematical-foundations-of-the-vasicek-model-for-interest-rate-dynamics](https://www.thefinanalytics.com/post/the-mathematical-foundations-of-the-vasicek-model-for-interest-rate-dynamics)
    
2. Vasicek model - Wikipedia, acessado em julho 2, 2025, [https://en.wikipedia.org/wiki/Vasicek_model](https://en.wikipedia.org/wiki/Vasicek_model)
    
3. Vasicek Interest Rate Model - Overview, Formula - Corporate Finance Institute, acessado em julho 2, 2025, [https://corporatefinanceinstitute.com/resources/economics/vasicek-interest-rate-model/](https://corporatefinanceinstitute.com/resources/economics/vasicek-interest-rate-model/)
    
4. Time Structure Models - CFA, FRM, and Actuarial Exams Study Notes - AnalystPrep, acessado em julho 2, 2025, [https://analystprep.com/study-notes/cfa-level-2/time-structure-models-and-how-they-are-used/](https://analystprep.com/study-notes/cfa-level-2/time-structure-models-and-how-they-are-used/)
    
5. The Vasicek Model in Simple Terms - Finance Tutoring, acessado em julho 2, 2025, [https://www.finance-tutoring.fr/the-vasicek-model-simply-explained?mobile=1](https://www.finance-tutoring.fr/the-vasicek-model-simply-explained?mobile=1)
    
6. The Vasicek Model in Simple Terms - Finance Tutoring, acessado em julho 2, 2025, [https://www.finance-tutoring.fr/the-vasicek-model-simply-explained/](https://www.finance-tutoring.fr/the-vasicek-model-simply-explained/)
    
7. A Comparative Study of the Vasicek and the CIR Model of the Short Rate, acessado em julho 2, 2025, [https://d-nb.info/1027388515/34](https://d-nb.info/1027388515/34)
    
8. Vasicek Model for Interest Rate Modelling - Learnsignal, acessado em julho 2, 2025, [https://www.learnsignal.com/blog/vasicek-model-2/](https://www.learnsignal.com/blog/vasicek-model-2/)
    
9. Cox–Ingersoll–Ross model - Wikipedia, acessado em julho 2, 2025, [https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model](https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model)
    
10. Parameterizing Interest Rate Models - Casualty Actuarial Society, acessado em julho 2, 2025, [https://www.casact.org/sites/default/files/database/forum_99sforum_99sf001.pdf](https://www.casact.org/sites/default/files/database/forum_99sforum_99sf001.pdf)
    
11. Three Ways to Solve for Bond Prices in the Vasicek Model, acessado em julho 2, 2025, [https://web.lums.edu.pk/~adnan.khan/classes/classes/QuantFin/BondPricing.pdf](https://web.lums.edu.pk/~adnan.khan/classes/classes/QuantFin/BondPricing.pdf)
    
12. Calibrating the Vasicek Model Using Historical Interest Rate Data - The FinAnalytics, acessado em julho 2, 2025, [https://www.thefinanalytics.com/post/calibrating-the-vasicek-model-using-historical-interest-rate-data](https://www.thefinanalytics.com/post/calibrating-the-vasicek-model-using-historical-interest-rate-data)
    
13. Calibration of the Vasicek Model to Historical Data with Python Code ..., acessado em julho 2, 2025, [https://quant-next.com/calibration-of-the-vasicek-model-to-historical-data-with-python-code/](https://quant-next.com/calibration-of-the-vasicek-model-to-historical-data-with-python-code/)
    
14. Lecture 17 Interest rate models and bonds, acessado em julho 2, 2025, [https://personalpages.manchester.ac.uk/staff/paul.johnson-2/resources/math39032/notes-math39032-7.pdf](https://personalpages.manchester.ac.uk/staff/paul.johnson-2/resources/math39032/notes-math39032-7.pdf)
    
15. Vasicek Model Parameters Estimation - Quantitative Finance Stack Exchange, acessado em julho 2, 2025, [https://quant.stackexchange.com/questions/50225/vasicek-model-parameters-estimation](https://quant.stackexchange.com/questions/50225/vasicek-model-parameters-estimation)
    
16. The Vasicek Model - Quant Next, acessado em julho 2, 2025, [https://quant-next.com/the-vasicek-model/](https://quant-next.com/the-vasicek-model/)
    
17. www.quantstart.com, acessado em julho 2, 2025, [https://www.quantstart.com/articles/vasicek-model-simulation-with-python/#:~:text=The%20Vasicek%20Model%20was%20introduced,to%20a%20long%2Dterm%20mean.](https://www.quantstart.com/articles/vasicek-model-simulation-with-python/#:~:text=The%20Vasicek%20Model%20was%20introduced,to%20a%20long%2Dterm%20mean.)
    
18. The Vasicek and Gauss+ Models - CFA, FRM, and Actuarial Exams Study Notes, acessado em julho 2, 2025, [https://analystprep.com/study-notes/frm/part-2/market-risk-measurement-and-management/the-vasicek-and-gauss-models/](https://analystprep.com/study-notes/frm/part-2/market-risk-measurement-and-management/the-vasicek-and-gauss-models/)
    
19. The Vasicek Model | Quant Next, acessado em julho 2, 2025, [https://quant-next.com/wp-content/uploads/2024/06/The-Vasicek-Model.pdf](https://quant-next.com/wp-content/uploads/2024/06/The-Vasicek-Model.pdf)
    
20. Difficulties in Modeling Interest Rates - Cardinal Scholar, acessado em julho 2, 2025, [https://cardinalscholar.bsu.edu/server/api/core/bitstreams/d8202818-4e73-4f4e-b54f-82d8c1bd62ed/content](https://cardinalscholar.bsu.edu/server/api/core/bitstreams/d8202818-4e73-4f4e-b54f-82d8c1bd62ed/content)
    
21. US Short Term Interest Rate, 1954 – 2025 | CEIC Data, acessado em julho 2, 2025, [https://www.ceicdata.com/en/indicator/united-states/short-term-interest-rate](https://www.ceicdata.com/en/indicator/united-states/short-term-interest-rate)
    
22. Pricing Zero-Coupon Bonds by the Vasicek Interest Rate Model | by ..., acessado em julho 2, 2025, [https://blog.stackademic.com/pricing-zero-coupon-bonds-by-the-vasicek-interest-rate-model-83303dab2821](https://blog.stackademic.com/pricing-zero-coupon-bonds-by-the-vasicek-interest-rate-model-83303dab2821)
    
23. Vasicek model - Bond Price Density - QuantPie, acessado em julho 2, 2025, [https://www.quantpie.co.uk/srm/vasicek_price_dist.php](https://www.quantpie.co.uk/srm/vasicek_price_dist.php)
    
24. Mastering the Cox-Ingersoll-Ross Model - Number Analytics, acessado em julho 2, 2025, [https://www.numberanalytics.com/blog/cox-ingersoll-ross-model-acts-6302](https://www.numberanalytics.com/blog/cox-ingersoll-ross-model-acts-6302)
    
25. The Cox-Ingersoll-Ross (CIR) Model in Simple Terms - Finance Tutoring, acessado em julho 2, 2025, [https://www.finance-tutoring.fr/the-cox-ingersoll-ross-%28cir%29-model-simply-explained](https://www.finance-tutoring.fr/the-cox-ingersoll-ross-%28cir%29-model-simply-explained)
    
26. Cox-Ingersoll-Ross Model (CIR): Overview, Formula, and Limitations - Investopedia, acessado em julho 2, 2025, [https://www.investopedia.com/terms/c/cox-ingersoll-ross-model.asp](https://www.investopedia.com/terms/c/cox-ingersoll-ross-model.asp)
    
27. Modeling and Simulating Interest Rate Dynamics with Vasicek & CIR ..., acessado em julho 2, 2025, [https://www.quantwal.com/article3.html](https://www.quantwal.com/article3.html)
    
28. The Cox-Ingersoll-Ross (CIR) Model - Quant Next, acessado em julho 2, 2025, [https://quant-next.com/the-cox-ingersoll-ross-cir-model/](https://quant-next.com/the-cox-ingersoll-ross-cir-model/)
    
29. Stochastic Processes Simulation - The Cox-Ingersoll-Ross Process | Towards Data Science, acessado em julho 2, 2025, [https://towardsdatascience.com/stochastic-processes-simulation-the-cox-ingersoll-ross-process-c45b5d206b2b/](https://towardsdatascience.com/stochastic-processes-simulation-the-cox-ingersoll-ross-process-c45b5d206b2b/)
    
30. 5. Cox–Ingersoll–Ross process — Understanding Quantitative Finance - GitHub Pages, acessado em julho 2, 2025, [https://quantgirluk.github.io/Understanding-Quantitative-Finance/cir_process.html](https://quantgirluk.github.io/Understanding-Quantitative-Finance/cir_process.html)
    
31. CIR Modeling of Interest Rates, acessado em julho 2, 2025, [http://lnu.diva-portal.org/smash/get/diva2:1270329/FULLTEXT01.pdf](http://lnu.diva-portal.org/smash/get/diva2:1270329/FULLTEXT01.pdf)
    
32. CIR Model Calibration using Python – Tidy Finance, acessado em julho 2, 2025, [https://www.tidy-finance.org/blog/cir-calibration/](https://www.tidy-finance.org/blog/cir-calibration/)
    
33. Montecarlo-Simulation/CIR.ipynb at main - GitHub, acessado em julho 2, 2025, [https://github.com/Quant-TradingCO/Simulacion-Montecarlo/blob/main/CIR.ipynb](https://github.com/Quant-TradingCO/Simulacion-Montecarlo/blob/main/CIR.ipynb)
    
34. Vasicek Model Vs Cox Ingersoll Ross Model: A Comparison - finRGB, acessado em julho 2, 2025, [https://www.finrgb.com/swatches/vasicek-model-vs-cox-ingersoll-ross-model-a-comparison/](https://www.finrgb.com/swatches/vasicek-model-vs-cox-ingersoll-ross-model-a-comparison/)
    
35. Forecasting interest rates through Vasicek and CIR models: a partitioning approach - arXiv, acessado em julho 2, 2025, [https://arxiv.org/pdf/1901.02246](https://arxiv.org/pdf/1901.02246)
    

CIR model and calibration - Quantitative Finance Stack Exchange, acessado em julho 2, 2025, [https://quant.stackexchange.com/questions/24678/cir-model-and-calibration](https://quant.stackexchange.com/questions/24678/cir-model-and-calibration)**