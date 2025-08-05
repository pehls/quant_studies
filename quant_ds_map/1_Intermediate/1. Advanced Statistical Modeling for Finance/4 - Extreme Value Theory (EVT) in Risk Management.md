# Chapter 4: Advanced Statistical Modeling for Finance: Extreme Value Theory (EVT) in Risk Management

## 4.1 The Tyranny of the Bell Curve: Why Traditional Risk Models Fail

The history of financial markets is punctuated by sudden, violent events that defy conventional statistical explanation. From the Black Monday crash of 1987 to the Global Financial Crisis (GFC) of 2008, these episodes have repeatedly demonstrated the profound inadequacy of traditional risk management frameworks.1 The core failing of these frameworks lies in their foundational assumptions about the nature of asset returns. For decades, many risk models were built upon the convenient and mathematically tractable assumption of a normal (or Gaussian) distribution. However, real-world financial data tells a starkly different story.

The normal distribution, with its familiar bell shape, assigns vanishingly small probabilities to events that are many standard deviations away from the mean. A "six-sigma" event, for instance, is considered so rare under a normal distribution that it should occur less than once in a million years. Yet, in financial markets, such events seem to occur with unsettling regularity. This discrepancy arises because the empirical distribution of asset returns is not normal; it is characterized by **leptokurtosis**, or "fat tails." This means that extreme outcomes—both large gains and, more critically, catastrophic losses—are far more probable than a Gaussian model would suggest.3

This empirical reality was a key driver for the development and adoption of more sophisticated risk management techniques. The 2008 GFC, in particular, intensified the need for robust methods among financial institutions and insurance companies, as models that worked well under "normal market conditions" proved disastrously wrong during periods of high stress.1 The challenge is that extreme events are not merely larger versions of normal fluctuations; they often appear to be drawn from a different statistical process altogether.

To illustrate this fundamental mismatch, consider the daily log-returns of the S&P 500 index. By plotting a histogram of these returns against a fitted normal distribution, the divergence becomes visually apparent.


```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Download S&P 500 data
sp500 = yf.download('^GSPC', start='2000-01-01', end='2023-12-31')

# Calculate daily log returns
sp500 = np.log(sp500['Adj Close'] / sp500['Adj Close'].shift(1))
sp500.dropna(inplace=True)

# Plotting the histogram and the fitted normal distribution
plt.figure(figsize=(12, 7))
plt.hist(sp500, bins=100, density=True, alpha=0.6, label='S&P 500 Daily Log Returns')

# Fit a normal distribution to the data
mu, std = norm.fit(sp500)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')

title = f"S&P 500 Daily Log Returns vs. Normal Distribution\nFit Results: mu = {mu:.4f},  std = {std:.4f}"
plt.title(title)
plt.xlabel("Log Return")
plt.ylabel("Density")
plt.legend()
plt.show()

# Print Kurtosis
print(f"Kurtosis of S&P 500 Log Returns: {sp500.kurtosis():.2f}")
print("A normal distribution has a kurtosis of 0 (or 3, depending on convention).")
```

The output of this code will show a distribution that is more peaked at the center (higher kurtosis) and has noticeably fatter tails than the overlaid normal distribution curve. The calculated kurtosis will be significantly greater than that of a normal distribution, providing quantitative proof of the "fat tail" phenomenon.

This is where **Extreme Value Theory (EVT)** enters the picture. EVT is a branch of statistics that provides a dedicated, principled framework for analyzing the behavior of extreme values, rather than focusing on the central tendency of a distribution.5 It offers a set of tools to model the tail of a distribution directly, allowing for a more accurate assessment of the probability and magnitude of rare events.7 In the landscape of quantitative finance, EVT represents a critical evolution in thinking, ranking in importance alongside the foundational portfolio theory of Markowitz and the option pricing models of Black-Scholes-Merton.1 It provides a means to move beyond managing risk in "normal" times and to start quantifying the catastrophic potential of "extreme" market conditions.

## 4.2 The Twin Pillars: Foundational Theorems of EVT

The theoretical power of Extreme Value Theory rests on two fundamental theorems, which play a role for extreme values analogous to the role the Central Limit Theorem (CLT) plays for sample averages.3 The CLT is powerful because it states that, for a sufficiently large sample, the distribution of the sample mean will be approximately normal, regardless of the underlying distribution from which the sample was drawn (provided it has a finite variance). This allows for powerful statistical inference about averages.

Similarly, the foundational theorems of EVT describe the limiting distributions for extreme values, providing a solid theoretical basis for modeling the tails of distributions without needing to make strong assumptions about the entire underlying distribution.

### Pillar 1: The Fisher-Tippett-Gnedenko Theorem and Block Maxima

The first pillar of EVT is the **Fisher-Tippett-Gnedenko Theorem**, also known as the extreme value theorem.8 Developed through the work of Fréchet (1927), Fisher and Tippett (1928), and Gnedenko (1943), this theorem describes the asymptotic distribution of the maximum value of a sample.8

Let $X1​,X2​,…,Xn$​ be a sequence of independent and identically-distributed (i.i.d.) random variables. Let $Mn​=max(X1​,X2​,…,Xn​)$ be the maximum of this sample. The Fisher-Tippett-Gnedenko theorem states that if there exist sequences of normalizing constants an​>0 (a scale parameter) and bn​ (a location parameter) such that the distribution of the normalized maximum converges to a non-degenerate distribution G as n→∞:

![[Pasted image 20250628233739.png]]

then the limiting distribution G(z) must belong to one of only three possible families: the Gumbel, the Fréchet, or the Weibull distribution.8

Crucially, these three distinct distributions can be unified into a single, flexible family of distributions known as the **Generalized Extreme Value (GEV) distribution**.8 This powerful result means that, under very general conditions, the behavior of the maximum of a large sample can be modeled by the GEV distribution, regardless of the original "parent" distribution of the individual data points. This is the theoretical basis for the

**Block Maxima (BM)** approach to EVT, where data is divided into blocks and the maximum of each block is modeled.3

### Pillar 2: The Pickands-Balkema-de Haan Theorem and Threshold Exceedances

While the Block Maxima approach is foundational, it can be inefficient, as it discards all data within a block except for the single maximum value.11 This led to the development of a more modern and data-efficient approach known as

**Peaks-Over-Threshold (POT)**. The theoretical underpinning for this method is the **Pickands-Balkema-de Haan Theorem**.3

This theorem focuses not on the maximum of a large block, but on the distribution of values that exceed a certain high threshold, u. Let X be a random variable with a cumulative distribution function (CDF) F(x). The distribution of the "excesses" over the threshold u is defined by the conditional probability:

![[Pasted image 20250628225631.png]]

The Pickands-Balkema-de Haan theorem states that for a large class of underlying distributions F, as the threshold u is raised, the conditional excess distribution function Fu​(y) converges to a Generalized Pareto Distribution (GPD).7

This result is profoundly useful. It gives us a theoretical justification to select all observations above a high threshold and model them with a specific, well-defined distribution—the GPD. This forms the basis of the POT method, which is generally considered more efficient and practical for financial applications because it utilizes the available data on extreme events more effectively.7

The two theorems, while distinct, are deeply connected and describe the same underlying tail behavior of the parent distribution. The GEV distribution is the limiting law for block maxima, while the GPD is the limiting law for threshold exceedances. As will be shown, their parameters are intrinsically linked, reflecting this shared theoretical foundation.

## 4.3 Modeling Block Maxima with the Generalized Extreme Value (GEV) Distribution

The Block Maxima (BM) method is the classical approach in Extreme Value Theory, directly applying the implications of the Fisher-Tippett-Gnedenko theorem. The procedure involves partitioning a time series into non-overlapping blocks of equal size (e.g., years) and collecting the maximum observation from each block. These block maxima are then assumed to follow a Generalized Extreme Value (GEV) distribution.16

### The GEV Distribution in Detail

The GEV distribution is a continuous probability distribution that unifies the Gumbel, Fréchet, and Weibull families into a single parametric form. Its cumulative distribution function (CDF) is given by 19:

![[Pasted image 20250628225641.png]]

This formula is valid for $1+ξ((x−μ​)/σ)>0$. For the special case where ξ=0, the CDF is:

![[Pasted image 20250628225712.png]]

The GEV distribution is defined by three parameters 19:

- μ (Location Parameter): This parameter shifts the distribution along the x-axis, controlling its central position. It can be any real number.
    
- σ (Scale Parameter): This parameter must be positive (σ>0) and controls the spread or width of the distribution.
    
- ξ (Shape Parameter): This is the most critical parameter in financial applications. It governs the tail behavior of the distribution and determines which of the three extreme value types the distribution belongs to. It is also known as the tail index.
    

### The Three Types of Extreme Value Distributions

The value of the shape parameter, ξ, dictates the nature of the distribution's tail, which has profound implications for risk assessment. The three cases are summarized below.19

**Table 4.1: The Three Families of the GEV Distribution**

|Shape Parameter (ξ)|Distribution Name|Tail Behavior|Relevance to Financial Returns|
|---|---|---|---|
|ξ>0|**Fréchet**|**Heavy-tailed** (power-law decay). The tail is unbounded and decays slowly, allowing for very large extreme values.|**Most relevant.** Financial asset returns are empirically found to be heavy-tailed. A positive shape parameter indicates a high probability of extreme market crashes or spikes.21|
|ξ=0|**Gumbel**|**Light-tailed** (exponential decay). The tail is unbounded but decays much faster than the Fréchet.|Corresponds to parent distributions like the Normal or Lognormal. Often underestimates financial risk because it assigns lower probability to extreme events.21|
|ξ<0|**Weibull (reversed)**|**Bounded tail** (finite upper endpoint). The distribution has a maximum possible value.|Of limited use for modeling financial losses, which are typically considered unbounded. It might be applicable in specific contexts where a variable has a physical upper limit.24|

### Practical Implementation: The Bias-Variance Tradeoff in Block Size Selection

When applying the BM method, the choice of block size is a critical decision that involves a delicate tradeoff between statistical bias and variance.26

- **Small Blocks:** Using a small block size (e.g., monthly maxima) yields more data points. This reduces the variance of the parameter estimates, making them more stable. However, the blocks may not be large enough for the asymptotic theory of the Fisher-Tippett-Gnedenko theorem to hold true. This can lead to bias, as the GEV distribution may be a poor approximation of the true distribution of maxima.
    
- **Large Blocks:** Using a large block size (e.g., annual maxima) ensures the GEV approximation is more likely to be valid, thus reducing bias. However, this results in very few data points, which increases the variance of the parameter estimates, making them highly sensitive to the specific sample of maxima obtained.26
    

In practice, an annual block size is a common choice, especially for data with seasonal patterns (like meteorological data), but for financial data, this often results in too few observations for robust analysis.11

### Python Example: Fitting a GEV to S&P 500 Maximum Annual Losses

Let's apply the Block Maxima method to real financial data. We will analyze the maximum daily losses of the S&P 500 index for each year.

```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme

# 1. Data Acquisition and Preparation
sp500 = yf.download('^GSPC', start='1990-01-01', end='2023-12-31')
sp500 = np.log(sp500['Adj Close'] / sp500['Adj Close'].shift(1))
# Define losses as positive values (negative of returns)
sp500['Loss'] = -sp500
sp500.dropna(inplace=True)

# 2. Extract Block Maxima
# We use an annual block size
annual_max_losses = sp500['Loss'].resample('Y').max()
print("Annual Maximum Losses (First 5 years):")
print(annual_max_losses.head())
print(f"\nNumber of data points (years): {len(annual_max_losses)}")

# 3. Fit GEV Distribution to Block Maxima
# The genextreme.fit function from scipy returns shape (xi), location (mu), and scale (sigma)
xi_gev, mu_gev, sigma_gev = genextreme.fit(annual_max_losses)

print("\nFitted GEV Parameters:")
print(f"Shape (xi): {xi_gev:.4f}")
print(f"Location (mu): {mu_gev:.4f}")
print(f"Scale (sigma): {sigma_gev:.4f}")

# 4. Visualization of the Fit
plt.figure(figsize=(12, 7))
# Plot histogram of the data
plt.hist(annual_max_losses, bins=15, density=True, alpha=0.7, label='Empirical Annual Max Losses')

# Plot the PDF of the fitted GEV distribution
x = np.linspace(annual_max_losses.min(), annual_max_losses.max(), 200)
plt.plot(x, genextreme.pdf(x, xi_gev, mu_gev, sigma_gev), 'r-', lw=2, label='Fitted GEV PDF')

plt.title('GEV Distribution Fitted to S&P 500 Annual Maximum Daily Losses')
plt.xlabel('Maximum Daily Loss')
plt.ylabel('Density')
plt.legend()
plt.show()
```

Interpretation of Results:

The code first downloads historical S&P 500 data, calculates daily losses, and then extracts the maximum loss for each year. This series of annual maxima is our dataset for the GEV model. The genextreme.fit function estimates the three parameters using Maximum Likelihood Estimation (MLE).

The key parameter to inspect is the shape, ξ. For most financial asset returns, we expect to find ξ^​>0, indicating a Fréchet-type distribution with heavy tails.21 A positive shape parameter confirms that the distribution of maximum losses is fat-tailed, implying that extremely large losses, while rare, are more probable than a light-tailed distribution like the Gumbel would suggest. The visualization helps assess the quality of the fit; a good fit will show the GEV probability density function (PDF) closely matching the shape of the empirical histogram of annual maximum losses.

## 4.4 Modeling Threshold Exceedances with the Generalized Pareto Distribution (GPD)

The practical limitations of the Block Maxima method, particularly its inefficient use of data, led to the widespread adoption of the Peaks-Over-Threshold (POT) approach in financial risk management.3 Instead of considering only one maximum value per block, the POT method utilizes all observations that exceed a sufficiently high threshold,

_u_. The theoretical justification, as established by the Pickands-Balkema-de Haan theorem, is that the distribution of these exceedances converges to a Generalized Pareto Distribution (GPD).7

### The GPD in Detail

The GPD is a two-parameter distribution used to model the tail of another distribution. Its cumulative distribution function (CDF) is given by 14:

![[Pasted image 20250628225820.png]]

where x represents the excess over the threshold u. The distribution is defined for x≥0 when ξ≥0, and for 0≤x≤−σ/ξ when ξ<0.

The GPD is characterized by two parameters:

- σ (Scale Parameter): A positive parameter that determines the scale of the distribution of excesses.
    
- ξ (Shape Parameter): The tail index, which is the same shape parameter found in the GEV distribution. This shared parameter forms the crucial link between the BM and POT approaches and governs the tail behavior of the underlying data.21 A positive
    
    ξ indicates a heavy, Pareto-type tail, ξ=0 corresponds to an exponential tail, and ξ<0 implies a short, bounded tail.
    

### The Art and Science of Threshold Selection: A Critical Step

The most crucial, and often most challenging, step in any practical POT analysis is the selection of the threshold, _u_.28 This choice embodies a fundamental

**bias-variance tradeoff**:

- **A low threshold** results in a large number of exceedances, which reduces the variance of the parameter estimates. However, if the threshold is too low, the asymptotic theory may not apply, and the GPD will be a poor model for the excesses, leading to biased estimates.14
    
- **A high threshold** ensures that the GPD is a valid approximation (low bias), but it leaves very few data points for estimation. This small sample size leads to high variance in the parameter estimates, making them unstable and unreliable.7
    

The goal is to find a threshold that is high enough for the GPD approximation to be reasonable, but low enough to retain a sufficient number of observations for robust estimation. This is more of an art guided by science, and several graphical methods are used to inform this decision.

#### Graphical Diagnostic 1: Mean Residual Life (MRL) Plot

The Mean Residual Life (MRL) plot, also known as a mean excess plot, is a primary tool for threshold selection.30 It plots the mean of the excesses over a threshold

_u_ against a range of possible thresholds.

- **Concept:** The MRL is defined as E(X−u∣X>u). A key property of the GPD is that its mean excess function is a linear function of the threshold _u_.
    
- **Interpretation:** Therefore, the MRL plot for data that follows a GPD should be approximately linear for all thresholds above the "true" threshold where the GPD behavior begins.29 The analyst looks for the lowest threshold at which the plot begins to exhibit a stable linear trend.
    
    - An **upward-sloping** linear region suggests a heavy-tailed distribution (ξ>0).
        
    - A **downward-sloping** linear region suggests a thin-tailed distribution (ξ<0).
        
    - A **horizontal** linear region suggests an exponential tail (ξ=0).7
        

#### Graphical Diagnostic 2: Parameter Stability Plot

Another essential diagnostic tool is the parameter stability plot.7

- **Concept:** This method involves fitting the GPD to the data for a range of candidate thresholds and plotting the estimated shape parameter (ξ^​) and a modified scale parameter against the threshold values.
    
- **Interpretation:** According to the theory, for any threshold u0​ above which the GPD model is valid, the shape parameter ξ should be constant. Therefore, the analyst looks for a region in the plot where the parameter estimates appear stable (i.e., the plot becomes flat) and are accompanied by reasonably tight confidence intervals. The optimal threshold is typically chosen as the lowest value at the beginning of this stable region.28
    

### Python Example: Interactive Threshold Selection with `pyextremes`

The `pyextremes` library is specifically designed for EVT and provides excellent tools for these graphical diagnostics.34 Let's use it to analyze the daily losses of a volatile stock, Tesla (TSLA), to select an appropriate threshold.


```python
import yfinance as yf
import pandas as pd
from pyextremes import plot_mean_residual_life, plot_parameter_stability

# 1. Get data for a volatile asset (e.g., Tesla)
tsla = yf.download('TSLA', start='2015-01-01', end='2023-12-31')
tsla = np.log(tsla['Adj Close'] / tsla['Adj Close'].shift(1))
# We focus on losses (positive values)
tsla_losses = -tsla.dropna()

# 2. Generate and Interpret the Mean Residual Life Plot
plot_mean_residual_life(tsla_losses)
plt.suptitle("Mean Residual Life Plot for TSLA Daily Losses")
plt.show()

# 3. Generate and Interpret the Parameter Stability Plot
plot_parameter_stability(tsla_losses)
plt.suptitle("Parameter Stability Plot for TSLA Daily Losses")
plt.show()
```

Interpretation of Plots:

When running the code, the MRL plot for TSLA losses will likely show an upward trend, confirming heavy-tailed behavior. We would look for the point where this trend straightens out. The parameter stability plot will show the estimated GPD shape parameter, ξ^​, across a range of thresholds. We would look for a region where the estimate for ξ^​ stabilizes. For a volatile stock like TSLA, this might occur around the 95th percentile of losses, which could correspond to a daily loss of around 4-5%. This value would be our chosen threshold, u.

### Ensuring Independence: Declustering

A core assumption of the basic POT method is that the exceedances are independent events. However, in financial markets, extreme events often exhibit **volatility clustering**, where one large loss is quickly followed by others. This violates the independence assumption. To address this, a **declustering** process is applied.36

A common method is to group consecutive exceedances into clusters. Any two exceedances separated by less than a specified time window (e.g., 24 or 48 hours) are considered part of the same cluster. Then, only the single maximum value from each cluster is retained for the GPD analysis, ensuring the events are treated as independent.36 The

`pyextremes` library handles this automatically via the `r` parameter, which sets the minimum time distance between clusters.26

### Python Example: Fitting a GPD to Threshold Exceedances

After selecting a threshold and a declustering window, we can fit the GPD model.


```python
from pyextremes import EVA
from scipy.stats import genpareto
import matplotlib.pyplot as plt

# Assume we selected a threshold of 0.04 (4% daily loss) from the plots above
threshold = 0.04

# Use the EVA class from pyextremes to extract declustered extremes
model = EVA(tsla_losses)
model.get_extremes(method="POT", threshold=threshold, r="24H")

# The extracted extremes are stored in model.extremes
exceedances = model.extremes - threshold
print(f"Number of exceedances above {threshold:.2f}: {len(model.extremes)}")
print(f"Number of declustered extreme events: {len(exceedances)}")

# Fit a GPD to the exceedances
# The genpareto.fit function returns shape (xi), location (mu), and scale (sigma)
# For exceedances, the location parameter should be fixed at 0.
xi_gpd, loc_gpd, sigma_gpd = genpareto.fit(exceedances, floc=0)

print("\nFitted GPD Parameters:")
print(f"Shape (xi): {xi_gpd:.4f}")
print(f"Scale (sigma): {sigma_gpd:.4f}")

# Visualize the fit with a Q-Q plot
from scipy import stats
plt.figure(figsize=(8, 6))
stats.probplot(exceedances, dist=genpareto, sparams=(xi_gpd, 0, sigma_gpd), plot=plt)
plt.title(f"Q-Q Plot of TSLA Exceedances vs. Fitted GPD")
plt.show()
```

Interpretation of Results:

The code first uses pyextremes to extract all losses exceeding 4% and declusters them. The resulting series of excesses (value above the threshold) is then fitted with a GPD using scipy.stats.genpareto. The floc=0 argument is crucial, as the theory states the excesses themselves should be modeled starting from zero. The resulting shape parameter, ξ^​, should be positive and its value should be reasonably close to the ξ^​ estimated from the GEV model, as they represent the same underlying tail index.27 The Q-Q plot provides a visual check of the goodness-of-fit. If the GPD is a good model, the points on the plot should lie closely along the reference line.

## 4.5 From Distributions to Decisions: Calculating VaR and Expected Shortfall

The ultimate goal of fitting GEV or GPD models in finance is not just parameter estimation; it is to derive actionable risk measures. The two most prominent tail risk measures are Value at Risk (VaR) and Expected Shortfall (ES).

### Defining the Key Risk Measures

- **Value at Risk (VaR):** For a given confidence level α (e.g., 99%), the VaR is the quantile of the loss distribution. It represents the minimum loss that is expected to be exceeded with a probability of only (1−α).27 For example, a 1-day 99% VaR of $1 million means there is a 1% chance of losing more than $1 million over the next day. While widely used, VaR has been criticized because it provides no information about the
    
    _magnitude_ of the loss if the VaR threshold is breached. Furthermore, VaR is not always a **coherent risk measure** because it can fail the property of subadditivity, meaning the VaR of a portfolio can sometimes be greater than the sum of the VaRs of its components.27
    
- **Expected Shortfall (ES):** Also known as Conditional VaR (CVaR) or Expected Tail Loss (ETL), ES measures the expected loss _given that the loss is greater than or equal to the VaR_. It answers the more pertinent question: "If we have a bad day (a VaR breach), what is our average expected loss?".27 ES is considered a superior and coherent risk measure because it accounts for the severity of losses in the tail and is always sub-additive.38
    

### Calculating Risk Measures from Fitted Models

Once we have the estimated parameters for our GEV or GPD models, we can use analytical formulas to calculate VaR and ES.

#### From the GEV Distribution

For a GEV model fitted to block maxima (e.g., annual maximum losses), the quantile, or **Return Level** (Rm​), which is expected to be exceeded on average once every m blocks, is the VaR for that period. The VaR at confidence level p is given by inverting the GEV CDF 42:

![[Pasted image 20250628225850.png]]

The formula for ES from a GEV distribution is more complex and involves the incomplete gamma function.43 For

ξ<1, it can be expressed as:
![[Pasted image 20250628225900.png]]

where ![[Pasted image 20250628225928.png]] is the lower incomplete gamma function.

#### From the GPD Distribution

The POT approach is more commonly used for calculating daily VaR and ES. Given a threshold u, the number of total observations n, the number of exceedances Nu​, and the fitted GPD parameters ξ^​ and σ^, the VaR for a confidence level p (where p is high, e.g., > 95%) is 15:

$$VaR_p = u + \frac{\hat{\sigma}}{\hat{\xi}} \left[ \left(\frac{n}{N_u}(1-p)\right)^{-\hat{\xi}} - 1 \right] $$The corresponding Expected Shortfall is then calculated as 15:$$ ES_p = \frac{VaR_p}{1-\hat{\xi}} + \frac{\hat{\sigma} - \hat{\xi}u}{1-\hat{\xi}} \quad (\text{for } \xi < 1)$$

This formula elegantly combines the VaR estimate with the GPD parameters to provide the expected loss beyond the VaR threshold.

### Comparative Analysis of VaR Methodologies

EVT is not the only method for calculating VaR. It is instructive to compare it with more traditional approaches.

**Table 4.2: A Comparative Analysis of VaR Methodologies**

|Methodology|Core Assumption|Pros|Cons|Best Use Case|
|---|---|---|---|---|
|**Variance-Covariance**|Returns are normally distributed.|Simple, fast, analytical formula.|Fails for non-normal returns; severely underestimates tail risk, especially during crises.4|Portfolios with assets that are reasonably close to normal, or for low-confidence VaR calculations.|
|**Historical Simulation (HS)**|The recent past is a good predictor of the near future.|Non-parametric, no distributional assumptions, easy to implement and understand.45|Results are entirely dependent on the historical data window; cannot extrapolate beyond the worst historical loss; slow to adapt to new volatility regimes.4|Quick estimation, useful when distributional assumptions are highly uncertain. Performs surprisingly well at moderate confidence levels.47|
|**EVT (POT-GPD)**|The tail of the distribution follows a GPD.|Accurately models fat tails, provides robust estimates for very high confidence levels, can extrapolate beyond observed data.47|More complex, requires subjective threshold selection, sensitive to the i.i.d. assumption.5|**Regulatory capital calculation, stress testing, and any application requiring accurate estimation of extreme, rare losses**.2|

### Python Example: Comparing VaR and ES Estimates

Let's continue with our TSLA example and compare the 1-day 99% and 99.5% VaR and ES estimates from our fitted GPD model with those from Historical Simulation and the Variance-Covariance method.

```python
import pandas as pd
from scipy.stats import norm

# --- Parameters from previous examples ---
# TSLA daily losses data
# tsla_losses = -tsla.dropna()

# POT-GPD fitted parameters
# threshold = 0.04
# xi_gpd, sigma_gpd were fitted
# n = len(tsla_losses)
# Nu = len(exceedances)

# For demonstration, let's assume these values were obtained:
# This is for illustrative purposes; use your actual fitted values.
n = len(tsla_losses)
threshold = 0.04
exceedances_count = model.extremes.shape # Nu
# xi_gpd, _, sigma_gpd = genpareto.fit(model.extremes - threshold, floc=0)

# Confidence levels
alphas = [0.99, 0.995]

# --- 1. EVT (POT-GPD) Calculation ---
def calculate_gpd_var_es(p, u, xi, sigma, n, Nu):
    # VaR formula
    var_p = u + (sigma / xi) * ((((n / Nu) * (1 - p)) ** -xi) - 1)
    
    # ES formula
    es_p = (var_p / (1 - xi)) + ((sigma - xi * u) / (1 - xi))
    
    return var_p, es_p

evt_results = {}
for alpha in alphas:
    var, es = calculate_gpd_var_es(alpha, threshold, xi_gpd, sigma_gpd, n, exceedances_count)
    evt_results = var
    evt_results = es

# --- 2. Historical Simulation (HS) Calculation ---
hs_results = {}
for alpha in alphas:
    hs_results = tsla_losses.quantile(alpha)
    # ES for HS is the mean of losses exceeding the VaR
    var_hs = hs_results
    hs_results = tsla_losses[tsla_losses > var_hs].mean()

# --- 3. Variance-Covariance (Normal) Calculation ---
norm_results = {}
mu_norm = tsla_losses.mean()
std_norm = tsla_losses.std()
for alpha in alphas:
    z = norm.ppf(alpha)
    norm_results = mu_norm + z * std_norm
    # ES for Normal distribution
    es_norm = mu_norm + std_norm * (norm.pdf(z) / (1 - alpha))
    norm_results = es_norm

# --- 4. Compare Results ---
comparison_df = pd.DataFrame([evt_results, hs_results, norm_results], 
                             index=)

print("Comparison of 1-Day Risk Measures for TSLA Losses (%):")
print(comparison_df * 100)
```

Expected Output and Interpretation:

The resulting DataFrame will clearly show that for the 99% and especially the 99.5% confidence levels, the EVT-GPD method produces significantly higher (more conservative and likely more accurate) VaR and ES estimates than the other two methods. The Variance-Covariance method, assuming normality, will produce the lowest and most dangerously optimistic figures. Historical Simulation will be limited by the worst day in its dataset. EVT, by modeling the tail explicitly, provides a more realistic picture of the potential for extreme losses in a volatile asset like TSLA, demonstrating its superiority for high-quantile risk estimation.

## 4.6 Real-World Challenges and Advanced Frontiers

While EVT provides a powerful framework for modeling tail risk, its classical formulation rests on a critical assumption: that the extreme events are independent and identically distributed (i.i.d.).5 In the context of financial markets, this assumption is frequently violated.

### The IID Assumption vs. Volatility Clustering

Financial asset returns exhibit well-documented **volatility clustering**: periods of high volatility tend to be followed by more high volatility, and calm periods are followed by calm periods. This means that large losses (extreme events) often arrive in clusters, violating the assumption of independence.6 Applying classical EVT directly to raw financial returns can lead to an underestimation of risk because it fails to account for the time-varying nature of volatility. If today is a high-volatility day, the probability of an extreme loss tomorrow is higher than it would be on an average day.

This limitation is not a reason to discard EVT but rather serves as a motivation to integrate it with models that can capture this time-dependence.

### Advanced Topic 1: Conditional EVT (GARCH-EVT)

The most popular solution to the problem of volatility clustering is the **conditional EVT** or **GARCH-EVT** approach, pioneered by McNeil and Frey (2000).38 This method combines the strengths of GARCH models for volatility forecasting with the power of EVT for tail modeling in a two-step process 49:

1. **Fit a Volatility Model:** First, an econometric model capable of capturing time-varying volatility, such as a GARCH(1,1) model, is fitted to the asset return series.
    
2. **Apply EVT to Residuals:** The standardized residuals from the GARCH model, zt​=σt​rt​−μt​​, are extracted. These residuals are, by construction, stripped of their time-varying volatility and are much closer to being i.i.d. than the raw returns. EVT (typically the POT method) is then applied to this series of standardized residuals to model their tail distribution.
    
3. **Forecast Risk:** To calculate the 1-day ahead VaR, one first forecasts the next day's volatility, σ^t+1​, using the GARCH model. Then, the VaR of the standardized residuals, VaRp​(Z), is calculated from the fitted GPD. The final conditional VaR is the product of these two components: VaRp​(Rt+1​)=μ^​t+1​+σ^t+1​×VaRp​(Z).
    

This approach yields dynamic risk estimates that adapt to current market volatility, providing a much more realistic and responsive risk management tool.38

### Advanced Topic 2: Multivariate EVT

Another significant challenge is extending EVT from a single asset to a portfolio of multiple assets. This is the domain of **Multivariate Extreme Value Theory (MEVT)**.2 The primary difficulties are:

- **Defining a Multivariate Extreme:** In one dimension, the extreme is simply the maximum or minimum. In multiple dimensions, it is not obvious how to define the "most extreme" event. Is it when one asset crashes, or when all assets crash together?.25
    
- **Modeling Tail Dependence:** The key to portfolio risk is the correlation between assets. In the context of extremes, this is called **tail dependence**. It measures the likelihood that one asset will experience an extreme loss given that another has. Standard correlation breaks down during market crises. MEVT seeks to model this tail dependence structure, often using mathematical tools called **copulas** to separate the modeling of the marginal distributions of each asset from the dependence structure that links them together.17
    

These advanced topics are beyond the scope of this introductory chapter but represent the active frontiers of research and practice in quantitative risk management.

## 4.7 Capstone Project: Extreme Risk Analysis of the CBOE Volatility Index (VIX)

### Project Brief

The CBOE Volatility Index (VIX) is a real-time market index that represents the market's expectation of 30-day forward-looking volatility. Derived from S&P 500 index option prices, it is widely known as the "investor fear gauge".51 The VIX typically trades in a low range but is characterized by sudden, extreme upward spikes during periods of market stress and panic. Its distribution is highly non-normal, positively skewed, and possesses extremely fat tails, making it an ideal and challenging candidate for analysis using Extreme Value Theory.

In this capstone project, you will perform a comprehensive risk analysis of the VIX using the Peaks-Over-Threshold (POT) methodology. You will acquire the data, select an appropriate threshold for defining an extreme event, fit a Generalized Pareto Distribution (GPD), and calculate high-quantile Value at Risk (VaR) and Expected Shortfall (ES). Finally, you will compare your results to a more naive method to understand the value added by EVT.

### Part 1: Data Acquisition and Exploration

#### **Question 1:**

Acquire the daily historical data for the VIX index (`^VIX`) from 2004 to the present day. Calculate the daily percentage changes. Plot the time series of these percentage changes and their corresponding histogram. What do you observe about the distribution's characteristics, and how do they compare to the assumptions of a normal distribution?

#### **Response 1:**

First, we will use the `yfinance` library to download the required data. We then calculate the percentage change in the daily closing price to analyze its volatility.

```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

# --- Data Acquisition ---
vix_data = yf.download('^VIX', start='2004-01-01', end=pd.to_datetime('today'))

# --- Calculate Percentage Changes ---
vix_data['Pct_Change'] = vix_data['Close'].pct_change()
vix_data.dropna(inplace=True)

# --- Visualization ---
# Plot 1: Time Series of VIX Percentage Changes
plt.figure(figsize=(15, 6))
plt.plot(vix_data.index, vix_data['Pct_Change'], label='VIX Daily % Change', color='darkblue', alpha=0.8)
plt.title('Daily Percentage Change of the VIX Index (2004-Present)')
plt.ylabel('Percentage Change')
plt.xlabel('Date')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plot 2: Histogram of VIX Percentage Changes
plt.figure(figsize=(12, 7))
plt.hist(vix_data['Pct_Change'], bins=150, density=True, alpha=0.7, label='VIX % Change Distribution')

# Overlay a fitted normal distribution for comparison
mu_vix, std_vix = norm.fit(vix_data['Pct_Change'])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)
p = norm.pdf(x, mu_vix, std_vix)
plt.plot(x, p, 'r', linewidth=2, label='Fitted Normal Distribution')

plt.title('Histogram of VIX Daily Percentage Changes')
plt.xlabel('Percentage Change')
plt.ylabel('Density')
plt.legend()
plt.xlim(-0.4, 0.4) # Zoom in to see the distribution body
plt.show()

# --- Statistical Analysis ---
skewness = skew(vix_data['Pct_Change'])
kurt = kurtosis(vix_data['Pct_Change']) # Fisher's kurtosis (normal=0)

print("--- Statistical Properties of VIX Daily % Changes ---")
print(f"Mean: {mu_vix:.4f}")
print(f"Standard Deviation: {std_vix:.4f}")
print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurt:.4f}")
```

Observations and Interpretation:

The time series plot clearly shows periods of relative calm punctuated by massive spikes in the VIX's daily percentage change, a classic representation of volatility clustering. These spikes correspond to major market crises, such as the 2008 GFC and the COVID-19 pandemic in 2020.

The histogram and statistical properties reveal a distribution that is starkly non-normal:

1. **High Kurtosis:** The kurtosis value is extremely high, indicating exceptionally fat tails. This means that large daily changes (both positive and negative) occur far more frequently than a normal distribution would ever predict.
    
2. **Positive Skewness:** The distribution is positively skewed. This is a key feature of the VIX; while it can have large negative daily changes, the positive spikes (representing sudden increases in fear) are far more extreme in magnitude.
    
3. **Poor Normal Fit:** The red line representing the fitted normal distribution is a very poor match for the empirical data. It dramatically underestimates the peak of the distribution and, most importantly, fails to capture the significant mass in the tails.
    

These characteristics make the VIX percentage changes a textbook case for applying Extreme Value Theory, as any risk model based on the assumption of normality would be dangerously misleading.

### Part 2: Peaks-Over-Threshold (POT) Analysis

#### **Question 2:**

We are interested in the risk of extreme upward spikes in the VIX. Using the daily percentage changes, perform a POT analysis to determine an appropriate threshold for defining such an event. Generate and interpret both the Mean Residual Life (MRL) plot and the parameter stability plot for the GPD. Based on these diagnostics, select and justify a threshold value.

#### **Response 2:**

We will use the `pyextremes` library to generate the standard diagnostic plots for threshold selection. Our focus is on the positive tail of the distribution (large positive percentage changes).

```python
from pyextremes import plot_mean_residual_life, plot_parameter_stability

# We use the positive VIX percentage changes as our series of interest
vix_spikes = vix_data['Pct_Change'][vix_data['Pct_Change'] > 0]

# --- Mean Residual Life Plot ---
plot_mean_residual_life(vix_spikes)
plt.suptitle("Mean Residual Life Plot for VIX Daily Spikes")
plt.show()

# --- Parameter Stability Plot ---
plot_parameter_stability(vix_spikes)
plt.suptitle("GPD Parameter Stability Plot for VIX Daily Spikes")
plt.show()
```

**Interpretation and Threshold Selection:**

1. **Mean Residual Life Plot:** The MRL plot will show a clear, strong, and persistent upward trend. This is indicative of a very heavy-tailed distribution (ξ>0). We are looking for the point at which the plot begins to straighten into a stable linear trend. Initially, the plot may be noisy and curved. We should observe it starting to become more linear around a threshold of approximately 0.10 (a 10% daily increase). The upward slope confirms that the average spike _above_ a given high threshold tends to be even larger, a classic sign of extreme-risk behavior.
    
2. **Parameter Stability Plot:** This plot shows the estimated GPD shape parameter (ξ) and modified scale parameter for a range of thresholds. We are looking for the lowest threshold value beyond which the parameter estimates become reasonably stable. The plot for ξ will likely be volatile for low thresholds but should start to stabilize in a positive range (e.g., between 0.2 and 0.4) for thresholds above a certain point. The scale parameter should also show a similar stabilization.
    

Justification of Threshold Choice:

Based on the visual evidence from both plots, a threshold of 0.12 (a 12% daily increase) appears to be a reasonable choice. Below this level, the MRL plot is more curved, and the parameter stability plot shows significant volatility. Above 0.12, the MRL plot exhibits a more stable linear trend, and the GPD parameter estimates begin to settle into a stable region without sacrificing too many data points for the estimation. This choice represents a sound compromise in the bias-variance tradeoff.

### Part 3: Quantifying Extreme Risk

#### **Question 3:**

Fit a Generalized Pareto Distribution (GPD) to the VIX percentage changes that exceed your chosen threshold of 12%. Use a declustering window of 24 hours to ensure event independence. Using the fitted parameters, calculate the 1-day VaR and ES at 99.0%, 99.5%, and 99.9% confidence levels. Interpret the 99.5% VaR and ES figures in the context of VIX risk.

#### **Response 3:**

Now we will apply our chosen threshold, perform declustering, fit the GPD, and calculate the risk measures using the analytical formulas.

```python
from pyextremes import EVA
from scipy.stats import genpareto

# --- Model Fitting ---
threshold = 0.12
# Use EVA to extract declustered extremes
vix_model = EVA(vix_data['Pct_Change'])
vix_model.get_extremes(method="POT", threshold=threshold, r="24H", extremes_type="high")

# Get the exceedances (excess over threshold)
exceedances = vix_model.extremes - threshold

# Fit GPD to the exceedances, fixing location to 0
xi_gpd_vix, loc_gpd_vix, sigma_gpd_vix = genpareto.fit(exceedances, floc=0)

print("--- Fitted GPD Parameters for VIX Spikes ---")
print(f"Chosen Threshold (u): {threshold:.4f}")
print(f"Number of Exceedances (Nu): {len(vix_model.extremes)}")
print(f"Shape (xi): {xi_gpd_vix:.4f}")
print(f"Scale (sigma): {sigma_gpd_vix:.4f}")

# --- VaR and ES Calculation ---
n = len(vix_data)
Nu = len(vix_model.extremes)
conf_levels = [0.99, 0.995, 0.999]
risk_measures =

for p in conf_levels:
    # Calculate VaR
    var_p = threshold + (sigma_gpd_vix / xi_gpd_vix) * ((((n / Nu) * (1 - p)) ** -xi_gpd_vix) - 1)
    
    # Calculate ES
    es_p = (var_p / (1 - xi_gpd_vix)) + ((sigma_gpd_vix - xi_gpd_vix * threshold) / (1 - xi_gpd_vix))
    
    risk_measures.append({'Confidence Level': f"{p*100}%", 'VaR': var_p, 'ES': es_p})

risk_df = pd.DataFrame(risk_measures)
print("\n--- EVT-Based Risk Measures for VIX Daily % Change ---")
print(risk_df)

# Interpretation for 99.5%
var_995 = risk_df[risk_df['Confidence Level'] == '99.5%'].iloc
es_995 = risk_df[risk_df['Confidence Level'] == '99.5%'].iloc
```

Interpretation of 99.5% Risk Measures:

The output will provide the VaR and ES values. For example, if the calculated 99.5% VaR is 0.35 (35%) and the ES is 0.50 (50%), the interpretation would be:

- **99.5% VaR:** "Based on our EVT model, there is a 0.5% probability (or a 1-in-200 chance on any given trading day) that the VIX index will spike by more than **35%** in a single day. This figure represents the plausible worst-case loss for a short volatility position under extreme market stress."
    
- **99.5% ES:** "In the unlikely event that a 99.5% VaR breach occurs (i.e., the VIX spikes by more than 35%), our model predicts that the average increase in the VIX on those days would be **50%**. This ES figure gives a more complete picture of the catastrophic risk involved, quantifying the expected magnitude of the tail event itself."
    

### Part 4: Comparative Analysis

#### **Question 4:**

Calculate the 1-day VaR at 99.0%, 99.5%, and 99.9% confidence levels using the simple Historical Simulation (HS) method. Compare these values to your EVT-based VaR estimates. Why is the EVT approach fundamentally more reliable for an asset like the VIX, especially for applications like regulatory capital calculation or stress testing?

#### **Response 4:**

The Historical Simulation method is non-parametric and simply involves taking the empirical quantiles from the historical data.


```python
# --- Historical Simulation Calculation ---
hs_var =
for p in conf_levels:
    var_hs = vix_data['Pct_Change'].quantile(p)
    hs_var.append({'Confidence Level': f"{p*100}%", 'VaR_HS': var_hs})

hs_df = pd.DataFrame(hs_var)

# --- Comparison ---
comparison_final = pd.merge(risk_df], hs_df, on='Confidence Level')
comparison_final.rename(columns={'VaR': 'VaR_EVT'}, inplace=True)

print("\n--- Comparison of VaR Estimates (EVT vs. Historical Simulation) ---")
print(comparison_final)
```

Comparison and Reliability Analysis:

The comparison table will likely show that for the 99% level, the EVT and HS VaR estimates might be somewhat close. However, as we move to higher confidence levels like 99.5% and especially 99.9%, the EVT-based VaR will be significantly higher than the HS VaR.

The EVT approach is fundamentally more reliable for several key reasons:

1. **Extrapolation Beyond Historical Data:** Historical Simulation is inherently constrained by the data it has seen. Its 99.9% VaR can be no greater than the second or third largest spike in the entire multi-decade dataset.4 EVT, by fitting a parametric model (the GPD) to the tail, can mathematically extrapolate to estimate the probability of events even more extreme than any that have been observed historically. For an asset like the VIX, where the "next crisis" could produce an unprecedented spike, this is a critical advantage.
    
2. **More Efficient Use of Tail Data:** HS uses only a few data points at the extreme end of the distribution to determine its quantiles. The POT method, in contrast, uses all data points above the threshold to estimate the parameters of the GPD. This larger effective sample size for the tail leads to more stable and statistically robust estimates of the tail's shape.14
    
3. **Provides a Smoother Tail Distribution:** HS results in a clunky, step-wise tail distribution. EVT provides a continuous, smooth parametric function (the GPD) to describe the tail. This allows for the calculation of risk measures at any confidence level, not just those directly observable in the data, and provides a more coherent theoretical framework for risk.
    

For regulatory capital and stress testing, the objective is to ensure solvency during events of extreme, unprecedented stress. A model that is blind to anything worse than what has happened in the past (Historical Simulation) is inadequate for this task. EVT, by providing a theoretically-grounded method for modeling and extrapolating extreme risks, offers a far more prudent and reliable foundation for making such critical financial decisions.38

## References
**

1. Extreme Value Theory in Financial Risk Management: The Random Walk Approach, acessado em junho 28, 2025, [http://article.sapub.org/10.5923.j.ijps.20150401.03.html](http://article.sapub.org/10.5923.j.ijps.20150401.03.html)
    
2. Mastering Extreme Value Theory - Number Analytics, acessado em junho 28, 2025, [https://www.numberanalytics.com/blog/extreme-value-theory-computational-finance](https://www.numberanalytics.com/blog/extreme-value-theory-computational-finance)
    
3. Extreme Value Theory: the Block-Maxima approach and the Peak-Over-Threshold approach, acessado em junho 28, 2025, [https://www.simtrade.fr/blog_simtrade/extreme-value-theory-block-maxima-peak-over-threshold/](https://www.simtrade.fr/blog_simtrade/extreme-value-theory-block-maxima-peak-over-threshold/)
    
4. Using Extreme Value Theory to Estimate Value-at-Risk, acessado em junho 28, 2025, [https://www.agrar.hu-berlin.de/de/institut/departments/daoe/lbl/dokumente/literatur/VAREVT/@@download/file/VaRUsingEVT.PDF](https://www.agrar.hu-berlin.de/de/institut/departments/daoe/lbl/dokumente/literatur/VAREVT/@@download/file/VaRUsingEVT.PDF)
    
5. What is Extreme Value Theory? | CQF, acessado em junho 28, 2025, [https://www.cqf.com/blog/quant-finance-101/what-is-extreme-value-theory](https://www.cqf.com/blog/quant-finance-101/what-is-extreme-value-theory)
    
6. Pitfalls and Opportunities in the Use of Extreme Value Theory in Risk ..., acessado em junho 28, 2025, [http://www.ssc.upenn.edu/~fdiebold/papers/paper21/dss-f.pdf](http://www.ssc.upenn.edu/~fdiebold/papers/paper21/dss-f.pdf)
    
7. Extreme Value Theory For a 1-in-200 event - NoCA, acessado em junho 28, 2025, [https://www.noca.uk/wp-content/uploads/2020/11/EVT_NOCA.pdf](https://www.noca.uk/wp-content/uploads/2020/11/EVT_NOCA.pdf)
    
8. Fisher–Tippett–Gnedenko theorem - Wikipedia, acessado em junho 28, 2025, [https://en.wikipedia.org/wiki/Fisher%E2%80%93Tippett%E2%80%93Gnedenko_theorem](https://en.wikipedia.org/wiki/Fisher%E2%80%93Tippett%E2%80%93Gnedenko_theorem)
    
9. Fisher-Tippett-Gnedenko Theorem: Generalizing Three Types of Extreme Value Distributions | Wolfram Demonstrations Project, acessado em junho 28, 2025, [https://demonstrations.wolfram.com/FisherTippettGnedenkoTheoremGeneralizingThreeTypesOfExtremeV](https://demonstrations.wolfram.com/FisherTippettGnedenkoTheoremGeneralizingThreeTypesOfExtremeV)
    
10. Fisher-Tippett theorem with an historical perspective | Freakonometrics - Hypotheses.org, acessado em junho 28, 2025, [https://freakonometrics.hypotheses.org/2321](https://freakonometrics.hypotheses.org/2321)
    
11. Extreme Value Theory 3. Main Block maxima results and the Fisher-Tippett, Gnedenko theorem - Nematrian, acessado em junho 28, 2025, [http://www.nematrian.com/ExtremeValueTheory3](http://www.nematrian.com/ExtremeValueTheory3)
    
12. On the block maxima method in extreme value theory, acessado em junho 28, 2025, [https://personal.eur.nl/ldehaan/pwm_blocks.pdf](https://personal.eur.nl/ldehaan/pwm_blocks.pdf)
    
13. Extreme Value Theory in Finance - mediaTUM, acessado em junho 28, 2025, [https://mediatum.ub.tum.de/doc/1072087/190304.pdf](https://mediatum.ub.tum.de/doc/1072087/190304.pdf)
    
14. Mastering Peaks-Over-Threshold in Actuarial Science, acessado em junho 28, 2025, [https://www.numberanalytics.com/blog/peaks-over-threshold-method-acts-4302](https://www.numberanalytics.com/blog/peaks-over-threshold-method-acts-4302)
    
15. Value at Risk Estimation using Extreme Value Theory - MSSANZ, acessado em junho 28, 2025, [https://mssanz.org.au/modsim2011/D6/singh.pdf](https://mssanz.org.au/modsim2011/D6/singh.pdf)
    
16. An Application of Extreme Value Theory for Measuring Financial Risk 1 - ResearchGate, acessado em junho 28, 2025, [https://www.researchgate.net/profile/Manfred-Gilli/publication/5144622_An_Application_of_Extreme_Value_Theory_for_Measuring_Financial_Risk/links/09e41506c756fc2c0f000000/An-Application-of-Extreme-Value-Theory-for-Measuring-Financial-Risk.pdf](https://www.researchgate.net/profile/Manfred-Gilli/publication/5144622_An_Application_of_Extreme_Value_Theory_for_Measuring_Financial_Risk/links/09e41506c756fc2c0f000000/An-Application-of-Extreme-Value-Theory-for-Measuring-Financial-Risk.pdf)
    
17. Block Maxima Method: Theory and Practice - Number Analytics, acessado em junho 28, 2025, [https://www.numberanalytics.com/blog/block-maxima-method-theory-practice-acts-4302](https://www.numberanalytics.com/blog/block-maxima-method-theory-practice-acts-4302)
    
18. Mastering Block Maxima Method - Number Analytics, acessado em junho 28, 2025, [https://www.numberanalytics.com/blog/block-maxima-method-acts-4302-guide](https://www.numberanalytics.com/blog/block-maxima-method-acts-4302-guide)
    
19. Generalized Extreme Value distribution and calculation of Return value - NASA GMAO, acessado em junho 28, 2025, [https://gmao.gsfc.nasa.gov/research/subseasonal/atlas/GEV-RV-html/GEV-RV-description.html](https://gmao.gsfc.nasa.gov/research/subseasonal/atlas/GEV-RV-html/GEV-RV-description.html)
    
20. Generalized extreme value distribution - Wikipedia, acessado em junho 28, 2025, [https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution)
    
21. Parametric Approaches (II): Extreme Value - AnalystPrep, acessado em junho 28, 2025, [https://analystprep.com/study-notes/frm/part-2/operational-and-integrated-risk-management/parametric-approaches-ii-extreme-value/](https://analystprep.com/study-notes/frm/part-2/operational-and-integrated-risk-management/parametric-approaches-ii-extreme-value/)
    
22. Modeling Data with the Generalized Extreme Value Distribution - MATLAB &, acessado em junho 28, 2025, [https://www.mathworks.com/help/stats/modelling-data-with-the-generalized-extreme-value-distribution.html](https://www.mathworks.com/help/stats/modelling-data-with-the-generalized-extreme-value-distribution.html)
    
23. Generalized Extreme Value Distribution Explained - Trajectory Hub, acessado em junho 28, 2025, [https://trajdash.usc.edu/generalized-extreme-value-distribution](https://trajdash.usc.edu/generalized-extreme-value-distribution)
    
24. Assessing the importance of the choice threshold in quantifying market risk under the POT approach (EVT) - PubMed Central, acessado em junho 28, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9818059/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9818059/)
    
25. Extreme value theory - Wikipedia, acessado em junho 28, 2025, [https://en.wikipedia.org/wiki/Extreme_value_theory](https://en.wikipedia.org/wiki/Extreme_value_theory)
    
26. Block Maxima - pyextremes, acessado em junho 28, 2025, [https://georgebv.github.io/pyextremes/user-guide/3-block-maxima/](https://georgebv.github.io/pyextremes/user-guide/3-block-maxima/)
    
27. A comparative study of VaR and ES using extreme value theory Klara Andersson - Lund University Publications, acessado em junho 28, 2025, [https://lup.lub.lu.se/student-papers/record/9013722/file/9013723.pdf](https://lup.lub.lu.se/student-papers/record/9013722/file/9013723.pdf)
    
28. Optimal threshold selection for the peak-over-threshold approach of ..., acessado em junho 28, 2025, [https://www.simtrade.fr/blog_simtrade/optimal-threshold-selection-peak-over-threshold-approach-extreme-value-theory/](https://www.simtrade.fr/blog_simtrade/optimal-threshold-selection-peak-over-threshold-approach-extreme-value-theory/)
    
29. Threshold Selection - pyextremes, acessado em junho 28, 2025, [https://georgebv.github.io/pyextremes/user-guide/5-threshold-selection/](https://georgebv.github.io/pyextremes/user-guide/5-threshold-selection/)
    
30. Excess over Threshold: A STAT 476 Deep Dive - Number Analytics, acessado em junho 28, 2025, [https://www.numberanalytics.com/blog/excess-over-threshold-stat-476-deep-dive](https://www.numberanalytics.com/blog/excess-over-threshold-stat-476-deep-dive)
    
31. Threshold Selection in Extreme Value Analysis - CEAUL, acessado em junho 28, 2025, [http://ceaul.org/wp-content/uploads/2018/10/NotaCom07.pdf](http://ceaul.org/wp-content/uploads/2018/10/NotaCom07.pdf)
    
32. georgebv.github.io, acessado em junho 28, 2025, [https://georgebv.github.io/pyextremes/user-guide/5-threshold-selection/#:~:text=Mean%20residual%20life%20plot%20plots,Pareto%20Distribution%20model%20is%20valid.](https://georgebv.github.io/pyextremes/user-guide/5-threshold-selection/#:~:text=Mean%20residual%20life%20plot%20plots,Pareto%20Distribution%20model%20is%20valid.)
    
33. Threshold selection • mev, acessado em junho 28, 2025, [https://lbelzile.github.io/mev/articles/02-threshold.html](https://lbelzile.github.io/mev/articles/02-threshold.html)
    
34. pyextremes, acessado em junho 28, 2025, [https://georgebv.github.io/pyextremes/](https://georgebv.github.io/pyextremes/)
    
35. georgebv/pyextremes: Extreme Value Analysis (EVA) in Python - GitHub, acessado em junho 28, 2025, [https://github.com/georgebv/pyextremes](https://github.com/georgebv/pyextremes)
    
36. Peaks Over Threshold - pyextremes, acessado em junho 28, 2025, [https://georgebv.github.io/pyextremes/user-guide/4-peaks-over-threshold/](https://georgebv.github.io/pyextremes/user-guide/4-peaks-over-threshold/)
    
37. VIVIANA FERNANDEZ* EXTREME VALUE THEORY AND VALUE AT RISK, acessado em junho 28, 2025, [https://www.rae-ear.org/index.php/rae/article/download/24/47/](https://www.rae-ear.org/index.php/rae/article/download/24/47/)
    
38. Ranking of VaR and ES Models: Performance in developed and emerging markets - EconStor, acessado em junho 28, 2025, [https://www.econstor.eu/bitstream/10419/66873/1/730411532.pdf](https://www.econstor.eu/bitstream/10419/66873/1/730411532.pdf)
    
39. Expected shortfall - Wikipedia, acessado em junho 28, 2025, [https://en.wikipedia.org/wiki/Expected_shortfall](https://en.wikipedia.org/wiki/Expected_shortfall)
    
40. Defining Expected Shortfall And Its Calculation - FasterCapital, acessado em junho 28, 2025, [https://fastercapital.com/topics/defining-expected-shortfall-and-its-calculation.html/1](https://fastercapital.com/topics/defining-expected-shortfall-and-its-calculation.html/1)
    
41. Expected Shortfall, acessado em junho 28, 2025, [https://personal.ntu.edu.sg/nprivault/MH8331/expected_shortfall.pdf](https://personal.ntu.edu.sg/nprivault/MH8331/expected_shortfall.pdf)
    
42. (PDF) Generalized Extreme Value Distribution and Extreme ..., acessado em junho 28, 2025, [https://www.researchgate.net/publication/226073080_Generalized_Extreme_Value_Distribution_and_Extreme_Economic_Value_at_Risk_EE-VaR](https://www.researchgate.net/publication/226073080_Generalized_Extreme_Value_Distribution_and_Extreme_Economic_Value_at_Risk_EE-VaR)
    
43. gev: Generalized extreme value distribution in VaRES: Computes ..., acessado em junho 28, 2025, [https://rdrr.io/cran/VaRES/man/gev.html](https://rdrr.io/cran/VaRES/man/gev.html)
    
44. Backtesting Value-at-Risk (VaR): The Basics - Investopedia, acessado em junho 28, 2025, [https://www.investopedia.com/articles/professionals/081215/backtesting-valueatrisk-var-basics.asp](https://www.investopedia.com/articles/professionals/081215/backtesting-valueatrisk-var-basics.asp)
    
45. A comprehensive review of Value at Risk methodologies | The Spanish Review of Financial Economics - Elsevier, acessado em junho 28, 2025, [https://www.elsevier.es/en-revista-the-spanish-review-financial-economics-332-articulo-a-comprehensive-review-value-at-S217312681300017X?redirectNew=true](https://www.elsevier.es/en-revista-the-spanish-review-financial-economics-332-articulo-a-comprehensive-review-value-at-S217312681300017X?redirectNew=true)
    
46. The Empirical Comparison of Risk Models in Estimating Value at Risk and Expected Shortfall - Atlantis Press, acessado em junho 28, 2025, [https://www.atlantis-press.com/article/125947482.pdf](https://www.atlantis-press.com/article/125947482.pdf)
    
47. COMPARING THE PRECISION OF DIFFERENT METHODS OF ..., acessado em junho 28, 2025, [http://www.acrn-journals.eu/resources/jofrp0501f.pdf](http://www.acrn-journals.eu/resources/jofrp0501f.pdf)
    
48. Hybrid Historical Simulation VaR and ES: Performance in Developed and Emerging Markets, acessado em junho 28, 2025, [https://www.researchgate.net/publication/45137968_Hybrid_Historical_Simulation_VaR_and_ES_Performance_in_Developed_and_Emerging_Markets](https://www.researchgate.net/publication/45137968_Hybrid_Historical_Simulation_VaR_and_ES_Performance_in_Developed_and_Emerging_Markets)
    
49. Expected Shortfall Estimation Using Extreme Theory - Research India Publications, acessado em junho 28, 2025, [https://www.ripublication.com/gjfm16/gjfmv8n1_07.pdf](https://www.ripublication.com/gjfm16/gjfmv8n1_07.pdf)
    
50. (PDF) Expected Shortfall Estimation Using Extreme Theory - ResearchGate, acessado em junho 28, 2025, [https://www.researchgate.net/publication/313314928_Expected_Shortfall_Estimation_Using_Extreme_Theory](https://www.researchgate.net/publication/313314928_Expected_Shortfall_Estimation_Using_Extreme_Theory)
    
51. VIX - CBOE Volatility Index - Kaggle, acessado em junho 28, 2025, [https://www.kaggle.com/datasets/joebeachcapital/vix-cboe-volatility-index](https://www.kaggle.com/datasets/joebeachcapital/vix-cboe-volatility-index)
    

**