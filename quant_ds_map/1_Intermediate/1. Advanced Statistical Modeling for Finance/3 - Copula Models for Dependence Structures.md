## 1.1 Introduction: The Limits of Correlation in Finance

In the landscape of quantitative finance, the concept of dependence between assets is a cornerstone of portfolio theory, risk management, and derivative pricing. For decades, the primary tool for measuring and modeling this dependence has been the linear, or Pearson, correlation coefficient. Foundational models such as the Capital Asset Pricing Model (CAPM) and Arbitrage Pricing Theory (APT) rely on correlation to describe the relationships between financial instruments. Similarly, the pricing of multi-asset derivatives, like basket options, hinges on understanding the joint movements of the underlying assets, a task historically delegated to correlation matrices.

However, the reliance on linear correlation, while convenient, is fraught with peril. It provides an incomplete and often dangerously misleading picture of risk, particularly during periods of market stress.2 The limitations of correlation are not merely academic; they represent a significant source of model risk that has contributed to major financial crises. The core issue is that Pearson correlation measures only the strength of a

_linear_ relationship between two variables. Financial markets, however, are rife with non-linearities. Two assets might exhibit little to no correlation during normal market conditions, only to move in lockstep during a crash—a phenomenon that linear correlation is structurally incapable of capturing.4 This is especially problematic as financial returns data often exhibit "fat tails," meaning extreme events occur far more frequently than predicted by the normal distribution, and it is precisely the dependence during these extreme events that risk managers are most concerned about.6

The most prominent illustration of this failure is the role of the Gaussian copula model in the 2008 global financial crisis.7 Collateralized Debt Obligations (CDOs) are complex financial products whose value depends on the joint default behavior of a pool of debt instruments. To price these products, financial engineers widely adopted the Gaussian copula model. This model extends the logic of linear correlation, allowing for non-normal marginal default probabilities for individual assets but imposing a dependence structure derived from the multivariate normal distribution.8 A key property of this Gaussian dependence is its lack of

_tail dependence_. It assumes that the probability of many assets defaulting simultaneously is extremely low, akin to multiple independent coin flips all landing on heads.5

This assumption was fundamentally at odds with reality. Empirical evidence demonstrates that financial assets exhibit significant tail dependence; they are far more likely to crash together than they are to boom together.4 The Gaussian copula, by its very construction, ignored this possibility of systemic contagion. As a result, models used to price CDOs systematically underestimated the risk of widespread, simultaneous defaults, leading to an illusion of safety and a severe underpricing of risk.7 When the U.S. housing market turned, defaults cascaded through the system in a highly correlated manner that the models had deemed virtually impossible, with catastrophic consequences.

This historical failure serves as a powerful motivation for moving beyond simple correlation. It highlights the critical need for more sophisticated statistical tools that can flexibly model the true, often non-linear and asymmetric, dependence structures observed in financial markets. Copula functions provide such a framework. They allow for the separation of the individual behavior of assets from their joint dependence, enabling the creation of far more realistic and robust models of financial risk.2 This chapter provides a comprehensive introduction to the theory and practical application of copula models, equipping the modern quantitative analyst with the tools necessary to avoid the pitfalls of the past and build more resilient financial models.

## 1.2 The Theoretical Cornerstone: Sklar's Theorem

The mathematical foundation that enables the entire field of copula modeling is a powerful result known as Sklar's Theorem.11 To fully appreciate the theorem, one must first understand a fundamental statistical concept: the probability integral transform (PIT).

### The Probability Integral Transform

The probability integral transform states that if X is a continuous random variable with a cumulative distribution function (CDF) $FX​(x)=P(X≤x)$, then the new random variable $U=FX​(X)$ is uniformly distributed on the interval .12 In essence, by applying a variable's own CDF to it, we can map it to a standardized, uniform scale. This transformation is crucial because it allows us to convert any set of random variables, regardless of their original distributions (Normal, Student's t, Gamma, etc.), onto a common, distribution-free domain—the unit hypercube

d. This process is the gateway to modeling their dependence structure independently of their individual distributional properties.

### Sklar's Theorem: Decomposing the Joint Distribution

In 1959, Abe Sklar proved a theorem that revolutionized multivariate statistics by formalizing the link between a multivariate distribution, its univariate marginals, and a function that "couples" them together—the copula.14

Sklar's Theorem: Let F be a d-dimensional joint cumulative distribution function with univariate marginal distribution functions $F1​,F2​,…,Fd$​. Then there exists a d-dimensional copula C such that for all $x1​,x2​,…,xd$​ in $R$:

![[Pasted image 20250628233002.png]]

If the marginals F1​,…,Fd​ are all continuous, then the copula C is unique. Conversely, if C is a copula and F1​,…,Fd​ are univariate CDFs, then the function F defined above is a joint CDF with marginals F1​,…,Fd​.11

A **copula** C is itself a multivariate CDF defined on the unit hypercube d with uniform marginals.14 By letting

$ui​=Fi​(xi​)$, we can see the theorem in its more common form: $F(x1​,…,xd​)=C(u1​,…,ud​)$. This equation reveals that the copula contains all the information about the dependence structure between the variables, separate from their marginal properties.15

For continuous variables with probability density functions (PDFs), we can differentiate the equation from Sklar's Theorem to obtain a relationship for the densities. The joint PDF $f(x1​,…,xd​)$ can be expressed as the product of the marginal PDFs fi​(xi​) and the copula density $c(u1​,…,ud​)$:

![[Pasted image 20250628233120.png]]

where ![[Pasted image 20250628233137.png]]​.14 This density decomposition is fundamental for statistical inference, as it allows for the construction of the log-likelihood function for parameter estimation.

### The Modularity Paradigm: Flexibility in Modeling

The true power of Sklar's theorem lies in the practical modeling flexibility it affords.15 Traditional multivariate models, such as the multivariate normal distribution, are monolithic; they force a single distributional family (e.g., Normal) onto all variables and restrict their dependence to a single type (e.g., linear correlation).8 Sklar's theorem shatters this rigidity by decoupling the modeling process into two independent, more manageable steps 21:

1. **Modeling the Marginals:** The practitioner can choose the best-fitting univariate distribution for each asset or risk factor individually. One stock's returns might be best described by a skewed Student's t-distribution to capture its heavy tails and asymmetry, while an exchange rate might follow a different process entirely. This is a level of specificity unattainable in classical multivariate frameworks.15
    
2. **Modeling the Dependence:** After specifying the marginals, the practitioner can select a copula function that best captures the joint dependence structure, particularly the behavior in the tails. This choice is not constrained by the choices made for the marginals.
    

This "separation of concerns" is a powerful paradigm. It allows for a modular approach where expertise in time series analysis (for the marginals) can be combined with expertise in dependence modeling (for the copula) to build more granular, realistic, and ultimately more robust financial models.2

### Python Example: Constructing a Joint Distribution

The following Python code demonstrates the principles of the probability integral transform and Sklar's theorem. We will use `scipy.stats` for our marginal distributions and `statsmodels` to construct a joint distribution from a Gaussian copula.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.distributions.copula.api import GaussianCopula, CopulaDistribution

# Set a seed for reproducibility
np.random.seed(42)

# 1. Demonstrate the Probability Integral Transform (PIT)
# Define a non-uniform marginal distribution, e.g., a Beta distribution
beta_dist = stats.beta(a=2, b=5)
# Generate 1000 random samples from this distribution
original_samples = beta_dist.rvs(size=1000)
# Apply the PIT by passing the samples through the CDF
uniform_samples = beta_dist.cdf(original_samples)

# Visualize the transformation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(original_samples, kde=True, ax=axes, stat='density', color='blue')
axes.set_title('Original Beta(2, 5) Samples')
sns.histplot(uniform_samples, kde=True, ax=axes, stat='density', color='green')
axes.set_title('Transformed Uniform(0, 1) Samples (PIT)')
plt.tight_layout()
plt.show()

# 2. Construct a Bivariate Distribution using Sklar's Theorem
# Define two different marginal distributions
marginals = [stats.norm(loc=0, scale=1), stats.t(df=5)]

# Define a copula to describe the dependence structure.
# We'll use a Gaussian copula with a correlation of 0.7.
# The `corr` parameter for a bivariate Gaussian copula is a scalar.
copula = GaussianCopula(corr=0.7)

# Create the joint distribution by "coupling" the marginals with the copula
# This is a direct application of Sklar's Theorem
joint_dist = CopulaDistribution(copula=copula, marginals=marginals)

# Generate 2000 samples from the newly constructed joint distribution
joint_samples = joint_dist.rvs(2000, random_state=42)

# Visualize the joint distribution and its marginals
g = sns.jointplot(x=joint_samples[:, 0], y=joint_samples[:, 1], kind="scatter", alpha=0.5)
g.ax_joint.set_xlabel('Samples from Normal(0,1) Marginal')
g.ax_joint.set_ylabel("Samples from Student's t(5) Marginal")
g.fig.suptitle("Joint Distribution from Gaussian Copula and Dissimilar Marginals", y=1.02)
plt.show()
```

The first part of the code clearly shows how applying the CDF of a Beta distribution to its own samples results in a uniform distribution. The second part demonstrates the constructive power of Sklar's theorem: we take two completely different distributions (a standard Normal and a Student's t with 5 degrees of freedom) and join them using a Gaussian copula to create a new, custom bivariate distribution. The resulting scatter plot shows data that follows the specified marginals but is bound together by the dependence structure imposed by the copula.

## 1.3 A Taxonomy of Copula Families and the Concept of Tail Dependence

Once Sklar's theorem provides the framework, the practical task becomes choosing an appropriate copula function from a vast library of possibilities. Copulas are typically grouped into families based on their mathematical construction and, more importantly, their dependence properties. For financial applications, the most critical of these properties is **tail dependence**, which describes the tendency of variables to experience extreme events together.6

Linear correlation measures the average co-movement across the entire range of outcomes. Tail dependence, in contrast, focuses exclusively on the behavior in the tails of the distribution.5 It answers questions like: "Given that one asset has experienced an extreme loss (e.g., in its bottom 5% of outcomes), what is the probability that another asset also experiences an extreme loss?" This is a fundamentally different and more relevant question for risk management than what correlation can answer. Two assets can have a low overall correlation but be highly dependent during a market crash.26

Tail dependence is quantified by the lower and upper tail dependence coefficients, λL​ and λU​. For two continuous random variables X1​ and X2​ with marginal CDFs F1​ and F2​, they are defined as limits of conditional probabilities 6:

- Upper Tail Dependence (λU​):
    
    ![[Pasted image 20250628233212.png]]
- Lower Tail Dependence (λL​):
    
    ![[Pasted image 20250628233219.png]]

Where F−1(q) is the quantile function (inverse CDF) at probability level q. A coefficient greater than zero indicates the presence of tail dependence. Crucially, these coefficients are properties of the copula alone, not the marginals.27 The choice of copula family is therefore a direct choice about the type of extreme event co-movement to be modeled. The most prominent families are Elliptical and Archimedean copulas.

### Elliptical Copulas

Elliptical copulas are derived from elliptical distributions, such as the multivariate normal and multivariate Student's t distributions. They are characterized by symmetric dependence structures, meaning the dependence in the upper tail is identical to the dependence in the lower tail.28

- **Gaussian Copula:** This is the most basic copula, derived from the multivariate normal distribution. Its dependence is entirely characterized by a linear correlation matrix. The Gaussian copula is simple and tractable but has a major drawback for financial modeling: it exhibits **zero tail dependence** (λL​=λU​=0) for any correlation less than 1.8 As discussed, this property makes it notoriously poor at modeling financial contagion and systemic risk.
    
- **Student's t-Copula:** Derived from the multivariate Student's t-distribution, this copula is a significant improvement over the Gaussian. It is parameterized by a correlation-like matrix and an additional parameter: the degrees of freedom (ν). The t-copula exhibits **symmetric, positive tail dependence** (λL​=λU​>0), where the strength of the tail dependence is controlled by ν. As ν decreases, the tails of the distribution become heavier, and the tail dependence increases. As ν→∞, the t-copula converges to the Gaussian copula.10 This makes it a flexible choice for modeling assets that tend to crash and boom together.
    

### Archimedean Copulas

Archimedean copulas are a large and flexible class of copulas constructed from a mathematical function called a **generator**, ϕ.29 The bivariate Archimedean copula has the general form

![[Pasted image 20250628233304.png]]The choice of generator function determines the properties of the copula, and this construction allows for a wide variety of dependence structures, including asymmetry, which is often observed in financial markets.10

- **Clayton Copula:** The Clayton copula is defined by a generator that produces strong **lower tail dependence** and weak (zero) upper tail dependence $(λL​>0,λU​=0)$.4 This makes it exceptionally well-suited for modeling phenomena where joint negative events are more likely than joint positive events, such as simultaneous market crashes or correlated defaults in a credit portfolio.32
    
- **Gumbel Copula:** The Gumbel copula is, in a sense, the mirror image of the Clayton. It exhibits strong **upper tail dependence** and weak (zero) lower tail dependence $(λU​>0,λL​=0)$.4 It is useful for modeling variables that tend to experience extreme positive events together, such as the returns of assets in a booming sector.
    
- **Frank Copula:** The Frank copula allows for both positive and negative dependence and has a symmetric structure, but unlike the elliptical copulas, it exhibits **zero tail dependence** in both tails (λL​=λU​=0).8 It tends to capture stronger dependence in the center of the distribution compared to the Gaussian copula.
    

The choice among these copulas is a critical modeling decision. It is not merely a statistical fitting exercise but an explicit assumption about the nature of risk. Selecting a Clayton copula over a Gaussian copula is a deliberate choice to model the "contagion" effect during market downturns, a choice that will have a profound impact on any resulting risk measure.

### Table 1: Properties of Common Bivariate Copulas

The following table serves as a quick reference guide for the properties of the most common copula families used in finance.

| Copula Family   | Specific Copula | Formula C(u,v)             | Parameter(s)                                     | Lower Tail Dep. (λL​)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Upper Tail Dep. (λU​)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------- | --------------- | -------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Elliptical**  | Gaussian        | Φρ​(Φ−1(u),Φ−1(v))         | ρ∈(−1,1)                                         | 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|                 | Student's t     | tρ,ν​(tν−1​(u),tν−1​(v))   | ρ∈(−1,1),ν>2                                     | 2tν+1​(−1+ρ(ν+1)(1−ρ)​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.88em" viewBox="0 0 400000 1944" preserveAspectRatio="xMinYMin slice"><path d="M983 90%0Al0 -0%0Ac4,-6.7,10,-10,18,-10 H400000v40%0AH1013.1s-83.4,268,-264.1,840c-180.7,572,-277,876.3,-289,913c-4.7,4.7,-12.7,7,-24,7%0As-12,0,-12,0c-1.3,-3.3,-3.7,-11.7,-7,-25c-35.3,-125.3,-106.7,-373.3,-214,-744%0Ac-10,12,-21,25,-33,39s-32,39,-32,39c-6,-5.3,-15,-14,-27,-26s25,-30,25,-30%0Ac26.7,-32.7,52,-63,76,-91s52,-60,52,-60s208,722,208,722%0Ac56,-175.3,126.3,-397.3,211,-666c84.7,-268.7,153.8,-488.2,207.5,-658.5%0Ac53.7,-170.3,84.5,-266.8,92.5,-289.5z%0AM1001 80h400000v40h-400000z"></path></svg>)​) | 2tν+1​(−1+ρ(ν+1)(1−ρ)​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.88em" viewBox="0 0 400000 1944" preserveAspectRatio="xMinYMin slice"><path d="M983 90%0Al0 -0%0Ac4,-6.7,10,-10,18,-10 H400000v40%0AH1013.1s-83.4,268,-264.1,840c-180.7,572,-277,876.3,-289,913c-4.7,4.7,-12.7,7,-24,7%0As-12,0,-12,0c-1.3,-3.3,-3.7,-11.7,-7,-25c-35.3,-125.3,-106.7,-373.3,-214,-744%0Ac-10,12,-21,25,-33,39s-32,39,-32,39c-6,-5.3,-15,-14,-27,-26s25,-30,25,-30%0Ac26.7,-32.7,52,-63,76,-91s52,-60,52,-60s208,722,208,722%0Ac56,-175.3,126.3,-397.3,211,-666c84.7,-268.7,153.8,-488.2,207.5,-658.5%0Ac53.7,-170.3,84.5,-266.8,92.5,-289.5z%0AM1001 80h400000v40h-400000z"></path></svg>)​) |
| **Archimedean** | Clayton         | (u−θ+v−θ−1)−1/θ            | θ∈[−1,∞)∖{0}                                     | 2−1/θ for θ>0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|                 | Gumbel          | exp(−[(−lnu)θ+(−lnv)θ]1/θ) | $\theta \in.params = params['Gaussian']['corr']$ |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |




```python
copulas.params = (params['corr'], params['df'])
copulas['Gumbel'].params = params['Gumbel']['theta']
copulas['Frank'].params = params['Frank']['theta']

# Generate and plot samples

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes = axes.flatten()

for i, (name, cop) in enumerate(copulas.items()):

# Generate 2000 random samples

samples = cop.random(2000, seed=42)


# Create scatter plot
ax = axes[i]
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
ax.set_title(f'{name} Copula')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


# Hide the last subplot as it's unused

axes[-1].axis('off')

plt.suptitle("Comparison of Bivariate Copula Dependence Structures (Kendall's Tau ≈ 0.5)", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()

```

The output plots vividly illustrate the theoretical concepts. The Gaussian plot is a symmetric, elliptical cloud. The Student's t plot is similar but with more points clustered in the lower-left and upper-right corners, showing its symmetric tail dependence. The Clayton plot shows a distinct concentration of points in the lower-left corner (lower tail dependence), while the Gumbel plot shows concentration in the upper-right corner (upper tail dependence). The Frank plot is symmetric but more concentrated in the center than the Gaussian. These visual signatures are crucial for developing an intuition for which copula might be appropriate for a given dataset.

## 1.4 The Copula-GARCH Modeling Workflow in Python

Applying copulas to real-world financial time series requires a multi-step process that respects the statistical properties of the data. A frequent and critical mistake is to apply a copula directly to raw asset returns. Financial returns are not independent and identically distributed (i.i.d.); they exhibit well-known stylized facts such as autocorrelation and, most importantly, **volatility clustering** (heteroskedasticity), where periods of high volatility are followed by more high volatility, and vice versa.[6, 24] Applying a copula to data that violates the i.i.d. assumption leads to a misspecified model and unreliable results.[24]

The industry-standard solution is the **Copula-GARCH** methodology.[34, 35, 36] This approach recognizes that an asset's return can be decomposed into a predictable component (conditional mean and volatility) and an unpredictable, i.i.d. random shock. The GARCH model is used to filter out the predictable time-series dynamics, isolating the random shocks. The copula is then applied to model the dependence structure of these shocks, not the raw returns themselves. This isolates the "pure" dependence from the confounding effects of individual asset dynamics.

The workflow can be summarized in the following steps:

1.  **Data Preparation:** Obtain time series of asset prices and calculate their log returns.
2.  **Marginal Distribution Modeling:** For each asset's return series, fit an appropriate univariate time-series model. A GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model, or one of its variants like GJR-GARCH, is typically used to capture volatility clustering and leverage effects. The model should also specify a distribution for the innovations (shocks), such as a skewed Student's t-distribution, to account for fat tails and asymmetry.[37]
3.  **Residual Extraction and Transformation:** From each fitted GARCH model, extract the standardized residuals, $z_t = (R_t - \mu_t) / \sigma_t$. By construction, these residuals should be approximately i.i.d. with a mean of zero and a variance of one. Then, transform these standardized residuals into uniform "pseudo-observations" on using the probability integral transform. Since the true CDF of the residuals is unknown, the empirical CDF is commonly used for this transformation.[24]
4.  **Copula Fitting and Selection:** Fit several candidate copula families (e.g., Gaussian, t, Clayton, Gumbel) to the multivariate dataset of uniform pseudo-observations.
5.  **Model Selection:** Select the best-fitting copula based on statistical criteria such as the maximum log-likelihood, Akaike Information Criterion (AIC), or Bayesian Information Criterion (BIC).[38] Goodness-of-fit tests can also be employed.

### Python Example: A Step-by-Step Copula-GARCH Implementation

This tutorial demonstrates the complete workflow using Python. We will model the dependence between the daily returns of the S&P 500 ETF (SPY) and a Gold ETF (GLD).

**Libraries:**
*   `yfinance`: For fetching financial data.
*   `arch`: For fitting GARCH models.
*   `numpy`, `pandas`: For data manipulation.
*   `scipy.stats`: For transformations.
*   `statsmodels.distributions.copula.api`: For fitting copulas.
*   `matplotlib.pyplot`, `seaborn`: For plotting.

```python
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import rankdata
import statsmodels.api as sm
from statsmodels.distributions.copula.api import (
    GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Preparation ---
tickers =
start_date = '2010-01-01'
end_date = '2022-12-31'

# Fetch daily closing prices
prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate log returns and drop missing values
log_returns = np.log(prices / prices.shift(1)).dropna()

# --- 2. Marginal Distribution Modeling (GARCH) ---
# We'll fit a GJR-GARCH(1,1) model with a skewed Student's t distribution
# This is a robust choice for financial returns
garch_models = {}
standardized_residuals = pd.DataFrame()

for ticker in tickers:
    print(f"Fitting GARCH model for {ticker}...")
    # Model specification: p=1, q=1 for GARCH, o=1 for leverage effect (GJR-GARCH)
    model = arch_model(log_returns[ticker] * 100, # Scale for better convergence
                       vol='Garch', p=1, o=1, q=1, dist='skewt')
    
    # Fit the model
    garch_result = model.fit(disp='off')
    garch_models[ticker] = garch_result
    
    # Extract standardized residuals
    standardized_residuals[ticker] = garch_result.std_resid
    print(garch_result.summary())

# --- 3. Residual Transformation to Uniform ---
# Use the empirical CDF (via rankdata) to transform residuals to pseudo-observations
# This is a non-parametric approach for the PIT
pseudo_obs = pd.DataFrame()
for ticker in tickers:
    pseudo_obs[ticker] = rankdata(standardized_residuals[ticker]) / (len(standardized_residuals) + 1)

# Visualize the pseudo-observations
sns.jointplot(x=pseudo_obs, y=pseudo_obs, kind='scatter', alpha=0.2)
plt.suptitle('Uniform Pseudo-Observations from Standardized Residuals', y=1.02)
plt.show()

# --- 4 & 5. Copula Fitting and Selection ---
# Data must be a numpy array for statsmodels
data_for_copula = pseudo_obs.values

# Candidate copulas
candidate_copulas = {
    'Gaussian': GaussianCopula(),
    'Student-t': StudentTCopula(),
    'Clayton': ClaytonCopula(),
    'Gumbel': GumbelCopula(),
    'Frank': FrankCopula()
}

results =
for name, copula in candidate_copulas.items():
    try:
        # Fit the copula to the data
        # For elliptical copulas (Gaussian, t), we first estimate the correlation param
        if name in:
            copula.fit_corr_param(data_for_copula)
        # For Archimedean copulas, we fit the theta parameter
        else:
            copula.fit_corr_param(data_for_copula) # `fit_corr_param` estimates theta via Kendall's tau

        # Calculate log-likelihood
        log_likelihood = copula.logpdf(data_for_copula).sum()
        
        # Get number of parameters (k)
        if name == 'Student-t':
            k = 2 # corr and df
        else:
            k = 1 # corr or theta
            
        # Calculate AIC and BIC
        n = len(data_for_copula)
        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + np.log(n) * k
        
        results.append({
            'Copula': name,
            'Log-Likelihood': log_likelihood,
            'AIC': aic,
            'BIC': bic
        })
        print(f"Fitted {name} Copula. LL: {log_likelihood:.2f}, AIC: {aic:.2f}, BIC: {bic:.2f}")

    except Exception as e:
        print(f"Could not fit {name} copula: {e}")

# Display results in a DataFrame for easy comparison
results_df = pd.DataFrame(results).sort_values(by='AIC').set_index('Copula')
print("\n--- Copula Selection Results ---")
print(results_df)

best_copula_name = results_df.index
print(f"\nBest fitting copula based on AIC: {best_copula_name}")
````

This code provides a complete, practical template for the Copula-GARCH workflow. It begins with data acquisition and proceeds through marginal modeling with a sophisticated GARCH specification. It then correctly transforms the resulting i.i.d. residuals into uniform pseudo-observations before fitting and comparing several copula models. The final output is a ranked list of copulas, allowing the analyst to make a data-driven choice for the dependence structure. This process forms the essential core of most advanced financial risk models that employ copulas.

## 1.5 Capstone Project: Advanced Portfolio Risk Management with Copula-GARCH

This capstone project synthesizes all the concepts covered in the chapter into a complete, end-to-end application. The objective is to perform a sophisticated risk analysis of a two-asset portfolio, calculating its Value-at-Risk (VaR) and Expected Shortfall (ES). The central theme is to demonstrate quantitatively how the choice of copula—specifically, a tail-dependent copula versus a simple Gaussian copula—dramatically alters the perception of portfolio risk.

**Problem Statement:** An investor holds an equally-weighted portfolio of the SPDR S&P 500 ETF (SPY) and the Invesco QQQ Trust (QQQ). The task is to estimate the 1-day 99% VaR and ES of this portfolio. We will compare the risk estimates generated by a model using the best-fitting copula against a model that naively assumes a Gaussian dependence structure.

The project will proceed by answering a series of guiding questions.

---

### Q1: How do we set up the problem and model the individual asset risks?

**Response:** The first step is to acquire the historical data for SPY and QQQ and model their individual return dynamics. Since financial returns are known to exhibit volatility clustering and fat tails, a GJR-GARCH model with a skewed Student's t distribution is an appropriate choice for each marginal distribution. This captures the unique risk profile of each asset before considering their joint behavior.


```python
# Continue from the previous section's imports
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import rankdata, t
from statsmodels.distributions.copula.api import GaussianCopula, StudentTCopula

# --- Step 1: Data Fetching and Marginal Modeling ---
tickers =
start_date = '2010-01-01'
end_date = '2022-12-31'

prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
log_returns = np.log(prices / prices.shift(1)).dropna()

garch_models = {}
standardized_residuals = pd.DataFrame()

for ticker in tickers:
    model = arch_model(log_returns[ticker] * 100, vol='Garch', p=1, o=1, q=1, dist='skewt')
    garch_result = model.fit(disp='off')
    garch_models[ticker] = garch_result
    standardized_residuals[ticker] = garch_result.std_resid

print("GARCH models fitted for SPY and QQQ.")
```

### Q2: How do we model the dependence between the asset shocks?

**Response:** With the individual time-series dynamics filtered out, we now model the dependence structure of the i.i.d. standardized residuals. We first transform them into uniform pseudo-observations using their empirical CDFs. Then, we fit several candidate copulas and use AIC and BIC to select the one that best describes the data.


```python
# --- Step 2: Transform Residuals and Select Best Copula ---
pseudo_obs = pd.DataFrame({
    ticker: rankdata(standardized_residuals[ticker]) / (len(standardized_residuals) + 1)
    for ticker in tickers
})
data_for_copula = pseudo_obs.values

# Fit and compare copulas
candidate_copulas_fit = {
    'Gaussian': GaussianCopula(),
    'Student-t': StudentTCopula(),
    'Clayton': ClaytonCopula(),
    'Gumbel': GumbelCopula(),
    'Frank': FrankCopula()
}

fit_results =
for name, copula in candidate_copulas_fit.items():
    copula.fit_corr_param(data_for_copula)
    ll = copula.logpdf(data_for_copula).sum()
    k = 2 if name == 'Student-t' else 1
    n = len(data_for_copula)
    aic = -2 * ll + 2 * k
    bic = -2 * ll + np.log(n) * k
    fit_results.append({'Copula': name, 'Log-Likelihood': ll, 'AIC': aic, 'BIC': bic})

fit_results_df = pd.DataFrame(fit_results).sort_values(by='AIC').set_index('Copula')
print("\n--- Copula Goodness-of-Fit Comparison ---")
print(fit_results_df)

best_copula_name = fit_results_df.index
best_copula = candidate_copulas_fit[best_copula_name]
print(f"\nSelected Copula: {best_copula_name}")

# For comparison, we will also use the Gaussian copula
gaussian_copula_comp = candidate_copulas_fit['Gaussian']
```

### Table 2: Copula Goodness-of-Fit Comparison

The output of the code above will generate a table similar to this, allowing for rigorous model selection.

|Copula|Log-Likelihood|AIC|BIC|
|---|---|---|---|
|**Student-t**|1350.75|-2697.50|-2684.65|
|**Gumbel**|1295.30|-2588.60|-2581.98|
|**Frank**|1250.10|-2498.20|-2491.58|
|**Gaussian**|1185.45|-2368.90|-2362.28|
|**Clayton**|1150.22|-2298.44|-2291.82|
|_(Note: These are illustrative values. Actual results will depend on the data.)_||||

Based on these illustrative results, the Student's t-copula provides the best fit to the dependence structure of the shocks between SPY and QQQ, as indicated by its lowest AIC and BIC scores.

### Q3: How do we simulate future portfolio returns using our full Copula-GARCH model?

**Response:** To calculate VaR and ES, we must simulate a large number of possible next-day portfolio returns. The simulation algorithm combines all the pieces we have built: the fitted GARCH models for the marginals and the fitted copula for the dependence.

The Monte Carlo simulation process is as follows:

1. **Simulate from the Copula:** Generate N (e.g., 100,000) random samples (u1​,u2​) from the chosen copula (both the best-fitting t-copula and the Gaussian copula for comparison).
    
2. **Inverse Transform to Shocks:** Convert these uniform samples back into standardized residuals (z1​,z2​). This is done by applying the inverse CDF (or quantile function) of the marginals' innovation distribution (skewed Student's t).
    
3. **Forecast Volatility:** Use the fitted GARCH models to produce a one-step-ahead forecast of the conditional volatility (σt+1​) for each asset.
    
4. **Simulate Asset Returns:** Construct the simulated next-day log returns for each asset using the GARCH equation: Rt+1​=μt+1​+σt+1​⋅zt+1​. The conditional mean forecast, μt+1​, is also obtained from the GARCH model.
    
5. **Calculate Portfolio Returns:** For each of the N simulations, calculate the portfolio return using the specified weights (50% SPY, 50% QQQ).
    

```python
# --- Step 3: Monte Carlo Simulation of Portfolio Returns ---
def simulate_portfolio_returns(garch_models, copula, n_sims=100000):
    # Get parameters from fitted GARCH models
    params_spy = garch_models.params
    params_qqq = garch_models['QQQ'].params
    
    # Get last values for forecasting
    res_spy = garch_models.resid[-1]
    res_qqq = garch_models['QQQ'].resid[-1]
    var_spy = garch_models.conditional_volatility[-1]**2
    var_qqq = garch_models['QQQ'].conditional_volatility[-1]**2

    # Forecast next-day variance
    forecast_var_spy = params_spy['omega'] + params_spy['alpha'] * res_spy**2 + \
                       params_spy['gamma'] * res_spy**2 * (res_spy < 0) + params_spy['beta'] * var_spy
    forecast_var_qqq = params_qqq['omega'] + params_qqq['alpha'] * res_qqq**2 + \
                       params_qqq['gamma'] * res_qqq**2 * (res_qqq < 0) + params_qqq['beta'] * var_qqq

    # Forecast next-day conditional standard deviation and mean
    forecast_std_spy = np.sqrt(forecast_var_spy)
    forecast_std_qqq = np.sqrt(forecast_var_qqq)
    forecast_mean_spy = params_spy['mu']
    forecast_mean_qqq = params_qqq['mu']
    
    # Simulate from the copula
    copula_samples = copula.rvs(n_sims, random_state=42)
    
    # Inverse transform to get standardized residuals (shocks)
    # Using the skewed-t distribution from the GARCH models
    df_spy, skew_spy = params_spy['nu'], params_spy['lambda']
    df_qqq, skew_qqq = params_qqq['nu'], params_qqq['lambda']
    
    shocks_spy = t.ppf(copula_samples[:, 0], df=df_spy, loc=0, scale=1) * (1 + skew_spy * np.sign(copula_samples[:, 0] - 0.5))
    shocks_qqq = t.ppf(copula_samples[:, 1], df=df_qqq, loc=0, scale=1) * (1 + skew_qqq * np.sign(copula_samples[:, 1] - 0.5))

    # Simulate asset returns (rescaled back from %)
    sim_returns_spy = (forecast_mean_spy + forecast_std_spy * shocks_spy) / 100
    sim_returns_qqq = (forecast_mean_qqq + forecast_std_qqq * shocks_qqq) / 100
    
    # Calculate portfolio returns (50/50 weights)
    portfolio_returns = 0.5 * sim_returns_spy + 0.5 * sim_returns_qqq
    
    return portfolio_returns

# Simulate using both the best-fit (t) and Gaussian copulas
portfolio_sim_t = simulate_portfolio_returns(garch_models, best_copula)
portfolio_sim_gauss = simulate_portfolio_returns(garch_models, gaussian_copula_comp)

print(f"\nSimulated {len(portfolio_sim_t)} portfolio returns using {best_copula_name} and Gaussian copulas.")
```

### Q4: What are the portfolio's VaR and ES, and how does the choice of copula impact them?

**Response:** Now that we have two sets of simulated portfolio returns—one from the more realistic Student's t-copula model and one from the naive Gaussian copula model—we can calculate the risk measures. VaR is the quantile of the simulated return distribution, while ES is the average of all simulated returns that are worse than the VaR.

The comparison of these measures will reveal the quantitative impact of modeling tail dependence. The Student's t-copula, by generating more frequent and more severe joint extreme negative events, will result in a distribution of portfolio returns with a fatter left tail. This will lead to higher (more negative) VaR and substantially higher ES estimates compared to the Gaussian copula model.

This difference is not merely academic. For a financial institution, underestimating VaR and ES could lead to insufficient capital reserves, leaving it vulnerable to unexpected losses during a market crisis. The results demonstrate that properly modeling tail dependence is a critical component of prudent risk management.

A particularly important observation is that the Expected Shortfall (ES) is more sensitive to the choice of copula than the Value-at-Risk (VaR).39 VaR is a single point—the threshold of the tail—while ES is the average of all outcomes

_within_ the tail. The t-copula doesn't just make tail events more likely (pushing the VaR threshold out); it makes the events in that tail much more severe. Since ES averages these more severe losses, its value increases more dramatically than VaR when moving from a Gaussian to a t-copula. This highlights why regulators and sophisticated practitioners increasingly favor ES as a risk measure: it provides a more complete picture of the potential magnitude of tail losses, a feature that is captured more accurately by tail-dependent copulas.41

```python
# --- Step 4: Calculate VaR and ES and Compare ---
def calculate_risk_measures(returns, confidence_level=0.99):
    alpha = 1 - confidence_level
    var = np.percentile(returns, alpha * 100)
    es = returns[returns <= var].mean()
    return var, es

confidence_level = 0.99

# Calculate risk measures for the t-copula model
var_t, es_t = calculate_risk_measures(portfolio_sim_t, confidence_level)

# Calculate risk measures for the Gaussian copula model
var_gauss, es_gauss = calculate_risk_measures(portfolio_sim_gauss, confidence_level)

# Create comparison table
risk_comparison = pd.DataFrame({
    "GARCH-Student's t Model": [f"{var_t:.4%}", f"{es_t:.4%}"],
    "GARCH-Gaussian Model": [f"{var_gauss:.4%}", f"{es_gauss:.4%}"],
    "Difference (%)": [
        f"{(var_t/var_gauss - 1):.2%}",
        f"{(es_t/es_gauss - 1):.2%}"
    ]
}, index=)

print("\n--- Portfolio Risk Measure Comparison ---")
print(risk_comparison)

# Visualize the distributions
plt.figure(figsize=(10, 6))
sns.histplot(portfolio_sim_gauss, color='blue', label='Gaussian Copula', stat='density', bins=100)
sns.histplot(portfolio_sim_t, color='red', label=f'{best_copula_name} Copula', stat='density', bins=100, alpha=0.7)
plt.axvline(var_gauss, color='blue', linestyle='--', label=f'Gaussian VaR: {var_gauss:.2%}')
plt.axvline(var_t, color='red', linestyle='--', label=f'{best_copula_name} VaR: {var_t:.2%}')
plt.title('Distribution of Simulated 1-Day Portfolio Returns')
plt.xlabel('Portfolio Log Return')
plt.legend()
plt.show()
```

### Table 3: Portfolio Risk Measure Comparison (99% Confidence Level)

The final output of the project is a clear, quantitative comparison of the risk measures derived from the two models.

|Risk Measure|GARCH-Student's t Model|GARCH-Gaussian Model|Difference (%)|
|---|---|---|---|
|**VaR (99%)**|-3.52%|-2.98%|18.12%|
|**ES (99%)**|-4.89%|-3.71%|31.81%|
|_(Note: These are illustrative values. The key takeaway is the direction and relative magnitude of the differences.)_||||

The results are stark. The model using the more realistic Student's t-copula estimates a 99% VaR that is over 18% larger than the estimate from the naive Gaussian model. Even more dramatically, the Expected Shortfall is nearly 32% larger. This is the quantifiable value of advanced dependence modeling. By failing to account for tail dependence, the Gaussian model provides a dangerously optimistic view of the portfolio's risk, while the Copula-GARCH model with a Student's t-copula offers a more prudent and realistic assessment, which is indispensable for robust financial risk management.

## References
**

1. Modelling Copulas: An Overview, acessado em junho 28, 2025, [https://sias.org.uk/media/1188/modelling-copulas-an-overview.pdf](https://sias.org.uk/media/1188/modelling-copulas-an-overview.pdf)
    
2. Copulas and trading strategies - Macrosynergy, acessado em junho 28, 2025, [https://macrosynergy.com/research/copulas-and-trading-strategies/](https://macrosynergy.com/research/copulas-and-trading-strategies/)
    
3. Copula - Meaning, Explained, Examples, Applications In Finance - WallStreetMojo, acessado em junho 28, 2025, [https://www.wallstreetmojo.com/copula/](https://www.wallstreetmojo.com/copula/)
    
4. Unlocking Copula Methods in Finance, acessado em junho 28, 2025, [https://www.numberanalytics.com/blog/ultimate-guide-copula-methods-financial-mathematics](https://www.numberanalytics.com/blog/ultimate-guide-copula-methods-financial-mathematics)
    
5. The Difference Between Correlation and Tail Dependence in Simple Terms, acessado em junho 28, 2025, [https://www.finance-tutoring.fr/the-difference-between-correlation-and-tail-dependence-in-simple-terms/?mobile=1](https://www.finance-tutoring.fr/the-difference-between-correlation-and-tail-dependence-in-simple-terms/?mobile=1)
    
6. Unconditional Copula-Based Simulation of Tail Dependence for Co-movement of International Equity Markets, acessado em junho 28, 2025, [https://methods.stat.kit.edu/download/doc_secure1/Comovement-JAE.pdf](https://methods.stat.kit.edu/download/doc_secure1/Comovement-JAE.pdf)
    
7. An Introduction to Copulas, acessado em junho 28, 2025, [http://www.columbia.edu/~mh2078/QRM/Copulas.pdf](http://www.columbia.edu/~mh2078/QRM/Copulas.pdf)
    
8. Copula for Pairs Trading: A Detailed, But Practical Introduction - Hudson & Thames, acessado em junho 28, 2025, [https://hudsonthames.org/copula-for-pairs-trading-introduction/](https://hudsonthames.org/copula-for-pairs-trading-introduction/)
    
9. A Deeper Intro to Copulas — arbitragelab 1.0.0 documentation, acessado em junho 28, 2025, [https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/copula_approach/copula_deeper_intro.html](https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/copula_approach/copula_deeper_intro.html)
    
10. What are Copulas? - The Certificate in Quantitative Finance | CQF, acessado em junho 28, 2025, [https://www.cqf.com/blog/quant-finance-101/what-are-copulas](https://www.cqf.com/blog/quant-finance-101/what-are-copulas)
    
11. Financial Correlation Modeling – Bottom-Up Approaches | AnalystPrep, acessado em junho 28, 2025, [https://analystprep.com/study-notes/frm/part-2/market-risk-measurement-and-management/financial-correlation-modeling-bottom-up-approaches/](https://analystprep.com/study-notes/frm/part-2/market-risk-measurement-and-management/financial-correlation-modeling-bottom-up-approaches/)
    
12. Copulas: An Introduction I - Fundamentals - Columbia University, acessado em junho 28, 2025, [http://www.columbia.edu/~rf2283/Conference/1Fundamentals%20(1)Seagers.pdf](http://www.columbia.edu/~rf2283/Conference/1Fundamentals%20\(1\)Seagers.pdf)
    
13. An intuitive, visual guide to copulas - Thomas Wiecki, acessado em junho 28, 2025, [https://twiecki.io/blog/2018/05/03/copulas/](https://twiecki.io/blog/2018/05/03/copulas/)
    
14. Copula (statistics) - Wikipedia, acessado em junho 28, 2025, [https://en.wikipedia.org/wiki/Copula_(statistics)](https://en.wikipedia.org/wiki/Copula_\(statistics\))
    
15. Copula-Based Models for Financial Time Series1 ... - Duke Economics, acessado em junho 28, 2025, [https://public.econ.duke.edu/~ap172/Patton_copula_handbook_19nov07.pdf](https://public.econ.duke.edu/~ap172/Patton_copula_handbook_19nov07.pdf)
    
16. Lecture 12: Copula 12.1 Introduction 12.2 Sklar's theorem and copulas, acessado em junho 28, 2025, [https://faculty.washington.edu/yenchic/21Sp_stat542/Lec12_copula.pdf](https://faculty.washington.edu/yenchic/21Sp_stat542/Lec12_copula.pdf)
    
17. How to prove Sklar's Theorem - ResearchGate, acessado em junho 28, 2025, [https://www.researchgate.net/profile/Juan-Fernandez-Sanchez-2/publication/266003919_How_to_Prove_Sklar's_Theorem/links/5429c60a0cf277d58e86fe1b/How-to-Prove-Sklars-Theorem.pdf](https://www.researchgate.net/profile/Juan-Fernandez-Sanchez-2/publication/266003919_How_to_Prove_Sklar's_Theorem/links/5429c60a0cf277d58e86fe1b/How-to-Prove-Sklars-Theorem.pdf)
    
18. An Introduction to Copula Theory - NTNU Open, acessado em junho 28, 2025, [https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2980279/no.ntnu%3Ainspera%3A79432288%3A35311762.pdf?sequence=1](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2980279/no.ntnu%3Ainspera%3A79432288%3A35311762.pdf?sequence=1)
    
19. papers.neurips.cc, acessado em junho 28, 2025, [http://papers.neurips.cc/paper/4082-copula-processes.pdf](http://papers.neurips.cc/paper/4082-copula-processes.pdf)
    
20. Derivation of Sklar's theorem for copula - Cross Validated - Stats Stackexchange, acessado em junho 28, 2025, [https://stats.stackexchange.com/questions/485219/derivation-of-sklars-theorem-for-copula](https://stats.stackexchange.com/questions/485219/derivation-of-sklars-theorem-for-copula)
    
21. Key Statistics Terms # 28:Part 1 Key Concepts of Copula | by Rajiv Gopinath | Medium, acessado em junho 28, 2025, [https://medium.com/@mail2rajivgopinath/key-statistics-terms-28-part-1-key-concepts-of-copula-4338eebf73ae](https://medium.com/@mail2rajivgopinath/key-statistics-terms-28-part-1-key-concepts-of-copula-4338eebf73ae)
    
22. Correlation & Dependency Structures | Actuaries, acessado em junho 28, 2025, [https://www.actuaries.org.uk/documents/correlation-and-dependency-structures-slides](https://www.actuaries.org.uk/documents/correlation-and-dependency-structures-slides)
    
23. How do we separate marginals from dependence using copulas, and why do we assume uniform marginals? - Stats Stackexchange, acessado em junho 28, 2025, [https://stats.stackexchange.com/questions/191304/how-do-we-separate-marginals-from-dependence-using-copulas-and-why-do-we-assume](https://stats.stackexchange.com/questions/191304/how-do-we-separate-marginals-from-dependence-using-copulas-and-why-do-we-assume)
    
24. Copula Modelling to Analyse Financial Data - MDPI, acessado em junho 28, 2025, [https://www.mdpi.com/1911-8074/15/3/104](https://www.mdpi.com/1911-8074/15/3/104)
    
25. Methods of Tail Dependence Estimation - eScholarship, acessado em junho 28, 2025, [https://escholarship.org/content/qt07x6p3bk/qt07x6p3bk_noSplash_c6faf62d3de3c34d81607b2465a48c15.pdf](https://escholarship.org/content/qt07x6p3bk/qt07x6p3bk_noSplash_c6faf62d3de3c34d81607b2465a48c15.pdf)
    
26. Tail dependence - Wikipedia, acessado em junho 28, 2025, [https://en.wikipedia.org/wiki/Tail_dependence](https://en.wikipedia.org/wiki/Tail_dependence)
    
27. 1 Tail dependence, acessado em junho 28, 2025, [https://wisostat.uni-koeln.de/fileadmin/sites/statistik/pdf_publikationen/TDCSchmidt.pdf](https://wisostat.uni-koeln.de/fileadmin/sites/statistik/pdf_publikationen/TDCSchmidt.pdf)
    
28. Classification of Copulas | Forum | Bionic Turtle, acessado em junho 28, 2025, [https://forum.bionicturtle.com/threads/classification-of-copulas.5597/](https://forum.bionicturtle.com/threads/classification-of-copulas.5597/)
    
29. Dependence Modeling with Archimedean Copulas - Portland State University, acessado em junho 28, 2025, [https://web.pdx.edu/~fountair/seminar/arch.pdf](https://web.pdx.edu/~fountair/seminar/arch.pdf)
    
30. Archimedean Copulas - Uni Ulm, acessado em junho 28, 2025, [https://www.uni-ulm.de/fileadmin/website_uni_ulm/mawi.inst.110/Seminar__Copulas_and_Applications_WS2021/David_Ziener_-_Notes__Archimedean_Copulas.pdf](https://www.uni-ulm.de/fileadmin/website_uni_ulm/mawi.inst.110/Seminar__Copulas_and_Applications_WS2021/David_Ziener_-_Notes__Archimedean_Copulas.pdf)
    
31. www.numberanalytics.com, acessado em junho 28, 2025, [https://www.numberanalytics.com/blog/ultimate-guide-copula-methods-financial-mathematics#:~:text=Archimedean%20copulas%20are%20a%20family,credit%20risk%20and%20portfolio%20risk.](https://www.numberanalytics.com/blog/ultimate-guide-copula-methods-financial-mathematics#:~:text=Archimedean%20copulas%20are%20a%20family,credit%20risk%20and%20portfolio%20risk.)
    
32. Copula models in Python using sympy, acessado em junho 28, 2025, [https://williamsantos.me/posts/2022/copula-from-scratch-with-sympy/](https://williamsantos.me/posts/2022/copula-from-scratch-with-sympy/)
    
33. Dependence Patterns across Financial Markets: a Mixed Copula Approach - Portland State University, acessado em junho 28, 2025, [https://web.pdx.edu/~fountair/seminar/CopulaAppHu.pdf](https://web.pdx.edu/~fountair/seminar/CopulaAppHu.pdf)
    
34. Using Extreme Value Theory and Copulas to Evaluate Market Risk - MATLAB &, acessado em junho 28, 2025, [https://www.mathworks.com/help/econ/using-extreme-value-theory-and-copulas-to-evaluate-market-risk.html](https://www.mathworks.com/help/econ/using-extreme-value-theory-and-copulas-to-evaluate-market-risk.html)
    
35. The Copula GARCH Model, acessado em junho 28, 2025, [https://cran.r-project.org/web/packages/copula/vignettes/copula_GARCH.html](https://cran.r-project.org/web/packages/copula/vignettes/copula_GARCH.html)
    
36. A Simple Copula-GARCH Example — MUArch 0.0.4 documentation, acessado em junho 28, 2025, [https://muarch.readthedocs.io/en/latest/examples/Copula-GARCH.html](https://muarch.readthedocs.io/en/latest/examples/Copula-GARCH.html)
    
37. VaR: Value at Risk: Estimating VaR with GARCH: A Comprehensive Guide - FasterCapital, acessado em junho 28, 2025, [https://fastercapital.com/content/VaR--Value-at-Risk---Estimating-VaR-with-GARCH--A-Comprehensive-Guide.html](https://fastercapital.com/content/VaR--Value-at-Risk---Estimating-VaR-with-GARCH--A-Comprehensive-Guide.html)
    
38. Key Concepts of Copula: Dependence Modeling & Python Implementation - Rajiv Gopinath, acessado em junho 28, 2025, [https://www.rajivgopinath.com/blogs/statistics-and-data-science-hub/advanced-statistical-methods/multivariate-distributions/key-concepts-of-copula](https://www.rajivgopinath.com/blogs/statistics-and-data-science-hub/advanced-statistical-methods/multivariate-distributions/key-concepts-of-copula)
    
39. Expected shortfall - Wikipedia, acessado em junho 28, 2025, [https://en.wikipedia.org/wiki/Expected_shortfall](https://en.wikipedia.org/wiki/Expected_shortfall)
    
40. Dive into Expected Shortfall ES for Quant Risk - Number Analytics, acessado em junho 28, 2025, [https://www.numberanalytics.com/blog/dive-into-expected-shortfall-es-quant-risk](https://www.numberanalytics.com/blog/dive-into-expected-shortfall-es-quant-risk)
    

Conditional Value at Risk (CVaR) or Expected Shortfall: Formula and Calculation in Python and Excel - QuantInsti Blog, acessado em junho 28, 2025, [https://blog.quantinsti.com/cvar-expected-shortfall/](https://blog.quantinsti.com/cvar-expected-shortfall/)**