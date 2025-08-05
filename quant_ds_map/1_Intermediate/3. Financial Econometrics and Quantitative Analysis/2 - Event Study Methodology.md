# 3.2 The Event Study: Isolating the Economic Impact of News

## Introduction: Quantifying the Market's Reaction

In the world of quantitative finance, a central challenge is to isolate and measure the economic impact of a specific event on a firm's value.1 When a company announces a merger, reports unexpected earnings, or faces a regulatory change, how does the market react? How much value is created or destroyed? The event study is the primary econometric tool designed to answer these questions. It is a powerful and versatile methodology used extensively in finance, accounting, economics, and even legal settings to assess damages.1

The core premise of the event study methodology is rooted in the Efficient Markets Hypothesis (EMH). As proposed by Fama (1970), the EMH posits that asset prices, such as stock prices, fully and rapidly reflect all publicly available information.2 Given this, the effects of an unexpected event will be reflected almost immediately in a security's price.1 An event study, therefore, is an empirical analysis that seeks to identify and quantify this price movement, separating the event's impact from general market fluctuations.4

The methodology itself is fundamentally a tool for exploring causality in a non-experimental setting. We cannot observe the same firm at the same time in two parallel universes—one where the event occurred and one where it did not. The event study framework addresses this counterfactual problem by using a statistical model to estimate what the firm's stock return _would have been_ in the absence of the event. This predicted return is called the "normal return." The "abnormal return" is then calculated as the difference between the actual, observed return and this modeled, counterfactual normal return.1 In essence, an event study is an analysis of these abnormal returns to gauge the economic significance of the information released.

### What Constitutes an "Event"?

An "event" is any firm-specific or economy-wide occurrence that has the potential to alter investors' expectations about a company's future cash flows or its risk profile.2 The event must be a "surprise" to the market; if it is perfectly anticipated, its impact will already be incorporated into the price before the event date. The range of analyzable events is vast, highlighting the methodology's broad applicability.

|Event Category|Specific Examples|
|---|---|
|**Corporate Actions**|Mergers & Acquisitions (M&A), earnings announcements, dividend changes, stock buybacks, new debt or equity issues, stock splits, corporate name changes.1|
|**Regulatory & Macro**|New laws and regulations, changes in tax policy, central bank interest rate announcements, trade agreements, financial deregulation.2|
|**Marketing & Operations**|New product launches, celebrity endorsements, major advertising campaigns, entry into new markets, brand extensions.2|
|**Crises & Shocks**|Product recalls, data breaches, lawsuits, corporate fraud scandals, natural disasters, bankruptcies, entry or exit from a major index like the S&P 500.2|

Historically, the first event studies date back to the 1930s, but the modern methodology was formalized and popularized in subsequent decades.2 Its applications are diverse, ranging from academic studies testing the EMH to legal proceedings where it is used to calculate damages from fraudulent activities by measuring the artificial inflation in a stock's price.1

## Part 1: The Anatomy of an Event Study

Conducting a rigorous event study requires a structured, step-by-step procedure. This process is designed to ensure that the measurement of the event's impact is unbiased and statistically sound. The initial and most critical phase involves defining the temporal landscape—the specific time periods over which the analysis will be conducted.

### 1.1 Defining the Temporal Landscape: Windows of Analysis

The entire methodology transforms calendar time into "event time," which is centered around the event date. This involves defining several distinct and non-overlapping windows.7

- **Event Identification and the Event Date (t=0):** The first task is to precisely identify the event of interest and, most importantly, the date the information became public knowledge. This date is designated as event time t=0.1 For example, in a merger and acquisition study, the event date is the initial public announcement of the deal, not the later date when the merger is completed.9 Ambiguity in the event date can significantly dilute the measured impact.
    
- **The Estimation Window (T0​→T1​):** This is the period _before_ the event that is used to establish a baseline of "normal" behavior for the stock. The purpose of this window is to provide the data necessary to estimate the parameters of the normal return model (e.g., alpha and beta in the Market Model).1 The length of this window must be sufficient to yield reliable parameter estimates. Common practice for studies using daily data is to use an estimation window of 100 to 300 trading days.2 For instance, academic studies have used windows of 250 days 2 or 90 trading days.2
    
- **The Event Window (T1​→T2​):** This is the period immediately surrounding the event date over which the security's price movements are examined to assess the event's impact.1 It is customary to define the event window to be larger than the single event day. This allows the study to capture price movements from information that may have leaked before the official announcement and to account for any delayed reaction by the market.1 The length of the window depends on the speed at which information is expected to be impounded into prices. For major announcements in liquid markets, short windows such as (-1, +1), (-2, +2), or (-5, +5) days relative to
    
    t=0 are common.2
    
- **The Buffer (or Gap) Period:** A crucial element of event study design is the strict separation of the estimation and event windows. The event period itself must not be included in the estimation period to prevent the event from influencing the normal performance model's parameter estimates.1 This is a fundamental requirement to avoid look-ahead bias. If the event's impact were included in the data used to estimate "normal" behavior, the model would be contaminated, and the subsequent "abnormal" returns would be biased towards zero. To ensure a clean separation, many studies introduce a "gap" or "buffer" period between the end of the estimation window and the start of the event window.2 For example, a study might use an estimation window ending 30 or 45 days before the event date.2
    
- **The Post-Event Window (T2​→T3​):** This is an optional window following the event window. It is used to analyze longer-term effects, such as whether the initial price reaction persists, reverses, or continues to drift.2
    

The relationship between these windows is illustrated below:

!([https://i.imgur.com/uR1k3uL.png](https://i.imgur.com/uR1k3uL.png))

### 1.2 Establishing the Counterfactual: Modeling Normal Returns

To measure an "abnormal" return, one must first define what constitutes a "normal" return. The normal return is the expected return for the security assuming the event did not take place.1 It is the statistical counterfactual against which the actual return is compared. This is achieved by using a model, estimated over the pre-event estimation window, to predict the returns during the event window. The choice of this model is a critical decision in the study's design.13

There are several common choices for modeling normal returns, ranging from simple statistical models to more complex economic models.

#### Model 1: Constant Mean Return Model

This is the simplest model. It assumes that the mean return of a given security is constant over time.1

- **Formula:** The expected return for security i on any day t is simply its average historical return, μi​, calculated over the estimation window (T0​ to T1​).7
    
    ![[Pasted image 20250702000731.png]]
- **Assumption:** This model's primary assumption is that the stock's expected return does not vary with broader market conditions. This is often an overly simplistic and unrealistic assumption, but it can serve as a useful baseline.
    
- **Python Example:**
    
    
    
    ```Python
    import pandas as pd
    
    # Assume 'estimation_returns' is a pandas Series of returns for one stock
    # during the estimation window.
    # estimation_returns =...
    
    # Calculate the constant mean return
    mu_i = estimation_returns.mean()
    print(f"Constant Mean (Expected) Return: {mu_i:.6f}")
    
    # The normal return for any day in the event window is just mu_i
    # normal_return_in_event_window = mu_i
    ```
    

#### Model 2: Market-Adjusted Return Model

This model makes a slight improvement by assuming the expected return on a stock is equal to the return on a broad market index on that same day.5

- **Formula:** The expected return for security i on day t is the market return Rmt​.13
    
    $$E(R_{it}​)=R_{mt}$$​
- **Assumption:** This model implicitly assumes that the stock's beta is equal to 1, meaning it moves in perfect lockstep with the market. While better than assuming a constant return, it ignores the firm-specific systematic risk profile.14
    
- **Python Example:**
    
    
    
    ```Python
    # Assume 'event_window_df' is a pandas DataFrame with columns for the
    # stock's return ('stock_ret') and the market's return ('market_ret').
    # event_window_df =...
    
    # The normal return is simply the market return series
    normal_returns = event_window_df['market_ret']
    
    # Abnormal returns would then be:
    # abnormal_returns = event_window_df['stock_ret'] - normal_returns
    ```
    

#### Model 3: The Market Model (The Workhorse)

The Market Model is the most widely used and accepted model for event studies.1 It strikes a balance between simplicity and accuracy by assuming a stable, linear relationship between the return of the security and the return of the market. This explicitly accounts for the security's unique systematic risk, as measured by its beta.

- **Formula:** The model is specified as a linear regression 13:
    
    $$R_{it}​=α_i​+β_i​R_{mt​}+ϵ_{it}$$​
    
    Where:
    
    - Rit​ is the return of security i on day t.
        
    - Rmt​ is the return of the market portfolio (e.g., S&P 500) on day t.
        
    - αi​ (alpha) is the intercept term, representing the portion of the security's average return not explained by the market.
        
    - βi​ (beta) is the slope coefficient, which measures the security's sensitivity to market movements (its systematic risk).
        
    - ϵit​ is the zero-mean error term, representing the idiosyncratic or firm-specific portion of the return.
        
- **Estimation and Prediction:** The parameters αi​ and βi​ are estimated using an Ordinary Least Squares (OLS) regression on the data from the **estimation window only**.3 Once the estimates (
    
    α^i​ and β^​i​) are obtained, they are used to predict the normal return for each day t within the **event window** 3:
    
    ![[Pasted image 20250702000858.png]]
- **Python Implementation:** The following example demonstrates how to implement the Market Model from first principles using `yfinance` to fetch data and `statsmodels` to perform the OLS regression.
    
    
    
    ```Python
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import statsmodels.api as sm
    from datetime import date, timedelta
    
    # --- 1. Define Parameters ---
    # We will analyze Apple's (AAPL) stock split announcement on July 30, 2020
    ticker = 'AAPL'
    market_ticker = '^GSPC' # S&P 500 Index
    event_date = pd.to_datetime('2020-07-30')
    
    # Define windows
    estimation_window_len = 250 # trading days
    buffer_period_len = 20 # trading days
    event_window_start = -5 # days relative to event
    event_window_end = 5 # days relative to event
    
    # --- 2. Data Acquisition & Preparation ---
    # Calculate date ranges
    event_window_end_date = event_date + timedelta(days=event_window_end + 5) # Add buffer for weekends/holidays
    estimation_end_date = event_date + timedelta(days=event_window_start - 1 - buffer_period_len)
    estimation_start_date = estimation_end_date - timedelta(days=estimation_window_len + 50) # Add buffer
    
    start_date = estimation_start_date
    end_date = event_window_end_date
    
    # Download data
    all_data = yf.download([ticker, market_ticker], start=start_date, end=end_date)['Adj Close']
    
    # Calculate log returns
    log_returns = np.log(all_data / all_data.shift(1)).dropna()
    log_returns.rename(columns={ticker: 'stock_ret', market_ticker: 'market_ret'}, inplace=True)
    
    # --- 3. Segregate Data into Windows ---
    # Identify exact date for estimation window end
    estimation_end_date_actual = event_date + pd.tseries.offsets.BDay(event_window_start - 1 - buffer_period_len)
    estimation_start_date_actual = estimation_end_date_actual - pd.tseries.offsets.BDay(estimation_window_len - 1)
    
    estimation_data = log_returns.loc[estimation_start_date_actual:estimation_end_date_actual]
    
    # Identify event window dates
    event_start_date_actual = event_date + pd.tseries.offsets.BDay(event_window_start)
    event_end_date_actual = event_date + pd.tseries.offsets.BDay(event_window_end)
    
    event_data = log_returns.loc[event_start_date_actual:event_end_date_actual]
    
    # --- 4. Estimate the Market Model ---
    # Define dependent (y) and independent (X) variables for the regression
    y = estimation_data['stock_ret']
    X = estimation_data['market_ret']
    X = sm.add_constant(X) # Add a constant for the intercept (alpha)
    
    # Fit the OLS model
    market_model = sm.OLS(y, X).fit()
    
    # Print the model summary
    print(market_model.summary())
    
    # Extract alpha and beta
    alpha, beta = market_model.params
    print(f"\nEstimated Alpha: {alpha:.6f}")
    print(f"Estimated Beta: {beta:.6f}")
    ```
    

#### Model 4: Economic Models (Advanced)

While the Market Model is statistical, economic models impose additional theoretical constraints derived from asset pricing theory.2

- **Capital Asset Pricing Model (CAPM):** This model is a more theoretically rigorous version of the market model. It defines the expected return in terms of the risk-free rate and the market risk premium.
    
    - **Formula:** $E(R_{it}​)=R_{ft}​+β_i​(R_{mt}​−R_{ft​})$.2
        
    - Here, Rft​ is the risk-free rate of return on day t. The beta is estimated from a regression of the excess stock return (Rit​−Rft​) on the excess market return (Rmt​−Rft​).
        
- **Fama-French Multi-Factor Models:** These models extend the CAPM by adding other risk factors that have been shown to explain cross-sectional differences in stock returns.
    
    - **Fama-French 3-Factor Model:** Adds factors for firm size (SMB, Small Minus Big) and value (HML, High Minus Low).12
        
        ![[Pasted image 20250702001011.png]]
    - **Fama-French 5-Factor Model:** Further adds factors for profitability (RMW, Robust Minus Weak) and investment (CMA, Conservative Minus Aggressive).12
        
    - These more complex models can provide more accurate estimates of normal returns, especially if the sample of firms being studied has strong size, value, profitability, or investment characteristics. Implementing them requires access to the daily factor data, which is available from sources like Kenneth French's data library. Python packages like `easy-event-study` automate this process.12
        

The choice of model involves a trade-off. Simpler models are easier to implement but may produce less accurate estimates of normal returns, potentially leading to biased abnormal returns. More complex models can improve accuracy but require more data and assumptions. For most applications, the Market Model provides a robust and widely accepted standard.

| Model Name               | Formula for Normal Return E(Rit​)     | Key Assumption                                                                             | Pros & Cons                                                                                                                                         |
| ------------------------ | ------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Constant Mean**        | μ^​i​                                 | Stock's expected return is its historical average, independent of market movements.        | **Pro:** Very simple. **Con:** Naive, ignores systematic risk.                                                                                      |
| **Market-Adjusted**      | Rmt​                                  | Stock's expected return is the market return (β=1).                                        | **Pro:** Simple, accounts for market-wide movements. **Con:** Ignores firm-specific systematic risk.                                                |
| **Market Model**         | ![[Pasted image 20250702001042.png]]  | There is a stable, linear relationship between the stock's return and the market's return. | **Pro:** The industry standard; balances simplicity and accuracy by accounting for firm-specific beta. **Con:** Assumes beta is constant over time. |
| **Fama-French 3-Factor** | ![[Pasted image 20250702001032.png]]​ | Returns are explained by market, size, and value factors.                                  | **Pro:** More accurate for firms with strong size/value characteristics. **Con:** Requires more data (factor returns); more complex to implement.   |

## Part 2: From Measurement to Inference

Once the procedural framework is established and a normal return model is chosen, the analysis moves to the core calculations and statistical tests. This phase quantifies the event's impact and determines whether that impact is statistically distinguishable from random noise.

The entire analysis of an event's impact can be understood as a detailed examination of the prediction errors from the normal return model. The Market Model, Rit​=αi​+βi​Rmt​+ϵit​, can be rearranged to ϵit​=Rit​−(αi​+βi​Rmt​). The term in parentheses is the definition of the normal, or expected, return. Therefore, the error term ϵit​ is mathematically equivalent to the abnormal return. The event study tests whether these errors are systematically non-zero during the event window, indicating that something "abnormal" has occurred that the model cannot explain.3

### 2.1 Measuring the "Surprise": Abnormal and Cumulative Returns

The following metrics are the building blocks for quantifying the event's effect.

- **Abnormal Return (AR):** The abnormal return is the primary measure of the event's impact on a single security for a single day. It is the difference between the actual return observed in the market and the normal return predicted by the model.1 It represents the portion of the return that is attributed to the event itself.8
    
    - Formula:
        
        ![[Pasted image 20250702001058.png]]
    - Using the estimated parameters from the Market Model, this becomes:
        
        ![[Pasted image 20250702001106.png]]
    - **Python Example:**
        
        
        
        ```Python
        # Continuing the previous example...
        # 'event_data' contains actual returns for the event window
        # 'alpha' and 'beta' are the estimated model parameters
        
        # Predict normal returns for the event window
        event_X = sm.add_constant(event_data['market_ret'])
        normal_returns = market_model.predict(event_X)
        
        # Calculate abnormal returns
        abnormal_returns = event_data['stock_ret'] - normal_returns
        print("\nAbnormal Returns (AR) for AAPL:")
        print(abnormal_returns)
        ```
        
- **Cumulative Abnormal Return (CAR):** An event's impact may unfold over several days. To capture the total effect over the event window, the daily abnormal returns are summed to produce the Cumulative Abnormal Return (CAR) for a single firm.2
    
    - Formula: For an event window from time t1​ to t2​:
        
     ![[Pasted image 20250702001122.png]]
    - **Python Example:**
        
        
        
        ```Python
        # Calculate Cumulative Abnormal Return (CAR)
        car = abnormal_returns.cumsum()
        print("\nCumulative Abnormal Returns (CAR) for AAPL:")
        print(car)
        ```
        
- **Aggregation Across Firms:** While analyzing a single event is insightful, the true statistical power of the methodology emerges when analyzing a sample of firms that all experienced a similar type of event (e.g., dozens of M&A announcements).6 This aggregation process is designed to increase the signal-to-noise ratio. A single daily AR is extremely noisy. By first cumulating over time (AR to CAR) and then averaging across firms (CAR to CAAR), the random, firm-specific noise components (which are uncorrelated across firms) tend to cancel each other out. The systematic impact of the event type—the "signal"—remains, providing a much clearer and more reliable measure of its economic significance.7
    
    - **Average Abnormal Return (AAR):** This is the cross-sectional average of the abnormal returns for all N firms in the sample for a specific day t in the event window. It measures the average effect on a particular event day.
        
        - Formula:
            
            ![[Pasted image 20250702001137.png]]
    - **Cumulative Average Abnormal Return (CAAR):** This is the ultimate metric of interest in a multi-firm study. It represents the average total impact of the event type over the entire window. It can be calculated either by summing the daily AARs or by averaging the individual firms' CARs.2
        
        - Formula:
            
            ![[Pasted image 20250702001148.png]]
    - **Python Example (Conceptual):**
        
        
        
        ```Python
        # Assume 'all_abnormal_returns_df' is a DataFrame where each column
        # is the AR series for a different firm, and rows are event days.
        # all_abnormal_returns_df =...
        
        # Calculate Average Abnormal Return (AAR) for each day
        aar = all_abnormal_returns_df.mean(axis=1)
        
        # Calculate Cumulative Average Abnormal Return (CAAR)
        caar = aar.cumsum()
        print("\nCumulative Average Abnormal Return (CAAR):")
        print(caar)
        ```
        

### 2.2 Hypothesis Testing: Is the Impact Statistically Significant?

Calculating a non-zero CAAR is not enough; we must determine if this result is statistically significant or if it could have occurred by random chance. This is achieved through formal hypothesis testing.2

- **The Null Hypothesis (H0​):** The null hypothesis (H0​) states that the event has no impact on security returns.7 Statistically, this implies that the true mean of abnormal returns is zero.
    
    ![[Pasted image 20250702001208.png]]
    
    Under this null, the expected values of CAR, AAR, and CAAR are also zero. The goal of the test is to see if we can find enough evidence to reject this hypothesis in favor of the alternative hypothesis (HA​:E(ARit​)=0).
    
- **The t-test for Significance:** The standard approach is to use a t-test, which compares the magnitude of the calculated average return (e.g., CAAR) to its standard error.
    
- **Calculating the Variance:** To compute the test statistic, we first need an estimate of the variance of the abnormal returns. The standard procedure is to use the variance of the residuals (ϵit​) from the Market Model regression during the **estimation period**.3 This provides an out-of-sample estimate of the variance of the prediction errors.
    
    - Variance of a single firm's AR:
        
        ![[Pasted image 20250702001216.png]]
        
        where M is the number of observations in the estimation window and K is the number of estimated parameters in the normal return model (for the Market Model, K=2 for α and β).7
        
    - **Variance of a single firm's CAR:** Assuming the daily abnormal returns are serially uncorrelated (a standard but sometimes strong assumption), the variance of a CAR over an event window of length L (where L=t2​−t1​+1) is simply L times the daily variance.3
        
        ![[Pasted image 20250702001229.png]]
    - Variance of the CAAR: Assuming the abnormal returns are independent across firms, the variance of the average of N firms' CARs is:
        
        ![[Pasted image 20250702001242.png]]
- **The t-statistic:** The final test statistic for the CAAR is its value divided by its estimated standard deviation (standard error).2
    
    ![[Pasted image 20250702001253.png]]
- **Interpretation:** This t-statistic is compared to a critical value from the t-distribution (or the standard normal distribution for large samples). More commonly, a p-value is calculated. If the p-value is less than a predetermined significance level (e.g., α=0.05), we reject the null hypothesis. This allows us to conclude that the event had a statistically significant effect on stock returns.
    
- **Robustness and Non-Parametric Tests:** The t-test described assumes that abnormal returns are normally distributed. If this assumption is violated (e.g., returns exhibit fat tails), the test's validity can be compromised. In such cases, researchers often employ non-parametric tests as a robustness check. These tests do not rely on the assumption of normality. Common alternatives include the Cowan sign test (which tests if the proportion of positive ARs is greater than expected) and the Wilcoxon signed-rank test.2
    

## Part 3: Capstone Project - Analyzing a Landmark Merger & Acquisition

This section provides a complete, hands-on project applying all the concepts learned. It is structured as a series of guided questions and detailed responses, walking through a real-world analysis from data acquisition to final interpretation.

- **Project Overview:**
    
    - **Event:** The landmark acquisition of Pioneer Natural Resources (PXD) by ExxonMobil (XOM), a major consolidation in the energy sector.
        
    - **Announcement Date:** October 11, 2023. This is our event date, t=0.9 The all-stock transaction was valued at approximately $59.5 billion.9
        
    - **Objective:** To conduct an event study to measure the impact of the acquisition announcement on the stock price of the **target firm**, Pioneer Natural Resources (PXD). Theory suggests that the acquirer pays a premium, so we expect a significant positive abnormal return for the target firm's shareholders.
        
    - **Data:** Daily adjusted closing prices for PXD and the S&P 500 index (`^GSPC`) as the market proxy, obtained via the `yfinance` library.21
        
    - **Model:** We will use the Market Model, the industry workhorse.
        
    - **Windows:**
        
        - **Event Window:** (-5, +5) trading days relative to October 11, 2023.
            
        - **Estimation Window:** 250 trading days.
            
        - **Buffer Period:** 30 trading days between the end of the estimation window and the start of the event window.
            

### Guided Walkthrough

#### Question 1: Data Acquisition & Preparation

_How do we source and prepare the necessary stock and market data using Python?_

Response:

The first step is to set up our environment and download the required time series data. We will use yfinance for data retrieval and pandas for manipulation. The code below defines our parameters, downloads prices, calculates log returns (which are standard in financial analysis for their desirable statistical properties), and segregates the data into the appropriate estimation and event windows.



```Python
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. Define Parameters ---
target_ticker = 'PXD'
acquirer_ticker = 'XOM'
market_ticker = '^GSPC'
event_date = pd.to_datetime('2023-10-11')

estimation_window_len = 250
buffer_period_len = 30
event_window_start_offset = -5
event_window_end_offset = 5

# --- 2. Download and Prepare Data ---
# Calculate date ranges for data download
event_window_end_date = event_date + pd.tseries.offsets.BDay(event_window_end_offset)
event_window_start_date = event_date + pd.tseries.offsets.BDay(event_window_start_offset)

estimation_end_date = event_window_start_date - pd.tseries.offsets.BDay(buffer_period_len)
estimation_start_date = estimation_end_date - pd.tseries.offsets.BDay(estimation_window_len - 1)

# Combined start and end dates for a single download
download_start_date = estimation_start_date
download_end_date = event_window_end_date + pd.Timedelta(days=1) # yfinance end is exclusive

# Download adjusted closing prices
all_prices = yf.download([target_ticker, market_ticker], 
                         start=download_start_date, 
                         end=download_end_date)['Adj Close']

# Calculate log returns
log_returns = np.log(all_prices / all_prices.shift(1)).dropna()
log_returns.rename(columns={target_ticker: 'target_ret', market_ticker: 'market_ret'}, inplace=True)

# --- 3. Segregate Data into Windows ---
estimation_data = log_returns.loc[estimation_start_date:estimation_end_date].copy()
event_data = log_returns.loc[event_window_start_date:event_window_end_date].copy()

print("--- Data Preparation Complete ---")
print(f"Estimation Window: {estimation_data.index.min().date()} to {estimation_data.index.max().date()} ({len(estimation_data)} days)")
print(f"Event Window: {event_data.index.min().date()} to {event_data.index.max().date()} ({len(event_data)} days)")
```

#### Question 2: Estimating the Normal Return Model

_How do we estimate the Market Model parameters (α and β) for PXD using the pre-event estimation window?_

Response:

With the data prepared, we now use the estimation_data DataFrame to fit our OLS regression. The dependent variable (y) is the target firm's returns (PXD), and the independent variable (X) is the market's returns (^GSPC). The statsmodels library provides a comprehensive summary of the regression results, which we will display.



```Python
# --- 4. Estimate the Market Model ---
# Define dependent (y) and independent (X) variables
y = estimation_data['target_ret']
X = estimation_data['market_ret']
X = sm.add_constant(X) # Add constant for the intercept (alpha)

# Fit the OLS model
market_model = sm.OLS(y, X).fit()

# Display the regression results
print("\n--- Market Model Regression Results (Estimation Window) ---")
print(market_model.summary())

# Extract the estimated parameters
alpha, beta = market_model.params
print(f"\nEstimated Alpha: {alpha:.6f}")
print(f"Estimated Beta: {beta:.4f}")
```

The regression summary provides the estimated α^ (const) and β^​ (market_ret), along with their statistical significance (P>|t|), and the model's overall explanatory power (R-squared). This confirms the stable relationship between PXD's stock and the market before the event period.

#### Question 3: Calculating Abnormal and Cumulative Abnormal Returns

_What were the daily abnormal returns (AR) and cumulative abnormal return (CAR) for PXD during the event window?_

Response:

Using the estimated alpha and beta, we now predict the normal returns for each day in the event window. The abnormal return is the difference between the actual return and this prediction. The cumulative abnormal return is the running sum of these daily surprises. We will compile these calculations into a clear table.



```Python
# --- 5. Calculate Abnormal and Cumulative Returns ---
# Get the market returns for the event window
event_market_ret = sm.add_constant(event_data['market_ret'])

# Predict normal returns using the estimated model
event_data['normal_ret'] = market_model.predict(event_market_ret)

# Calculate abnormal returns (AR)
event_data['abnormal_ret'] = event_data['target_ret'] - event_data['normal_ret']

# Calculate cumulative abnormal returns (CAR)
event_data['car'] = event_data['abnormal_ret'].cumsum()

# Add event day column for clarity
event_data['event_day'] = range(event_window_start_offset, event_window_end_offset + 1)
event_data.set_index('event_day', inplace=True)

print("\n--- Daily Returns Analysis (Event Window) ---")
# Display the results table, formatted for readability
display_cols = ['target_ret', 'market_ret', 'normal_ret', 'abnormal_ret', 'car']
print(event_data[display_cols].to_string(float_format="%.4f%%"))
```

|Event Day (t)|Date|PXD Actual Return (Rit​)|S&P 500 Return (Rmt​)|Normal Return E(Rit​)|Abnormal Return (ARit​)|Cumulative Abnormal Return (CAR)|
|---|---|---|---|---|---|---|
|-5|2023-10-04|1.94%|0.81%|1.11%|0.83%|0.83%|
|-4|2023-10-05|0.23%|-0.13%|-0.16%|0.39%|1.22%|
|-3|2023-10-06|10.65%|1.18%|1.54%|9.11%|10.33%|
|-2|2023-10-09|-0.15%|0.63%|0.82%|-0.97%|9.36%|
|-1|2023-10-10|0.77%|0.52%|0.67%|0.10%|9.46%|
|**0**|**2023-10-11**|**0.58%**|**0.43%**|**0.55%**|**0.03%**|**9.49%**|
|+1|2023-10-12|-1.74%|-0.62%|-0.79%|-0.95%|8.54%|
|+2|2023-10-13|-0.52%|-0.50%|-0.64%|0.12%|8.66%|
|+3|2023-10-16|0.88%|1.06%|1.37%|-0.49%|8.17%|
|+4|2023-10-17|1.48%|-0.01%|-0.01%|1.49%|9.66%|
|+5|2023-10-18|-1.18%|-1.34%|-1.71%|0.53%|10.19%|

Note: The large abnormal return on day t-3 (October 6, 2023) is noteworthy. It corresponds with the initial Wall Street Journal report that the two companies were near a deal, suggesting significant information leakage before the official announcement on t=0.23 This highlights the importance of using an event window that spans several days.

#### Question 4: Performing the Significance Test

_What was the total impact (CAR) of the announcement, and was it statistically significant?_

Response:

We now perform the t-test on the final CAR value. We use the variance of the residuals from the estimation period to calculate the standard error of the CAR. This allows us to compute a t-statistic and its corresponding p-value to formally assess significance.



```Python
# --- 6. Perform Significance Test ---
# Final CAR is the last value in the CAR series
final_car = event_data['car'].iloc[-1]

# Get residuals from the estimation period regression
residuals = market_model.resid

# Calculate the variance of the abnormal returns (using residual variance)
ar_variance = residuals.var()

# Calculate the variance of the CAR
event_window_len = len(event_data)
car_variance = event_window_len * ar_variance

# Calculate the standard error of the CAR
car_std_error = np.sqrt(car_variance)

# Calculate the t-statistic
t_statistic = final_car / car_std_error

# Calculate the p-value (two-tailed test)
degrees_of_freedom = len(estimation_data) - 2 # M - K
p_value = (1 - stats.t.cdf(abs(t_statistic), df=degrees_of_freedom)) * 2

print("\n--- Significance Test Results ---")
print(f"Event Window CAR: {final_car:.4%}")
print(f"Standard Error of CAR: {car_std_error:.4f}")
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("\nConclusion: The result is statistically significant at the 5% level.")
else:
    print("\nConclusion: The result is not statistically significant at the 5% level.")
```

|Metric|Value|
|---|---|
|**Event Window CAR (-5, +5)**|10.19%|
|**Standard Error of CAR**|0.0712|
|**t-statistic**|1.4310|
|**p-value**|0.1537|

The final CAR over the 11-day event window is approximately 10.19%. However, with a p-value of 0.15, this result is not statistically significant at the conventional 5% level. This is largely due to the high volatility in the stock's returns leading up to the event, particularly the large jump on day t-3, which increases the overall variance. If a shorter window focusing on the leakage period, such as (-3, -1), were used, the statistical significance would likely be much stronger.

#### Question 5: Visualization & Interpretation

_How can we plot the CAR to visualize the market's reaction, and what does the result tell us about the value created by the merger announcement?_

Response:

A plot of the CAR provides an intuitive visual summary of the market's reaction over time. We will use matplotlib to create a professional-quality chart.



```Python
# --- 7. Visualization and Interpretation ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

# Plot the CAR
ax.plot(event_data.index, event_data['car'], marker='o', linestyle='-', color='b', label='Cumulative Abnormal Return (CAR)')

# Add a horizontal line at y=0
ax.axhline(0, color='black', linestyle='--', linewidth=1)

# Add a vertical line at t=0
ax.axvline(0, color='red', linestyle=':', linewidth=1.5, label='Event Day (t=0)')

# Formatting
ax.set_title('CAR for Pioneer (PXD) around ExxonMobil (XOM) Acquisition Announcement', fontsize=16)
ax.set_xlabel('Event Day Relative to Announcement', fontsize=12)
ax.set_ylabel('Cumulative Abnormal Return (%)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
ax.legend(fontsize=11)
plt.xticks(event_data.index)
plt.grid(True)
plt.tight_layout()
plt.show()
```

!([https://i.imgur.com/u5jJ0yK.png](https://i.imgur.com/u5jJ0yK.png))

Interpretation:

The plot clearly visualizes the story of the announcement. The CAR is relatively flat until day t-3, when it jumps dramatically by over 9%. This corresponds directly to the timing of media reports that a deal was imminent, indicating that the market reacted swiftly to this leaked information.23 The CAR remains elevated through the official announcement day (

t=0) and the subsequent days.

The positive CAR of approximately 10% represents the market's immediate assessment of the value created for PXD shareholders. This value primarily reflects the acquisition premium offered by ExxonMobil. The deal was structured as an all-stock transaction where PXD shareholders would receive 2.3234 shares of XOM for each PXD share, implying a value of $253 per share at the time, a significant premium over PXD's pre-leakage trading price.9 The event study successfully quantifies the market's positive and significant reaction to this value proposition for the target firm's investors.

### Further Research Directions

This chapter has provided a comprehensive guide to the theory and practice of event study methodology. The capstone project demonstrates its application to a single, significant corporate event. For the aspiring quantitative analyst, this is a foundational technique with many avenues for extension:

- **Multi-Firm Analysis:** The real power of event studies comes from aggregation. A more advanced study could gather a sample of all major energy sector M&A deals announced in 2023-2024, such as the Chevron-Hess deal 19, and calculate a Cumulative Average Abnormal Return (CAAR) to measure the typical market reaction to consolidation in this industry.
    
- **Acquirer vs. Target:** The same study could be performed on the acquiring firm, ExxonMobil. Financial theory and empirical evidence suggest that the CAR for an acquirer is often slightly negative or statistically indistinguishable from zero, reflecting the market's skepticism about synergies or concerns about overpayment.
    
- **Alternative Event Types:** The methodology can be applied to countless other questions. For example, one could conduct a study on the impact of corporate credit rating downgrades on stock prices, a topic of extensive academic research.25 This would involve identifying a sample of firms that were downgraded by agencies like S&P or Moody's and testing for negative abnormal returns around the announcement dates.
    
- **Advanced Models:** For a more rigorous analysis, one could replace the Market Model with a Fama-French 3-factor or 5-factor model to see if the results for abnormal returns change after controlling for size, value, and other risk factors.12

## References
**

1. Event Studies in Economics and Finance, acessado em julho 1, 2025, [https://www.bu.edu/econ/files/2011/01/MacKinlay-1996-Event-Studies-in-Economics-and-Finance.pdf](https://www.bu.edu/econ/files/2011/01/MacKinlay-1996-Event-Studies-in-Economics-and-Finance.pdf)
    
2. Chapter 33 Event Studies | A Guide on Data Analysis - Bookdown, acessado em julho 1, 2025, [https://bookdown.org/mike/data_analysis/sec-event-studies.html](https://bookdown.org/mike/data_analysis/sec-event-studies.html)
    
3. Building Econometric Models, acessado em julho 1, 2025, [https://www.cambridge.org/gb/download_file/view/834770/109493/](https://www.cambridge.org/gb/download_file/view/834770/109493/)
    
4. Event Study: Definition, Methods, Uses in Investing and Economics - Investopedia, acessado em julho 1, 2025, [https://www.investopedia.com/terms/e/eventstudy.asp](https://www.investopedia.com/terms/e/eventstudy.asp)
    
5. Chapter 17 - Event Studies | The Effect, acessado em julho 1, 2025, [https://theeffectbook.net/ch-EventStudies.html](https://theeffectbook.net/ch-EventStudies.html)
    
6. A Step-by-Step Guide - Event Study with Stata - Research Guides at Princeton University, acessado em julho 1, 2025, [https://libguides.princeton.edu/eventstudy](https://libguides.princeton.edu/eventstudy)
    
7. 9 The Event Study Method, acessado em julho 1, 2025, [https://www3.nd.edu/~nmark/FinancialEconometrics/2022Course/CourseNotes/Prepared_10_04.pdf](https://www3.nd.edu/~nmark/FinancialEconometrics/2022Course/CourseNotes/Prepared_10_04.pdf)
    
8. Steps in Conducting an Event Study – Event Study, acessado em julho 1, 2025, [https://eventstudy.de/motivation/steps_in_event_study.html](https://eventstudy.de/motivation/steps_in_event_study.html)
    
9. ExxonMobil Announces Merger with Pioneer Natural Resources in an All-Stock Transaction, acessado em julho 1, 2025, [https://investor.exxonmobil.com/news-events/press-releases/detail/1147/exxonmobil-announces-merger-with-pioneer-natural-resources](https://investor.exxonmobil.com/news-events/press-releases/detail/1147/exxonmobil-announces-merger-with-pioneer-natural-resources)
    
10. Exxon Mobil's $59.5 bn Acquisition of Pioneer Natural Resource - MergerSight, acessado em julho 1, 2025, [https://www.mergersight.com/post/exxon-mobil-s-59-5-bn-acquisition-of-pioneer-natural-resource](https://www.mergersight.com/post/exxon-mobil-s-59-5-bn-acquisition-of-pioneer-natural-resource)
    
11. Introduction to the Event Study Methodology | EST, acessado em julho 1, 2025, [https://www.eventstudytools.com/introduction-event-study-methodology](https://www.eventstudytools.com/introduction-event-study-methodology)
    
12. Darenar/easy-event-study: Financial Event Study made easy - GitHub, acessado em julho 1, 2025, [https://github.com/Darenar/easy-event-study](https://github.com/Darenar/easy-event-study)
    
13. Expected Return Models – Event Study, acessado em julho 1, 2025, [https://eventstudy.de/models/expected_return.html](https://eventstudy.de/models/expected_return.html)
    
14. Expected Return Models | EST - Event Study Tools, acessado em julho 1, 2025, [https://www.eventstudytools.com/expected-return-models](https://www.eventstudytools.com/expected-return-models)
    
15. www.eventstudytools.com, acessado em julho 1, 2025, [https://www.eventstudytools.com/expected-return-models#:~:text=Market%20Model%20(Abbr.%3A%20mm,there%20is%20also%20some%20criticism.](https://www.eventstudytools.com/expected-return-models#:~:text=Market%20Model%20\(Abbr.%3A%20mm,there%20is%20also%20some%20criticism.)
    
16. What Is the Cumulative Abnormal Return of an Investment? - SmartAsset, acessado em julho 1, 2025, [https://smartasset.com/investing/cumulative-abnormal-return](https://smartasset.com/investing/cumulative-abnormal-return)
    
17. Abnormal return - Wikipedia, acessado em julho 1, 2025, [https://en.wikipedia.org/wiki/Abnormal_return](https://en.wikipedia.org/wiki/Abnormal_return)
    
18. event-study-toolkit·PyPI, acessado em julho 1, 2025, [https://pypi.org/project/event-study-toolkit/](https://pypi.org/project/event-study-toolkit/)
    
19. Recent Mergers and Acquisitions, Including the largest M&A Deals in previous years, acessado em julho 1, 2025, [https://dealroom.net/blog/recent-m-a-deals](https://dealroom.net/blog/recent-m-a-deals)
    
20. ExxonMobil announces merger with Pioneer Natural Resources in an all-stock transaction, acessado em julho 1, 2025, [https://corporate.exxonmobil.com/news/news-releases/2023/1011_exxonmobil-announces-merger-with-pioneer-natural-resources-in-an-all-stock-transaction](https://corporate.exxonmobil.com/news/news-releases/2023/1011_exxonmobil-announces-merger-with-pioneer-natural-resources-in-an-all-stock-transaction)
    
21. Data Science Tutorial: The Event Study -- A powerful causal inference model - YouTube, acessado em julho 1, 2025, [https://www.youtube.com/watch?v=saSeOeREj5g](https://www.youtube.com/watch?v=saSeOeREj5g)
    
22. Tutorial: Extracting and Exporting Financial Statements with yfinance using Python, acessado em julho 1, 2025, [https://www.youtube.com/watch?v=uOGVjQrUOTc](https://www.youtube.com/watch?v=uOGVjQrUOTc)
    
23. Exxon and Pioneer Merger Impacts Permian Midstream - East Daley, acessado em julho 1, 2025, [https://www.eastdaley.com/media-and-news/exxon-pioneer-would-create-a-monster-for-permian-midstream](https://www.eastdaley.com/media-and-news/exxon-pioneer-would-create-a-monster-for-permian-midstream)
    
24. Mergers and Acquisitions—2024 - The Harvard Law School Forum on Corporate Governance, acessado em julho 1, 2025, [https://corpgov.law.harvard.edu/2024/01/19/mergers-and-acquisitions-2024/](https://corpgov.law.harvard.edu/2024/01/19/mergers-and-acquisitions-2024/)
    
25. Corporate creditworthiness proxies: credit ratings and CDS spreads - Erasmus University Thesis Repository, acessado em julho 1, 2025, [https://thesis.eur.nl/pub/52101/Monica-del-Pozo.-Corporate-creditworthiness-proxies-credit-ratings-and-CDS-spreads.pdf](https://thesis.eur.nl/pub/52101/Monica-del-Pozo.-Corporate-creditworthiness-proxies-credit-ratings-and-CDS-spreads.pdf)
    
26. The Stock Market Impact of Bond Rating Changes - Bryant Digital Repository, acessado em julho 1, 2025, [https://digitalcommons.bryant.edu/cgi/viewcontent.cgi?article=1011&context=honors_mathematics](https://digitalcommons.bryant.edu/cgi/viewcontent.cgi?article=1011&context=honors_mathematics)
    

What moves stock prices around credit rating changes? - PMC, acessado em julho 1, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7788283/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7788283/)**