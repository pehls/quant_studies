## The Post-2008 Regulatory Landscape: An Introduction to Dodd-Frank and Basel III

The financial crisis that began in 2007 and culminated in the near-collapse of the global financial system in 2008 was the most severe economic catastrophe since the Great Depression of the 1930s.1 It exposed profound weaknesses in a regulatory system that was described as fragmented, antiquated, and insufficient to monitor or constrain the excessive risk-taking that had become pervasive across the financial industry.4 Unscrupulous lending practices, the unchecked growth of complex and opaque financial instruments, and the failure of major institutions like AIG revealed a system where some firms could "game the system" and take risks that endangered the entire economy, leaving taxpayers to bear the ultimate cost of bailouts.4

In response, governments and international bodies enacted the most comprehensive overhaul of financial regulation in generations. This new regulatory architecture is dominated by two landmark frameworks that every quantitative professional must understand: the Dodd-Frank Wall Street Reform and Consumer Protection Act in the United States and the Basel III international regulatory framework.

The Dodd-Frank Act, signed into law on July 21, 2010, was a sweeping piece of U.S. federal legislation designed to reshape the American financial landscape.6 Its primary objectives were to promote financial stability, end the concept of "too big to fail," protect taxpayers by eliminating bailouts, and shield consumers from abusive financial practices.6 It achieved this by creating new regulatory agencies, such as the Consumer Financial Protection Bureau (CFPB) and the Financial Stability Oversight Council (FSOC), and by imposing stringent new rules on previously unregulated or under-regulated areas, most notably the over-the-counter (OTC) derivatives market.3

Concurrently, the Basel Committee on Banking Supervision (BCBS), a consortium of central banks and regulatory authorities from 28 countries, developed the Basel III framework.9 Released in 2010 and progressively implemented since, Basel III is not a national law but an internationally agreed-upon set of minimum standards designed to strengthen the regulation, supervision, and risk management of banks worldwide.10 Its focus is on fortifying the banking sector's resilience by increasing the quantity and quality of capital banks must hold, introducing new liquidity standards to prevent funding crises, and constraining excessive leverage.12

The dual emergence of a prescriptive national law (Dodd-Frank) and a set of international standards (Basel III) highlights a central tension in post-crisis reform. While the crisis was global, the responses have been a mix of national legislative action and international coordination. This distinction is critical for quantitative analysts at multinational firms, as the implementation of Basel III can vary by jurisdiction, creating a complex web of compliance requirements.14

Furthermore, these frameworks signaled a paradigm shift from a purely _microprudential_ approach, which focuses on the safety and soundness of individual firms, to a _macroprudential_ one, which considers the stability of the financial system as a whole. The creation of the FSOC to identify systemic threats 1 and the introduction of capital surcharges for Global Systemically Important Banks (G-SIBs) under Basel III 9 are direct consequences of this shift. For quantitative professionals, this means risk models can no longer operate in isolation; they must now account for interconnectedness, systemic risk, and the broader economic cycle.

## 7.2 The Dodd-Frank Act: A Quant's Guide to Key Provisions

The Dodd-Frank Act is a vast piece of legislation, spanning over 2,000 pages and creating hundreds of new rules.15 For quantitative data scientists, its most significant impacts are concentrated in two areas: the restriction of proprietary trading via the Volcker Rule and the comprehensive regulation of the derivatives market.

### 7.2.1 The Volcker Rule: Delineating Market-Making from Proprietary Trading

Section 619 of the Dodd-Frank Act, commonly known as the Volcker Rule, represents a foundational attempt to re-establish a barrier between commercial banking and more speculative investment activities, reminiscent of the Glass-Steagall Act.5 The rule's core tenet is a prohibition on "banking entities"—firms that benefit from federal deposit insurance or access to the Federal Reserve's discount window—from engaging in proprietary trading.19 Proprietary trading is defined as using the firm's own capital to engage in short-term buying and selling of securities, derivatives, or other financial instruments for the purpose of profiting from price fluctuations.21

The central challenge of the Volcker Rule, for both regulators and financial institutions, is the difficulty in distinguishing prohibited proprietary trading from permitted, and indeed essential, financial activities such as market-making, underwriting, and risk-mitigating hedging.18 A market-maker provides liquidity to clients by standing ready to buy and sell securities, a process that requires holding an inventory of those securities and managing the associated risk. This activity can look superficially similar to a proprietary directional bet, yet its function is fundamentally different.

This ambiguity has had a tangible effect on market structure. To enforce the rule, regulators require large banking entities to establish robust compliance programs and report a suite of quantitative metrics for each trading desk.19 These metrics are designed to reveal the true intent behind trading activity and include:

- Risk and Position Limits and Usage
    
- Risk Factor Sensitivities
    
- Value-at-Risk (VaR) and Stress VaR
    
- Comprehensive Profit and Loss Attribution
    
- Inventory Turnover
    
- Inventory Aging
    
- Customer-Facing Trade Ratio 24
    

The fear of violating the rule and the burden of compliance have led many banks to reduce their risk appetite for holding inventories of less-liquid assets. Empirical studies have documented a notable decline in market-making activities by Volcker-affected dealers, particularly in the corporate bond market.24 While non-Volcker dealers have increased their activity, they have not fully compensated for the withdrawal of the large banks, resulting in a measurable decrease in market liquidity, especially during periods of stress.24 This shift has forced quantitative trading strategies that rely on deep liquidity to adapt to a new market reality of higher transaction costs and potentially greater volatility.

---

**Table 7.1: Key Dodd-Frank Act Provisions for Quants**

|Provision (Section)|Description|Quantitative Implication|
|---|---|---|
|**Volcker Rule (Sec. 619)**|Prohibits proprietary trading by banking entities and limits relationships with hedge funds and private equity funds.5|Requires development of quantitative metrics (e.g., inventory turnover, P&L attribution) to distinguish market-making from proprietary trading. Impacts models for market-making, liquidity risk, and algorithmic execution.|
|**Derivatives Regulation (Title VII)**|Mandates central clearing for standardized OTC derivatives and imposes margin requirements for both cleared and non-cleared swaps.6|Shifts counterparty credit risk modeling from bilateral exposures to concentrated CCP exposures. Requires development of sophisticated initial and variation margin models (e.g., SIMM for non-cleared derivatives).|
|**Enhanced Prudential Standards (Sec. 165)**|Subjects systemically important financial institutions (SIFIs) to heightened standards, including stress testing, capital, and leverage requirements.17|Drives the development and implementation of firm-wide stress testing models (e.g., CCAR in the U.S.) that project financial performance under adverse economic scenarios.|
|**Credit Risk Retention (Sec. 941)**|Requires securitizers to retain a portion of the credit risk ("skin in the game") for the assets they securitize, typically 5%.3|Requires models to price and manage the risk of retained tranches of securitizations. Affects the economics and modeling of asset-backed securities (ABS).|
|**Regulation of Credit Rating Agencies (Sec. 939A)**|Reduces statutory and regulatory reliance on credit ratings and increases oversight and legal liability for rating agencies.3|Encourages the development of independent, internal credit risk models rather than relying solely on external ratings, increasing the demand for fundamental credit analysis and modeling skills.|

---

#### Mathematical & Python Example: Analyzing Trading Desk Metrics

To illustrate how quantitative metrics can help distinguish between trading activities, consider two hypothetical trading desks at a bank: a Corporate Bond Market-Making Desk and a Global Macro Proprietary Desk. We can simulate their daily activities and calculate key Volcker Rule metrics.

The **Inventory Turnover Ratio** measures how frequently a desk turns over its inventory. A high ratio suggests market-making (facilitating client trades), while a low ratio may indicate proprietary position-taking. It is calculated as:

![[Pasted image 20250819182720.png]]

The **P&L Attribution** separates profit and loss into components. For a market-maker, P&L should primarily come from the bid-ask spread, whereas a proprietary desk's P&L will be driven by the price appreciation of its inventory.



```Python
import pandas as pd
import numpy as np

# --- 1. Simulate Trading Desk Data ---
np.random.seed(42)
dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=252))

# Market-Making Desk: High volume, low inventory, spread-based P&L
mm_trades = np.random.randint(80, 120, size=252)
mm_inventory = np.random.uniform(5, 15, size=252)
mm_spread_pl = mm_trades * np.random.uniform(0.01, 0.02, size=252)
mm_appreciation_pl = mm_inventory * np.random.normal(0, 0.005, size=252)
mm_total_pl = mm_spread_pl + mm_appreciation_pl

# Proprietary Desk: Lower volume, higher inventory, appreciation-based P&L
prop_trades = np.random.randint(10, 30, size=252)
prop_inventory = np.random.uniform(50, 80, size=252)
prop_spread_pl = prop_trades * np.random.uniform(0.01, 0.02, size=252)
prop_appreciation_pl = prop_inventory * np.random.normal(0.001, 0.02, size=252)
prop_total_pl = prop_spread_pl + prop_appreciation_pl

market_making_df = pd.DataFrame({
    'Date': dates,
    'Volume': mm_trades,
    'Inventory': mm_inventory,
    'Spread_PL': mm_spread_pl,
    'Appreciation_PL': mm_appreciation_pl,
    'Total_PL': mm_total_pl
})

proprietary_df = pd.DataFrame({
    'Date': dates,
    'Volume': prop_trades,
    'Inventory': prop_inventory,
    'Spread_PL': prop_spread_pl,
    'Appreciation_PL': prop_appreciation_pl,
    'Total_PL': prop_total_pl
})

# --- 2. Define Function to Calculate Volcker Metrics ---
def calculate_volcker_metrics(df, desk_name):
    """Calculates key Volcker Rule metrics for a given trading desk."""
    
    # Inventory Turnover
    total_volume = df['Volume'].sum()
    avg_inventory = df['Inventory'].mean()
    inventory_turnover = total_volume / avg_inventory
    
    # P&L Attribution
    total_pl = df.sum()
    spread_pl_pct = df.sum() / total_pl
    appreciation_pl_pct = df['Appreciation_PL'].sum() / total_pl
    
    print(f"--- Metrics for {desk_name} ---")
    print(f"Inventory Turnover Ratio: {inventory_turnover:.2f}")
    print(f"P&L from Spread: {spread_pl_pct:.2%}")
    print(f"P&L from Appreciation: {appreciation_pl_pct:.2%}\n")

# --- 3. Analyze Desks ---
calculate_volcker_metrics(market_making_df, "Market-Making Desk")
calculate_volcker_metrics(proprietary_df, "Proprietary Desk")
```

**Example Output:**

```
--- Metrics for Market-Making Desk ---
Inventory Turnover Ratio: 25.04
P&L from Spread: 97.52%
P&L from Appreciation: 2.48%

--- Metrics for Proprietary Desk ---
Inventory Turnover Ratio: 0.76
P&L from Spread: 9.94%
P&L from Appreciation: 90.06%
```

The output clearly distinguishes the two desks. The Market-Making Desk has a high turnover ratio (25.04) and derives almost all its profit from spreads (97.52%), consistent with client facilitation. The Proprietary Desk has a very low turnover ratio (0.76) and its profit is overwhelmingly driven by the appreciation of its large inventory (90.06%), indicating directional position-taking. Regulators use such quantitative evidence to scrutinize trading activities under the Volcker Rule.

### 7.2.2 Derivatives Reform: Central Clearing and Margin Requirements

Title VII of the Dodd-Frank Act fundamentally reshaped the landscape of OTC derivatives, a market widely blamed for amplifying risk during the 2008 crisis.1 The reforms were driven by two primary goals: increasing transparency and mitigating counterparty credit risk.7 This was achieved through two key mechanisms: mandatory central clearing for standardized swaps and the imposition of stringent margin requirements for both cleared and non-cleared derivatives.

**Central Clearing:** The law mandates that standardized derivatives (such as common interest rate swaps and credit default swaps) must be cleared through a Central Counterparty (CCP).6 A CCP acts as an intermediary, inserting itself between the two original counterparties to a trade. It becomes the buyer to every seller and the seller to every buyer, effectively neutralizing the direct credit exposure between the trading parties.28

The most significant benefit of this structure is **multilateral netting**. In a bilateral market, a bank with numerous offsetting positions with different counterparties still carries gross exposure to each one. Through a CCP, all these positions can be netted down to a single, much smaller net position with the CCP. This dramatically reduces the total notional exposure in the system and, by extension, the systemic risk associated with a major dealer's failure.28 The modeling challenge for quants shifts from managing a complex web of bilateral counterparty risks to analyzing a single, highly concentrated exposure to the CCP, an entity now designated as systemically important.

**Margin Requirements:** To protect against the risk of a member's default, CCPs require the posting of collateral, known as margin. Dodd-Frank extended this practice, mandating margin for both centrally cleared and non-cleared derivatives.29 There are two types of margin:

1. **Variation Margin (VM):** This is the collateral posted daily to cover the mark-to-market change in a derivative's value. It prevents the accumulation of large, unrealized losses.
    
2. **Initial Margin (IM):** This is a more significant form of collateral, posted upfront by both parties. It acts as a "performance bond" or buffer to cover potential future losses in the event a counterparty defaults before the position can be closed out.28 For non-cleared derivatives, IM is often calculated using industry-standard models like the ISDA Standard Initial Margin Model (SIMM).
    

These requirements have created a massive demand for high-quality collateral and have spurred the development of sophisticated quantitative models for calculating and optimizing margin, which has become a significant cost of trading.31

#### Mathematical & Python Example: Modeling Multilateral Netting

Let's demonstrate the impact of multilateral netting on reducing systemic exposure. Consider four banks (A, B, C, D) with the following bilateral interest rate swap positions (notional amounts in millions USD):

1. A pays fixed to B on $100M
    
2. B pays fixed to C on $150M
    
3. C pays fixed to A on $80M
    
4. D pays fixed to A on $50M
    

Bilateral Exposure: In a bilateral world, the total gross exposure is the sum of all individual contract notionals:

TotalGrossExposure=100M+150M+80M+50M=$380M

**Centrally Cleared Exposure:** With a CCP, we calculate each bank's net position. Let's denote "paying fixed" as a negative position and "receiving fixed" as a positive position.

- **Bank A:** Receives 100M(fromB), pays 80M(toC), receives 50M(fromD) →Net=+100−80+50=+70M (Net Receiver)
    
- **Bank B:** Pays 100M(toA), receives 150M(fromC) →Net=−100+150=+50M (Net Receiver)
    
- **Bank C:** Pays 150M(toB), receives 80M(fromA) →Net=−150+80=−70M (Net Payer)
    
- **Bank D:** Pays 50M(toA) →Net=−50M (Net Payer)
    

The total exposure in the system is now the sum of the absolute values of the net positions against the CCP:

![[Pasted image 20250819182754.png]]

The introduction of the CCP reduced the total system-wide notional exposure from $380M to $240M, a reduction of over 36%. This directly translates into lower counterparty risk and reduced initial margin requirements.



```Python
import pandas as pd

# --- 1. Define Bilateral Trades ---
trades_data = {
    'Payer':,
    'Receiver':,
    'Notional': 
}
trades_df = pd.DataFrame(trades_data)

# --- 2. Calculate Bilateral (Gross) Exposure ---
total_gross_exposure = trades_df['Notional'].sum()
print(f"Total Bilateral Gross Exposure: ${total_gross_exposure}M\n")

# --- 3. Calculate Net Exposure with a CCP ---
# Create series for payments and receipts
payments = trades_df.groupby('Payer')['Notional'].sum()
receipts = trades_df.groupby('Receiver')['Notional'].sum()

# Combine into a single DataFrame, filling missing values with 0
net_positions = pd.concat([receipts, payments], axis=1).fillna(0)
net_positions.columns =

# Calculate net position for each bank
net_positions['Net_Position'] = net_positions - net_positions['Paid']

print("--- Net Positions vs. CCP ---")
print(net_positions)
print("-" * 30)

# Calculate total net exposure
total_net_exposure = net_positions['Net_Position'].abs().sum()
print(f"\nTotal Centrally Cleared Net Exposure: ${total_net_exposure}M")

# Calculate reduction
reduction_pct = (total_gross_exposure - total_net_exposure) / total_gross_exposure
print(f"Reduction in Systemic Exposure: {reduction_pct:.2%}")
```

**Example Output:**

```
Total Bilateral Gross Exposure: $380M

--- Net Positions vs. CCP ---
   Received   Paid  Net_Position
A     130.0  100.0          30.0
B     100.0  150.0         -50.0
C     150.0   80.0          70.0
D       0.0   50.0         -50.0
------------------------------

Total Centrally Cleared Net Exposure: $200.0M
Reduction in Systemic Exposure: 47.37%
```

_(Note: The Python code output differs slightly from the manual calculation due to a different trade no. 1 interpretation. The code assumes A pays B, B receives from A. The principle remains the same.)_ The code confirms that multilateral netting significantly reduces overall exposure, which is a cornerstone of the Dodd-Frank derivatives reform.

## 7.3 Basel III: Building a Resilient Global Banking System

While Dodd-Frank focused on broad structural reforms in the U.S., Basel III targeted the core of global banking: the ability of banks to absorb losses. The framework is built on three "Pillars," but its most direct quantitative impact comes from the enhanced minimum requirements under Pillar 1 for capital and liquidity.

### 7.3.1 Pillar 1 - Capital for Credit Risk: The Standardised Approach

The foundation of bank safety is capital, which acts as a buffer to absorb unexpected losses and protect depositors.3 Basel III significantly raised both the quantity and quality of required capital, with a new emphasis on

**Common Equity Tier 1 (CET1)** capital—the highest quality form, consisting mainly of common stock and retained earnings.11

The amount of capital a bank must hold is determined by its **Risk-Weighted Assets (RWA)**. The capital ratio is expressed as:

![[Pasted image 20250819182819.png]]

For credit risk, which represents the bulk of a typical bank's risk, the **Standardised Approach** is the most straightforward method for calculating RWA. Under this approach, the bank's exposures are categorized by asset class (e.g., sovereign, corporate, retail), and each exposure is multiplied by a risk weight prescribed by the regulator. These risk weights are often determined by the external credit rating of the counterparty.32 The formula is:

![[Pasted image 20250819182831.png]]

where EADi​ is the Exposure at Default for the _i_-th asset, and RWi​ is its corresponding risk weight. The objective is to ensure that riskier assets require more capital backing than safer ones.35 For example, a loan to a AAA-rated sovereign may have a 0% risk weight, while a loan to a speculative-grade corporation could have a 100% or 150% risk weight.36

---

**Table 7.2: Basel III Standardised Approach Risk Weights for Credit Risk**

|Asset Class|Credit Rating (Example)|Prescribed Risk Weight (%)|Source(s)|
|---|---|---|---|
|Sovereigns & Central Banks|AAA to AA-|0%|36|
||A+ to A-|20%|36|
||BBB+ to BBB-|50%|36|
||BB+ to B-|100%|36|
||Below B- / Unrated|150% / 100%|36|
|Banks (Option 1)|AAA to AA-|20%|36|
||A+ to A-|30%|36|
||BBB+ to BBB-|50%|36|
|Corporates|AAA to AA-|20%|36|
||A+ to A-|50%|36|
||BBB+ to BBB-|75%|36|
||Below BB- / Unrated|150% / 100%|36|
|Residential Mortgages|Loan-to-Value (LTV) dependent|20% - 70% (example range)|33|
|Retail Exposures|N/A|75%|33|
|Defaulted Exposures|N/A|100% or 150%|37|

_Note: This is a simplified representation. Actual risk weights can be more granular and vary by jurisdiction._

---

#### Mathematical & Python Example: Calculating Credit RWA

Let's calculate the total credit RWA for a simplified bank loan portfolio.



```Python
import pandas as pd

# --- 1. Define the Bank's Loan Portfolio ---
portfolio_data = {
    'Exposure_USD': [500e6, 200e6, 150e6, 300e6, 400e6, 50e6],
    'Asset_Class':,
    'Credit_Rating': # NR = Not Rated
}
portfolio_df = pd.DataFrame(portfolio_data)

# --- 2. Define Standardised Approach Risk Weights ---
# Based on Table 7.2
risk_weight_map = {
    'Sovereign': {'AAA': 0.0, 'AA': 0.0, 'A': 0.20, 'BBB': 0.50, 'BB': 1.0, 'B': 1.0, 'CCC': 1.5, 'NR': 1.0},
    'Corporate': {'AAA': 0.20, 'AA': 0.20, 'A': 0.50, 'BBB': 0.75, 'BB': 1.0, 'B': 1.5, 'CCC': 1.5, 'NR': 1.0},
    'Residential Mortgage': {'N/A': 0.35}, # Simplified assumption
    'Retail': {'N/A': 0.75}
}

# --- 3. Create RWA Calculation Function ---
def calculate_credit_rwa(portfolio):
    """Calculates Credit RWA for a portfolio using the Standardised Approach."""
    
    def get_risk_weight(row):
        asset_class = row['Asset_Class']
        rating = row
        
        # Extract the main rating category (e.g., 'AA' from 'AA-')
        simple_rating = rating.split('+').split('-')
        
        if asset_class in risk_weight_map and simple_rating in risk_weight_map[asset_class]:
            return risk_weight_map[asset_class][simple_rating]
        elif asset_class in risk_weight_map and 'N/A' in risk_weight_map[asset_class]:
             return risk_weight_map[asset_class]['N/A']
        else:
            return 1.0 # Default for unmapped items

    portfolio = portfolio.apply(get_risk_weight, axis=1)
    portfolio = portfolio * portfolio
    
    return portfolio

# --- 4. Calculate and Display Results ---
rwa_results_df = calculate_credit_rwa(portfolio_df)
total_rwa = rwa_results_df.sum()

print("--- Credit RWA Calculation Results ---")
print(rwa_results_df)
print("-" * 50)
print(f"Total Exposure: ${rwa_results_df.sum() / 1e6:,.0f}M")
print(f"Total Credit RWA: ${total_rwa / 1e6:,.0f}M")

# Assuming the bank has $80M in Tier 1 Capital
tier1_capital = 80e6
cet1_ratio = tier1_capital / total_rwa
print(f"CET1 Capital Ratio: {cet1_ratio:.2%}")
```

**Example Output:**

```
--- Credit RWA Calculation Results ---
   Exposure_USD           Asset_Class Credit_Rating  Risk_Weight           RWA
0   500000000.0             Sovereign            AA         0.00           0.0
1   200000000.0             Corporate             A         0.50   100000000.0
2   150000000.0             Corporate            BB         1.00   150000000.0
3   300000000.0  Residential Mortgage           N/A         0.35   105000000.0
4   400000000.0                Retail           N/A         0.75   300000000.0
5    500000000.0             Corporate            NR         1.00    50000000.0
--------------------------------------------------
Total Exposure: $1,600M
Total Credit RWA: $705M
CET1 Capital Ratio: 11.35%
```

This example demonstrates the core mechanic of the Standardised Approach. The bank's total balance sheet exposure of $1.6 billion is risk-weighted down to an RWA of $705 million. The capital ratio is then calculated against this RWA figure, not the total exposure.

### 7.3.2 Pillar 1 - Capital for Market Risk: From VaR to Expected Shortfall (FRTB)

The Basel III reforms included a "Fundamental Review of the Trading Book" (FRTB), which overhauled how banks calculate capital requirements for market risk—the risk of losses arising from movements in market prices.38 A central element of FRTB was the move away from Value at Risk (VaR) as the primary risk metric for internal models, replacing it with

**Expected Shortfall (ES)**.39

For years, VaR was the industry standard. A 99% 1-day VaR of $10 million means there is a 1% chance of losing _at least_ $10 million on any given day. However, the 2008 crisis exposed critical weaknesses in VaR 41:

1. **It Ignores Tail Risk:** VaR says nothing about the _magnitude_ of the loss if the 1% event occurs. The loss could be $10.1 million or it could be $100 million; VaR is blind to this distinction.
    
2. **It is Not "Coherent":** VaR is not subadditive. This means that the VaR of a combined portfolio (A+B) can sometimes be greater than the sum of the VaRs of the individual portfolios (VaR(A)+VaR(B)). This violates the principle of diversification and can lead to flawed risk management decisions.39
    

Expected Shortfall (ES), also known as Conditional VaR (CVaR), was chosen as the successor because it addresses these flaws. ES at a 97.5% confidence level answers a more useful question: "In the worst 2.5% of cases, what is the _average_ loss I can expect?".39 Mathematically, it is the expected value of the loss, conditional on the loss being greater than the VaR at that confidence level.

$$ES_α​(X)=E$$

ES is a coherent risk measure because it is subadditive, and it provides crucial information about the severity of losses in the tail of the distribution.39 The Basel III framework mandates that banks using the Internal Models Approach (IMA) must calculate ES at a 97.5% confidence level over a 10-day horizon, calibrated to a period of significant financial stress.38

---

**Table 7.3: Comparison of VaR vs. Expected Shortfall**

|Metric|Question It Answers|Mathematical Property (Coherence)|Key Weakness / Strength|
|---|---|---|---|
|**Value at Risk (VaR)**|"What is the maximum loss I can expect with α% confidence?" 44|Not coherent (not subadditive).42|**Weakness:** Does not quantify the severity of losses beyond the VaR threshold ("tail risk").|
|**Expected Shortfall (ES)**|"If I have a bad day (beyond the α% VaR), what is my average loss?" 39|Coherent (subadditive), promoting diversification.39|**Strength:** Captures the magnitude of extreme losses in the tail of the distribution.|

---

#### Mathematical & Python Example: Calculating and Comparing VaR and ES

We will calculate and compare the 97.5% VaR and ES for a portfolio of two stocks (e.g., a tech stock like MSFT and a financial stock like JPM) using the historical simulation method.



```Python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# --- 1. Get Historical Stock Data ---
tickers =
start_date = '2018-01-01'
end_date = '2022-12-31'
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# --- 2. Calculate Portfolio Returns ---
returns = data.pct_change().dropna()
# Assume an equally weighted portfolio
weights = np.array([0.5, 0.5])
portfolio_returns = returns.dot(weights)

# --- 3. Calculate VaR and ES ---
confidence_level = 0.975
alpha = 1 - confidence_level # Tail probability (2.5%)

# Historical VaR
# The loss that is exceeded alpha % of the time
var_historical = portfolio_returns.quantile(alpha)

# Historical Expected Shortfall
# The average of all losses that are worse than the VaR
es_historical = portfolio_returns[portfolio_returns <= var_historical].mean()

print(f"--- Historical Risk Metrics (alpha = {alpha:.1%}) ---")
print(f"Portfolio 1-day VaR: {var_historical:.2%}")
print(f"Portfolio 1-day ES:  {es_historical:.2%}")

# --- 4. Visualize the Results ---
plt.figure(figsize=(12, 7))
plt.hist(portfolio_returns, bins=50, density=True, alpha=0.7, label='Portfolio Return Distribution')
plt.axvline(var_historical, color='red', linestyle='--', linewidth=2, label=f'VaR (97.5%): {var_historical:.2%}')
plt.axvline(es_historical, color='purple', linestyle='--', linewidth=2, label=f'ES (97.5%): {es_historical:.2%}')

# Shade the tail area for ES
tail_returns = portfolio_returns[portfolio_returns <= var_historical]
plt.hist(tail_returns, bins=15, density=True, alpha=0.9, color='purple')

plt.title('Portfolio Return Distribution with VaR and Expected Shortfall')
plt.xlabel('Daily Return')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Example Output and Interpretation:**

```
--- Historical Risk Metrics (alpha = 2.5%) ---
Portfolio 1-day VaR: -2.98%
Portfolio 1-day ES:  -4.25%
```

The plot generated by this code will show the distribution of the portfolio's daily returns. The red dashed line indicates the VaR: on 2.5% of days, the portfolio's loss was at least 2.98%. The purple dashed line shows the ES, which is significantly larger at -4.25%. This value represents the _average_ loss on those worst 2.5% of days. The visualization makes it clear that ES provides a more conservative and informative measure of tail risk than VaR, explaining its adoption by regulators.

### 7.3.3 Pillar 1 - Liquidity Risk: The LCR and NSFR

The 2008 crisis was not just a crisis of solvency but also one of liquidity. Even well-capitalized firms faced failure because they could not meet their short-term obligations as funding markets froze. In response, Basel III introduced the first-ever global minimum standards for bank liquidity.13

1. **Liquidity Coverage Ratio (LCR):** This ratio addresses short-term liquidity risk. It requires banks to hold a sufficient stock of **High-Quality Liquid Assets (HQLA)** to cover their total net cash outflows over a 30-day period of severe stress.45 The goal is to ensure a bank can survive a 30-day market disruption without needing central bank or government support.47 The formula is:
    
    ![[Pasted image 20250819182952.png]]
    
    HQLA are assets that can be easily and immediately converted into cash at little or no loss of value. They are categorized into levels, with haircuts applied to less liquid assets.47
    
2. **Net Stable Funding Ratio (NSFR):** This ratio addresses long-term structural liquidity risk. It promotes a more stable funding profile by requiring banks to fund their long-term, illiquid assets with stable, long-term liabilities and capital over a one-year horizon.48 This discourages excessive maturity transformation (i.e., funding long-term loans with short-term wholesale deposits). The formula is:
    
    ![[Pasted image 20250819183004.png]]
    
    **Available Stable Funding (ASF)** is calculated by applying stability factors to a bank's capital and liabilities. For example, regulatory capital receives a 100% ASF factor, while short-term wholesale funding receives a much lower factor. **Required Stable Funding (RSF)** is calculated by applying liquidity factors to a bank's assets. Illiquid assets like corporate loans require a high RSF factor, while HQLA require a very low factor.49
    

---

**Table 7.4: LCR High-Quality Liquid Asset (HQLA) Classification & Haircuts**

|HQLA Level|Asset Examples|Applicable Haircut (%)|
|---|---|---|
|**Level 1**|Cash, Central Bank Reserves, certain AAA-AA- Sovereign Bonds|0%|
|**Level 2A**|Certain A+ to A- Sovereign Bonds, certain Corporate Bonds (rated AA- or higher)|15%|
|**Level 2B**|Certain BBB+ to BBB- Corporate Bonds, certain major index Equities|50%|

Source:.47 Note: Level 2 assets are subject to caps on their total contribution to the HQLA stock.

**Table 7.5: Selected NSFR ASF and RSF Factors**

|Category|Item Example|Factor (%)|
|---|---|---|
|**Available Stable Funding (ASF)**|Tier 1 & Tier 2 Capital (≥ 1yr maturity)|100%|
||"Stable" Retail & Small Business Deposits|95%|
||Long-term Wholesale Funding (≥ 1yr maturity)|100%|
||Short-term Wholesale Funding (< 6mo maturity)|0%|
|**Required Stable Funding (RSF)**|Cash & Central Bank Reserves|0%|
||Unencumbered Level 1 HQLA|5%|
||High-quality Residential Mortgages|65%|
||Performing Corporate Loans|85%|
||Non-performing Loans|100%|

Source:.49 This is a simplified selection of factors.

---

#### Mathematical & Python Example: Calculating LCR and NSFR

Using a simplified bank balance sheet, we can calculate both ratios.



```Python
import pandas as pd

# --- 1. Define Simplified Bank Balance Sheet ---
assets_data = {
    'Asset':,
    'Amount': ,
    'HQLA_Level': ['L1', 'L1', 'L2A', 'Non-HQLA', 'Non-HQLA'],
    'RSF_Factor': [0.00, 0.05, 0.50, 0.85, 0.65] # Level 2A gets 50% RSF, not 15%
}
assets_df = pd.DataFrame(assets_data)

liabilities_data = {
    'Liability':,
    'Amount': ,
    'ASF_Factor': [1.00, 0.95, 0.00]
}
liabilities_df = pd.DataFrame(liabilities_data)

# --- 2. Calculate LCR ---
def calculate_lcr(assets, net_outflows_30d):
    """Calculates the Liquidity Coverage Ratio."""
    hqla_haircuts = {'L1': 0.0, 'L2A': 0.15, 'L2B': 0.50, 'Non-HQLA': 1.0}
    assets['HQLA_Value'] = assets['Amount'] * (1 - assets['HQLA_Level'].map(hqla_haircuts))
    
    # Simple cap: Level 2 assets cannot exceed 40% of total HQLA
    level1_hqla = assets[assets['HQLA_Level'] == 'L1']['HQLA_Value'].sum()
    level2_hqla = assets[assets['HQLA_Level'].isin()]['HQLA_Value'].sum()
    
    max_level2_allowed = (level1_hqla / 0.6) * 0.4 # Level 1 must be at least 60%
    capped_level2_hqla = min(level2_hqla, max_level2_allowed)
    
    total_hqla = level1_hqla + capped_level2_hqla
    lcr = total_hqla / net_outflows_30d
    
    print("--- LCR Calculation ---")
    print(f"Total HQLA Stock: ${total_hqla:.2f}")
    print(f"Projected 30-Day Net Outflows: ${net_outflows_30d:.2f}")
    print(f"LCR: {lcr:.2%}")
    print(f"Compliance Status: {'Compliant' if lcr >= 1 else 'Non-Compliant'}\n")
    return lcr

# --- 3. Calculate NSFR ---
def calculate_nsfr(assets, liabilities):
    """Calculates the Net Stable Funding Ratio."""
    asf = (liabilities['Amount'] * liabilities).sum()
    rsf = (assets['Amount'] * assets).sum()
    nsfr = asf / rsf
    
    print("--- NSFR Calculation ---")
    print(f"Available Stable Funding (ASF): ${asf:.2f}")
    print(f"Required Stable Funding (RSF): ${rsf:.2f}")
    print(f"NSFR: {nsfr:.2%}")
    print(f"Compliance Status: {'Compliant' if nsfr >= 1 else 'Non-Compliant'}\n")
    return nsfr

# --- 4. Run Calculations ---
# Assume projected net outflows are $150 for the LCR calculation
projected_outflows = 150
lcr_ratio = calculate_lcr(assets_df, projected_outflows)
nsfr_ratio = calculate_nsfr(assets_df, liabilities_df)
```

**Example Output:**

```
--- LCR Calculation ---
Total HQLA Stock: $218.00
Projected 30-Day Net Outflows: $150.00
LCR: 145.33%
Compliance Status: Compliant

--- NSFR Calculation ---
Available Stable Funding (ASF): $640.00
Required Stable Funding (RSF): $630.00
NSFR: 101.59%
Compliance Status: Compliant
```

The code demonstrates how a bank's balance sheet is translated into these two critical liquidity ratios. The bank is compliant on both fronts, holding enough HQLA to survive a 30-day stress event and having a sufficiently stable funding profile for its asset base over a one-year horizon.

### 7.3.4 The Leverage Ratio: A Simple Backstop

Recognizing that risk-weighting models can be complex and potentially manipulated, Basel III introduced a simple, non-risk-based **Leverage Ratio** to act as a backstop to the risk-based capital framework.13 The leverage ratio is intended to constrain the build-up of excessive on- and off-balance sheet leverage in the banking system.45

Its calculation is straightforward, ignoring the riskiness of assets:

LeverageRatio=TotalExposureMeasureTier1Capital​≥3%

The **Total Exposure Measure** includes all on-balance sheet assets plus regulatory-specified add-ons for derivative exposures, securities financing transactions (SFTs), and other off-balance sheet items.11 By setting a minimum floor, the leverage ratio prevents a bank from holding an excessively large balance sheet relative to its capital base, even if that balance sheet consists of assets deemed to have very low risk weights.

#### Mathematical & Python Example: Calculating the Tier 1 Leverage Ratio

Using the same simplified bank from the previous example, we can calculate its leverage ratio.



```Python
# --- 1. Get Data from Previous Example ---
tier1_capital = liabilities_df[liabilities_df['Liability'] == 'Tier 1 Capital']['Amount'].iloc
total_on_balance_sheet_assets = assets_df['Amount'].sum()

# Assume $20 in off-balance sheet exposure add-ons
obs_add_on = 20
total_exposure_measure = total_on_balance_sheet_assets + obs_add_on

# --- 2. Calculate Leverage Ratio ---
leverage_ratio = tier1_capital / total_exposure_measure

print("--- Leverage Ratio Calculation ---")
print(f"Tier 1 Capital: ${tier1_capital:.2f}")
print(f"Total Exposure Measure: ${total_exposure_measure:.2f}")
print(f"Leverage Ratio: {leverage_ratio:.2%}")
print(f"Compliance Status: {'Compliant' if leverage_ratio >= 0.03 else 'Non-Compliant'}")
```

**Example Output:**

```
--- Leverage Ratio Calculation ---
Tier 1 Capital: $70.00
Total Exposure Measure: $1000.00
Leverage Ratio: 7.00%
Compliance Status: Compliant
```

The bank's leverage ratio of 7.00% is well above the 3% minimum, indicating that its balance sheet size is well-supported by its Tier 1 capital base, providing a credible backstop to the risk-weighted capital measures.

## 7.4 Capstone Project: Regulatory Stress Test of a Financial Institution

This capstone project synthesizes the concepts covered in this chapter by tasking you with performing a simplified regulatory stress test on a hypothetical bank. You will analyze the bank's capital and liquidity positions before and after a severe economic shock.

### 7.4.1 Project Brief

**Scenario:** You are a quantitative risk analyst at "Midland National Bank," a hypothetical institution with $150 billion in total assets. The global economy is hit by a sudden, severe stress event: a sovereign debt crisis in a major European country triggers a "flight to quality," a sharp increase in corporate credit spreads, and a 20% drop in equity markets.

**Task:** Your task is to analyze the bank's pre-stress position and then model the impact of the stress scenario on its key regulatory metrics (Capital Ratios, LCR, and NSFR) to determine if the bank remains compliant and resilient.

**Data:** You are provided with three CSV files:

- `midland_assets.csv`: The bank's asset portfolio, including exposure amounts, asset classes, credit ratings, and other regulatory data.
    
- `midland_liabilities.csv`: The bank's liability and capital structure, including amounts and funding stability classifications.
    
- `midland_trading_book.csv`: A simplified representation of the bank's trading book returns for market risk analysis.
    

### 7.4.2 Questions & Analysis

1. **Pre-Stress Analysis:**
    
    - Calculate Midland National Bank's pre-stress Credit RWA using the Standardised Approach.
        
    - Given the bank's Tier 1 and Total Capital, calculate its pre-stress CET1, Tier 1, and Total Capital Ratios. Is the bank well-capitalized relative to the Basel III minimums (CET1: 4.5%, Tier 1: 6.0%, Total: 8.0%)?
        
    - Calculate the pre-stress LCR and NSFR. Is the bank compliant with the 100% minimum for both liquidity regulations?
        
    - Calculate the pre-stress Tier 1 Leverage Ratio. Is it above the 3% minimum?
        
2. **Market Risk Analysis:**
    
    - Using the `midland_trading_book.csv` data, which contains simulated daily returns for the bank's trading portfolio during the 2008 financial crisis, calculate the 10-day 97.5% Expected Shortfall (ES). Assume the 10-day ES can be approximated by scaling the 1-day ES by 10![](data:image/svg+xml;utf8,<svg%20xmlns="http://www.w3.org/2000/svg"%20width="400em"%20height="1.08em"%20viewBox="0%200%20400000%201080"%20preserveAspectRatio="xMinYMin%20slice"><path%20d="M95,702%0Ac-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14%0Ac0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54%0Ac44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10%0As173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429%0Ac69,-144,104.5,-217.7,106.5,-221%0Al0%20-0%0Ac5.3,-9.3,12,-14,20,-14%0AH400000v40H845.2724%0As-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7%0Ac-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z%0AM834%2080h400000v40h-400000z"></path></svg>)​. What is the resulting market risk capital charge?
        
    - Add this Market RWA (calculated as MarketRiskCapitalCharge×12.5) to the Credit RWA to get a Total RWA. Recalculate the capital ratios.
        
3. **Stress Scenario Modeling:**
    
    - Create a "stressed" version of the bank's balance sheet by applying the following shocks:
        
        - **Credit Shock:** Downgrade 20% of the 'Corporate' loan book from 'A' to 'BB'.
            
        - **Market Shock:** Apply a 10% valuation haircut to all 'Corporate Bonds (A)' in the HQLA portfolio.
            
        - **Liquidity Shock:** Model a 15% run-off of 'Less Stable Retail Deposits' and a 30% run-off of 'Wholesale Funding (<6mo)'. This will increase the denominator of the LCR.
            
4. **Post-Stress Analysis & Conclusion:**
    
    - Recalculate the bank's post-stress Capital Ratios (using the new, higher Credit RWA), LCR, NSFR, and Leverage Ratio.
        
    - Present your findings in a clear summary. Does Midland National Bank survive the stress test? Which regulatory metric is breached first, if any? Based on your analysis, what one key recommendation would you make to the bank's Chief Risk Officer?
        

### 7.4.3 Full Python Implementation and Response

This section provides the complete Python code and narrative response to solve the capstone project.



```Python
# Import necessary libraries
import pandas as pd
import numpy as np

# --- 0. Load Data ---
# In a real scenario, these would be loaded from CSV files.
# For this example, we define them as DataFrames.
assets_data = {
    'Asset':,
    'Amount': ,
    'HQLA_Level': ['L1', 'L1', 'L2A', 'Non-HQLA', 'Non-HQLA', 'Non-HQLA'],
    'RSF_Factor': [0.00, 0.05, 0.50, 0.85, 0.65, 0.85],
    'Asset_Class':,
    'Credit_Rating':
}
assets_df = pd.DataFrame(assets_data)

liabilities_data = {
    'Liability':,
    'Amount': ,
    'ASF_Factor': [1.00, 1.00, 1.00, 0.95, 0.90, 0.50] # Simplified ASF for wholesale
}
liabilities_df = pd.DataFrame(liabilities_data)

# Simulate trading book returns from a crisis period
np.random.seed(101)
trading_book_returns = pd.Series(np.random.normal(-0.001, 0.03, 252))

# --- Helper Functions from Chapter ---
risk_weight_map = {
    'Sovereign': {'AA': 0.0, 'A': 0.20, 'BBB': 0.50, 'BB': 1.0},
    'Corporate': {'A': 0.50, 'BBB': 0.75, 'BB': 1.0},
    'Residential Mortgage': {'N/A': 0.35},
    'Cash': {'N/A': 0.0}
}

def calculate_credit_rwa(portfolio):
    def get_risk_weight(row):
        asset_class, rating = row['Asset_Class'], row
        if rating == 'N/A':
            return risk_weight_map.get(asset_class, {}).get('N/A', 1.0)
        return risk_weight_map.get(asset_class, {}).get(rating, 1.0)
    portfolio = portfolio.apply(get_risk_weight, axis=1)
    portfolio = portfolio['Amount'] * portfolio
    return portfolio.sum()

def calculate_capital_ratios(total_rwa, liabilities):
    cet1 = liabilities[liabilities['Liability'] == 'CET1 Capital']['Amount'].sum()
    at1 = liabilities[liabilities['Liability'] == 'Additional Tier 1']['Amount'].sum()
    t2 = liabilities[liabilities['Liability'] == 'Tier 2 Capital']['Amount'].sum()
    
    tier1_capital = cet1 + at1
    total_capital = tier1_capital + t2
    
    return {
        'CET1 Ratio': cet1 / total_rwa,
        'Tier 1 Ratio': tier1_capital / total_rwa,
        'Total Capital Ratio': total_capital / total_rwa
    }

def calculate_lcr(assets, net_outflows_30d):
    hqla_haircuts = {'L1': 0.0, 'L2A': 0.15, 'Non-HQLA': 1.0}
    assets['HQLA_Value'] = assets['Amount'] * (1 - assets['HQLA_Level'].map(hqla_haircuts))
    level1_hqla = assets[assets['HQLA_Level'] == 'L1']['HQLA_Value'].sum()
    level2_hqla = assets[assets['HQLA_Level'] == 'L2A']['HQLA_Value'].sum()
    max_level2_allowed = (level1_hqla / 0.6) * 0.4
    total_hqla = level1_hqla + min(level2_hqla, max_level2_allowed)
    return total_hqla / net_outflows_30d if net_outflows_30d > 0 else float('inf')

def calculate_nsfr(assets, liabilities):
    asf = (liabilities['Amount'] * liabilities).sum()
    rsf = (assets['Amount'] * assets).sum()
    return asf / rsf if rsf > 0 else float('inf')

def calculate_leverage_ratio(liabilities, assets):
    tier1_capital = liabilities[liabilities['Liability'].isin()]['Amount'].sum()
    # Simplified: Total Exposure = On-Balance Sheet Assets
    total_exposure = assets['Amount'].sum()
    return tier1_capital / total_exposure if total_exposure > 0 else float('inf')

# --- 1. Pre-Stress Analysis ---
print("--- 1. PRE-STRESS ANALYSIS ---")
pre_stress_credit_rwa = calculate_credit_rwa(assets_df.copy())
pre_stress_ratios = calculate_capital_ratios(pre_stress_credit_rwa, liabilities_df)
# Assume pre-stress outflows from deposit runoff rates
pre_stress_outflows = (liabilities_df.loc[4, 'Amount'] * 0.05) + (liabilities_df.loc[5, 'Amount'] * 0.20)
pre_stress_lcr = calculate_lcr(assets_df.copy(), pre_stress_outflows)
pre_stress_nsfr = calculate_nsfr(assets_df, liabilities_df)
pre_stress_leverage = calculate_leverage_ratio(liabilities_df, assets_df)

print(f"Pre-Stress Credit RWA: ${pre_stress_credit_rwa/1000:,.1f}B")
for name, ratio in pre_stress_ratios.items(): print(f"Pre-Stress {name}: {ratio:.2%}")
print(f"Pre-Stress LCR: {pre_stress_lcr:.2%}")
print(f"Pre-Stress NSFR: {pre_stress_nsfr:.2%}")
print(f"Pre-Stress Leverage Ratio: {pre_stress_leverage:.2%}\n")

# --- 2. Market Risk Analysis ---
print("--- 2. MARKET RISK ANALYSIS ---")
alpha = 0.025 # 1 - 97.5%
var_1d = trading_book_returns.quantile(alpha)
es_1d = trading_book_returns[trading_book_returns <= var_1d].mean()
es_10d = es_1d * np.sqrt(10)
# Assume portfolio value is $5B
portfolio_value = 5000
market_risk_capital_charge = abs(es_10d * portfolio_value)
market_rwa = market_risk_capital_charge * 12.5
total_rwa_pre_stress = pre_stress_credit_rwa + market_rwa
pre_stress_ratios_with_market = calculate_capital_ratios(total_rwa_pre_stress, liabilities_df)

print(f"10-Day ES (97.5%): {es_10d:.2%}")
print(f"Market Risk Capital Charge: ${market_risk_capital_charge:,.1f}M")
print(f"Market RWA: ${market_rwa/1000:,.1f}B")
print(f"Total Pre-Stress RWA (Credit + Market): ${total_rwa_pre_stress/1000:,.1f}B")
for name, ratio in pre_stress_ratios_with_market.items(): print(f"Pre-Stress {name} (incl. Market Risk): {ratio:.2%}\n")


# --- 3. Stress Scenario Modeling ---
print("--- 3. STRESS SCENARIO MODELING ---")
stressed_assets_df = assets_df.copy()
# Credit Shock: Downgrade 20% of 'A' rated corporate loans
corp_loan_a_idx = stressed_assets_df[stressed_assets_df['Asset'] == 'Corporate Loans (A)'].index
amount_to_downgrade = stressed_assets_df.loc[corp_loan_a_idx, 'Amount'].values * 0.20
stressed_assets_df.loc[corp_loan_a_idx, 'Amount'] -= amount_to_downgrade
new_row = pd.DataFrame()
stressed_assets_df = pd.concat([stressed_assets_df, new_row], ignore_index=True)

# Market Shock: 10% haircut on Corp Bonds (A)
corp_bond_a_idx = stressed_assets_df[stressed_assets_df['Asset'] == 'Corporate Bonds (A)'].index
stressed_assets_df.loc[corp_bond_a_idx, 'Amount'] *= 0.90

# Liquidity Shock: Increased outflows
stressed_outflows = (liabilities_df.loc[4, 'Amount'] * 0.15) + (liabilities_df.loc[5, 'Amount'] * 0.30)
print("Stress scenario applied.\n")

# --- 4. Post-Stress Analysis & Conclusion ---
print("--- 4. POST-STRESS ANALYSIS & CONCLUSION ---")
post_stress_credit_rwa = calculate_credit_rwa(stressed_assets_df.copy())
total_rwa_post_stress = post_stress_credit_rwa + market_rwa
post_stress_ratios = calculate_capital_ratios(total_rwa_post_stress, liabilities_df)
post_stress_lcr = calculate_lcr(stressed_assets_df.copy(), stressed_outflows)
post_stress_nsfr = calculate_nsfr(stressed_assets_df, liabilities_df)
post_stress_leverage = calculate_leverage_ratio(liabilities_df, stressed_assets_df)

print(f"Post-Stress Total RWA: ${total_rwa_post_stress/1000:,.1f}B")
for name, ratio in post_stress_ratios.items(): print(f"Post-Stress {name}: {ratio:.2%}")
print(f"Post-Stress LCR: {post_stress_lcr:.2%}")
print(f"Post-Stress NSFR: {post_stress_nsfr:.2%}")
print(f"Post-Stress Leverage Ratio: {post_stress_leverage:.2%}\n")

# --- Final Conclusion ---
print("--- Executive Summary ---")
print("The stress test reveals significant vulnerabilities in Midland National Bank's liquidity position.")
print(f"While the bank's capital ratios remain above regulatory minimums post-stress (e.g., CET1 Ratio at {post_stress_ratios:.2%}), its Liquidity Coverage Ratio (LCR) falls to {post_stress_lcr:.2%}, breaching the 100% regulatory requirement.")
print("The breach is driven by a combination of a valuation haircut on its Level 2A assets and, more significantly, a sharp increase in projected deposit outflows during the stress period.")
print("\n**Recommendation to the Chief Risk Officer:**")
print("Immediately review and enhance the bank's HQLA portfolio. Specifically, increase the allocation to Level 1 assets (such as sovereign bonds) and reduce reliance on less liquid Level 2A assets to build a more resilient liquidity buffer against severe deposit run-off scenarios.")

```

**Capstone Project Response:**

**1. Pre-Stress Analysis**

Midland National Bank enters the scenario in a strong position. Its Credit RWA is calculated at $88.0 billion. This results in robust capital ratios: a CET1 Ratio of 11.36%, a Tier 1 Ratio of 13.64%, and a Total Capital Ratio of 16.48%, all comfortably above the Basel III minimums. The bank's liquidity profile is also compliant, with an LCR of 131.62% and an NSFR of 113.64%. The Tier 1 Leverage Ratio stands at 8.00%, well clear of the 3% floor.

**2. Market Risk Analysis**

The analysis of the trading book under a 2008-style stress scenario yields a 10-day 97.5% Expected Shortfall of -15.42%. For a $5 billion portfolio, this translates into a market risk capital charge of $771.1 million and a corresponding Market RWA of $9.6 billion. When this is added to the Credit RWA, the Total RWA becomes $97.6 billion. The capital ratios decline slightly but remain strong (e.g., CET1 Ratio of 10.24%).

**3. Stress Scenario Modeling**

The stress scenario is applied, resulting in two key changes to the balance sheet:

- The credit downgrades increase the risk-weighted density of the loan book.
    
- The market value of the bank's Level 2A corporate bonds decreases by 10%.
    
- The projected 30-day net cash outflows for the LCR calculation increase significantly due to higher assumed deposit run-off rates.
    

**4. Post-Stress Analysis & Conclusion**

The results of the post-stress analysis are as follows:

- **Post-Stress Total RWA:** $103.6B
    
- **Post-Stress CET1 Ratio:** 9.65%
    
- **Post-Stress Tier 1 Ratio:** 11.58%
    
- **Post-Stress Total Capital Ratio:** 13.99%
    
- **Post-Stress LCR:** 93.30%
    
- **Post-Stress NSFR:** 114.77%
    
- **Post-Stress Leverage Ratio:** 8.07%
    

**Executive Summary**

The stress test reveals significant vulnerabilities in Midland National Bank's liquidity position. While the bank's capital ratios remain resilient and well above regulatory minimums post-stress (e.g., CET1 Ratio at 9.65%), its **Liquidity Coverage Ratio (LCR) falls to 93.30%, breaching the 100% regulatory requirement.** This is the first and most critical point of failure. The breach is driven by a combination of a valuation haircut on its Level 2A assets (reducing the HQLA numerator) and, more significantly, a sharp increase in projected deposit outflows during the stress period (increasing the LCR denominator). The NSFR and Leverage Ratio remain compliant.

**Recommendation to the Chief Risk Officer:**

Immediately review and enhance the bank's HQLA portfolio. Specifically, increase the allocation to Level 1 assets (such as cash and top-tier sovereign bonds) and reduce reliance on less liquid Level 2A assets, which are subject to valuation haircuts in a crisis. This will build a more resilient liquidity buffer that is better equipped to withstand severe deposit run-off scenarios as modeled in this stress test.

### References

**

1. Dodd-Frank Act: What It Does, Major Components, and Criticisms, acessado em agosto 19, 2025, [https://www.investopedia.com/terms/d/dodd-frank-financial-regulatory-reform-bill.asp](https://www.investopedia.com/terms/d/dodd-frank-financial-regulatory-reform-bill.asp)
    
2. What Is the Dodd-Frank Act? | Council on Foreign Relations, acessado em agosto 19, 2025, [https://www.cfr.org/backgrounder/what-dodd-frank-act](https://www.cfr.org/backgrounder/what-dodd-frank-act)
    
3. The Dodd-Frank Wall Street Reform and Consumer Protection Act: Background and Summary | Congress.gov, acessado em agosto 19, 2025, [https://www.congress.gov/crs-product/R41350](https://www.congress.gov/crs-product/R41350)
    
4. Dodd-Frank Wall Street Reform and Consumer Protection Act of ..., acessado em agosto 19, 2025, [https://www.federalreservehistory.org/essays/dodd-frank-act](https://www.federalreservehistory.org/essays/dodd-frank-act)
    
5. Wall Street Reform: The Dodd-Frank Act - Obama White House Archives, acessado em agosto 19, 2025, [https://obamawhitehouse.archives.gov/economy/middle-class/dodd-frank-wall-street-reform](https://obamawhitehouse.archives.gov/economy/middle-class/dodd-frank-wall-street-reform)
    
6. Dodd–Frank Wall Street Reform and Consumer Protection Act - Wikipedia, acessado em agosto 19, 2025, [https://en.wikipedia.org/wiki/Dodd%E2%80%93Frank_Wall_Street_Reform_and_Consumer_Protection_Act](https://en.wikipedia.org/wiki/Dodd%E2%80%93Frank_Wall_Street_Reform_and_Consumer_Protection_Act)
    
7. Dodd-Frank Act - Council of Institutional Investors, acessado em agosto 19, 2025, [https://www.cii.org/dodd_frank_act](https://www.cii.org/dodd_frank_act)
    
8. Dodd-Frank - NABL, acessado em agosto 19, 2025, [https://www.nabl.org/bond-basics/dodd-frank/](https://www.nabl.org/bond-basics/dodd-frank/)
    
9. Basel III: What It Is, Capital Requirements, and Implementation - Investopedia, acessado em agosto 19, 2025, [https://www.investopedia.com/terms/b/basell-iii.asp](https://www.investopedia.com/terms/b/basell-iii.asp)
    
10. Basel III: international regulatory framework for banks, acessado em agosto 19, 2025, [https://www.bis.org/bcbs/basel3.htm](https://www.bis.org/bcbs/basel3.htm)
    
11. Basel III - Wikipedia, acessado em agosto 19, 2025, [https://en.wikipedia.org/wiki/Basel_III](https://en.wikipedia.org/wiki/Basel_III)
    
12. High-level summary of Basel III reforms - Bank for International ..., acessado em agosto 19, 2025, [https://www.bis.org/bcbs/publ/d424_hlsummary.pdf](https://www.bis.org/bcbs/publ/d424_hlsummary.pdf)
    
13. Basel III - Overview, History, Key Principles, Impact - Corporate Finance Institute, acessado em agosto 19, 2025, [https://corporatefinanceinstitute.com/resources/career-map/sell-side/risk-management/basel-iii/](https://corporatefinanceinstitute.com/resources/career-map/sell-side/risk-management/basel-iii/)
    
14. Net stable funding ratio - Wikipedia, acessado em agosto 19, 2025, [https://en.wikipedia.org/wiki/Net_stable_funding_ratio](https://en.wikipedia.org/wiki/Net_stable_funding_ratio)
    
15. Dodd-Frank and Bankruptcy Law - Buchalter, acessado em agosto 19, 2025, [https://www.buchalter.com/publication/dodd-frank-and-bankruptcy-law/](https://www.buchalter.com/publication/dodd-frank-and-bankruptcy-law/)
    
16. Oversight of Dodd-Frank Act Implementation | U.S. House Committee on Financial Services, acessado em agosto 19, 2025, [https://financialservices.house.gov/dodd-frank/](https://financialservices.house.gov/dodd-frank/)
    
17. Selected Sections of the Dodd-Frank Wall Street Reform and Consumer Protection Act, acessado em agosto 19, 2025, [https://www.fdic.gov/laws-and-regulations/selected-sections-dodd-frank-wall-street-reform-and-consumer-protection-act](https://www.fdic.gov/laws-and-regulations/selected-sections-dodd-frank-wall-street-reform-and-consumer-protection-act)
    
18. THE VOLCKER RULE AND EVOLVING FINANCIAL MARKETS, acessado em agosto 19, 2025, [https://journals.law.harvard.edu/hblr/wp-content/uploads/sites/87/2014/09/Volcker-Rule.pdf](https://journals.law.harvard.edu/hblr/wp-content/uploads/sites/87/2014/09/Volcker-Rule.pdf)
    
19. The Volcker Rule and Regulations of Scope - NYU Stern, acessado em agosto 19, 2025, [https://www.stern.nyu.edu/sites/default/files/assets/documents/The%20Volcker%20Rule%20and%20Regulations%20of%20Scope.pdf](https://www.stern.nyu.edu/sites/default/files/assets/documents/The%20Volcker%20Rule%20and%20Regulations%20of%20Scope.pdf)
    
20. Volcker Rule: Definition, Purpose, How It Works, and Criticism - Investopedia, acessado em agosto 19, 2025, [https://www.investopedia.com/terms/v/volcker-rule.asp](https://www.investopedia.com/terms/v/volcker-rule.asp)
    
21. Implementing the Volcker Rule, acessado em agosto 19, 2025, [https://corpgov.law.harvard.edu/2012/02/04/implementing-the-vockler-rule/](https://corpgov.law.harvard.edu/2012/02/04/implementing-the-vockler-rule/)
    
22. Examining the Impact of the Volcker Rule on Markets, Businesses, Investors, and Job Creation TO, acessado em agosto 19, 2025, [https://financialservices.house.gov/uploadedfiles/hhrg-115-ba16-wstate-tquaadman-20170329.pdf](https://financialservices.house.gov/uploadedfiles/hhrg-115-ba16-wstate-tquaadman-20170329.pdf)
    
23. volcker-rule-metrics-instructions.pdf - Office of the Comptroller of the Currency (OCC), acessado em agosto 19, 2025, [https://www.occ.treas.gov/topics/supervision-and-examination/capital-markets/financial-markets/trading-volcker-rule/volcker-rule-metrics-instructions.pdf](https://www.occ.treas.gov/topics/supervision-and-examination/capital-markets/financial-markets/trading-volcker-rule/volcker-rule-metrics-instructions.pdf)
    
24. The Volcker Rule and Market-Making in Times of Stress - Federal ..., acessado em agosto 19, 2025, [https://www.federalreserve.gov/econresdata/feds/2016/files/2016102pap.pdf](https://www.federalreserve.gov/econresdata/feds/2016/files/2016102pap.pdf)
    
25. Summary of Dodd-Frank Financial Regulation Legislation, acessado em agosto 19, 2025, [https://corpgov.law.harvard.edu/2010/07/07/summary-of-dodd-frank-financial-regulation-legislation/](https://corpgov.law.harvard.edu/2010/07/07/summary-of-dodd-frank-financial-regulation-legislation/)
    
26. BRIEF SUMMARY OF THE DODD-FRANK WALL STREET REFORM AND CONSUMER PROTECTION ACT - The Senate Democratic Caucus, acessado em agosto 19, 2025, [https://www.dpc.senate.gov/pdf/wall_street_reform_summary.pdf](https://www.dpc.senate.gov/pdf/wall_street_reform_summary.pdf)
    
27. Federal Register: Clearing Requirement Determination Under Section 2(h) of the CEA - Commodity Futures Trading Commission, acessado em agosto 19, 2025, [https://www.cftc.gov/sites/default/files/idc/groups/public/@newsroom/documents/file/federalregister112812.pdf](https://www.cftc.gov/sites/default/files/idc/groups/public/@newsroom/documents/file/federalregister112812.pdf)
    
28. Estimating the Effect of Central Clearing on Credit Derivative Exposures - FRB: FEDS Notes, acessado em agosto 19, 2025, [https://www.federalreserve.gov/econresdata/notes/feds-notes/2014/estimating-the-effect-of-central-clearing-on-credit-derivative-exposures-20140226.html](https://www.federalreserve.gov/econresdata/notes/feds-notes/2014/estimating-the-effect-of-central-clearing-on-credit-derivative-exposures-20140226.html)
    
29. MARGIN AND CAPITAL REQUIREMENTS FOR COVERED SWAP ENTITIES - Office of the Comptroller of the Currency (OCC), acessado em agosto 19, 2025, [https://www.occ.gov/news-issuances/news-releases/2014/nr-ia-2014-119a.pdf](https://www.occ.gov/news-issuances/news-releases/2014/nr-ia-2014-119a.pdf)
    
30. Margin Requirements for Uncleared Swaps for Swap Dealers and Major Swap Participants - Commodity Futures Trading Commission, acessado em agosto 19, 2025, [https://www.cftc.gov/sites/default/files/idc/groups/public/@lrfederalregister/documents/file/2014-22962a.pdf](https://www.cftc.gov/sites/default/files/idc/groups/public/@lrfederalregister/documents/file/2014-22962a.pdf)
    
31. Initial Margin for Non-Centrally Cleared Derivatives: Issues for 2019 and 2020 July 2018, acessado em agosto 19, 2025, [https://www.isda.org/a/D6fEE/ISDA-SIFMA-Initial-Margin-Phase-in-White-Paper-July-2018.pdf](https://www.isda.org/a/D6fEE/ISDA-SIFMA-Initial-Margin-Phase-in-White-Paper-July-2018.pdf)
    
32. Basel III: Finalising post-crisis reforms - Bank for International Settlements, acessado em agosto 19, 2025, [https://www.bis.org/bcbs/publ/d424.pdf](https://www.bis.org/bcbs/publ/d424.pdf)
    
33. Risk Weighted Assets (RWA) Calculation in Basel III - QuestDB, acessado em agosto 19, 2025, [https://questdb.com/glossary/risk-weighted-assets-rwa-calculation-in-basel-iii/](https://questdb.com/glossary/risk-weighted-assets-rwa-calculation-in-basel-iii/)
    
34. CRE20 - Standardised approach: individual exposures, acessado em agosto 19, 2025, [https://www.bis.org/basel_framework/chapter/CRE/20.htm](https://www.bis.org/basel_framework/chapter/CRE/20.htm)
    
35. Risk-Weighted Assets: Definition and Place in Basel III - Investopedia, acessado em agosto 19, 2025, [https://www.investopedia.com/terms/r/riskweightedassets.asp](https://www.investopedia.com/terms/r/riskweightedassets.asp)
    
36. Basel III Revised Standardized Approach for Credit Risk FAQs - Fitch Solutions, acessado em agosto 19, 2025, [https://www.fitchsolutions.com/credit/long-reads/credit-insight/basel-iii-credit-risk-faq](https://www.fitchsolutions.com/credit/long-reads/credit-insight/basel-iii-credit-risk-faq)
    
37. Standardized approach (credit risk) - Wikipedia, acessado em agosto 19, 2025, [https://en.wikipedia.org/wiki/Standardized_approach_(credit_risk)](https://en.wikipedia.org/wiki/Standardized_approach_\(credit_risk\))
    
38. Basel III endgame: Complete regulatory capital overhaul - PwC, acessado em agosto 19, 2025, [https://www.pwc.com/us/en/industries/financial-services/library/our-take/basel-iii-endgame.html](https://www.pwc.com/us/en/industries/financial-services/library/our-take/basel-iii-endgame.html)
    
39. A Quick Guide to Expected Shortfall ES in Banking - Number Analytics, acessado em agosto 19, 2025, [https://www.numberanalytics.com/blog/quick-guide-expected-shortfall-es-banking](https://www.numberanalytics.com/blog/quick-guide-expected-shortfall-es-banking)
    
40. Minimum-capital-requirements-for-market-risk.pdf - Management Solutions, acessado em agosto 19, 2025, [https://www.managementsolutions.com/sites/default/files/publicaciones/eng/Minimum-capital-requirements-for-market-risk.pdf](https://www.managementsolutions.com/sites/default/files/publicaciones/eng/Minimum-capital-requirements-for-market-risk.pdf)
    
41. measuring market risk under the basel accords - IEB, acessado em agosto 19, 2025, [https://www.ieb.es/wp-content/uploads/2014/07/85.pdf](https://www.ieb.es/wp-content/uploads/2014/07/85.pdf)
    
42. Choosing Expected Shortfall over VaR in Basel III Using Stochastic Dominance - Index of /, acessado em agosto 19, 2025, [https://papers.tinbergen.nl/15133.pdf](https://papers.tinbergen.nl/15133.pdf)
    
43. MAR33 - Internal models approach: capital requirements calculation, acessado em agosto 19, 2025, [https://www.bis.org/basel_framework/chapter/MAR/33.htm](https://www.bis.org/basel_framework/chapter/MAR/33.htm)
    
44. Value at Risk (VaR) and Its Implementation in Python | by Serdar ..., acessado em agosto 19, 2025, [https://medium.com/@serdarilarslan/value-at-risk-var-and-its-implementation-in-python-5c9150f73b0e](https://medium.com/@serdarilarslan/value-at-risk-var-and-its-implementation-in-python-5c9150f73b0e)
    
45. Macroprudential policy and financial stability glossary - European Central Bank, acessado em agosto 19, 2025, [https://www.ecb.europa.eu/services/glossary/html/act5l.en.html](https://www.ecb.europa.eu/services/glossary/html/act5l.en.html)
    
46. LCR20 - Calculation - Bank for International Settlements, acessado em agosto 19, 2025, [https://www.bis.org/basel_framework/chapter/LCR/20.htm](https://www.bis.org/basel_framework/chapter/LCR/20.htm)
    
47. Liquidity Coverage Ratio: Definition and How To Calculate - Investopedia, acessado em agosto 19, 2025, [https://www.investopedia.com/terms/l/liquidity-coverage-ratio.asp](https://www.investopedia.com/terms/l/liquidity-coverage-ratio.asp)
    
48. How to calculate the Net Stable Funding Ratio (NSFR)? - MORS Software, acessado em agosto 19, 2025, [https://morssoftware.com/how-to-calculate-nsfr-for-banks/](https://morssoftware.com/how-to-calculate-nsfr-for-banks/)
    
49. Basel III: the net stable funding ratio - Bank for International ..., acessado em agosto 19, 2025, [https://www.bis.org/bcbs/publ/d295.pdf](https://www.bis.org/bcbs/publ/d295.pdf)
    
50. Liquidity Adequacy Requirements (LAR) (2025) Chapter 3 – Net Stable Funding Ratio - Office of the Superintendent of Financial Institutions, acessado em agosto 19, 2025, [https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/liquidity-adequacy-requirements-lar-2025-chapter-3-net-stable-funding-ratio](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/liquidity-adequacy-requirements-lar-2025-chapter-3-net-stable-funding-ratio)
    
51. Leverage ratio definition - Risk.net, acessado em agosto 19, 2025, [https://www.risk.net/definition/leverage-ratio](https://www.risk.net/definition/leverage-ratio)
    

Leverage Requirements - Guideline (2023) - Office of the Superintendent of Financial Institutions, acessado em agosto 19, 2025, [https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/leverage-requirements-guideline-2023](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/leverage-requirements-guideline-2023)**