# Chapter 4.3: Credit Default Swaps: Pricing and Modeling

Credit derivatives represent a significant innovation in modern finance, allowing for the isolation and transfer of credit risk. Among these, the Credit Default Swap (CDS) stands out as the most prominent and widely utilized instrument. Originally conceived as a tool for banks to manage loan portfolio risk, the CDS evolved into a versatile instrument for hedging, speculation, and arbitrage, playing a central and controversial role in the global financial landscape. This chapter delves into the mechanics, pricing theory, and practical implementation of single-name CDS contracts, providing the quantitative analyst with the foundational knowledge required to model and value these critical instruments.

## The Architecture of a Credit Default Swap

At its core, a CDS is a bilateral financial contract designed to transfer the credit risk of a particular company or government—the "reference entity"—from one party to another.1 It functions as a form of financial insurance, where one party, the

**protection buyer**, makes periodic payments to another party, the **protection seller**. In return, the seller agrees to make a large, contingent payment to the buyer if the reference entity experiences a predefined "credit event," such as a default.3

This mechanism allows an investor holding a risky bond to offload the risk of the bond issuer defaulting. However, the design of the CDS contract permits applications far beyond simple hedging, a fact that contributed to its explosive growth. Invented by a team at J.P. Morgan in 1994, the CDS market grew to an astonishing notional value of over $62 trillion by 2007, creating a web of interconnected risk that was a key factor in the 2008 financial crisis.2

### Mechanics of the Contract

CDS transactions are highly standardized, governed by documentation from the International Swaps and Derivatives Association (ISDA), which defines the key terms and operational procedures.6

- **Protection Buyer:** This party is hedging an existing credit exposure or speculating on a decline in the reference entity's credit quality. By buying a CDS, they are considered "long the CDS" and "short the credit" of the reference entity, as they profit if the entity's creditworthiness deteriorates.6
    
- **Protection Seller:** This party receives periodic payments (the spread) in exchange for assuming the credit risk. They are "short the CDS" and "long the credit," benefiting if the reference entity's credit quality remains stable or improves.6
    
- **Reference Entity:** The corporation or sovereign whose debt is the subject of the CDS. Crucially, the reference entity is not a party to the CDS contract itself.6
    
- **Reference Obligation:** A specific debt instrument, such as a senior unsecured bond, issued by the reference entity. This obligation is used to determine if a credit event has occurred and to calculate the settlement value.7
    
- **Notional Principal:** The face value of the protection being bought. For single-name CDS, this amount is typically in the range of $10–$20 million.7
    
- **Maturity:** The tenor of the contract, usually ranging from one to ten years. The five-year maturity is the most liquid and commonly quoted tenor.7
    
- **Credit Events:** These are the triggers for the protection payment. ISDA definitions typically include bankruptcy, failure to pay on debt obligations, and, for many contracts, a significant debt restructuring.7 Sovereign CDS contracts may also include credit events like repudiation or moratorium on debt payments.7
    

### The CDS Spread: The Price of Protection

The cost of the protection is known as the **CDS spread**. It is an annualized premium, quoted in basis points (1 bp = 0.01%), that the protection buyer pays on the notional amount.7 For instance, a 5-year CDS on a $10 million notional with a spread of 100 bps means the protection buyer pays $100,000 per year to the seller. These payments are typically made quarterly in arrears and continue until the contract matures or a credit event occurs, whichever comes first.6

The CDS spread is a powerful, real-time indicator of the market's perception of a reference entity's credit risk. A widening spread signals that the market believes the probability of default is increasing, making protection more expensive. Conversely, a tightening spread indicates improving creditworthiness and greater market confidence.3

### Settlement Mechanisms

If a credit event is triggered, the contract is settled. The method of settlement determines how the protection buyer is compensated for the loss.

- **Physical Settlement:** The protection buyer delivers a defaulted bond of the reference entity to the protection seller. In return, the seller pays the buyer the full notional principal of the CDS contract. This was historically a common method but is less so today.6
    
- Cash Settlement: This is the standard method in modern markets. Following a credit event, ISDA oversees a formal auction process involving major dealers to determine the final market price of the defaulted debt. This price, expressed as a percentage of par, represents the recovery rate (R). The protection seller then makes a cash payment to the buyer calculated as:
    
    $$Payout=Notional Principal×(1−R)$$
    
    This amount is the Loss Given Default (LGD) on the notional.6
    

### CDS vs. Insurance: A Critical Distinction

While often described as insurance, a CDS contract has fundamental differences that are critical for a quantitative analyst to understand. These differences are what transform the instrument from a simple hedging tool into a vehicle for complex financial strategies.

The most profound distinction lies in the concept of "insurable interest." To buy a traditional insurance policy, one must own the asset being insured and stand to suffer a direct financial loss if it is damaged or destroyed. A CDS contract has no such requirement.6 An investor can buy protection on a company's debt without owning any of that debt. This gives rise to the

**"naked" CDS**, where the contract is a pure speculative bet on the creditworthiness of the reference entity. The majority of the CDS market is composed of such naked positions.7

This feature has systemic implications. Because the total notional value of CDS contracts referencing an entity is not constrained by the actual amount of its outstanding debt, a single default can trigger a payment cascade far exceeding the economic loss of the initial default. This potential for risk magnification, rather than distribution, was a key contributor to the severity of the 2008 financial crisis, as exemplified by the case of AIG, which had sold massive amounts of protection on mortgage-backed securities.1

The table below summarizes the key distinctions.

|Feature|Credit Default Swap (CDS)|Traditional Insurance|
|---|---|---|
|**Insurable Interest**|Not required. Can be used for speculation ("naked" CDS).|Required. Must own the asset and suffer a direct loss.|
|**Regulation**|Historically OTC. Now subject to clearing and reporting mandates (e.g., Dodd-Frank Act).1|Heavily regulated by dedicated insurance commissions.|
|**Tradability**|Highly tradable financial instrument. Can be bought, sold, or unwound at market value.6|Policy is generally not tradable.|
|**Payout Calculation**|Standardized, market-wide payout based on an auction-determined recovery rate.7|Payout is based on the actual, specific loss incurred by the policyholder.|

### Applications: Hedging, Speculation, and Arbitrage

The unique structure of the CDS allows for three primary uses in financial markets:

1. **Hedging:** This is the instrument's original purpose. A bank holding a large loan to a corporation can buy a CDS to protect against the borrower defaulting, thereby removing the credit risk from its balance sheet without having to sell the loan and potentially damage its client relationship.2
    
2. **Speculation:** An investor who believes a company is in financial trouble can buy CDS protection to profit from a default or a widening of its CDS spread. Conversely, an investor with a positive view on a company can sell CDS protection, collecting the premium payments as income, effectively taking a long position on the company's credit.2
    
3. **Arbitrage:** Quants can engage in capital structure arbitrage by looking for pricing discrepancies between an entity's different securities. For example, if a company's CDS spread implies a much higher probability of default than its bond yield spread or stock price suggests, an arbitrageur might construct a trade to profit from the eventual convergence of these prices.4
    

## The Mathematical Foundation of CDS Pricing

The valuation of a CDS contract is grounded in the no-arbitrage principle, which states that at inception, the contract should have a net present value (NPV) of zero for both the buyer and the seller.12 This equilibrium is achieved by setting the CDS spread,

s, to a level that equates the present value of the protection buyer's expected payments (the premium leg) with the present value of the protection seller's expected contingent payout (the protection leg).8

$$PV(Premium Leg)=PV(Protection Leg)$$

While market forces of supply and demand determine the observable CDS spread, quantitative models are indispensable for several tasks:

- Deconstructing a market spread to imply the underlying market-perceived probability of default.12
    
- Pricing non-standard or bespoke CDS contracts.
    
- Calculating the daily mark-to-market (MTM) value of existing positions for P&L and risk management.12
    

### Modeling Default: Hazard Rates and Survival Probability

The industry-standard approach for modeling default is the **reduced-form model**, also known as an intensity-based model.12 Unlike structural models (e.g., Merton model), which link default to a company's asset value falling below a certain threshold, reduced-form models treat the time of default,

τ, as an unpredictable event, akin to a "surprise." The arrival of this event is governed by a stochastic process.14

The core parameter of this model is the **hazard rate**, λ(t), also called the default intensity. It represents the instantaneous probability of default in an infinitesimally small time interval, dt, conditional on the entity having survived up to time t.14

$$P(t≤τ<t+dt∣τ≥t)=λ(t)dt$$

From the hazard rate, we can derive the **survival probability**, S(t), which is the probability that the reference entity will _not_ default before time t. For a constant hazard rate λ, the relationship is a simple exponential decay 16:

$$S(t)=e^{−λt}$$

If the hazard rate is assumed to be a deterministic function of time, λ(s), the survival probability is given by the integral 15:

$$S(t)=exp(−∫_0^t​λ(s)ds)$$

The cumulative probability of default by time T is simply PD(T)=1−S(T), and the marginal probability of default between times T1​ and T2​ is S(T1​)−S(T2​).12

This separation of the default process (governed by λ) from the risk-free interest rate process is a powerful modeling choice. The standard reduced-form model explicitly assumes these two processes are independent.12 This allows practitioners to use a standard risk-free yield curve (derived from government bonds or interest rate swaps) for discounting cash flows, and then solve for the credit-specific component—the hazard rate curve—separately. This simplification makes the model highly tractable. However, it is also a key source of model risk. In reality, credit risk and interest rates are often correlated, a phenomenon known as "wrong-way risk".9 For example, a sharp rise in interest rates can trigger economic distress and lead to a wave of defaults. More advanced models, which are beyond the scope of this intermediate text, address this by modeling the joint stochastic evolution of interest rates and hazard rates, for instance by using a Cox-Ingersoll-Ross (CIR) process for both.19 For the developing quant, recognizing that this independence is a deliberate modeling

_choice_ with significant consequences is a crucial step in their education.

#### Python Example: Calculating a Survival Probability Curve

The following Python code demonstrates how to compute a survival probability curve from a given piecewise-constant hazard rate curve.



```Python
import numpy as np
import pandas as pd

def calculate_survival_probability(hazard_rates, tenors):
    """
    Calculates survival probabilities from a piecewise-constant hazard rate curve.

    Args:
        hazard_rates (list): List of constant hazard rates for each period.
        tenors (list): List of tenor endpoints in years (e.g., ).

    Returns:
        pandas.DataFrame: A DataFrame with tenors and survival probabilities.
    """
    # Ensure tenors start from 0
    full_tenors =  + tenors
    
    survival_probs = [1.0]  # Survival probability at time 0 is 1
    
    for i in range(len(tenors)):
        # Time interval for this segment
        dt = full_tenors[i+1] - full_tenors[i]
        hazard_rate = hazard_rates[i]
        
        # Survival probability at the end of this segment is the survival from the
        # previous segment decayed by the current hazard rate.
        # S(t_i) = S(t_{i-1}) * exp(-lambda_i * (t_i - t_{i-1}))
        prev_survival_prob = survival_probs[-1]
        new_survival_prob = prev_survival_prob * np.exp(-hazard_rate * dt)
        survival_probs.append(new_survival_prob)
        
    # Remove the initial S(0)=1 for the final output table
    return pd.DataFrame({'Tenor (Years)': tenors, 'Survival Probability': survival_probs[1:]})

# Example usage:
tenors_example = 
# Hypothetical piecewise hazard rates (e.g., 1% for year 1, 1.5% for year 2, etc.)
hazard_rates_example = [0.01, 0.015, 0.02, 0.022, 0.025, 0.028]

survival_curve = calculate_survival_probability(hazard_rates_example, tenors_example)
print("--- Example Survival Probability Curve ---")
print(survival_curve)
```

### Valuing the Protection Leg (Default Leg)

The protection leg is the contingent payment of the Loss Given Default (LGD), which equals (1−R), where R is the recovery rate.12 Its present value is the sum of all possible discounted payouts, weighted by their respective probabilities of occurrence.

In a continuous-time framework, the PV is an integral of the discounted expected loss over the contract's life 15:

![[Pasted image 20250702102036.png]]

where tV​ is the valuation date, T is the maturity, Z(tV​,t) is the risk-free discount factor from t back to tV​, and Q(tV​,t) is the survival probability from tV​ to t.

For practical computation, this integral is approximated as a sum over discrete time steps (e.g., quarterly or monthly).8 The value is the sum of the discounted LGD for each period, weighted by the marginal probability of default in that period.

![[Pasted image 20250702102045.png]]

Here, N is the number of periods, Z(0,ti​) is the discount factor for the i-th period's payment date, and \$$ is the probability of defaulting between time ti−1​ and ti​.

### Valuing the Premium Leg

The premium leg consists of two components: the regular, fixed premium payments and the final accrued premium payment due upon default.

1. **PV of Periodic Payments:** The value is the sum of all future spread payments, with each payment discounted to present value and weighted by the probability that the reference entity survives long enough for the payment to be made.8
    
    ![[Pasted image 20250702102113.png]]
    
    where s is the annual CDS spread, αi​ is the day-count fraction for payment period i (e.g., approximately 0.25 for quarterly payments), Z(0,ti​) is the discount factor, and S(ti​) is the survival probability to the payment date ti​.
    
2. **PV of Accrued Premium:** When a default occurs between payment dates, the protection buyer typically owes the premium that has accrued from the last payment date up to the time of default. A common modeling simplification is to assume that default, if it occurs within a period, happens on average at the midpoint of that period.8
    
   ![[Pasted image 20250702102122.png]]
    
    where ti,mid​ is the midpoint of the period.
    

The total present value of the premium leg is the sum of these two components:

PVPremium Leg​=PVPremiums​+PVAccrual​

By setting PVPremium Leg​=PVProtection​, we can solve for the fair spread s.

## From Theory to Practice: Implementation and Calibration

With the theoretical framework established, we now turn to the practical steps of pricing a CDS, which involve calibrating the model to market data and implementing the valuation formulas.

### The Credit Curve and Bootstrapping

Just as a yield curve shows risk-free rates across different maturities, a **credit curve** shows the market CDS spreads for a single reference entity across a range of standard tenors (e.g., 1Y, 3Y, 5Y, 7Y, 10Y).11 This curve provides the market prices we must calibrate our model to.

The process of extracting the unobservable hazard rates from the observable market CDS spreads is called **bootstrapping**.12 It is an iterative process that builds a piecewise-constant hazard rate curve, ensuring that our pricing model reproduces the market price at each liquid tenor. The

`cdstools` Python package, for instance, explicitly implements this functionality based on a well-known JP Morgan model.20

The bootstrapping procedure is as follows:

1. **Start with the shortest tenor** (e.g., 1Y CDS). Assume the hazard rate, λ0−1Y​, is constant over this first year. Use a root-finding algorithm to solve for the value of λ0−1Y​ that makes the PV of the premium leg equal to the PV of the protection leg for the observed 1Y market spread.
    
2. **Move to the next tenor** (e.g., 3Y CDS). We now know λ0−1Y​. We assume the hazard rate is constant, but at a new level λ1−3Y​, for the period between year 1 and year 3. Using the observed 3Y market spread, we solve for the value of λ1−3Y​ that makes the pricing equation hold for the 3Y contract.
    
3. **Repeat for all tenors.** This process is continued out along the credit curve, building a complete term structure of forward hazard rates.15
    

### Python Implementation: Building a CDS Pricer from Scratch

To solidify these concepts, we will outline the construction of a basic CDS pricer in Python. This exercise is invaluable for understanding the model's mechanics. The implementation will use `NumPy` for numerical operations and `SciPy` for interpolation and root-finding. The logic is inspired by public examples and tutorials.21

The choice to build a pricer from scratch versus using a pre-built library represents a fundamental trade-off for a quantitative analyst. The from-scratch approach provides maximum transparency and customizability, which is essential for learning, debugging, and implementing novel model features (such as the advanced concepts discussed below). However, this path is often slow and risks missing subtle but crucial market conventions (e.g., specific day-count conventions, holiday calendars, IMM roll dates) that are embedded in industry-standard models.23 Libraries like QuantLib or the

`isda` Python wrapper encapsulate these conventions, ensuring consistency with market benchmarks and providing high-performance calculations by calling compiled C++ code.24 A professional quant must master both approaches: the deep understanding from building models themselves, and the practical wisdom to use high-performance, standardized libraries for production systems.



```Python
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

# --- Step 1: Yield Curve and Discount Factors ---
# In a real application, you would build this from Treasury or swap data.
# For this example, we assume a flat risk-free rate for simplicity.
RISK_FREE_RATE = 0.04

def get_discount_factor(t):
    """Calculates discount factor for a given time t."""
    return np.exp(-RISK_FREE_RATE * t)

# --- Step 2 & 3: Combined Bootstrapper and Pricer ---
def bootstrap_and_price(market_tenors, market_spreads, recovery_rate, pricing_tenor):
    """
    Bootstraps a hazard rate curve from market CDS spreads and prices a CDS.
    
    Args:
        market_tenors (list): Tenors of market CDS quotes (e.g., ).
        market_spreads (list): Corresponding market CDS spreads (as decimals).
        recovery_rate (float): Assumed recovery rate.
        pricing_tenor (float): The tenor of the CDS to be priced.
        
    Returns:
        tuple: (fair_spread, hazard_rates_curve)
    """
    
    hazard_rates =
    
    # Define the pricing function for a single CDS tenor
    def price_cds_pv_difference(hazard_rate_guess, tenor, known_spread, previous_tenors, previous_hrs):
        # Combine previous hazard rates with the current guess
        current_hrs = previous_hrs + [hazard_rate_guess]
        current_tenors = previous_tenors + [tenor]
        
        # Discretize time into quarterly steps
        time_steps = np.arange(0.25, tenor + 0.25, 0.25)
        
        # Build the full hazard rate and survival probability curves
        hr_interp = interp1d( + current_tenors, [current_hrs] + current_hrs, 
                             kind='previous', fill_value="extrapolate")
        
        integral_lambda_dt = np.array([hr_interp(t) * 0.25 for t in np.arange(0.25, tenor + 0.25, 0.25)]).cumsum()
        survival_probs = np.exp(-integral_lambda_dt)
        
        # Add S(0) = 1 for calculation
        survival_probs_with_zero = np.insert(survival_probs, 0, 1.0)
        
        # Calculate PV of Protection Leg
        marginal_defaults = survival_probs_with_zero[:-1] - survival_probs_with_zero[1:]
        discount_factors = get_discount_factor(time_steps)
        pv_protection = np.sum((1 - recovery_rate) * marginal_defaults * discount_factors)
        
        # Calculate PV of Premium Leg
        day_count_fraction = 0.25
        pv_premiums = np.sum(known_spread * day_count_fraction * survival_probs * discount_factors)
        
        # For simplicity, we ignore accrued interest in this basic bootstrapper
        # A full implementation would include it as in Section 2.4.
        
        return pv_premiums - pv_protection

    # --- Bootstrapping Loop ---
    for i, tenor in enumerate(market_tenors):
        spread = market_spreads[i]
        
        # Objective function for the root finder
        obj_func = lambda hr: price_cds_pv_difference(hr, tenor, spread, market_tenors[:i], hazard_rates)
        
        # Find the hazard rate that makes the PV difference zero
        # Initial guess for hazard rate: s / (1 - R)
        initial_guess = spread / (1 - recovery_rate)
        sol = root_scalar(obj_func, bracket=[0.0001, 1.0], x0=initial_guess, method='brentq')
        hazard_rates.append(sol.root)

    hazard_rate_curve = pd.DataFrame({
        'Tenor': market_tenors,
        'Implied Hazard Rate': hazard_rates
    })

    # --- Pricing the target CDS ---
    # Now use the bootstrapped curve to price the target tenor
    hr_interp_final = interp1d( + market_tenors, [hazard_rates] + hazard_rates, 
                               kind='previous', fill_value="extrapolate")
    
    time_steps_price = np.arange(0.25, pricing_tenor + 0.25, 0.25)
    
    integral_lambda_dt_price = np.array([hr_interp_final(t) * 0.25 for t in np.arange(0.25, pricing_tenor + 0.25, 0.25)]).cumsum()
    survival_probs_price = np.exp(-integral_lambda_dt_price)
    survival_probs_price_with_zero = np.insert(survival_probs_price, 0, 1.0)
    
    marginal_defaults_price = survival_probs_price_with_zero[:-1] - survival_probs_price_with_zero[1:]
    discount_factors_price = get_discount_factor(time_steps_price)
    
    # Risky Annuity (Denominator for spread calculation)
    risky_annuity = np.sum(0.25 * survival_probs_price * discount_factors_price)
    
    # Protection Leg PV (Numerator)
    protection_pv = np.sum((1 - recovery_rate) * marginal_defaults_price * discount_factors_price)
    
    fair_spread = protection_pv / risky_annuity
    
    return fair_spread, hazard_rate_curve

# Example: Bootstrap from market data and price a 5-year CDS
market_tenors_in = 
market_spreads_in = [0.006, 0.007, 0.0085, 0.011, 0.012, 0.013] # 60, 70, 85, 110, 120, 130 bps
rec_rate_in = 0.4
target_tenor = 5.0

fair_spread_5y, bootstrapped_hr_curve = bootstrap_and_price(
    market_tenors_in, market_spreads_in, rec_rate_in, target_tenor
)

print("--- Bootstrapped Hazard Rate Curve ---")
print(bootstrapped_hr_curve)
print(f"\nCalculated fair spread for a {target_tenor}-year CDS: {fair_spread_5y * 10000:.2f} bps")
# Note: The result should be very close to the input market spread of 110 bps, validating the model.
```

### Advanced Modeling Concepts (A Brief Overview)

The standard model provides a robust foundation, but professional quants often encounter situations that require more sophisticated approaches. These include:

- **Stochastic Intensity:** In reality, hazard rates are not deterministic. Advanced models treat λ(t) as a stochastic process itself, often using mean-reverting models like the Cox-Ingersoll-Ross (CIR) process, which is also used for interest rate modeling.19
    
- **Default Contagion and Clustering:** Defaults are not always independent events, especially during financial crises. The default of one entity can increase the likelihood of default for others in the same sector or with similar risk profiles. This "contagion" effect can be modeled using self-exciting point processes, such as the **Hawkes process**, where the default intensity jumps upwards following a default event.27
    
- **Counterparty Risk:** A CDS contract creates a new risk: the risk that the protection seller might default and be unable to pay. Valuing this requires pricing a "bilateral" contract, where the default probabilities of both the reference entity and the counterparty are considered simultaneously.1
    

### Industry-Standard Tools: An Introduction to QuantLib

For production-level work, analysts typically rely on specialized libraries like **QuantLib**, a free, open-source C++ library with Python bindings that is a benchmark in the industry.26 It provides pre-built, tested, and highly efficient tools for pricing a vast array of financial instruments, including CDS.

Using QuantLib abstracts away many of the low-level implementation details (like date calculations and calendar adjustments) and provides access to robust pricing engines, including one that is compliant with the ISDA Standard Model.29

The following snippet illustrates the general workflow for pricing a CDS in QuantLib-Python, highlighting its high-level, object-oriented approach.



```Python
import QuantLib as ql

# --- 1. Setup Dates and Market Conventions ---
valuation_date = ql.Date(26, 6, 2025)
ql.Settings.instance().evaluationDate = valuation_date
calendar = ql.UnitedStates()
day_counter = ql.Actual365Fixed()

# --- 2. Build Risk-Free Yield Curve ---
# (Using flat forward for simplicity)
risk_free_rate = 0.04
yield_curve = ql.FlatForward(valuation_date, risk_free_rate, day_counter)
yield_curve_handle = ql.YieldTermStructureHandle(yield_curve)

# --- 3. Build Default Probability Curve (from market CDS quotes) ---
recovery_rate = 0.4
cds_tenors = [ql.Period(y, ql.Years) for y in ]
cds_spreads = [s * 10000 for s in [0.006, 0.007, 0.0085, 0.011, 0.012, 0.013]] # in bps

# Create helper objects for bootstrapping
cds_helpers =

# Bootstrap the hazard rate curve
hazard_rate_curve = ql.PiecewiseDefaultCurve_HazardRate(
    valuation_date, cds_helpers, day_counter
)
hazard_rate_handle = ql.DefaultProbabilityTermStructureHandle(hazard_rate_curve)

# --- 4. Define the CDS Instrument ---
schedule = ql.MakeSchedule(from_=valuation_date, 
                           to=valuation_date + ql.Period(5, ql.Years),
                           tenor=ql.Period(ql.Quarterly),
                           calendar=calendar,
                           convention=ql.Following,
                           rule=ql.DateGeneration.TwentiethIMM)

cds_to_price = ql.CreditDefaultSwap(ql.Protection.Buyer, 10000000, # Notional
                                    0.0110, # 110 bps running spread
                                    schedule, ql.Following, day_counter)

# --- 5. Set Pricing Engine and Calculate ---
engine = ql.MidPointCdsEngine(hazard_rate_handle, recovery_rate, yield_curve_handle)
cds_to_price.setPricingEngine(engine)

# --- 6. Get Results ---
npv = cds_to_price.NPV()
fair_spread_bps = cds_to_price.fairSpread() * 10000

print(f"--- QuantLib CDS Pricing Results ---")
print(f"Mark-to-Market NPV of the CDS: ${npv:,.2f}")
print(f"Fair Spread for a 5-Year CDS: {fair_spread_bps:.2f} bps")
```

## Capstone Project: Hedging Corporate Bond Risk with a CDS

This capstone project applies the concepts discussed to a realistic financial scenario. We will analyze the process of hedging a corporate bond position using a single-name CDS, perform the necessary quantitative analysis, and evaluate the outcome.

### Scenario

An asset management fund holds a **$10 million** position in a corporate bond issued by **Ford Motor Company**. Recent analysis from rating agencies like S&P Global has affirmed Ford's investment-grade rating of 'BBB-' but revised the outlook to negative, citing concerns over profitability and cost-reduction progress.30 To mitigate the risk of credit deterioration, the portfolio manager decides to hedge the position by purchasing a

**5-year CDS** on Ford. The objective is to neutralize the specific credit risk of Ford, leaving the fund primarily exposed to general interest rate movements.

### Data Assembly

A real-world quantitative project begins with assembling accurate market data. The following table consolidates the necessary inputs for our analysis.

|Data Point|Value / Source|Rationale|
|---|---|---|
|**Valuation Date**|June 26, 2025|A specific, recent date for all calculations.|
|**Reference Entity**|Ford Motor Company|The company whose credit risk we are hedging.33|
|**Hedge Notional**|$10,000,000|The size of the bond position to be hedged.|
|**Hedge Tenor**|5 years|The desired length of the protection.7|
|**Recovery Rate**|40%|A standard market assumption for senior unsecured corporate debt, consistent with historical data.35|
|**Risk-Free Yield Curve**|US Treasury Rates (as of June 26, 2025) 37:|1Y: 3.96%, 2Y: 3.70%, 3Y: 3.68%, 5Y: 3.79%, 7Y: 4.00%, 10Y: 4.26%|Serves as the basis for risk-free discounting.|
|**Ford Market CDS Curve**|Hypothetical BBB- Auto Spreads (bps):<br><br>1Y: 70, 2Y: 85, 3Y: 100, 5Y: 120, 7Y: 135, 10Y: 150|Realistic spreads for a company with Ford's credit profile, based on index data 38 and single-name risk premium.|

### Project Tasks & Questions

1. **Construct the Risk-Free Discount Curve:** Using the provided US Treasury yield data, create a function that can provide a discount factor for any given future date via interpolation.
    
2. **Bootstrap Ford's Implied Hazard Rate Curve:** Implement the bootstrapping algorithm to derive Ford's piecewise-constant hazard rate curve from its market CDS spreads. Display the resulting term structure of hazard rates.
    
3. **Price the Hedge:** Calculate the annual cost of the 5-year CDS hedge. This involves computing the fair spread and the total annual premium payment on the $10 million notional.
    
4. **Analyze the Hedged Portfolio:** Create a table showing the projected annual cash flows for the next 5 years under a no-default scenario, comparing the bond coupons, the CDS premium payments, and the net cash flow of the combined position.
    
5. **Simulate a Default Scenario:** Assume Ford defaults exactly at the end of year 2. Calculate the total Profit & Loss (P&L) for an unhedged bond position versus the hedged position.
    
6. **Discussion of Residual Risks:** After implementing the hedge, what key risks remain in the portfolio?
    

### Full Python Solution

The following Python script provides a complete solution to the capstone project tasks.



```Python
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# --- 4.1 & 4.2: Data Assembly and Yield Curve Construction ---

# Market Data
VALUATION_DATE = pd.to_datetime('2025-06-26')
NOTIONAL = 10_000_000
RECOVERY_RATE = 0.4
HEDGE_TENOR = 5

# Risk-Free Curve Data
treasury_tenors = np.array()
treasury_yields = np.array([0.0396, 0.0370, 0.0368, 0.0379, 0.0400, 0.0426])

# Create an interpolation function for the risk-free rate
# We use linear interpolation on zero rates for simplicity.
# A more advanced approach would use log-linear on discount factors.
yield_curve_interp = interp1d(treasury_tenors, treasury_yields, kind='linear', fill_value='extrapolate')

def get_discount_factor(t):
    """Calculates discount factor for time t in years from valuation date."""
    rate = yield_curve_interp(t)
    return np.exp(-rate * t)

print("--- Task 1: Risk-Free Discount Curve ---")
print(f"Interpolated 4-year risk-free rate: {yield_curve_interp(4.0):.4%}")
print(f"Discount factor for 5 years: {get_discount_factor(5.0):.4f}\n")


# Ford Market CDS Data
ford_cds_tenors = np.array()
ford_cds_spreads = np.array([0.0070, 0.0085, 0.0100, 0.0120, 0.0135, 0.0150])

# --- 4.2: Bootstrap Ford's Hazard Rate Curve ---

def solve_hazard_rate(tenor, spread, prev_tenors, prev_hrs):
    """Function to be solved by the root finder for each bootstrapping step."""
    def pv_diff_for_hr(hr_guess):
        # Construct the full hazard rate curve for this iteration
        current_hrs = np.append(prev_hrs, hr_guess)
        current_tenors = np.append(prev_tenors, tenor)
        
        hr_func = interp1d(np.insert(current_tenors, 0, 0), 
                           np.insert(current_hrs, 0, current_hrs), 
                           kind='previous', fill_value='extrapolate')

        # Common calculations
        payment_dates = np.arange(0.25, tenor + 0.25, 0.25)
        survival_probs = np.exp(-hr_func(payment_dates) * payment_dates)
        
        # More precise survival prob calculation using piecewise integration
        integral_lambda = np.zeros_like(payment_dates)
        last_t = 0
        last_integral = 0
        for i, t in enumerate(payment_dates):
            # Find which hazard rate segment we are in
            hr_idx = np.searchsorted(current_tenors, t, side='right')
            
            integral = 0
            # Integrate up to the current payment date
            for j in range(hr_idx):
                start_t = current_tenors[j-1] if j > 0 else 0
                end_t = current_tenors[j]
                integral += current_hrs[j] * (end_t - start_t)
            
            # Add the final partial segment
            start_t_final = current_tenors[hr_idx-1] if hr_idx > 0 else 0
            integral += current_hrs[hr_idx] * (t - start_t_final)
            integral_lambda[i] = integral
            
        survival_probs = np.exp(-integral_lambda)
        
        survival_probs_with_zero = np.insert(survival_probs, 0, 1.0)
        marginal_defaults = survival_probs_with_zero[:-1] - survival_probs_with_zero[1:]
        discount_factors = np.array([get_discount_factor(t) for t in payment_dates])

        # Protection Leg PV
        protection_pv = np.sum((1 - RECOVERY_RATE) * marginal_defaults * discount_factors)
        
        # Premium Leg PV (including accrual)
        risky_annuity = np.sum(0.25 * survival_probs * discount_factors)
        accrual_pv_per_spread = np.sum(0.125 * marginal_defaults * discount_factors) # Assuming default at mid-point
        premium_leg_pv = spread * (risky_annuity + accrual_pv_per_spread)
        
        return premium_leg_pv - protection_pv

    # Use Brent's method for robust root finding
    initial_guess = spread / (1 - RECOVERY_RATE)
    hazard_rate = brentq(pv_diff_for_hr, a=0.0001, b=1.0, xtol=1e-8)
    return hazard_rate

bootstrapped_hrs =
for i, tenor in enumerate(ford_cds_tenors):
    hr = solve_hazard_rate(tenor, ford_cds_spreads[i], ford_cds_tenors[:i], bootstrapped_hrs)
    bootstrapped_hrs.append(hr)

ford_hazard_curve = pd.DataFrame({
    'Tenor (Years)': ford_cds_tenors,
    'Bootstrapped Hazard Rate': bootstrapped_hrs
})

print("--- Task 2: Bootstrapped Hazard Rate Curve for Ford Motor Co. ---")
print(ford_hazard_curve.to_string(formatters={'Bootstrapped Hazard Rate': '{:.4%}'.format}))
print("\n")


# --- 4.3: Price the Hedge ---
# The price is the market spread for a 5Y CDS
hedge_spread = ford_cds_spreads
annual_premium_payment = NOTIONAL * hedge_spread

print("--- Task 3: Price the 5-Year Hedge ---")
print(f"Fair Spread (Market Quote) for 5-Year CDS: {hedge_spread:.4%}")
print(f"Annual Premium Payment on $10M Notional: ${annual_premium_payment:,.2f}\n")


# --- 4.4: Analyze the Hedged Portfolio (No Default) ---
# Assuming a hypothetical Ford bond with a 5.5% coupon for cash flow analysis
bond_coupon_rate = 0.055
annual_bond_income = NOTIONAL * bond_coupon_rate
net_annual_cashflow = annual_bond_income - annual_premium_payment

cash_flow_data = {
    'Year': np.arange(1, HEDGE_TENOR + 1),
    'Bond Coupon Income': [annual_bond_income] * HEDGE_TENOR,
    'CDS Premium Outflow': [-annual_premium_payment] * HEDGE_TENOR,
    'Net Annual Cash Flow': [net_annual_cashflow] * HEDGE_TENOR
}
cash_flow_df = pd.DataFrame(cash_flow_data)

print("--- Task 4: Hedged Portfolio Cash Flow Analysis (No Default Scenario) ---")
print(cash_flow_df.to_string(formatters={col: '${:,.2f}'.format for col in cash_flow_df.columns if col!= 'Year'}))
print("\n")


# --- 4.5: Simulate a Default Scenario ---
default_time = 2 # End of year 2
premiums_paid = annual_premium_payment * default_time
loss_on_bond = NOTIONAL * (1 - RECOVERY_RATE)
cds_payout = NOTIONAL * (1 - RECOVERY_RATE)

pnl_unhedged = -loss_on_bond
pnl_hedged = cds_payout - premiums_paid

print("--- Task 5: Default Scenario Analysis (Default at Year 2) ---")
print(f"Loss on Unhedged Bond Position: ${pnl_unhedged:,.2f}")
print(f"Premiums Paid Before Default: ${premiums_paid:,.2f}")
print(f"CDS Payout Received: ${cds_payout:,.2f}")
print(f"Net P&L of Hedged Position (excluding bond coupons received): ${pnl_hedged - loss_on_bond:,.2f}")
print("Note: The hedged P&L is effectively just the cost of the premiums paid.\n")

# --- 4.6: Discussion of Residual Risks ---
print("--- Task 6: Discussion of Residual Risks ---")
print("Even with the CDS hedge in place, several risks remain:")
print("1. Counterparty Risk: The risk that the protection seller (the counterparty) defaults and cannot make the required payout. This risk was central to the 2008 crisis with AIG.")
print("2. Basis Risk: The risk of a mismatch between the hedge and the underlying asset. For example, the specific bond held by the fund might not be the 'cheapest-to-deliver' in the CDS settlement auction, leading to a recovery rate for the auction that differs from the recovery on the specific bond held.")
print("3. Liquidity Risk: In times of market stress, it may become difficult or expensive to unwind the CDS position before maturity if the fund's strategy changes.")
print("4. Interest Rate Risk: The hedge removes Ford's specific credit risk, but the fund still holds a bond exposed to moves in the general level of interest rates (duration risk).")
```

### Conclusion of the Project

The analysis demonstrates that a CDS can be an effective tool for isolating and neutralizing the credit risk of a specific corporate bond holding. In a default scenario, the payout from the CDS almost perfectly offsets the loss on the bond, with the net cost to the fund being the premiums paid for the protection.

However, the exercise also reveals a more profound truth about risk management. The hedge does not eliminate risk; it **transforms** it. The fund manager has exchanged a known, specific credit risk (Ford defaulting) for a new set of more complex and subtle risks. The primary new risk is **counterparty risk**—the possibility that the CDS seller defaults.1 Additionally,

**basis risk** arises from potential mismatches between the hedged bond and the broader pool of debt used to settle the CDS auction.39 The quant's role, therefore, extends beyond simply executing the hedge. It involves understanding, monitoring, and managing this new portfolio of risks. The CDS simplifies the fund's exposure by removing a large and easily understood risk, but the true work of financial engineering lies in managing the more nuanced risks that have been taken on in its place.

## References
**

1. credit default swap | Wex | US Law - Legal Information Institute, acessado em julho 2, 2025, [https://www.law.cornell.edu/wex/credit_default_swap](https://www.law.cornell.edu/wex/credit_default_swap)
    
2. Credit Default Swap: What It Is and How It Works - Investopedia, acessado em julho 2, 2025, [https://www.investopedia.com/terms/c/creditdefaultswap.asp](https://www.investopedia.com/terms/c/creditdefaultswap.asp)
    
3. What Are Credit Default Swaps (CDS) & How Do They Work ..., acessado em julho 2, 2025, [https://www.tastylive.com/concepts-strategies/credit-default-swap](https://www.tastylive.com/concepts-strategies/credit-default-swap)
    
4. Credit Default Swap - Defintion, How it Works, Risk - Corporate Finance Institute, acessado em julho 2, 2025, [https://corporatefinanceinstitute.com/resources/derivatives/credit-default-swap-cds/](https://corporatefinanceinstitute.com/resources/derivatives/credit-default-swap-cds/)
    
5. What does the CDS market imply for a U.S. default?; - Federal Reserve Bank of Chicago, acessado em julho 2, 2025, [https://www.chicagofed.org/-/media/publications/economic-perspectives/2023/ep2023-4.pdf?sc_lang=en](https://www.chicagofed.org/-/media/publications/economic-perspectives/2023/ep2023-4.pdf?sc_lang=en)
    
6. 3. Credit Default Swaps - Baruch MFE Program, acessado em julho 2, 2025, [https://mfe.baruch.cuny.edu/wp-content/uploads/2019/12/IRC_Lecture3_2019.pdf](https://mfe.baruch.cuny.edu/wp-content/uploads/2019/12/IRC_Lecture3_2019.pdf)
    
7. Credit default swap - Wikipedia, acessado em julho 2, 2025, [https://en.wikipedia.org/wiki/Credit_default_swap](https://en.wikipedia.org/wiki/Credit_default_swap)
    
8. Credit Default Swaps, acessado em julho 2, 2025, [http://www.princeton.edu/~markus/teaching/Eco467/10Lecture/CDS%20Presentation%20with%20References.pdf](http://www.princeton.edu/~markus/teaching/Eco467/10Lecture/CDS%20Presentation%20with%20References.pdf)
    
9. Credit Default Swap (CDS) Pricing - QuestDB, acessado em julho 2, 2025, [https://questdb.com/glossary/credit-default-swap-cds-pricing/](https://questdb.com/glossary/credit-default-swap-cds-pricing/)
    
10. Quick Look at CDS Spreads: A Finance Guide, acessado em julho 2, 2025, [https://www.numberanalytics.com/blog/quick-look-cds-spreads-finance-guide](https://www.numberanalytics.com/blog/quick-look-cds-spreads-finance-guide)
    
11. Credit Default Swaps - Learn Financial Modeling - Noble Desktop, acessado em julho 2, 2025, [https://www.nobledesktop.com/learn/financial-modeling/credit-default-swaps](https://www.nobledesktop.com/learn/financial-modeling/credit-default-swaps)
    
12. Single Name Credit Derivatives, acessado em julho 2, 2025, [https://didattica.unibocconi.it/mypage/dwload.php?nomefile=Lec_4_Credit_Default_Swaps20160221213301.pdf](https://didattica.unibocconi.it/mypage/dwload.php?nomefile=Lec_4_Credit_Default_Swaps20160221213301.pdf)
    
13. Reduced form models: Preliminaries, acessado em julho 2, 2025, [https://ifrogs.org/PDF/05_reduced_form_prelim.pdf](https://ifrogs.org/PDF/05_reduced_form_prelim.pdf)
    
14. Reduced Form (Intensity) Models, acessado em julho 2, 2025, [https://didattica.unibocconi.it/mypage/dwload.php?nomefile=Lecture_8_Intensity_Models20150302152822.pdf](https://didattica.unibocconi.it/mypage/dwload.php?nomefile=Lecture_8_Intensity_Models20150302152822.pdf)
    
15. Credit Curve Bootstrapping, acessado em julho 2, 2025, [https://cran.r-project.org/web/packages/credule/vignettes/credule.html](https://cran.r-project.org/web/packages/credule/vignettes/credule.html)
    
16. Quick Modern CDS Pricing Guide for Savvy Investors, acessado em julho 2, 2025, [https://www.numberanalytics.com/blog/quick-modern-cds-pricing-guide-savvy-investors](https://www.numberanalytics.com/blog/quick-modern-cds-pricing-guide-savvy-investors)
    
17. Hazard Rates and probability of survival - Forum | Bionic Turtle, acessado em julho 2, 2025, [https://forum.bionicturtle.com/threads/hazard-rates-and-probability-of-survival.9595/](https://forum.bionicturtle.com/threads/hazard-rates-and-probability-of-survival.9595/)
    
18. Survival Probability - Open Risk Manual, acessado em julho 2, 2025, [https://www.openriskmanual.org/wiki/Survival_Probability](https://www.openriskmanual.org/wiki/Survival_Probability)
    
19. Credit default swap pricing with counterparty risk in a reduced form model with a common jump process | Probability in the Engineering and Informational Sciences - Cambridge University Press, acessado em julho 2, 2025, [https://www.cambridge.org/core/journals/probability-in-the-engineering-and-informational-sciences/article/credit-default-swap-pricing-with-counterparty-risk-in-a-reduced-form-model-with-a-common-jump-process/ADF79CB813DE6DE27ED637A23A3A6586](https://www.cambridge.org/core/journals/probability-in-the-engineering-and-informational-sciences/article/credit-default-swap-pricing-with-counterparty-risk-in-a-reduced-form-model-with-a-common-jump-process/ADF79CB813DE6DE27ED637A23A3A6586)
    
20. cdstools·PyPI, acessado em julho 2, 2025, [https://pypi.org/project/cdstools/](https://pypi.org/project/cdstools/)
    
21. JazzikPeng/CDS_Pricing - GitHub, acessado em julho 2, 2025, [https://github.com/JazzikPeng/CDS_Pricing](https://github.com/JazzikPeng/CDS_Pricing)
    
22. CDS in Python; Extracting Israel Probability of Default implied by ..., acessado em julho 2, 2025, [https://medium.com/@polanitzer/cds-in-python-extracting-israel-probability-of-default-implied-by-israel-5-years-cds-spreads-ad732a3b4804](https://medium.com/@polanitzer/cds-in-python-extracting-israel-probability-of-default-implied-by-israel-5-years-cds-spreads-ad732a3b4804)
    
23. ISDA CDS Standard Model, acessado em julho 2, 2025, [https://www.cdsmodel.com/](https://www.cdsmodel.com/)
    
24. bakera1/CreditDefaultSwapPricer: Credit Default Swap Pricer - GitHub, acessado em julho 2, 2025, [https://github.com/bakera1/CreditDefaultSwapPricer](https://github.com/bakera1/CreditDefaultSwapPricer)
    
25. isda·PyPI, acessado em julho 2, 2025, [https://pypi.org/project/isda/1.0.9/](https://pypi.org/project/isda/1.0.9/)
    
26. QuantLib, a free/open-source library for quantitative finance, acessado em julho 2, 2025, [https://www.quantlib.org/](https://www.quantlib.org/)
    
27. Credit default swap pricing with counterparty risk in a reduced form model with Hawkes process - Taylor & Francis Online: Peer-reviewed Journals, acessado em julho 2, 2025, [https://www.tandfonline.com/doi/full/10.1080/03610926.2024.2349715](https://www.tandfonline.com/doi/full/10.1080/03610926.2024.2349715)
    
28. Credit default swap pricing with counterparty risk in a reduced form model with Hawkes process - IDEAS/RePEc, acessado em julho 2, 2025, [https://ideas.repec.org/a/taf/lstaxx/v54y2025i6p1813-1835.html](https://ideas.repec.org/a/taf/lstaxx/v54y2025i6p1813-1835.html)
    
29. Pricing Engines - QuantLib-Python - Read the Docs, acessado em julho 2, 2025, [https://quantlib-python-docs.readthedocs.io/en/latest/pricing_engines.html](https://quantlib-python-docs.readthedocs.io/en/latest/pricing_engines.html)
    
30. Ford Motor Credit Co. LLC - Ratings Actions | S&P Global Ratings, acessado em julho 2, 2025, [https://disclosure.spglobal.com/ratings/en/regulatory/org-details/sectorCode/FI/entityId/100881](https://disclosure.spglobal.com/ratings/en/regulatory/org-details/sectorCode/FI/entityId/100881)
    
31. Ford Motor Co. Credit Rating | S&P Global Ratings, acessado em julho 2, 2025, [https://disclosure.spglobal.com/ratings/en/regulatory/org-details/sectorCode/CORP/entityId/100880](https://disclosure.spglobal.com/ratings/en/regulatory/org-details/sectorCode/CORP/entityId/100880)
    
32. S&P Global Ratings revised outlook on Ford Motor Credit Company to negative and affirmed at - Cbonds, acessado em julho 2, 2025, [https://cbonds.com/news/3259505/](https://cbonds.com/news/3259505/)
    
33. Most Active Names in Credit and Equity Derivatives – August 2023 | - Clarus Financial Technology, acessado em julho 2, 2025, [https://www.clarusft.com/most-active-names-in-credit-and-equity-derivatives-august-2023/](https://www.clarusft.com/most-active-names-in-credit-and-equity-derivatives-august-2023/)
    
34. GFI releases most-traded CDS list - Risk.net, acessado em julho 2, 2025, [https://www.risk.net/ja/node/1502122](https://www.risk.net/ja/node/1502122)
    
35. Default, Transition, and Recovery: U.S. Recovery Study: Loan Recoveries Persist Below Their Trend | S&P Global Ratings, acessado em julho 2, 2025, [https://www.spglobal.com/ratings/en/research/articles/231215-default-transition-and-recovery-u-s-recovery-study-loan-recoveries-persist-below-their-trend-12947167](https://www.spglobal.com/ratings/en/research/articles/231215-default-transition-and-recovery-u-s-recovery-study-loan-recoveries-persist-below-their-trend-12947167)
    
36. Recovery Rate - Definition, Formula, Factors - Corporate Finance Institute, acessado em julho 2, 2025, [https://corporatefinanceinstitute.com/resources/commercial-lending/recovery-rate/](https://corporatefinanceinstitute.com/resources/commercial-lending/recovery-rate/)
    
37. US Treasury Yield Curve - ThaiBMA, acessado em julho 2, 2025, [https://www.thaibma.or.th/EN/Market/YieldCurve/USTreasury.aspx](https://www.thaibma.or.th/EN/Market/YieldCurve/USTreasury.aspx)
    
38. ICE BofA BBB US Corporate Index Option-Adjusted Spread (BAMLC0A4CBBB) - Federal Reserve Economic Data | FRED, acessado em julho 2, 2025, [https://fred.stlouisfed.org/series/BAMLC0A4CBBB](https://fred.stlouisfed.org/series/BAMLC0A4CBBB)
    

Risk Analysis for Corporate Bond Portfolios - Digital WPI, acessado em julho 2, 2025, [https://digital.wpi.edu/downloads/8g84mm38c?locale=pt-BR](https://digital.wpi.edu/downloads/8g84mm38c?locale=pt-BR)**