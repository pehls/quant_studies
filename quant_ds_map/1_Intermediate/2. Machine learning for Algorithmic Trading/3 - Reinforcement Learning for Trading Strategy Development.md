## 1. Introduction: Beyond Prediction to Intelligent Action

Previous chapters have explored the power of supervised machine learning to forecast financial outcomes—predicting stock prices, classifying market direction, or estimating volatility. These models answer the question, "What is likely to happen next?" However, a successful trading strategy requires more than just prediction; it demands a sequence of optimal decisions in the face of uncertainty. This is where Reinforcement Learning (RL) introduces a paradigm shift, moving from passive prediction to active, intelligent decision-making.1

### Shifting the Paradigm: Why Reinforcement Learning for Trading?

Reinforcement learning is a computational approach to goal-directed learning performed by an agent that interacts with a dynamic, and typically stochastic, environment.3 Instead of being trained on a dataset of "correct" labels, an RL agent learns by trial and error. It tries different actions, observes the outcomes, and receives feedback in the form of rewards or penalties. The agent's sole objective is to learn a strategy, known as a

**policy**, that maximizes its cumulative reward over the long term.1

This framework is intuitively analogous to training an agent to master a complex game like Chess or Go.5 In these games, a single move is not inherently "good" or "bad"; its quality depends on the sequence of moves that follow. The goal is not to win every piece but to win the game. Similarly, in trading, the objective is not to profit on every single trade but to maximize the portfolio's risk-adjusted return over a long horizon.6 RL is uniquely suited to this sequential, path-dependent nature of trading, where the value of an action (e.g., buying a stock) is realized through a subsequent, corresponding action (selling it).7

A key distinction from supervised learning is RL's ability to handle delayed rewards. A supervised model requires a label for every data point (e.g., the next day's return). An RL agent, however, can take an action (buy) and receive no immediate reward, only receiving feedback when the position is eventually closed, which could be many time steps later.1 This aligns perfectly with the mechanics of trading.

### Contrasting RL with Supervised Learning Approaches

To clarify the distinction, consider the questions each paradigm addresses:

- **Supervised Learning (Regression):** "Given the past 30 days of Apple's stock data, what will its closing price be tomorrow?"
    
- **Supervised Learning (Classification):** "Given the current technical indicators for the S&P 500, will the market close higher or lower tomorrow?"
    
- **Reinforcement Learning:** "Given the current market state (price history, indicators, volatility) and my current portfolio (cash balance, existing positions), what is the _optimal action_ (buy, sell, hold, or what percentage of capital to allocate) to take _right now_ to maximize my portfolio's Sharpe ratio over the next quarter?"
    

RL does not simply predict the market; it learns a complete, closed-loop trading strategy. It aims to unify the "signal generation" or "prediction" step with the "portfolio allocation" or "trade execution" step into a single, end-to-end optimization process. The result is a fully autonomous agent capable of interacting with its environment to make optimal decisions.8

### The Promise and Peril: A High-Level View of RL in the Financial Arena

The application of RL to algorithmic trading is a cutting-edge field that holds immense promise but is also fraught with significant challenges.9

The Promise:

RL agents can learn sophisticated, non-linear strategies that are difficult for humans to define explicitly. By interacting with market data, they can potentially uncover complex patterns and adapt their strategies to new information through online learning, making them more dynamic than static, rule-based systems.11 They have been successfully applied to a range of financial tasks, including portfolio optimization, market making, and optimal trade execution.13

The Peril:

Financial markets are arguably one of the most difficult environments for RL. Unlike a board game with fixed rules, markets are characterized by high levels of randomness, a very low signal-to-noise ratio, and, most critically, non-stationarity.5 The underlying "rules" of the market are constantly changing due to evolving macroeconomic conditions, geopolitical events, and shifts in investor sentiment. An agent that learns a profitable strategy during a bull market may fail catastrophically when the market regime shifts to a bear market.16

This gap between the simulated environment used for training (a static historical dataset) and the live, ever-changing market is the central challenge in applying RL to trading. An agent's actions in a historical simulation do not affect the market's trajectory, whereas in a live market, large trades can have a significant impact.17 Furthermore, a policy that is perfectly optimized for one historical period may be dangerously overfit and perform poorly on new, unseen data. Therefore, the focus of any serious practitioner must be not just on the power of the RL algorithm itself, but on the meticulous design of the environment, the careful crafting of the reward function, and the rigorous validation of the learned policy to ensure it is robust enough to bridge this "sim-to-real" gap.

## 2. The Language of the Market: Framing Trading as a Markov Decision Process (MDP)

To apply reinforcement learning to trading, we must first translate the problem into a formal mathematical framework. The standard for modeling sequential decision-making under uncertainty is the **Markov Decision Process (MDP)**.3 The MDP provides the language and structure needed to define the agent, its environment, and their interactions over time.

### The Core Interaction Loop: Agent, Environment, and Time

The RL paradigm is built on a simple but powerful feedback loop.18 At each discrete time step

t:

1. The **Agent** (our trading algorithm) observes the current state of the environment, St​.
    
2. Based on this state, the agent selects an **Action**, At​, from a set of possible actions.
    
3. The **Environment** (the financial market) receives the action, transitions to a new state, St+1​, and provides the agent with a scalar **Reward**, Rt+1​.
    

This loop continues, generating a trajectory of states, actions, and rewards. The agent's goal is to learn a policy that maximizes the cumulative discounted reward over this trajectory.20

### Mathematical Formulation of a Markov Decision Process (MDP)

An MDP is formally defined as a 5-tuple: (S,A,P,R,γ).3

- S: The **State Space**, a set of all possible states the environment can be in.
    
- A: The **Action Space**, a set of all possible actions the agent can take.
    
- P: The **State Transition Probability Function**, $P(s′∣s,a)=Pr(St+1​=s′∣St​=s,At​=a)$, which gives the probability of transitioning to state s′ after taking action a in state s. In a backtesting environment based on historical data, this is often deterministic: taking an action at time t always leads to the state at time t+1.
    
- R: The **Reward Function**, R(s,a,s′), which defines the immediate reward received after transitioning from state s to s′ due to action a.
    
- γ: The **Discount Factor**, where 0<γ≤1. It determines the present value of future rewards. A reward received k steps in the future is discounted by a factor of γk. This reflects the preference for immediate rewards over distant ones, which can represent the time value of money or the increasing uncertainty of the future.3
    

The agent's behavior is defined by its **policy**, $π(a∣s)=Pr(At​=a∣St​=s)$, which specifies the probability of taking action a in state s.3 The objective of RL is to find the optimal policy,

π∗, that maximizes the expected discounted return (the sum of all future discounted rewards).

### Defining the Components of a Trading MDP

Translating the abstract MDP into a concrete trading problem is the most critical design step. The choices made here will fundamentally determine what the agent can learn and how well it will perform.9

#### State Space (S): Crafting the Agent's Worldview

The state represents all the information the agent uses to make a decision at a given moment.2 A well-designed state space should be informative enough to capture the market's condition but not so complex that it becomes impossible to learn from (the "curse of dimensionality"). Key components include:

- **Market Data:** A window of recent historical data is essential. This typically includes Open, High, Low, Close prices and Volume (OHLCV) for a set number of past periods (e.g., the last 30 days).1
    
- **Technical Indicators:** Raw price data is often noisy. Technical indicators can distill this data into more meaningful features that capture concepts like momentum, trend, and volatility. Common choices include the Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Simple Moving Averages (SMA), and Bollinger Bands.1
    
- **Portfolio Status:** The agent must know its own status to make informed decisions. This includes its current position (e.g., long, short, flat), the amount of cash available, and the unrealized profit or loss on the current position.22 This information is crucial for risk management.
    
- **Data Preprocessing:** Before being fed to the agent, all state features must be cleaned (handling missing values) and, critically, **normalized** or **standardized**. Neural networks perform best when input features are on a similar scale (e.g., between -1 and 1 or with a mean of 0 and standard deviation of 1).9
    

#### Action Space (A): From Simple Decisions to Complex Portfolios

The action space defines the set of valid moves the agent can make.9

- **Discrete Actions:** This is the simplest formulation, where the agent chooses from a small, finite set of actions. A common example is `{0: Sell, 1: Hold, 2: Buy}`. This can be extended to include shorting, such as `{-1: Short, 0: Flat, 1: Long}`.5 This type of action space is required for value-based algorithms like DQN.
    
- **Continuous Actions:** This is a more flexible and realistic approach, where the action is a continuous value. For example, the action could be a number in `[-1, 1]`, representing the percentage of the portfolio to allocate to a short or long position, including leverage.26 Continuous action spaces require more advanced, policy-based algorithms like PPO or DDPG.
    

#### Reward Function (R): The Art and Science of Defining "Good" Performance

The reward function is the most critical element of the MDP design; it is the signal that guides the agent's learning process.6 A poorly designed reward function can lead to unintended and undesirable behaviors, a phenomenon known as "reward hacking."

- **Simple Profit-Based Rewards:** The most straightforward reward is the change in portfolio value from one step to the next, or the realized profit and loss (PnL) when a trade is closed.1 For example, a reward is given only when a
    
    `sell` action closes a `buy` position.1
    
- **Risk-Adjusted Rewards:** Maximizing profit alone is a dangerous objective, as it can encourage the agent to take on excessive risk. A more robust approach is to reward the agent based on a risk-adjusted performance metric. The **Sharpe Ratio**, which measures return per unit of volatility, is a popular choice.10 However, directly optimizing the Sharpe Ratio can be challenging as it is a non-convex function and can be "gamed" by agents.31 The
    
    **Sortino Ratio**, which only penalizes downside volatility, is another strong alternative.
    
- **Composite and Differential Rewards:** Advanced research focuses on more nuanced reward functions. The **differential Sharpe ratio** is a formulation designed for online, step-by-step learning.7 Another state-of-the-art approach is to use a
    
    **composite reward function**, which is a weighted sum of multiple objectives, such as maximizing returns, minimizing drawdown, and outperforming a benchmark.33 This explicitly frames trading as the multi-objective optimization problem that it is.
    

The table below summarizes how these abstract RL concepts map to the concrete domain of algorithmic trading.

|Component|Description|Trading Example|Key Design Considerations|
|---|---|---|---|
|**Agent**|The decision-maker.|The trading algorithm.|What RL algorithm to use (DQN, PPO)? What is its architecture (e.g., LSTM, MLP)? 10|
|**Environment**|The world the agent interacts with.|A market simulator built on historical data.|How to model transaction costs, slippage, and market impact? 9|
|**State (S)**|A snapshot of the environment.|Vector of.|What features to include? How to normalize them? How to avoid look-ahead bias? 9|
|**Action (A)**|A move the agent can make.|Discrete: {Buy, Sell, Hold}. Continuous: Allocate 35% of portfolio to asset X.|Should the action space be discrete or continuous? Does it include leverage? 27|
|**Reward (R)**|Feedback for an action.|Change in portfolio value; Sharpe ratio over a window.|How to balance risk and return? How to avoid "reward hacking"? 6|
|**Policy (π)**|The agent's strategy (state -> action).|A neural network that takes the state vector and outputs an action.|How to balance exploration and exploitation? 1|

## 3. Building the Virtual Trading Floor: A Custom `gymnasium` Environment

To train an RL agent, we must first construct a simulated environment that it can interact with. For trading, this means building a virtual market that processes the agent's actions (buy, sell, hold) and returns new states and rewards based on historical data.9 While many pre-built environments exist for classic RL problems like CartPole or Atari games, a custom environment is essential for financial applications to accurately model market mechanics like transaction costs, portfolio management, and specific data formats.34

We will use **`gymnasium`**, the maintained fork of OpenAI's Gym, which is the industry standard for developing and comparing RL environments.36 Building a

`gymnasium`-compatible environment ensures that it can be seamlessly used with powerful RL libraries like `stable-baselines3`.

### The Need for a High-Fidelity Simulation

The quality of the learned trading strategy is directly proportional to the fidelity of the simulation environment. A simplistic environment that ignores real-world frictions will produce an agent with an overly optimistic and ultimately unprofitable strategy. The environment is not just a data wrapper; it is a software component that must be engineered with care.

A robust trading environment is a significant software engineering challenge. It requires a class structure that adheres to the `gymnasium.Env` API, which mandates the implementation of several key methods: `__init__`, `step`, `reset`, `render`, and `close`.36 The

`step` function is the heart of the simulation, containing the logic for trade execution, portfolio value updates, reward calculation, and state transitions. This logic must be meticulously designed to account for:

- **Transaction Costs:** Every trade incurs a fee, which must be subtracted from the portfolio's value.27
    
- **Slippage:** In a real market, the price at which a trade is executed may differ from the last quoted price. This can be modeled as a small, random penalty or a fixed percentage.
    
- **Data Handling:** The environment must serve a sliding window of historical data as the observation at each step, ensuring there is no look-ahead bias (i.e., the agent cannot see future prices).
    
- **State Management:** The environment must track the agent's portfolio state—cash, number of shares held, current portfolio value—across time steps.
    

A subtle bug in any of these components, such as an incorrect reward calculation or a flawed cost model, will silently corrupt the entire training process. The agent will learn an optimal policy for a flawed game, which will not translate to real-world profitability. Therefore, building and thoroughly testing the environment is a critical prerequisite to any agent training.

### Step-by-Step Implementation of a `TradingEnv` Class in Python

The following code provides a complete, commented implementation of a basic `TradingEnv` class that inherits from `gymnasium.Env`. This environment will handle a single stock, a discrete action space (Hold, Buy, Sell), and transaction costs.



```Python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A custom stock trading environment for Reinforcement Learning.

    This environment simulates trading a single stock.
    - State: A window of past price data and the agent's current position.
    - Actions: Hold (0), Buy (1), Sell (2).
    - Reward: The change in portfolio value at each step.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=10, initial_balance=10000, transaction_cost=0.001):
        """
        Initializes the trading environment.

        Args:
            df (pd.DataFrame): DataFrame with stock prices (must contain 'Close' price).
            window_size (int): The number of past time steps to include in the observation.
            initial_balance (float): The starting cash balance.
            transaction_cost (float): The cost per transaction as a fraction.
        """
        super(TradingEnv, self).__init__()

        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # Define action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Define observation space: (window_size prices + cash_balance + shares_held)
        # The prices are normalized, so we use low=-np.inf, high=np.inf for flexibility
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size + 2,), dtype=np.float32
        )

        # Initialize state variables
        self.current_step = 0
        self.balance = 0
        self.shares_held = 0
        self.net_worth = 0

    def _get_obs(self):
        """Constructs the observation from the current state."""
        # Get the price window
        frame = self.df.iloc[self.current_step - self.window_size + 1 : self.current_step + 1]
        # Normalize the price data relative to the current price
        obs_prices = frame['Close'].values / self.df['Close'].iloc[self.current_step]
        
        # Append portfolio status
        obs = np.append(obs_prices, [self.balance / self.initial_balance, self.shares_held])
        return obs.astype(np.float32)

    def _get_info(self):
        """Returns auxiliary information about the current state."""
        return {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': self.df['Close'].iloc[self.current_step]
        }

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = self.window_size - 1

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        """
        Executes one time step within the environment.

        Args:
            action (int): The action to take (0: Hold, 1: Buy, 2: Sell).
        """
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Execute the action
        if action == 1:  # Buy
            # Buy one share if there is enough balance
            if self.balance > current_price * (1 + self.transaction_cost):
                self.shares_held += 1
                self.balance -= current_price * (1 + self.transaction_cost)
        elif action == 2:  # Sell
            # Sell one share if holding any
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price * (1 - self.transaction_cost)

        # Update portfolio net worth
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Calculate reward
        reward = self.net_worth - prev_net_worth

        # Move to the next time step
        self.current_step += 1

        # Check for termination
        terminated = self.net_worth <= 0 or self.current_step >= len(self.df) - 1
        truncated = False # Not using time limits for truncation in this simple env

        # Get the next observation and info
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Renders the environment (e.g., prints the current status)."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Net Worth: {self.net_worth:.2f}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Shares Held: {self.shares_held}")
            print(f"Current Price: {self.df['Close'].iloc[self.current_step]:.2f}")
            print("-" * 20)
            
    def close(self):
        """Performs any necessary cleanup."""
        pass

```

This class provides a solid foundation for a trading environment. It can be extended to include more complex features, such as continuous action spaces (for portfolio allocation), short selling, or more sophisticated reward functions, as we will explore in the capstone project.

## 4. Learning from Experience: Value-Based and Policy-Based Algorithms

With a trading environment established, the next step is to choose and implement an RL algorithm to train our agent. RL algorithms fall into two main categories: **value-based** methods, which learn the value of state-action pairs, and **policy-based** methods, which directly learn a policy. We will explore one prominent algorithm from each category: Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO).

### Deep Q-Networks (DQN): Learning the Value of Actions

Value-based methods attempt to find the optimal policy indirectly by first learning an optimal **action-value function**, denoted as Q∗(s,a). This function represents the maximum expected return an agent can achieve by taking action a in state s and acting optimally thereafter.

#### From Q-Tables to Neural Networks

The simplest value-based algorithm is Q-learning, which uses a lookup table (a "Q-table") to store the Q-value for every possible state-action pair.1 The table is updated iteratively using the

**Bellman equation**:

![[Pasted image 20250701083932.png]]

This equation states that the updated value of taking action a in state s is the immediate reward R(s,a) plus the discounted value of the best possible action a′ from the next state s′.1

However, for trading, the state space is continuous and high-dimensional (e.g., a window of 30 days of prices and indicators). A Q-table would be infinitely large and impossible to store or populate. **Deep Q-Networks (DQN)** solve this problem by replacing the Q-table with a deep neural network.10 This network acts as a function approximator: it takes the state

s as input and outputs a vector of Q-values, one for each possible action.10

#### Stabilizing the Learner

Training a neural network with the Bellman equation is inherently unstable because the target value $(R+γmaxa′​Q(s′,a′))$ is constantly changing as the network's own weights are updated. Two key innovations make DQN training feasible 4:

1. **Experience Replay:** Instead of training on consecutive samples as they occur, the agent stores its experiences—tuples of (s,a,r,s′)—in a fixed-size memory buffer. During training, it draws random mini-batches from this buffer.1 This technique breaks the strong temporal correlation between consecutive samples, leading to more stable and efficient learning.1
    
2. **Target Network:** A second, separate neural network, called the "target network," is used to calculate the target Q-values. The weights of this target network are not updated at every step; instead, they are periodically copied from the main "online" network. This creates a more stable target for the loss calculation, preventing the network from chasing a moving target and collapsing.4
    

#### Python Example: Training a DQN Agent with `stable-baselines3`

The `stable-baselines3` library provides a high-quality, easy-to-use implementation of DQN. Training an agent on our custom `TradingEnv` requires just a few lines of code.



```Python
import gymnasium as gym
import pandas as pd
from stable_baselines3 import DQN

# Assume 'TradingEnv' class from Section 3 is defined
# Assume 'spy_data' is a preloaded pandas DataFrame with 'Close' prices

# Create the trading environment
env = TradingEnv(df=spy_data, window_size=20, initial_balance=10000)

# Instantiate the DQN model
# "MlpPolicy" means we use a Multi-Layer Perceptron (a standard neural network)
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100000,      # Size of the experience replay buffer
    learning_starts=1000,    # Number of steps to collect before training starts
    batch_size=32,
    gamma=0.99,              # Discount factor
    tau=1.0,                 # The polyak update coefficient for the target network
    train_freq=4,            # Update the model every 4 steps
    gradient_steps=1,
    target_update_interval=1000, # Update the target network every 1000 steps
    verbose=1
)

# Train the agent
model.learn(total_timesteps=50000)

# Save the trained model
model.save("dqn_trading_agent")
```

### Proximal Policy Optimization (PPO): A More Stable Approach

Policy-based methods, in contrast to value-based ones, directly learn the policy π(a∣s) without needing to first learn a value function. **Proximal Policy Optimization (PPO)** is a state-of-the-art policy gradient algorithm that has become a go-to choice for many RL applications, including finance, due to its remarkable stability and solid performance.39

The popularity of PPO in finance stems from its design philosophy. Financial markets are incredibly noisy, and a large, misguided policy update based on random market fluctuations could be catastrophic. Older policy gradient methods were prone to such destructive updates. PPO addresses this by ensuring that policy updates are kept within a small, "trusted" region, preventing the new policy from deviating too drastically from the old one in a single step. This inherent stability is invaluable in a risk-averse domain like quantitative trading.40

#### The Actor-Critic Framework

PPO is an **Actor-Critic** algorithm, meaning it uses two neural networks that work in tandem 39:

- **The Actor (Policy Network):** This network takes the state as input and outputs a probability distribution over the actions. It is the component that decides which action to take.10
    
- **The Critic (Value Network):** This network takes the state as input and outputs a single scalar value, estimating the long-term value of being in that state. The critic does not choose actions; its purpose is to help train the actor by evaluating its decisions.10
    

The critic's output is used to compute the **advantage function**, A(s,a)=Q(s,a)−V(s), which measures how much better a specific action a is compared to the average action from state s. The actor is then updated to increase the probability of actions with a high positive advantage.

#### PPO's Core Innovation: The Clipped Surrogate Objective Function

The genius of PPO lies in its objective function, which constrains the size of the policy update. It uses a ratio, ![[Pasted image 20250701084015.png]]​, which measures how much the probability of taking a certain action has changed between the new policy (πθ​) and the old policy (πθold​​). The PPO objective function is then:

![[Pasted image 20250701084001.png]]

Where:

- A^t​ is the estimated advantage at time t.
    
- ϵ is a small hyperparameter (e.g., 0.2) that defines the clipping range.
    
- The `clip` function constrains the ratio rt​(θ) to be within the range [1−ϵ,1+ϵ].
    

The `min` operator ensures that the final objective is a pessimistic bound on the performance improvement. If the advantage is positive (a good action), the update is capped to prevent it from becoming too large. If the advantage is negative (a bad action), the update is also capped to prevent an over-correction. This simple clipping mechanism is what gives PPO its signature stability.44

#### Python Example: Training a PPO Agent with `stable-baselines3`

Training a PPO agent with `stable-baselines3` is just as straightforward as training a DQN.



```Python
import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Assume 'TradingEnv' class is defined and 'spy_data' is loaded

# It's often beneficial to use vectorized environments for on-policy algorithms like PPO
# This runs multiple environments in parallel to collect more diverse experiences
vec_env = make_vec_env(lambda: TradingEnv(df=spy_data, window_size=20), n_envs=4)

# Instantiate the PPO model
model = PPO(
    "MlpPolicy",
    vec_env,
    n_steps=2048,           # Number of steps to run for each environment per update
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    learning_rate=3e-4,
    verbose=1
)

# Train the agent
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_trading_agent")
```

The table below provides a high-level comparison to help guide the choice between these two powerful algorithms for a trading application.

|Feature|Deep Q-Network (DQN)|Proximal Policy Optimization (PPO)|
|---|---|---|
|**Algorithm Type**|Value-Based, Off-Policy|Policy-Based (Actor-Critic), On-Policy|
|**Action Space**|Primarily Discrete (e.g., Buy, Sell, Hold) 47|Discrete & Continuous (e.g., Portfolio Allocation %) 38|
|**Key Idea**|Learns the value Q(s,a) of each action in each state.|Directly learns the policy π(a∥s that maximizes reward.|
|**Sample Efficiency**|More sample efficient (can reuse old data via Experience Replay).|Less sample efficient (typically requires new data for each update).|
|**Stability**|Can be unstable with function approximation.|Very stable due to clipped objective function.39|
|**Best For...**|Simpler problems with a small, discrete set of actions.|Complex problems, continuous actions, and when stability is paramount.|

## 5. The Real-World Gauntlet: Overfitting, Non-Stationarity, and Backtesting

Developing an RL agent that performs well in a simulated environment is only the first step. The true test is whether its learned strategy can generalize to the live, unpredictable financial markets. This transition from simulation to reality is fraught with challenges, the most significant of which are overfitting and non-stationarity.5

### The Specter of Overfitting: Why Backtest Performance Can Be Deceiving

In the context of RL for trading, overfitting occurs when an agent learns a policy that is excessively tailored to the specific noise and idiosyncrasies of the historical data it was trained on, rather than a genuinely robust market logic.2 This is the algorithmic equivalent of "curve-fitting" a strategy. The result is often a model that shows spectacular, too-good-to-be-true performance in backtesting but fails catastrophically when deployed in a live market with new, unseen data.48 The infamous 2012 Knight Capital incident, where a faulty algorithm deployment led to a loss of $440 million in 45 minutes, serves as a stark reminder of the real-world consequences.48

**Causes of Overfitting in RL Trading:**

- **Excessive Complexity:** Using an overly complex state representation with too many features or a deep neural network with too many parameters can lead the agent to memorize the training data.49
    
- **Hyperparameter Over-Optimization:** Endlessly tweaking algorithm hyperparameters (learning rate, discount factor, etc.) to maximize performance on a single validation set can cause the model to fit the validation set's specific noise.
    
- **Data Snooping:** Repeatedly testing different strategy ideas on the same dataset can lead to the unintentional discovery of spurious correlations that do not hold in the future.
    

Mitigation Strategies:

The most effective defense against overfitting is a rigorous and disciplined validation process.

- **Strict Data Splitting:** Time-series data must be split chronologically into training, validation, and testing sets. The test set must be held out and used only once to evaluate the final, trained model.49
    
- **Simplicity:** Favor simpler models and state representations over complex ones unless there is a strong justification for the added complexity. Often, simpler strategies are more robust.48
    
- **Regularization:** Techniques like dropout in neural networks can help prevent the model from relying too heavily on any single feature, promoting better generalization.45
    

### The Challenge of Non-Stationarity: When the Market Changes its Rules

Non-stationarity is arguably the most profound and difficult challenge in quantitative finance. It refers to the fact that the statistical properties of financial time series—such as their mean, variance, and correlations—are not constant over time.51 Markets transition between different

**regimes**, such as bull markets, bear markets, periods of high volatility, and periods of low volatility.38

Standard RL algorithms are built on the assumption of a stationary MDP, meaning the "rules of the game" (the transition probabilities and reward dynamics) are fixed. When an agent is trained on data from one market regime, it learns a policy that is optimal for that specific set of rules. If the market then shifts to a new regime, the agent's learned policy may become obsolete or even disastrously wrong.16

Addressing non-stationarity is an active and advanced area of academic research.5 While a full treatment is beyond the scope of this chapter, it is crucial to be aware of the state-of-the-art approaches:

- **Ensemble and Online Learning:** One promising technique involves training multiple expert agents, each specialized for a different historical market regime (e.g., a "bull market agent," a "bear market agent"). A higher-level meta-learning algorithm then dynamically selects the most suitable expert agent to use based on recent performance, effectively adapting to the current regime in real-time.51
    
- **Causal Representation Learning:** Instead of learning surface-level statistical correlations, which are unstable, some methods attempt to learn the underlying causal structure of the market. The hypothesis is that these causal relationships may be more stable across different regimes.53
    
- **Adaptive Learning and Change-Point Detection:** These methods explicitly try to detect when a regime shift has occurred. Upon detection, the agent can adapt its policy, for example, by increasing its learning rate to adapt to new data more quickly or by employing a dynamic memory that gives more weight to recent experiences.52
    

### Best Practices for Robust Validation

Given these challenges, how can we gain confidence in an RL trading strategy? The answer lies in robust backtesting and validation methodologies.

- **Out-of-Sample (OOS) Testing:** As mentioned, this is the gold standard. The final model must be evaluated on a completely untouched segment of data that was not used in any part of the training or model selection process.48
    
- **Walk-Forward Analysis:** This is a more dynamic and realistic validation technique than a single train-test split. The process is as follows:
    
    1. Train the model on an initial window of historical data (e.g., 2010-2015).
        
    2. Test the model on the next, immediately following window of data (e.g., 2016).
        
    3. Slide the training window forward (e.g., to 2011-2016) and retrain the model.
        
    4. Test on the next window (e.g., 2017).
        
    5. Repeat this process until the end of the dataset.
        
        This method better simulates how a model would be periodically retrained and deployed in a live environment, testing its ability to adapt over time.48
        
- **Backtesting RL Agents:** It is important to note that for an RL agent, the "backtest" is simply the process of running an episode in the `gymnasium` environment. The environment itself, when loaded with historical data, _is_ the backtester. While dedicated backtesting libraries like `backtesting.py`, `Zipline`, or `Backtrader` are excellent for traditional, signal-based strategies, the natural way to evaluate a learned RL policy is to let it interact with the test data via the `env.step()` loop.55 The performance metrics can then be calculated from the history of portfolio values generated during this evaluation episode.
    

## 6. Capstone Project: A PPO-Based Agent for S&P 500 E-mini Futures

This capstone project synthesizes all the concepts covered in this chapter into a complete, end-to-end workflow. The objective is to build, train, and critically evaluate a sophisticated RL trading agent using the Proximal Policy Optimization (PPO) algorithm. We will apply it to a highly liquid financial instrument, using the SPDR S&P 500 ETF (SPY) as a proxy for the E-mini futures market. PPO is chosen for its stability and robust performance, which are desirable qualities for a noisy financial environment.40

### Phase 1: Data Acquisition and Environment Setup

First, we acquire historical data and set up our custom trading environment.

- **Data Acquisition:** We will use the `yfinance` library to download daily historical data for SPY.
    
- **Feature Engineering:** We will create a set of technical indicators to form our state space. These features will include the 14-day Relative Strength Index (RSI), the 50-day Simple Moving Average (SMA), and Bollinger Bands. These features must be normalized to be suitable for a neural network.
    
- **Environment Instantiation:** We will use the `TradingEnv` class developed in Section 3. The action space will be discrete: `{0: Hold, 1: Buy, 2: Sell}`.
    

### Phase 2: Agent Training on Historical Data

We will split our data chronologically and train two different agents to investigate the impact of the reward function design.

- **Data Splitting:** The data will be split into a training set (January 2010 to December 2020) and a test set (January 2021 to present day). This ensures our final evaluation is on completely unseen data.
    
- **Agent Training:** We will use the `stable-baselines3` implementation of PPO. Two agents will be trained on the same training data but with different reward functions:
    
    1. **Agent 1 (Profit-Maximizing):** The reward at each step will be the change in the portfolio's net worth. This agent's goal is simply to maximize profit.
        
    2. **Agent 2 (Risk-Adjusted):** The reward will be a simplified differential Sharpe ratio, calculated as the daily return divided by the rolling standard deviation of returns over the last 21 days. This encourages the agent to seek returns while penalizing it for taking on high volatility.30 A small constant is added to the denominator to prevent division by zero.
        

### Phase 3: Rigorous Backtesting and Performance Analysis

After training, we will evaluate both agents and a benchmark on the unseen test data.

- **Evaluation:** The trained policies of Agent 1 and Agent 2 will be run on the test environment.
    
- **Benchmark:** A simple "Buy-and-Hold" strategy will serve as our baseline for comparison.
    
- **Performance Metrics:** We will calculate a suite of standard performance metrics for all three strategies, including cumulative return, annualized return, annualized volatility, Sharpe ratio, Sortino ratio, and maximum drawdown. A library like `pyfolio` or `quantstats` can be used for this purpose, or the metrics can be calculated manually.16
    

### Python Code: Complete End-to-End Capstone Project

The following code block provides the complete, runnable Python script for this capstone project.



```Python
# Step 0: Install necessary libraries
#!pip install yfinance pandas numpy gymnasium stable-baselines3[extra] quantstats

import yfinance as yf
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_bas_lines3.common.vec_env import DummyVecEnv
import quantstats as qs

# --- Step 1: Custom Trading Environment ---
class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=20, initial_balance=10000, transaction_cost=0.001, reward_type='profit'):
        super(TradingEnv, self).__init__()

        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_type = reward_type

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size + 2,), dtype=np.float32)

    def _get_obs(self):
        frame = self.df.iloc[self.current_step - self.window_size + 1: self.current_step + 1]
        
        # Normalize features relative to the current step's value
        obs_features = frame.values / frame.values[-1, :]
        
        obs = np.append(obs_features.flatten(), [self.balance / self.initial_balance, self.shares_held / 100]) # Normalize shares held
        return obs.astype(np.float32)

    def _get_info(self):
        return {'net_worth': self.net_worth}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = self.window_size - 1
        self.history =
        return self._get_obs(), self._get_info()

    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        prev_net_worth = self.net_worth

        if action == 1:  # Buy
            if self.balance > current_price:
                num_shares_to_buy = self.balance // current_price
                self.shares_held += num_shares_to_buy
                self.balance -= num_shares_to_buy * current_price * (1 + self.transaction_cost)
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price * (1 - self.transaction_cost)
                self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * current_price
        self.history.append(self.net_worth)

        # Calculate reward
        if self.reward_type == 'profit':
            reward = self.net_worth - prev_net_worth
        elif self.reward_type == 'sharpe':
            returns = pd.Series(self.history).pct_change().dropna()
            if len(returns) > 1:
                sharpe = np.mean(returns) / (np.std(returns) + 1e-9)
                reward = sharpe
            else:
                reward = 0
        else:
            reward = 0

        self.current_step += 1
        terminated = self.net_worth <= 0 or self.current_step >= len(self.df) - 1
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

# --- Step 2: Data Preparation ---
# Download SPY data
data = yf.download('SPY', start='2010-01-01', end='2024-01-01')

# Feature Engineering
data = data['Close'].rolling(window=50).mean()
data = 100 - (100 / (1 + data['Close'].diff().clip(lower=0).rolling(window=14).mean() / \
                             -data['Close'].diff().clip(upper=0).rolling(window=14).mean()))
rolling_mean = data['Close'].rolling(window=20).mean()
rolling_std = data['Close'].rolling(window=20).std()
data = rolling_mean + (rolling_std * 2)
data = rolling_mean - (rolling_std * 2)

# Select features and drop missing values
features =
data = data[features].dropna()

# Split data
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# --- Step 3: Train Agents ---
# Create environments
env_profit = TradingEnv(train_data, window_size=20, reward_type='profit')
env_sharpe = TradingEnv(train_data, window_size=20, reward_type='sharpe')

vec_env_profit = DummyVecEnv([lambda: env_profit])
vec_env_sharpe = DummyVecEnv([lambda: env_sharpe])

# Train Profit-Maximizing Agent
print("Training Profit-Maximizing Agent...")
model_profit = PPO('MlpPolicy', vec_env_profit, verbose=0, n_steps=2048, batch_size=64, n_epochs=10)
model_profit.learn(total_timesteps=50000)
model_profit.save("ppo_profit_agent")

# Train Risk-Adjusted Agent
print("Training Risk-Adjusted Agent...")
model_sharpe = PPO('MlpPolicy', vec_env_sharpe, verbose=0, n_steps=2048, batch_size=64, n_epochs=10)
model_sharpe.learn(total_timesteps=50000)
model_sharpe.save("ppo_sharpe_agent")

print("Training complete.")

# --- Step 4: Evaluate Agents on Test Data ---
def evaluate_agent(model, test_df, reward_type):
    env = TradingEnv(test_df, window_size=20, reward_type=reward_type)
    obs, _ = env.reset()
    done = False
    net_worths = [env.initial_balance]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        net_worths.append(env.net_worth)
    return pd.Series(net_worths, index=test_df.index[19:]) # Align index

# Evaluate Profit Agent
profit_agent_returns = evaluate_agent(model_profit, test_data, 'profit').pct_change().dropna()

# Evaluate Sharpe Agent
sharpe_agent_returns = evaluate_agent(model_sharpe, test_data, 'sharpe').pct_change().dropna()

# Benchmark
benchmark_returns = test_data['Close'][19:].pct_change().dropna()

# --- Step 5: Performance Analysis ---
print("\n--- Performance Analysis ---")

qs.reports.html(profit_agent_returns, benchmark=benchmark_returns, output='profit_agent_report.html', title='Profit Agent vs. Benchmark')
qs.reports.html(sharpe_agent_returns, benchmark=benchmark_returns, output='sharpe_agent_report.html', title='Sharpe Agent vs. Benchmark')

print("Generated performance reports: 'profit_agent_report.html' and 'sharpe_agent_report.html'")

# Display key metrics in a table
metrics = {
    'Strategy':,
    'Cumulative Return (%)': [
        qs.stats.comp(profit_agent_returns) * 100,
        qs.stats.comp(sharpe_agent_returns) * 100,
        qs.stats.comp(benchmark_returns) * 100
    ],
    'Sharpe Ratio': [
        qs.stats.sharpe(profit_agent_returns),
        qs.stats.sharpe(sharpe_agent_returns),
        qs.stats.sharpe(benchmark_returns)
    ],
    'Max Drawdown (%)': [
        qs.stats.max_drawdown(profit_agent_returns) * 100,
        qs.stats.max_drawdown(sharpe_agent_returns) * 100,
        qs.stats.max_drawdown(benchmark_returns) * 100
    ],
    'Volatility (ann.) (%)':
}

results_df = pd.DataFrame(metrics)
print("\nCapstone Project Final Performance Metrics:")
print(results_df.to_string(index=False))

```

### Analysis and Interpretation (Questions & Responses)

After running the capstone project code, we can analyze the generated reports and metrics to answer key questions about the agents' performance.

**Q1: How do the PPO agents' performances (Cumulative Return, Sharpe Ratio, Max Drawdown) compare against a Buy-and-Hold benchmark?**

**A:** The final performance table provides a quantitative answer. We would typically analyze this table to see if either RL agent managed to outperform the simple Buy-and-Hold strategy. For instance, an agent might achieve a lower cumulative return but a significantly higher Sharpe ratio and a lower maximum drawdown. This would indicate that while it was less profitable in absolute terms, it generated its returns more efficiently and with less risk, which is often a desirable outcome for a trading strategy. Conversely, if an agent underperforms on all metrics, it suggests that the learned policy was not robust enough for the unseen test data.

**Q2: How does altering the reward function (from simple profit to a risk-adjusted reward) impact the agent's learned behavior?**

**A:** This is a crucial question that gets to the heart of reward engineering. By comparing the performance of the "Profit Agent" and the "Sharpe Agent," we can observe the direct impact of the reward signal. We would expect the **Profit Agent** to trade more aggressively, potentially capturing larger gains but also suffering from deeper drawdowns, leading to higher volatility. The **Sharpe Agent**, having been penalized for volatility during training, should exhibit more conservative behavior. It would likely trade less frequently, aim for more consistent but smaller gains, and demonstrate a lower maximum drawdown and higher Sharpe ratio than the Profit Agent. This comparison highlights that the reward function is a powerful lever for shaping the agent's risk appetite and overall strategy.

**Q3: Can we interpret the agent's learned policy? What market conditions trigger its buy/sell decisions?**

**A:** While deep neural networks are often considered "black boxes," we can gain some insight into the learned policy by visualizing its actions against the price chart and its input features. By plotting the buy and sell signals generated by an agent on the test data's price chart, we can look for recurring patterns. For example, we might observe that the agent consistently buys after the price crosses above its 50-day SMA and the RSI is below 70, suggesting it has learned a form of trend-following strategy with a momentum filter. Or, it might learn a mean-reversion strategy, selling when the price hits the upper Bollinger Band and buying at the lower band. This qualitative analysis helps build confidence that the agent has learned a logical, understandable strategy rather than just exploiting noise in the training data.61

### Capstone Project Final Performance Metrics

The final output of the project would be a table summarizing the key performance indicators for each strategy on the test set.

|Metric|PPO Agent (Profit-Max Reward)|PPO Agent (Risk-Adjusted Reward)|Buy-and-Hold Benchmark|
|---|---|---|---|
|Cumulative Return (%)|_Result from code_|_Result from code_|_Result from code_|
|Annualized Return (%)|_Result from code_|_Result from code_|_Result from code_|
|Annualized Volatility (%)|_Result from code_|_Result from code_|_Result from code_|
|Sharpe Ratio|_Result from code_|_Result from code_|_Result from code_|
|Sortino Ratio|_Result from code_|_Result from code_|_Result from code_|
|Max Drawdown (%)|_Result from code_|_Result from code_|_Result from code_|
|Total Trades|_Result from code_|_Result from code_|_N/A_|
|Win Rate (%)|_Result from code_|_Result from code_|_N/A_|

This chapter has provided a comprehensive journey into the application of reinforcement learning for algorithmic trading. From the foundational theory of MDPs to the practical implementation of a custom environment and the training of sophisticated agents like PPO, it is clear that RL offers a powerful new toolkit for quantitative finance. However, its power comes with significant challenges, especially the perils of overfitting and non-stationarity. A successful practitioner must be not only a data scientist but also a diligent software engineer and a skeptical empiricist, always prioritizing robust validation and a deep understanding of the market's dynamic nature.

## References
**

1. Reinforcement Learning in Trading: Build Smarter Strategies with Q-Learning & Experience Replay - QuantInsti Blog, acessado em julho 1, 2025, [https://blog.quantinsti.com/reinforcement-learning-trading/](https://blog.quantinsti.com/reinforcement-learning-trading/)
    
2. Reinforcement Learning Transforming Trading Strategies - PyQuant News, acessado em julho 1, 2025, [https://www.pyquantnews.com/free-python-resources/reinforcement-learning-transforming-trading-strategies](https://www.pyquantnews.com/free-python-resources/reinforcement-learning-transforming-trading-strategies)
    
3. Reinforcement learning - Wikipedia, acessado em julho 1, 2025, [https://en.wikipedia.org/wiki/Reinforcement_learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
    
4. machine-learning-for-trading/22_deep_reinforcement_learning ..., acessado em julho 1, 2025, [https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/22_deep_reinforcement_learning/README.md](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/22_deep_reinforcement_learning/README.md)
    
5. The Evolution of Reinforcement Learning in Quantitative Finance: A Survey - arXiv, acessado em julho 1, 2025, [https://arxiv.org/html/2408.10932v3](https://arxiv.org/html/2408.10932v3)
    
6. Implementing Reinforcement Learning in Trading Strategies | Gamify and Master Algorithmic Trading - YouTube, acessado em julho 1, 2025, [https://m.youtube.com/watch?v=hDhFRT8DcGY](https://m.youtube.com/watch?v=hDhFRT8DcGY)
    
7. Reinforcement Learning for Trading - NIPS, acessado em julho 1, 2025, [https://papers.nips.cc/paper/1551-reinforcement-learning-for-trading](https://papers.nips.cc/paper/1551-reinforcement-learning-for-trading)
    
8. Deep Reinforcement Learning Approach for Trading Automation in The Stock Market - arXiv, acessado em julho 1, 2025, [https://arxiv.org/abs/2208.07165](https://arxiv.org/abs/2208.07165)
    
9. Deep Reinforcement Learning in Algorithmic Trading: A Step-by-Step Guide | by Pham The Anh | Funny AI & Quant | Medium, acessado em julho 1, 2025, [https://medium.com/funny-ai-quant/deep-reinforcement-learning-in-algorithmic-trading-a-step-by-step-guide-197f39a8be9a](https://medium.com/funny-ai-quant/deep-reinforcement-learning-in-algorithmic-trading-a-step-by-step-guide-197f39a8be9a)
    
10. Deep Reinforcement Learning for Trading: Strategy Development & AutoML - MLQ.ai, acessado em julho 1, 2025, [https://blog.mlq.ai/deep-reinforcement-learning-trading-strategies-automl/](https://blog.mlq.ai/deep-reinforcement-learning-trading-strategies-automl/)
    
11. A Review of Reinforcement Learning in Financial Applications - arXiv, acessado em julho 1, 2025, [http://arxiv.org/pdf/2411.12746](http://arxiv.org/pdf/2411.12746)
    
12. Using Reinforcement Learning to Optimize Stock Trading Strategies | by Zhong Hong, acessado em julho 1, 2025, [https://medium.com/@zhonghong9998/using-reinforcement-learning-to-optimize-stock-trading-strategies-a77d35ea3308](https://medium.com/@zhonghong9998/using-reinforcement-learning-to-optimize-stock-trading-strategies-a77d35ea3308)
    
13. Systematic Review on Reinforcement Learning in the field of Fintech - arXiv, acessado em julho 1, 2025, [https://arxiv.org/pdf/2305.07466](https://arxiv.org/pdf/2305.07466)
    
14. [2112.04553] Recent Advances in Reinforcement Learning in Finance - arXiv, acessado em julho 1, 2025, [https://arxiv.org/abs/2112.04553](https://arxiv.org/abs/2112.04553)
    
15. [2101.07107] Deep Reinforcement Learning for Active High Frequency Trading - arXiv, acessado em julho 1, 2025, [https://arxiv.org/abs/2101.07107](https://arxiv.org/abs/2101.07107)
    
16. Trading Strategies using Reinforcement Learning, acessado em julho 1, 2025, [https://fsc.stevens.edu/trading-strategies-using-reinforcement-learning/](https://fsc.stevens.edu/trading-strategies-using-reinforcement-learning/)
    
17. The Limitations of Reinforcement Learning in Algorithmic Trading: A Closer Look - Medium, acessado em julho 1, 2025, [https://medium.com/@survexman/the-limitations-of-reinforcement-learning-in-algorithmic-trading-a-closer-look-7312d692ffe5](https://medium.com/@survexman/the-limitations-of-reinforcement-learning-in-algorithmic-trading-a-closer-look-7312d692ffe5)
    
18. Day 62: Reinforcement Learning Basics — Agent, Environment, Rewards - Medium, acessado em julho 1, 2025, [https://medium.com/@bhatadithya54764118/day-62-reinforcement-learning-basics-agent-environment-rewards-306b8e7e555c](https://medium.com/@bhatadithya54764118/day-62-reinforcement-learning-basics-agent-environment-rewards-306b8e7e555c)
    
19. Agent Environment interaction loop #reinforcementlearning #artificialintelligence - YouTube, acessado em julho 1, 2025, [https://m.youtube.com/shorts/NGuHUE26KZE](https://m.youtube.com/shorts/NGuHUE26KZE)
    
20. Reinforcement Learning - GeeksforGeeks, acessado em julho 1, 2025, [https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/](https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/)
    
21. Create a Custom Environment - Gymnasium Documentation, acessado em julho 1, 2025, [https://gymnasium.farama.org/introduction/create_custom_env/](https://gymnasium.farama.org/introduction/create_custom_env/)
    
22. [2406.08013] Deep reinforcement learning with positional context for intraday trading - arXiv, acessado em julho 1, 2025, [https://arxiv.org/abs/2406.08013](https://arxiv.org/abs/2406.08013)
    
23. Albert-Z-Guo/Deep-Reinforcement-Stock-Trading - GitHub, acessado em julho 1, 2025, [https://github.com/Albert-Z-Guo/Deep-Reinforcement-Stock-Trading](https://github.com/Albert-Z-Guo/Deep-Reinforcement-Stock-Trading)
    
24. What role does the environment play in reinforcement learning? - Milvus, acessado em julho 1, 2025, [https://milvus.io/ai-quick-reference/what-role-does-the-environment-play-in-reinforcement-learning](https://milvus.io/ai-quick-reference/what-role-does-the-environment-play-in-reinforcement-learning)
    
25. Deep reinforcement learning based stock trading (Stable baselines3 + Dow Jones) | by Ming Zhu | Medium, acessado em julho 1, 2025, [https://medium.com/@zhumingpassional/deep-reinforcement-learning-based-stock-trading-stable-baselines3-dow-jones-c7ed034eb9f0](https://medium.com/@zhumingpassional/deep-reinforcement-learning-based-stock-trading-stable-baselines3-dow-jones-c7ed034eb9f0)
    
26. [1911.10107] Deep Reinforcement Learning for Trading - arXiv, acessado em julho 1, 2025, [https://arxiv.org/abs/1911.10107](https://arxiv.org/abs/1911.10107)
    
27. Tutorial - Gym Trading Environment, acessado em julho 1, 2025, [https://gym-trading-env.readthedocs.io/en/latest/rl_tutorial.html](https://gym-trading-env.readthedocs.io/en/latest/rl_tutorial.html)
    
28. Reinforcement Learning for trading using Stable-Baselines3 - YouTube, acessado em julho 1, 2025, [https://www.youtube.com/watch?v=m_pmjaL_srg](https://www.youtube.com/watch?v=m_pmjaL_srg)
    
29. Mephistopheles-0/RL-trading-strategy: Utilizing ... - GitHub, acessado em julho 1, 2025, [https://github.com/Mephistopheles-0/RL-trading-strategy](https://github.com/Mephistopheles-0/RL-trading-strategy)
    
30. Deep Reinforcement Learning in Trading Algorithms - Digital Kenyon, acessado em julho 1, 2025, [https://digital.kenyon.edu/cgi/viewcontent.cgi?article=1008&context=dh_iphs_ai](https://digital.kenyon.edu/cgi/viewcontent.cgi?article=1008&context=dh_iphs_ai)
    
31. Sharpe ratio as a reward function for reinforcement learning trading agent : r/algotrading, acessado em julho 1, 2025, [https://www.reddit.com/r/algotrading/comments/8705zw/sharpe_ratio_as_a_reward_function_for/](https://www.reddit.com/r/algotrading/comments/8705zw/sharpe_ratio_as_a_reward_function_for/)
    
32. Reinforcement Learning for Trading Systems and Portfolios - CiteSeerX, acessado em julho 1, 2025, [https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=10f34407d0f7766cfb887334de4ce105d5aa8aae](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=10f34407d0f7766cfb887334de4ce105d5aa8aae)
    
33. Risk-Aware Reinforcement Learning Reward for Financial Trading - arXiv, acessado em julho 1, 2025, [https://arxiv.org/html/2506.04358v1](https://arxiv.org/html/2506.04358v1)
    
34. How to Build a Custom Trading Environment in OpenAI Gym,Step by Step Reinforcement Learning Tutorial - YouTube, acessado em julho 1, 2025, [https://www.youtube.com/watch?v=qSaNqp8Bcy4](https://www.youtube.com/watch?v=qSaNqp8Bcy4)
    
35. "Unlocking the Power of RL: Training a Bitcoin Trading Agent using Gymnasium & Stable Baselines3 - YouTube, acessado em julho 1, 2025, [https://www.youtube.com/watch?v=0xAZHHMjQVE](https://www.youtube.com/watch?v=0xAZHHMjQVE)
    
36. Make your own custom environment - Gymnasium Documentation, acessado em julho 1, 2025, [https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
    
37. Reinforcement Learning with Gymnasium: A Practical Guide - DataCamp, acessado em julho 1, 2025, [https://www.datacamp.com/tutorial/reinforcement-learning-with-gymnasium](https://www.datacamp.com/tutorial/reinforcement-learning-with-gymnasium)
    
38. Reinforcement Learning in Trading: A Perfect Match | by Leo Mercanti | Medium, acessado em julho 1, 2025, [https://medium.com/@leomercanti/mastering-the-markets-with-reinforcement-learning-62d99c4772b2](https://medium.com/@leomercanti/mastering-the-markets-with-reinforcement-learning-62d99c4772b2)
    
39. Reinforcement Learning (Part-8): Proximal Policy Optimization(PPO ..., acessado em julho 1, 2025, [https://medium.com/@sthanikamsanthosh1994/reinforcement-learning-part-8-proximal-policy-optimization-ppo-for-trading-9f1c3431f27d](https://medium.com/@sthanikamsanthosh1994/reinforcement-learning-part-8-proximal-policy-optimization-ppo-for-trading-9f1c3431f27d)
    
40. Stock Trading Strategy Developing Based on ... - Atlantis Press, acessado em julho 1, 2025, [https://www.atlantis-press.com/article/125989750.pdf](https://www.atlantis-press.com/article/125989750.pdf)
    
41. Proximal Policy Optimization (PPO) architecture | PPO Explained - YouTube, acessado em julho 1, 2025, [https://www.youtube.com/watch?v=7lDX7NZX94g](https://www.youtube.com/watch?v=7lDX7NZX94g)
    
42. Proximal Policy Optimisation from Scratch - RL - Kaggle, acessado em julho 1, 2025, [https://www.kaggle.com/code/auxeno/proximal-policy-optimisation-from-scratch-rl](https://www.kaggle.com/code/auxeno/proximal-policy-optimisation-from-scratch-rl)
    
43. Proximal Policy Optimization (PPO): From Control Systems to Bioengineering Applications | by Bay Chin | Medium, acessado em julho 1, 2025, [https://medium.com/@baychin/proximal-policy-optimization-ppo-from-control-systems-to-bioengineering-applications-3ff73bba4762](https://medium.com/@baychin/proximal-policy-optimization-ppo-from-control-systems-to-bioengineering-applications-3ff73bba4762)
    
44. Proximal Policy Optimization (PPO) for LLMs Explained Intuitively - YouTube, acessado em julho 1, 2025, [https://www.youtube.com/watch?v=8jtAzxUwDj0](https://www.youtube.com/watch?v=8jtAzxUwDj0)
    
45. Proximal Policy Optimization with PyTorch and Gymnasium - DataCamp, acessado em julho 1, 2025, [https://www.datacamp.com/tutorial/proximal-policy-optimization](https://www.datacamp.com/tutorial/proximal-policy-optimization)
    
46. Reinforcement Learning: A Practical Guide to Proximal Policy Optimization (PPO) - Medium, acessado em julho 1, 2025, [https://medium.com/@csobrinofm/reinforcement-learning-a-practical-guide-to-proximal-policy-optimization-ppo-276df3e5099e](https://medium.com/@csobrinofm/reinforcement-learning-a-practical-guide-to-proximal-policy-optimization-ppo-276df3e5099e)
    
47. Reinforcement Learning Tips and Tricks — Stable Baselines3 2.7 ..., acessado em julho 1, 2025, [https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
    
48. What Is Overfitting in Trading Strategies? - LuxAlgo, acessado em julho 1, 2025, [https://www.luxalgo.com/blog/what-is-overfitting-in-trading-strategies/](https://www.luxalgo.com/blog/what-is-overfitting-in-trading-strategies/)
    
49. What Is Overfitting in Algorithmic Trading? - Bookmap, acessado em julho 1, 2025, [https://bookmap.com/blog/what-is-overfitting-in-algorithmic-trading](https://bookmap.com/blog/what-is-overfitting-in-algorithmic-trading)
    
50. How to avoid overfitting in Reinforcement Learning - Data Science Stack Exchange, acessado em julho 1, 2025, [https://datascience.stackexchange.com/questions/60960/how-to-avoid-overfitting-in-reinforcement-learning](https://datascience.stackexchange.com/questions/60960/how-to-avoid-overfitting-in-reinforcement-learning)
    
51. Addressing Non-Stationarity in FX Trading with Online Model Selection of Offline RL Experts - RL@Polimi - Politecnico di Milano, acessado em julho 1, 2025, [https://rl.airlab.deib.polimi.it/wp-content/uploads/2022/10/ICAIF_2022___Expert_Learning-1.pdf](https://rl.airlab.deib.polimi.it/wp-content/uploads/2022/10/ICAIF_2022___Expert_Learning-1.pdf)
    
52. [1905.03970] Reinforcement Learning in Non-Stationary Environments - arXiv, acessado em julho 1, 2025, [https://arxiv.org/abs/1905.03970](https://arxiv.org/abs/1905.03970)
    
53. Tackling Non-Stationarity in Reinforcement Learning via Causal-Origin Representation, acessado em julho 1, 2025, [https://openreview.net/forum?id=HqmpIud9Uq](https://openreview.net/forum?id=HqmpIud9Uq)
    
54. [2206.13960] Dynamic Memory for Interpretable Sequential Optimisation - arXiv, acessado em julho 1, 2025, [https://arxiv.org/abs/2206.13960](https://arxiv.org/abs/2206.13960)
    
55. Python Backtesting Frameworks: Six Options to Consider - Pipekit, acessado em julho 1, 2025, [https://pipekit.io/blog/python-backtesting-frameworks-six-options-to-consider](https://pipekit.io/blog/python-backtesting-frameworks-six-options-to-consider)
    
56. Backtesting.py - Backtest trading strategies in Python, acessado em julho 1, 2025, [https://kernc.github.io/backtesting.py/](https://kernc.github.io/backtesting.py/)
    
57. Backtesting.py – An Introductory Guide to Backtesting with Python - Interactive Brokers, acessado em julho 1, 2025, [https://www.interactivebrokers.com/campus/ibkr-quant-news/backtesting-py-an-introductory-guide-to-backtesting-with-python/](https://www.interactivebrokers.com/campus/ibkr-quant-news/backtesting-py-an-introductory-guide-to-backtesting-with-python/)
    
58. ebrahimpichka/DeepRL-trade: Algorithmic Trading Using Deep Reinforcement Learning algorithms (PPO and DQN) - GitHub, acessado em julho 1, 2025, [https://github.com/ebrahimpichka/DeepRL-trade](https://github.com/ebrahimpichka/DeepRL-trade)
    
59. Trading with Reinforcement Learning in Python Part II: Application | Teddy Koker, acessado em julho 1, 2025, [https://teddykoker.com/2019/06/trading-with-reinforcement-learning-in-python-part-ii-application/](https://teddykoker.com/2019/06/trading-with-reinforcement-learning-in-python-part-ii-application/)
    
60. A Self-Rewarding Mechanism in Deep Reinforcement Learning for Trading Strategy Optimization - MDPI, acessado em julho 1, 2025, [https://www.mdpi.com/2227-7390/12/24/4020](https://www.mdpi.com/2227-7390/12/24/4020)
    

Deep Reinforcement Learning for Trading - IDEAS/RePEc, acessado em julho 1, 2025, [https://ideas.repec.org/p/arx/papers/1911.10107.html](https://ideas.repec.org/p/arx/papers/1911.10107.html)**