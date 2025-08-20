## The Quant's Imperative: Why Version Control is Non-Negotiable

In quantitative finance, the stakes are exceptionally high. A misplaced decimal, a flawed data cleaning script, or an outdated model parameter can lead to "misinformed decisions, regulatory penalties, or loss of trust among stakeholders".1 In this environment, version control is not merely a software development best practice; it is a fundamental pillar of risk management, regulatory compliance, and scientific rigor. It is the primary defense against the two greatest threats in the field: irreproducible results and catastrophic operational errors.

A Version Control System (VCS) is an essential process that "safeguards the integrity of your financial reporting" and, by extension, your entire quantitative research lifecycle.1 Its core function is to track every alteration made to a file or set of files over time, serving as a comprehensive "safety net" that gives teams the freedom to experiment without fear of causing irreparable harm.3 This system is built upon three pillars that are indispensable for any modern quantitative team.

1. **Reproducibility:** A quantitative strategy or research finding is fundamentally worthless if its results cannot be perfectly replicated. Version control is the only mechanism that can definitively link a specific backtest result, a risk report, or a research paper's figure to the exact versions of the code, data, and model parameters that produced it.4 This concept, known as "provenance," is critical.5 Without it, it becomes "impossible to know what version of the code and what version of the data was used to produce a particular figure" weeks or months after the fact, rendering the analysis scientifically invalid.5
    
2. **Auditability & Compliance:** The financial industry is heavily regulated. Every model, trading decision, and risk calculation must be defensible to internal compliance officers and external regulators like the SEC.1 A VCS provides a "clear audit trail" for every component of a trading strategy.1 Each change is recorded with a timestamp, the author's identity, and—through disciplined practice—a clear message explaining the reason for the change. This creates an immutable log that is invaluable during audits, allowing a firm to prove that its processes are rigorous and transparent.4
    
3. **Collaboration & Risk Mitigation:** Quantitative research is rarely a solo endeavor. It involves teams of researchers, data engineers, and portfolio managers.1 Without a VCS, collaboration descends into chaos, with team members emailing code snippets, working on different versions of a model, and risking "confusion, duplicated efforts, or even missing data".1 A VCS establishes a "single source of truth," ensuring that every team member is working from the same, most up-to-date codebase.3
    

Implementing a robust version control workflow does more than just improve a team's current processes; it is a strategic decision that enables future growth and complexity management. It is the technical and cultural shift that transforms quantitative research from a fragile, artisanal craft into a robust, industrial-scale process. A single researcher might manage a few models with ad-hoc file naming conventions (e.g., `strategy_final_v2_revised.py`), but a team of ten researchers working on dozens of interconnected models cannot operate this way without inviting disaster. A VCS provides the foundational layer of MLOps, allowing a quant team to scale its strategies and personnel without a corresponding explosion in operational risk.

## The Local Workflow: Tracking a Simple Trading Signal

This section provides a hands-on introduction to the fundamental Git commands, grounding them in the practical context of developing a simple quantitative signal. The core local workflow in Git revolves around a three-step cycle: modifying files in your working directory, staging those changes, and committing them to the project's history.

We will begin by creating a project to calculate the historical volatility of a stock, a common component in many financial models. First, create a project directory and a `src` folder within it.

### Initializing the Repository

To start tracking a project with Git, you must first initialize a repository. This is done with the `git init` command inside the project's root directory. This command creates a hidden `.git` subdirectory where Git stores all the metadata and object database for the project.6



```Bash
mkdir quant-volatility-project
cd quant-volatility-project
git init
```

### The Edit-Stage-Commit Cycle

Let's create a Python script to calculate historical volatility using the `yfinance` and `pandas` libraries. This script will download stock data and compute the annualized rolling volatility based on logarithmic returns.7

**Python Code Example: Calculating Historical Volatility**



```Python
# src/volatility.py
import yfinance as yf
import numpy as np
import pandas as pd

def calculate_historical_volatility(ticker, period="1y", window=252):
    """
    Downloads stock data and calculates annualized historical volatility.
    """
    # Download historical data using yfinance
    stock_data = yf.download(ticker, period=period, auto_adjust=True)
    
    # Calculate logarithmic daily returns
    log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    
    # Calculate rolling standard deviation of log returns
    rolling_volatility = log_returns.rolling(window=window).std()
    
    # Annualize the volatility
    annualized_volatility = rolling_volatility * np.sqrt(window)
    
    return annualized_volatility

if __name__ == '__main__':
    spy_volatility = calculate_historical_volatility("SPY")
    print("Annualized Volatility for SPY:")
    print(spy_volatility.tail())
```

1. **Checking the Status (`git status`):** After saving this file, you can check the state of your repository with `git status`. This is one of the most frequently used commands; it shows which files are new or modified and whether they are staged for the next commit.6
    
    
    
    ```Bash
    git status
    ```
    
    The output will indicate that `src/volatility.py` is an "untracked file."
    
2. **Staging Changes (`git add`):** Before you can commit a change, you must place it in the "staging area." This is an intermediate step that allows you to group related changes into a single, logical commit. The command for this is `git add`.6
    
    
    
    ```Bash
    git add src/volatility.py
    ```
    
    Running `git status` again will now show the file under "Changes to be committed." The staging area is not merely a technicality; it is a conceptual tool that encourages the creation of "atomic commits." For a quant, this means committing logically distinct pieces of research. For example, a bug fix in a data cleaning script and an experiment with a new alpha factor should be two separate commits. This discipline is crucial for isolating the impact of individual changes on model performance, making debugging and auditing far more effective. If a model's performance later degrades, it is much easier to pinpoint the exact change that caused the issue if commits are atomic.
    
3. **Committing Changes (`git commit`):** A commit permanently saves a snapshot of the staged changes to the repository's history. Each commit is a self-contained unit with a unique ID (a SHA-1 hash) and a descriptive message.6
    
    
    
    ```Bash
    git commit -m "feat: Implement historical volatility calculation"
    ```
    
    The `-m` flag allows for an inline commit message. A well-crafted message is vital for understanding the project's history.
    

### Inspecting History

Once commits are made, you can review the project's history.

- **`git log`:** This command displays a chronological list of commits, showing the hash, author, date, and commit message for each one.6 It provides a clear history of the project's evolution.
    
- **`git diff`:** This command is used to see the exact changes between commits, between a commit and the working directory, or between branches. For a quant, running `git diff` between the current version of a model and a version from three months ago can be an invaluable tool for debugging why its performance characteristics have changed.
    

## Branching Strategies for Alpha Exploration and Model Development

Branching is arguably the most powerful feature of Git for quantitative research. A branch is a "lightweight movable pointer to a commit".10 This design allows for the creation of isolated environments where new ideas—such as new alpha factors, alternative datasets, or different modeling techniques—can be developed and tested in parallel without jeopardizing the stability of the main, production-ready codebase.9

### Branching and Merging Mechanics

The primary commands for branching are straightforward:

- `git branch <branch-name>`: Creates a new branch.
    
- `git checkout <branch-name>`: Switches your working directory to the specified branch.11
    
- `git checkout -b <branch-name>`: A convenient shorthand that creates and switches to a new branch in one command.9
    

Once work on a branch is complete, the changes are integrated back into a primary branch (like `main` or `develop`) using the `git merge` command.12 Git performs merges in one of two ways:

1. **Fast-Forward Merge:** If the history of the branch you are merging into has not diverged, Git simply moves the pointer forward to the latest commit of the branch being merged. It's a linear progression.12
    
2. **Three-Way Merge:** If the two branches have diverged (i.e., commits have been made to both since they branched apart), Git creates a new "merge commit." This special commit has two parents and serves to tie the two independent histories back together.12
    

### Branching Workflows for Quant Teams

A disciplined branching strategy is a form of risk management. By isolating new, unproven code on separate branches, it prevents experimental work from destabilizing the core, validated models. This separation is the technical implementation of the conceptual wall between "research" and "production."

- **Feature Branching:** This is the most common and effective workflow. All new development happens on a dedicated branch (e.g., `feature/add-kalman-filter`) instead of directly on the main branch. Once the feature is complete and tested, it is merged back.9 For a quant, a "feature" is often an "experiment".9
    
- **Gitflow:** For more mature teams, the Gitflow workflow provides a robust structure for managing a complex development and release cycle.13 It uses several long-lived branches with specific roles, which can be adapted directly to a quantitative finance context.
    
    - `main`: This branch contains the code for the currently deployed, live trading strategy. It is the ultimate source of truth for production and should always be stable.
        
    - `develop`: This is the primary integration branch. Features and experiments that have been successfully backtested and peer-reviewed are merged here. It represents the "next" version of the strategy.
        
    - `feature/<feature-name>`: All new research and development occurs on these branches, which are created from `develop`.
        
    - `release/<version-number>`: When the `develop` branch is ready for a new production release, a `release` branch is created. This is where final, intensive backtesting, parameter tuning, and documentation updates occur.
        
    - `hotfix/<issue-name>`: If a critical bug is discovered in the live (`main`) trading model, a `hotfix` branch is created directly from `main` to address it immediately.
        

The following table maps the Gitflow model to a typical quantitative trading lifecycle, providing a clear blueprint for organizing work in a structured and safe manner.

|Branch Type|Purpose in Quant Finance|Example Branch Name|Branched From|Merged Into|
|---|---|---|---|---|
|`main`|Production-ready, live trading models. Tagged with versions.|`main`|-|`release/`, `hotfix/`|
|`develop`|Main integration branch for completed research. Represents the "next" version.|`develop`|`main`|`feature/`, `release/`, `hotfix/`|
|`feature/`|Isolate development of a new alpha signal, model, or data pipeline.|`feature/volatility-clustering-model`|`develop`|`develop`|
|`release/`|Prepare a new model version for deployment. Final backtesting and parameter lockdown.|`release/v2.1.0`|`develop`|`main`, `develop`|
|`hotfix/`|Address a critical bug in the live production model (e.g., incorrect order sizing).|`hotfix/fix-leverage-calculation`|`main`|`main`, `develop`|

## Team Collaboration with GitHub: From Solo Research to Team Alpha

While Git manages the local repository, a platform like GitHub is used to host the central, shared repository, enabling team collaboration. GitHub builds on Git's foundation by adding powerful features for code review, issue tracking, and automation.

### Remotes and Synchronization

A "remote" is a version of your repository that is hosted on the internet or a network, typically on GitHub. The three essential commands for synchronizing your local repository with a remote are:

1. `git clone <url>`: Creates a local copy of a remote repository on your machine. This is the first step when joining an existing project.15
    
2. `git pull`: Fetches changes from the remote repository and merges them into your current local branch. This is how you stay up-to-date with your teammates' work.8
    
3. `git push`: Uploads your committed local changes to the remote repository, sharing your work with the team.9
    

### The Pull Request Workflow: Peer Review for Quants

The heart of collaboration on GitHub is the **Pull Request (PR)**. A PR is a formal request to merge changes from one branch into another (e.g., merging a `feature/` branch into `develop`). In a quantitative context, a PR is far more than a simple code review; it is a critical _model validation_ and _knowledge transfer_ mechanism.

The workflow is as follows:

1. A quant completes their research on a feature branch and pushes it to GitHub.
    
2. They open a Pull Request on the GitHub website. The PR description is crucial and should contain a summary of the new alpha signal, key backtesting metrics (e.g., Sharpe ratio, max drawdown), and visualizations of performance.
    
3. Other team members are assigned as reviewers. They examine the code for correctness, critique the statistical methodology, and suggest improvements directly within the PR interface. This "in-built review process massively improves code quality" and, more importantly, the rigor of the research.9
    
4. This process forces the author to formalize and defend their work with data. Reviewers are not just asking, "Is this good Python code?" but also, "Is this a good strategy? Is the backtest methodology sound? Have you accounted for transaction costs?"
    
5. Once the PR is approved, it can be merged into the `develop` branch.
    

This workflow has a powerful secondary effect: it reduces "key-person risk." If a researcher leaves the firm, their entire thought process, the validation of their models, and the discussions with their peers are permanently documented in the closed PRs. The firm's intellectual property is retained in an auditable and understandable format.

### Resolving Merge Conflicts

A merge conflict occurs when two developers make competing changes to the same lines in the same file.11 Git cannot automatically decide which change to keep, so it pauses the merge and asks for human intervention.

To resolve a conflict:

1. Run `git status` to see which files are conflicted.
    
2. Open the conflicted file in a text editor. You will see conflict markers: `<<<<<<< HEAD` denotes the changes from your current branch, `=======` separates the conflicting sections, and `>>>>>>> <branch-name>` denotes the changes from the branch you are trying to merge.18
    
3. Manually edit the file to create the final, correct version, removing all conflict markers.
    
4. Use `git add <filename>` to mark the conflict as resolved.
    
5. Use `git commit` to finalize the merge.
    

If a merge becomes too complex, you can always abort it and return to the state before the merge began with `git merge --abort`.19

## Blueprint for a Professional Quant Repository

A well-structured repository is essential for clarity, reproducibility, and collaboration. It acts as a form of "defensive programming" for data science, preventing common and dangerous errors like committing sensitive credentials or bloating the repository with large data files.8

### Standard Directory Structure

Adopting a standard project layout ensures consistency and makes it easy for team members to navigate any project.22

```
project-root/
├──.github/workflows/         # GitHub Actions automation files
├──.gitignore                 # Files to ignore
├── README.md                  # Project overview, setup, and results
├── config/                    # Configuration files (e.g., model params)
├── data/
│   ├── raw/                   # Immutable raw data
│   └── processed/             # Cleaned and feature-engineered data
├── notebooks/                 # Exploratory analysis (Jupyter notebooks)
├── src/                       # Main source code (Python modules)
├── models/                    # Serialized/trained model files (e.g., using Git LFS)
└── requirements.txt           # Python dependencies
```

### The Art of the Atomic Commit

Clear, descriptive commit messages are vital for a readable project history.24

- **Structure:** Follow the 50/72 rule: a subject line of 50 characters or less, followed by a blank line, and a body with lines wrapped at 72 characters. Use the imperative mood (e.g., "Fix bug" not "Fixed bug").
    
- **Content:** The subject line should state _what_ changed. The body should explain _why_ the change was necessary.
    

**Example of a good quant commit message:**

```
feat: Add GARCH model for volatility forecasting

The previous historical volatility model was too simplistic and did not
capture volatility clustering effects observed in the data.

This commit replaces the simple rolling standard deviation with a GARCH(1,1)
model from the 'arch' library. This should provide more responsive
volatility estimates and improve risk management for the strategy.

Ref: #12 - Research new volatility models
```

### Mastering `.gitignore`

The `.gitignore` file tells Git which files or directories to ignore. This is critical for keeping the repository clean, secure, and performant.8

**What to Ignore:**

- **Data:** Never commit large data files. Add `data/raw/` and `data/processed/` to `.gitignore`.21 Committing a multi-gigabyte data file can permanently slow down repository operations for the entire team.
    
- **Secrets:** Never commit API keys, passwords, or credentials. Create a `.env` file for these and add `.env` to `.gitignore`.8 Bots constantly scan public GitHub repositories for exposed credentials.
    
- **Environment Artifacts:** Ignore Python virtual environments (`/venv/`), compiled files (`*.pyc`), and cache directories (`__pycache__/`).
    
- **Notebook Outputs:** Jupyter notebook outputs can bloat `.ipynb` files with non-code text and images, making diffs unreadable. It is a strong best practice to use a pre-commit hook to automatically strip these outputs before committing.8
    

### Handling Large Files: Git LFS

Git itself is not designed to handle large binary files like trained machine learning models or datasets.21 For this, the industry standard is

**Git LFS (Large File Storage)**. Git LFS replaces large files with small text pointers inside Git, while storing the actual file content on a remote server. This allows you to version models and smaller datasets without destroying your repository's performance.21

## Automating the Quant Pipeline: An Introduction to MLOps with GitHub Actions

Version control can be elevated from a passive record-keeping tool to an active automation engine using **GitHub Actions**. Actions provide a Continuous Integration and Continuous Deployment (CI/CD) platform built directly into GitHub, allowing you to automate workflows based on events like a `push` or the creation of a Pull Request.26 This democratizes MLOps (Machine Learning Operations), allowing even small teams to implement sophisticated automation that was previously the domain of dedicated infrastructure teams, significantly increasing research velocity and reliability.27

Workflows are defined in YAML files located in the `.github/workflows/` directory.

### Use Case 1: Automated Backtesting on Pull Requests

You can ensure that no new code is merged without passing a backtest. The following workflow triggers whenever a PR is opened against the `develop` branch. It checks out the code, installs dependencies, and runs a backtest script, providing immediate feedback on the PR.28

Example Backtest Script to Automate:

This simple script calculates a Sharpe ratio from strategy returns and asserts it meets a minimum threshold.



```Python
# src/backtest.py
import pandas as pd
import numpy as np

def run_backtest(strategy_returns_path):
    """
    A mock backtest function that calculates a Sharpe ratio.
    """
    strategy_returns = pd.read_csv(strategy_returns_path, index_col='Date', parse_dates=True)
    
    # Assume 252 trading days
    annualized_return = strategy_returns.mean() * 252
    annualized_volatility = strategy_returns.std() * np.sqrt(252)
    
    # Assume 2% risk-free rate
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    print(f"Backtest Complete. Sharpe Ratio: {sharpe_ratio.iloc:.2f}")
    
    # In a real scenario, we would assert the Sharpe ratio is above a threshold
    assert sharpe_ratio.iloc > 0.5, "Sharpe ratio is below the minimum threshold!"

if __name__ == '__main__':
    # Create a dummy returns file for the test to run
    dummy_returns = pd.DataFrame(np.random.normal(0.0005, 0.01, 1000), columns=['returns'])
    dummy_returns.to_csv('data/processed/strategy_returns.csv', index_label='Date')
    
    run_backtest('data/processed/strategy_returns.csv')
```

**YAML Workflow for Automated Backtesting:**



```YAML
#.github/workflows/backtest.yml
name: Run Backtest on PR

on:
  pull_request:
    branches: [ develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pandas numpy
        
    - name: Create dummy data directories
      run: mkdir -p data/processed
        
    - name: Run backtest
      run: python src/backtest.py
```

### Use Case 2: Scheduled Model Retraining

Actions can also be triggered on a schedule. This is perfect for tasks like weekly model retraining on new data.26 The

`on` key is modified to use a `cron` schedule.



```YAML
on:
  schedule:
    - cron: '0 0 * * 0' # Runs at midnight UTC on Sundays
```

### Use Case 3: Securely Using API Keys

Never hard-code API keys in your scripts. GitHub Secrets provides a secure way to store sensitive information. You can add a secret (e.g., `ALPHA_VANTAGE_API_KEY`) in your repository's settings, and then reference it in your workflow file as an environment variable.26



```YAML
- name: Run script with API key
  env:
    API_KEY: ${{ secrets.ALPHA_VANTAGE_API_KEY }}
  run: python src/data_pipeline.py
```

## Capstone Project: Building a Version-Controlled Markowitz Portfolio Optimizer

This capstone project will apply every concept from this chapter to a real-world quantitative finance problem. We will build, test, and automate a portfolio optimization model based on Harry Markowitz's Modern Portfolio Theory. Each logical step in the development process will correspond to a Git commit, demonstrating the version control workflow in action.

**Project Goal:** Develop a Python application that takes a list of stock tickers, calculates the optimal portfolio weights to maximize the Sharpe Ratio, and visualizes the efficient frontier.

---

### Step 1: Project Initialization

First, create the project directory structure as defined in Section 5 and initialize a Git repository.



```Bash
# Create directories
mkdir markowitz-optimizer
cd markowitz-optimizer
mkdir -p.github/workflows src data/raw data/processed notebooks config models

# Initialize Git
git init

# Create.gitignore
touch.gitignore
```

Populate `.gitignore` with a standard Python template, ensuring to include `data/`, `venv/`, `__pycache__/`, and `.env`.

**Commit 1:**



```Bash
git add.
git commit -m "init: Initial project structure and git setup"
```

---

### Step 2: Data Retrieval Module

Create a module in `src/data_fetcher.py` to download historical price data using `yfinance`.2



```Python
# src/data_fetcher.py
import yfinance as yf
import pandas as pd

def get_price_data(tickers, start_date, end_date):
    """Fetches adjusted close prices for a list of tickers."""
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']
```

**Commit 2:**



```Bash
git add src/data_fetcher.py
git commit -m "feat: Add module to fetch historical stock data"
```

---

### Step 3: Financial Calculations Module

Create a module `src/financial_metrics.py` to handle the core calculations for returns and risk.

The expected return of a portfolio is the weighted average of the individual asset returns:

$$E(R_p​)=w^Tμ$$

The portfolio variance (risk) is given by:

$$σ_p^2​=w^TΣw$$

Where w is the vector of asset weights, μ is the vector of expected asset returns, and Σ is the covariance matrix of asset returns.30



```Python
# src/financial_metrics.py
import numpy as np

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculates annualized portfolio performance."""
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std_dev
```

**Commit 3:**



```Bash
git add src/financial_metrics.py
git commit -m "feat: Implement portfolio return and volatility calculations"
```

---

### Step 4: Portfolio Optimization (Branching & Merging)

Now for the core logic. We will develop this on a new feature branch to isolate the work.



```Bash
git checkout -b feature/optimizer
```

The goal is to find the portfolio weights that maximize the Sharpe Ratio, defined as:

![[Pasted image 20250819182521.png]]

where Rf​ is the risk-free rate.32

We will use `scipy.optimize.minimize` to find the optimal weights. Since optimizers find minimums, we will minimize the _negative_ Sharpe Ratio.34



```Python
# src/optimizer.py
import numpy as np
from scipy.optimize import minimize
from.financial_metrics import calculate_portfolio_performance

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_returns, p_std_dev = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std_dev

def maximize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    """Finds the portfolio that maximizes the Sharpe ratio."""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]
    
    result = minimize(negative_sharpe_ratio, initial_guess, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x
```

Now, commit the changes on the branch, push to GitHub, open a Pull Request for review, and merge it into `develop` (or `main` for this simpler project).

**Commit 4 (on branch):**



```Bash
git add src/optimizer.py
git commit -m "feat: Implement Sharpe ratio maximization"
git push -u origin feature/optimizer
# --- Go to GitHub, open PR, review, and merge ---
```

---

### Step 5: Visualization and Reporting

Create a main script, `main.py`, to run the entire process and generate a plot of the efficient frontier using `matplotlib`.31



```Python
# main.py
# (This script would import from src modules, run the optimization,
# and generate a plot showing the efficient frontier and the optimal portfolio)
#... implementation details omitted for brevity...
```

**Commit 5:**



```Bash
git add main.py
git commit -m "feat: Add efficient frontier visualization and main script"
```

---

### Step 6: Automation with GitHub Actions

Finally, create a workflow to run this optimization automatically every week, generating a new report.



```YAML
#.github/workflows/scheduled_run.yml
name: Weekly Portfolio Optimization

on:
  schedule:
    - cron: '0 8 * * 1' # Run at 8 AM UTC every Monday

jobs:
  run-optimization:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install yfinance pandas numpy scipy matplotlib
      - name: Run main script
        run: python main.py
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: efficient-frontier-plot
          path: efficient_frontier.png
```

**Commit 6:**



```Bash
git add.github/workflows/scheduled_run.yml
git commit -m "ci: Add weekly scheduled portfolio optimization run"
```

This completes the capstone project, demonstrating a full, version-controlled workflow from idea to automated execution.

---

### Common Questions and Best Practices

- **Question:** My trained model file is 500MB. Should I commit it to Git?
    
    - **Answer:** No. Standard Git is not designed for large binary files. Doing so will severely degrade repository performance for the entire team. The correct approach is to use **Git LFS (Large File Storage)**. After installing Git LFS, you would track the model files by running `git lfs track "*.pkl"` (for pickle files, for example) and committing the `.gitattributes` file that this command creates. This ensures the large file is stored efficiently while still being version-controlled.21
        
- **Question:** I accidentally committed my API key to a public repository. What should I do?
    
    - **Answer:** You must assume the key is compromised. **Step 1: Immediately revoke the key** with your service provider. **Step 2: Remove the key from your project's history.** Merely deleting the key in a new commit is insufficient, as the old commit containing the key still exists in the history. You must use a tool like `git-filter-repo` or BFG Repo-Cleaner to completely purge the sensitive data from every commit in your history. This is a destructive operation and should be performed with care.8
        
- **Question:** Our team's research is highly experimental. Does a strict workflow like Gitflow slow us down?
    
    - **Answer:** While it may introduce more process steps than a completely unstructured approach, a disciplined workflow like Gitflow is designed to increase long-term velocity by preventing costly errors and rework. The structure of `feature` branches provides complete freedom to experiment in isolation. The PR process ensures that only validated, high-quality research is integrated into the main codebase. This structure prevents the "stable" `develop` branch from being polluted by half-finished or broken experiments, which ultimately saves significant time on debugging and integration, allowing the team to move faster and more safely.9

### References

**

1. Financial Reporting Document Versioning | Document Management ..., acessado em agosto 19, 2025, [https://docsvault.com/blog/version-control-in-finance/](https://docsvault.com/blog/version-control-in-finance/)
    
2. Stock Market Data: Obtaining Data, Visualization & Analysis in Python, acessado em agosto 19, 2025, [https://blog.quantinsti.com/stock-market-data-analysis-python/](https://blog.quantinsti.com/stock-market-data-analysis-python/)
    
3. What is version control? - GitLab, acessado em agosto 19, 2025, [https://about.gitlab.com/topics/version-control/](https://about.gitlab.com/topics/version-control/)
    
4. The importance of Data Versioning / Version Control | IoA - Institute of Analytics, acessado em agosto 19, 2025, [https://ioaglobal.org/blog/importance-of-data-versioning/](https://ioaglobal.org/blog/importance-of-data-versioning/)
    
5. Version Control - The Turing Way, acessado em agosto 19, 2025, [https://book.the-turing-way.org/reproducible-research/vcs](https://book.the-turing-way.org/reproducible-research/vcs)
    
6. How to Create a Git Repository | Atlassian Git Tutorial, acessado em agosto 19, 2025, [https://www.atlassian.com/git/tutorials/setting-up-a-repository](https://www.atlassian.com/git/tutorials/setting-up-a-repository)
    
7. Volatility And Measures Of Risk-Adjusted Return With Python - QuantInsti Blog, acessado em agosto 19, 2025, [https://blog.quantinsti.com/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/](https://blog.quantinsti.com/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/)
    
8. Git for Data Science: What every data scientist should know about Git, acessado em agosto 19, 2025, [https://valohai.com/blog/git-for-data-science/](https://valohai.com/blog/git-for-data-science/)
    
9. Branching Out: 4 Git Workflows for Collaborating on ML | Towards Data Science, acessado em agosto 19, 2025, [https://towardsdatascience.com/branching-out-4-git-workflows-for-collaborating-on-ml/](https://towardsdatascience.com/branching-out-4-git-workflows-for-collaborating-on-ml/)
    
10. Git Branch | Atlassian Git Tutorial, acessado em agosto 19, 2025, [https://www.atlassian.com/git/tutorials/using-branches](https://www.atlassian.com/git/tutorials/using-branches)
    
11. Git Branching and Merging Made Easy | by Abdullah Al Mamun | The Startup | Medium, acessado em agosto 19, 2025, [https://medium.com/swlh/git-branching-and-merging-made-easy-f7dacd4aa75e](https://medium.com/swlh/git-branching-and-merging-made-easy-f7dacd4aa75e)
    
12. Git Merge | Atlassian Git Tutorial, acessado em agosto 19, 2025, [https://www.atlassian.com/git/tutorials/using-branches/git-merge](https://www.atlassian.com/git/tutorials/using-branches/git-merge)
    
13. Gitflow Workflow | Atlassian Git Tutorial, acessado em agosto 19, 2025, [https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
    
14. Principled Git-based Workflow in Collaborative Data Science Projects, acessado em agosto 19, 2025, [https://ericmjl.github.io/essays-on-data-science/workflow/gitflow/](https://ericmjl.github.io/essays-on-data-science/workflow/gitflow/)
    
15. OpenBB-finance/OpenBB: Financial data platform for analysts, quants and AI agents., acessado em agosto 19, 2025, [https://github.com/OpenBB-finance/OpenBB](https://github.com/OpenBB-finance/OpenBB)
    
16. Git Remote | Atlassian Git Tutorial, acessado em agosto 19, 2025, [https://www.atlassian.com/git/tutorials/syncing](https://www.atlassian.com/git/tutorials/syncing)
    
17. A Step by Step Guide for How to Resolve Git Merge Conflicts - DevCamp, acessado em agosto 19, 2025, [https://vtm.devcamp.com/full-stack-development-javascript-python/guide/step-by-step-guide-how-to-resolve-git-merge-conflicts](https://vtm.devcamp.com/full-stack-development-javascript-python/guide/step-by-step-guide-how-to-resolve-git-merge-conflicts)
    
18. Resolving a merge conflict using the command line - GitHub Docs, acessado em agosto 19, 2025, [https://docs.github.com/articles/resolving-a-merge-conflict-using-the-command-line](https://docs.github.com/articles/resolving-a-merge-conflict-using-the-command-line)
    
19. medium.com, acessado em agosto 19, 2025, [https://medium.com/@python-javascript-php-html-css/resolving-git-merge-conflicts-aborting-a-merge-and-keeping-pulled-changes-b9e95eed14db#:~:text=By%20running%20git%20merge%20%E2%80%94%20abort,text%3DTrue%20parameters%20in%20subprocess.](https://medium.com/@python-javascript-php-html-css/resolving-git-merge-conflicts-aborting-a-merge-and-keeping-pulled-changes-b9e95eed14db#:~:text=By%20running%20git%20merge%20%E2%80%94%20abort,text%3DTrue%20parameters%20in%20subprocess.)
    
20. Git Merge Conflict Resolution: Cancelling a Merge and Maintaining Pulled Changes | by Denis Bélanger | Medium, acessado em agosto 19, 2025, [https://medium.com/@python-javascript-php-html-css/resolving-git-merge-conflicts-aborting-a-merge-and-keeping-pulled-changes-b9e95eed14db](https://medium.com/@python-javascript-php-html-css/resolving-git-merge-conflicts-aborting-a-merge-and-keeping-pulled-changes-b9e95eed14db)
    
21. 10 Best Practices for Data Science | by Benedict Neo - Medium, acessado em agosto 19, 2025, [https://medium.com/bitgrit-data-science-publication/10-best-practices-for-data-science-21a748a410e4](https://medium.com/bitgrit-data-science-publication/10-best-practices-for-data-science-21a748a410e4)
    
22. How to Use GitHub and Git for Collaborative Data Science Projects: A Complete Guide for Algerian Data Scientists, acessado em agosto 19, 2025, [https://arounddatascience.com/blog/coding/how-to-use-github-and-git-for-collaborative-data-science-projects-a-complete-guide-for-algerian-data-scientists/](https://arounddatascience.com/blog/coding/how-to-use-github-and-git-for-collaborative-data-science-projects-a-complete-guide-for-algerian-data-scientists/)
    
23. MAANG's Top 10 Git Practices for Streamlined Development - Data Science Dojo, acessado em agosto 19, 2025, [https://datasciencedojo.com/blog/maang-top-10-git-practices/](https://datasciencedojo.com/blog/maang-top-10-git-practices/)
    
24. Git best practices - Python for Data Science 24.3.0, acessado em agosto 19, 2025, [https://www.python4data.science/en/24.3.0/productive/git/best-practices.html](https://www.python4data.science/en/24.3.0/productive/git/best-practices.html)
    
25. Git for Data Science | DAGsHub, acessado em agosto 19, 2025, [https://dagshub.com/blog/how-to-use-git-for-data-science/](https://dagshub.com/blog/how-to-use-git-for-data-science/)
    
26. 4 Levels of GitHub Actions: A Guide to Data Workflow Automation, acessado em agosto 19, 2025, [https://towardsdatascience.com/4-levels-of-github-actions-a-guide-to-data-workflow-automation/](https://towardsdatascience.com/4-levels-of-github-actions-a-guide-to-data-workflow-automation/)
    
27. Using GitHub Actions for MLOps & Data Science, acessado em agosto 19, 2025, [https://github.blog/ai-and-ml/machine-learning/using-github-actions-for-mlops-data-science/](https://github.blog/ai-and-ml/machine-learning/using-github-actions-for-mlops-data-science/)
    
28. Introduction to GitHub Actions for Python Projects - PyImageSearch, acessado em agosto 19, 2025, [https://pyimagesearch.com/2024/09/30/introduction-to-github-actions-for-python-projects/](https://pyimagesearch.com/2024/09/30/introduction-to-github-actions-for-python-projects/)
    
29. Welcome to GitHub Actions for Scientific Data Workflows — GitHub Actions for Scientific Data Workflows (SciPy 2024), acessado em agosto 19, 2025, [https://scipy2024-githubactionstutorial.readthedocs.io/](https://scipy2024-githubactionstutorial.readthedocs.io/)
    
30. 5.3 Markowitz portfolio optimization, acessado em agosto 19, 2025, [https://mobook.github.io/MO-book/notebooks/05/03-markowitz-portfolio.html](https://mobook.github.io/MO-book/notebooks/05/03-markowitz-portfolio.html)
    
31. Markowitz portfolio optimization in Python/v3 - Plotly, acessado em agosto 19, 2025, [https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/](https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/)
    
32. How To Calculate The Sharpe Ratio In Python For Your Trading Strategy, acessado em agosto 19, 2025, [https://www.quantifiedstrategies.com/how-to-calculate-the-sharpe-ratio-in-python/](https://www.quantifiedstrategies.com/how-to-calculate-the-sharpe-ratio-in-python/)
    
33. Volatility And Measures Of Risk-Adjusted Return With Python, acessado em agosto 19, 2025, [https://www.quantinsti.com/blog/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/](https://www.quantinsti.com/blog/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/)
    

An Introduction to Portfolio Optimization in Python | Built In, acessado em agosto 19, 2025, [https://builtin.com/data-science/portfolio-optimization-python](https://builtin.com/data-science/portfolio-optimization-python)**