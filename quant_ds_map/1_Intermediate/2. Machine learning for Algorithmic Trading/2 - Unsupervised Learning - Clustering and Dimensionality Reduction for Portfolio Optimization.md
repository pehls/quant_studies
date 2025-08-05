## Introduction: Beyond Markowitz - The Quest for Robust Portfolios

The cornerstone of modern quantitative finance was laid in 1952 with Harry Markowitz's Nobel Prize-winning work on Mean-Variance Optimization (MVO).1 The framework is elegant in its simplicity: construct a portfolio that maximizes expected return for a given level of risk (variance) or, conversely, minimizes risk for a given level of expected return. For decades, this has been the textbook approach to asset allocation. However, practitioners who have attempted to implement MVO in the real world have often encountered what can be called the "Markowitz Paradox": a theoretically optimal model that frequently produces financially suboptimal results.

The practical failings of MVO are well-documented and stem primarily from its acute sensitivity to its inputs. The model requires precise estimates of the expected return for every asset, the variance of every asset, and the covariance between every pair of assets. In practice, these parameters are estimated from historical data and are notoriously noisy and unstable.2 The MVO framework, being a quadratic optimizer, tends to amplify the impact of these estimation errors. This phenomenon, often termed "error maximization," causes the optimizer to aggressively over-allocate to assets with spuriously high historical returns and underestimated risks, while shunning those with the opposite characteristics. The result is often portfolios that are unintuitive, highly concentrated in a few assets, and perform poorly out-of-sample.1

Compounding this problem is a fundamental challenge in modern finance: the **Curse of Dimensionality**.5 As the universe of investable assets expands to include hundreds or even thousands of stocks, ETFs, and other securities, the number of parameters required by MVO explodes. For

N assets, one must estimate N means, N variances, and N(N−1)/2 unique covariances.6 This leads to a cascade of issues:

- **Data Sparsity:** The feature space becomes so vast that the available historical data—even decades' worth—is insufficient to derive statistically reliable estimates for all parameters. In high-dimensional space, data points become sparse, and the concept of distance, which underpins correlation, becomes less meaningful.7
    
- **Computational Complexity:** The sheer size of the covariance matrix and the complexity of the optimization problem can become computationally intractable for very large N.7
    
- **Overfitting:** When the number of assets (p) is large relative to the number of time-series observations (N), a condition known as p>>N, models are highly susceptible to overfitting. They learn the noise and random fluctuations in the historical data rather than the true underlying signal, leading to poor generalization and disappointing real-world performance.5
    

This chapter presents a paradigm shift away from the rigid assumptions of classical optimization. We will explore how **unsupervised learning** provides a powerful, data-driven toolkit to address these challenges. Instead of imposing a strict mathematical model on noisy data, the unsupervised approach seeks to first _learn the inherent structure_ of the market's risk and correlation network. This philosophy acknowledges the high degree of uncertainty in financial markets and focuses on discovering more stable, structural properties within the data.1 The move is from a fragile

_prediction-then-optimization_ workflow to a more robust _structure-discovery-then-allocation_ process.9

We will investigate two pillars of this modern approach. First, we will use **Dimensionality Reduction** via Principal Component Analysis (PCA) to tame the curse of dimensionality, distilling the complex web of asset interactions into a few key, interpretable risk factors. Second, we will employ **Clustering** algorithms like K-Means and Hierarchical Clustering to identify natural groupings of assets, revealing the market's hidden taxonomy. Finally, we will combine these techniques to build sophisticated and robust portfolio construction methodologies, such as Hierarchical Risk Parity (HRP), that are designed for the complexity of real-world financial markets.

## Part 1: Unveiling Latent Risk Factors with Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a cornerstone of unsupervised learning and a powerful technique for dimensionality reduction. Its primary objective is to transform a dataset of potentially correlated variables into a new set of uncorrelated variables, called **principal components**, which are ordered by the amount of original variance they capture.11 In finance, this mathematical transformation provides a profound lens through which to view and understand the hidden drivers of portfolio risk.

### 1.1 The Mathematical Engine of PCA

At its core, PCA is an application of linear algebra, specifically the eigendecomposition of a covariance matrix. To understand how it works, we must follow a precise mathematical sequence.

#### Step 1: Data Standardization

Before applying PCA, it is crucial to standardize the input data. For a portfolio of assets, this means taking the time series of asset returns and transforming each series so that it has a mean of zero and a standard deviation of one.13 This step is critical because PCA is sensitive to the variance of the initial variables. Without standardization, an asset with high volatility would dominate the first principal component simply because of its scale, not because it represents a more important source of common risk.15

For a given asset return series Ri​ with mean μi​ and standard deviation σi​, the standardized return Zi​ is calculated as:

![[Pasted image 20250630223821.png]]

This ensures that each asset contributes equally to the analysis, allowing PCA to identify the underlying correlation structure without being biased by differing volatility levels.

#### Step 2: The Covariance Matrix

The next step is to compute the covariance matrix of the standardized asset returns. The covariance matrix, denoted by Σ, is a square matrix where each element Σij​ represents the covariance between asset i and asset j. The diagonal elements Σii​ represent the variance of each asset. Since we are using standardized data, the covariance matrix is identical to the correlation matrix.14

This matrix is the heart of the analysis, as it numerically captures the complete web of linear relationships between all pairs of assets in the portfolio. For a matrix of standardized returns Z with T observations and N assets, the covariance matrix is computed as:

![[Pasted image 20250630223829.png]]

#### Step 3: Eigendecomposition

Eigendecomposition is the mathematical process that breaks down the covariance matrix into its fundamental components: its eigenvectors and eigenvalues. This is the central operation of PCA.16

An eigenvector of the covariance matrix Σ is a non-zero vector v that, when multiplied by Σ, results in a scaled version of the same vector. The scaling factor is the eigenvalue, denoted by λ. The relationship is defined by the fundamental equation:

$$Σv=λv$$

This equation tells us that eigenvectors represent special directions in the data space. When the data is transformed by the covariance matrix, the eigenvectors do not change their direction; they are only stretched or compressed by a factor of their corresponding eigenvalue.

For an N×N covariance matrix (representing N assets), there will be N eigenvectors, each with a corresponding eigenvalue. These eigenvectors are orthogonal to each other, meaning they represent uncorrelated directions of variance in the data. The eigenvalues quantify the amount of variance captured along each of these eigenvector directions. By sorting the eigenvalues in descending order, we can identify the principal components in order of their importance.15

- The **1st Principal Component (PC1)** is the eigenvector corresponding to the largest eigenvalue (λ1​). It is the direction in the data that captures the maximum possible variance.
    
- The **2nd Principal Component (PC2)** is the eigenvector corresponding to the second-largest eigenvalue (λ2​). It captures the maximum remaining variance, subject to being orthogonal (uncorrelated) to PC1.
    
- This continues for all N components.
    

### 1.2 From Mathematics to Markets: Interpreting Eigen-Portfolios

The true power of PCA in finance comes from its economic interpretation. The abstract mathematical concepts of eigenvectors and eigenvalues translate directly into tangible portfolio concepts.19

- **Eigenvectors as Eigen-Portfolios:** Each eigenvector can be interpreted as a set of portfolio weights. Since an eigenvector has N components (one for each asset), it represents a specific, fixed-weight portfolio. These are called **eigen-portfolios**.20
    
- **Eigenvalues as Portfolio Risk:** The corresponding eigenvalue for each eigen-portfolio is its variance, or risk. A high eigenvalue means the associated eigen-portfolio is a significant source of risk in the system, while a low eigenvalue indicates a low-risk combination of assets.20
    

This interpretation allows us to decompose the total risk of the market into a set of uncorrelated sources of risk, each represented by an eigen-portfolio.

#### The Market Portfolio (PC1)

Empirical studies consistently show that for a broad basket of stocks, the first principal component (the eigenvector with the largest eigenvalue) has a clear economic meaning: it represents the **market factor**.20 The components of this eigenvector are typically all positive, meaning it represents a long-only portfolio. An investment in this eigen-portfolio is essentially an investment in the broad market. The corresponding largest eigenvalue,

λ1​, represents the variance of the market itself and often accounts for a substantial portion of the total variance in the system. The weight of each stock in this eigenvector indicates its sensitivity to the overall market movement.

#### Subsequent Portfolios (PC2, PC3,...)

Because all eigenvectors are orthogonal, the subsequent eigen-portfolios (PC2, PC3, etc.) are, by construction, uncorrelated with the market portfolio and with each other. They represent systematic risk factors beyond the broad market movement.19 These are often long-short portfolios that capture industry, style, or strategy-based effects. For instance, PC2 might be a portfolio that is long technology stocks and short utility stocks, representing a "growth vs. value" or "risk-on vs. risk-off" factor.22 PC3 might isolate the behavior of financial stocks against industrials.

This reveals a profound connection: applying PCA to the asset covariance matrix is equivalent to building a custom, data-driven factor model. Unlike traditional models like Fama-French, which pre-specify factors based on economic theory (e.g., size, value), PCA derives the statistically most dominant risk factors directly from the data for a given period and asset universe. It allows an analyst to discover the _true_ sources of risk driving their specific portfolio, rather than assuming it conforms to a generic academic model.

Furthermore, the eigen-portfolios associated with the _smallest_ eigenvalues are also informative. These represent portfolios with extremely low variance, indicating combinations of assets that are nearly risk-free relative to each other. Such portfolios form the basis of statistical arbitrage strategies, which seek to profit from temporary deviations from these stable, low-risk relationships.23

### 1.3 Practical Implementation: PCA for Asset Returns in Python

Let's translate this theory into practice. We will use Python to download data for a selection of stocks, perform PCA, and interpret the results. We will use `yfinance` for data, `pandas` for data handling, and `scikit-learn` for the PCA implementation.

First, we set up our environment and download historical price data for a diverse set of tickers.



```Python
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Define a universe of assets from different sectors
tickers =

# Download historical data
prices = yf.download(tickers, start='2019-01-01', end='2023-12-31')['Adj Close']

# Calculate daily returns
returns = prices.pct_change().dropna()

print("Asset Returns Head:")
print(returns.head())
```

With the returns data ready, we can now perform the PCA analysis. This involves standardizing the data, fitting the PCA model, and examining the components.11



```Python
# Step 1: Standardize the data
scaler = StandardScaler()
scaled_returns = scaler.fit_transform(returns)

# Step 2: Fit the PCA model
pca = PCA()
pca.fit(scaled_returns)

# Step 3: Analyze the results
explained_variance_ratio = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

print(f"\nExplained Variance Ratio for each component:\n{explained_variance_ratio}")
print(f"\nNumber of components: {pca.n_components_}")
```

A crucial part of PCA is determining how many components are significant. The **scree plot** helps visualize this by plotting the eigenvalues (or the explained variance) for each component in descending order. We look for an "elbow" in the plot, after which the eigenvalues drop off significantly.15



```Python
# Visualize the explained variance (Scree Plot)
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center',
        label='Individual explained variance')
plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.title('Scree Plot of Principal Components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
```

The scree plot will likely show that the first few components capture the vast majority of the total variance. Now, let's interpret the composition of the most important eigen-portfolios by examining their corresponding eigenvectors.20



```Python
# Create a DataFrame for the eigenvectors (Eigen-portfolios)
eigen_portfolios = pd.DataFrame(eigenvectors, columns=returns.columns, 
                                index=[f'PC{i+1}' for i in range(len(tickers))])

print("\nComposition of Eigen-portfolios:")
print(eigen_portfolios.head())

# Visualize the first few eigen-portfolios
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
axes = axes.flatten()

for i in range(4):
    pc_weights = eigen_portfolios.iloc[i]
    pc_weights.plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'Eigen-portfolio {i+1} (Risk: {eigenvalues[i]:.4f})')
    axes[i].set_ylabel('Weight')

plt.tight_layout()
plt.show()
```

From the visualizations, we can draw conclusions:

- **Eigen-portfolio 1 (PC1):** We would expect to see all bars being positive (or negative, as the sign of an eigenvector is arbitrary), confirming its role as the market portfolio. Assets with higher weights are more sensitive to broad market movements. Its eigenvalue will be significantly larger than the others.
    
- **Eigen-portfolio 2 (PC2):** This will be a long-short portfolio. For example, we might see positive weights on technology and consumer discretionary stocks and negative weights on financials and energy, representing a factor that pits growth sectors against value/cyclical sectors.
    
- **Subsequent Portfolios:** These will reveal other, more nuanced uncorrelated risk factors present in our chosen universe of assets.
    

By using PCA, we have successfully transformed a complex, correlated set of asset returns into a simplified, orthogonal set of risk factors, providing a much clearer picture of the forces driving our portfolio's behavior.

## Part 2: Grouping Assets with Clustering Algorithms

While PCA reveals latent factors that drive the entire market, clustering algorithms help us answer a different but related question: can we identify distinct groups or segments of assets that behave similarly? By grouping assets, we can simplify portfolio construction, enhance diversification, and better understand the taxonomy of the market. We will explore two primary methods: K-Means and Hierarchical Clustering.

### 2.1 K-Means Clustering: Segmenting by Similarity

K-Means is a popular and intuitive clustering algorithm that aims to partition a dataset into a pre-defined number (k) of distinct, non-overlapping clusters. Each data point belongs to only one cluster.24 In finance, this can be used to group stocks based on characteristics like their risk-return profiles, momentum, or valuation metrics.

#### Algorithmic Deep Dive

K-Means operates through a simple iterative process, often described as a form of the Expectation-Maximization (E-M) algorithm 26:

1. **Initialization:** First, we must choose the number of clusters, k. Then, k initial **centroids** (the center points of the clusters) are chosen. While they can be selected randomly from the data points, this can lead to poor results. A superior method is **k-means++**, which intelligently spreads out the initial centroids, leading to better and more consistent convergence.24
    
2. **Assignment Step (Expectation):** Each data point (e.g., each stock) is assigned to the nearest centroid. The "nearness" is typically measured using the squared Euclidean distance. This step forms k initial clusters.25
    
3. **Update Step (Maximization):** The position of each of the k centroids is recalculated by taking the mean of all the data points assigned to its cluster.24
    
4. **Repetition:** Steps 2 and 3 are repeated until a stopping criterion is met, such as the centroids no longer changing their positions, the cluster assignments stabilizing, or a maximum number of iterations being reached.24
    

The algorithm's objective is to minimize the **Within-Cluster Sum of Squares (WCSS)**, also known as inertia. This is the sum of the squared distances between each data point and its assigned cluster's centroid. For a set of clusters S={S1​,S2​,...,Sk​}, with centroids μi​ for each cluster Si​, the objective function is 25:

![[Pasted image 20250630223913.png]]

Minimizing WCSS leads to clusters that are internally compact and cohesive.

#### A Crucial Decision: How to Determine the Optimal Number of Clusters (k)

The biggest challenge in K-Means is choosing the right value for k. Since it's an unsupervised algorithm, we don't have a "correct" answer. Instead, we use heuristics to find a value of k that provides a good balance between capturing the data's structure and avoiding overfitting. Two popular methods are the Elbow Method and Silhouette Analysis.

The Elbow Method

This method involves running the K-Means algorithm for a range of k values (e.g., from 1 to 10) and calculating the WCSS for each run. A plot is then created with k on the x-axis and WCSS on the y-axis. As k increases, the WCSS will always decrease, because more clusters will naturally fit the data better. However, the rate of decrease slows down. We look for the "elbow" point on the graph—the point of inflection where adding another cluster does not lead to a significant reduction in WCSS. This point is considered a good candidate for the optimal k.30

Silhouette Analysis

Silhouette analysis is a more robust method that measures how well-separated the resulting clusters are. For each data point i, the silhouette score s(i) is calculated as 33:

![[Pasted image 20250630223926.png]]

Where:

- a(i): The average distance from point i to all other points in its own cluster (a measure of intra-cluster cohesion).
    
- b(i): The average distance from point i to all points in the _nearest neighboring cluster_ (a measure of inter-cluster separation).
    

The silhouette score ranges from -1 to 1 33:

- A score near **+1** indicates that the point is well-clustered, far from neighboring clusters.
    
- A score near **0** indicates that the point is on or very close to the decision boundary between two clusters.
    
- A score near **-1** indicates that the point may have been assigned to the wrong cluster.
    

To find the optimal k, we can compute the average silhouette score for all data points for different values of k. The k that yields the highest average silhouette score is often the best choice. Additionally, visualizing the silhouette plots for each k can be very insightful. A good clustering will show plots where most clusters have scores above the average and the clusters are of relatively uniform thickness (representing size).32

#### Code Example: Clustering Stocks by Risk and Return

Let's apply K-Means to cluster our assets based on their annualized return and volatility. This can help segment them into categories like "Low Risk, Low Return," "High Risk, High Return," etc..35



```Python
# Calculate annualized return and volatility
annual_returns = returns.mean() * 252
annual_volatility = returns.std() * np.sqrt(252)

# Create a DataFrame for clustering
risk_return_df = pd.DataFrame({'volatility': annual_volatility, 'return': annual_returns})

# --- Finding Optimal K using Elbow Method ---
from sklearn.cluster import KMeans

wcss =
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(risk_return_df)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Let's assume the elbow is at k=4 based on the plot
optimal_k = 4

# --- Apply K-Means with optimal K ---
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(risk_return_df)

# Add cluster labels to our DataFrame
risk_return_df['cluster'] = clusters

# --- Visualize the Clusters ---
plt.figure(figsize=(12, 8))
sns.scatterplot(data=risk_return_df, x='volatility', y='return', hue='cluster', palette='viridis', s=150, legend='full')

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')

# Annotate points with ticker names
for i, txt in enumerate(risk_return_df.index):
    plt.annotate(txt, (risk_return_df['volatility'][i], risk_return_df['return'][i]), xytext=(5,-5), textcoords='offset points')

plt.title('Stock Clusters based on Risk-Return Profile')
plt.xlabel('Annualized Volatility (Risk)')
plt.ylabel('Annualized Return')
plt.legend()
plt.grid(True)
plt.show()
```

This analysis partitions the assets into distinct groups, allowing a portfolio manager to, for example, ensure they select assets from different clusters to achieve better diversification beyond what simple correlation might suggest.35

### 2.2 Hierarchical Clustering: Building an Asset Taxonomy

Unlike K-Means, which produces a flat partitioning of data, Hierarchical Clustering builds a hierarchy of clusters, often visualized as a tree-like structure called a **dendrogram**. This method does not require us to specify the number of clusters beforehand and can reveal deeper, nested relationships between assets.37

#### The Agglomerative Approach

The most common form of hierarchical clustering is the **agglomerative** or "bottom-up" approach.38 The algorithm proceeds as follows:

1. **Initialization:** Start by treating each data point (each asset) as its own individual cluster.
    
2. **Iterative Merging:** Find the two closest clusters in the dataset and merge them into a single new cluster.
    
3. **Repetition:** Repeat Step 2 until all data points have been merged into one single, all-encompassing cluster.
    

The result is a complete hierarchy, from individual assets at the bottom to the entire market at the top. The dendrogram visualizes this process. The y-axis of the dendrogram represents the distance or dissimilarity at which the clusters were merged. By "cutting" the dendrogram at a specific height, one can obtain a specific number of clusters.38

#### The Importance of Linkage: Defining Cluster Distance

A critical choice in hierarchical clustering is the **linkage criterion**, which defines how the "distance" between two clusters is measured. This choice significantly impacts the shape and composition of the resulting clusters.38 The primary linkage methods are summarized below.

|**Linkage Method**|**Description**|
|:--|:--|
|**Single**|Distance between the closest pair of points in two clusters (minimum distance). Tends to produce elongated, chain-like clusters.|
|**Complete**|Distance between the farthest pair of points in two clusters (maximum distance). Tends to create compact, spherical clusters.|
|**Average**|Average distance between all pairs of points in two clusters. Provides a compromise between single and complete linkage.|
|**Ward's**|Minimizes the total within-cluster variance. Merges clusters that lead to the smallest increase in total within-cluster variance after merging. Favors compact, similarly sized clusters.|
#### Code Example: Generating an Asset Dendrogram

Let's use hierarchical clustering to explore the correlation structure of our asset universe. We will use the correlation distance as our metric, which is defined as![[Pasted image 20250630223955.png]], where ρij​ is the correlation between assets i and j. This distance is 0 for perfectly correlated assets and 1 for perfectly anti-correlated assets.



```Python
import scipy.cluster.hierarchy as sch

# Calculate the correlation matrix from returns
corr_matrix = returns.corr()

# Calculate the correlation distance matrix
# The distance is 0 for perfect correlation (1) and 1 for perfect anti-correlation (-1)
dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))

# Perform hierarchical clustering using Ward's linkage
# The condensed distance matrix is required by the linkage function
condensed_dist = sch.distance.squareform(dist_matrix)
linkage_matrix = sch.linkage(condensed_dist, method='ward')

# Plot the dendrogram
plt.figure(figsize=(15, 8))
dendrogram = sch.dendrogram(linkage_matrix, labels=returns.columns, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram of Assets (Ward Linkage)')
plt.xlabel('Assets')
plt.ylabel('Distance (Dissimilarity)')
plt.grid(axis='y')
plt.show()
```

The resulting dendrogram provides a rich visualization of the market's taxonomy. We can clearly see which assets are most similar (those that merge at low distances) and how they form larger super-clusters representing sectors or styles. For instance, we would expect to see `AAPL` and `MSFT` merge early, as would `JPM` and `GS`, and `XOM` and `CVX`. The dendrogram reveals not just _that_ assets are correlated, but it exposes the _nested structure_ of these correlations. This hierarchical information is precisely what advanced portfolio construction methods, which we explore next, are designed to leverage.37

## Part 3: Unsupervised Learning in Modern Portfolio Construction

Having explored the tools of dimensionality reduction and clustering, we now turn to their direct application in constructing investment portfolios. These modern techniques move beyond the limitations of MVO by incorporating the learned structure of the market directly into the allocation process. We will focus on two powerful approaches: Hierarchical Risk Parity (HRP) and a hybrid strategy combining PCA with clustering.

### 3.1 Hierarchical Risk Parity (HRP): A Structural Approach to Diversification

Hierarchical Risk Parity, developed by Marcos López de Prado, is a sophisticated portfolio optimization method inspired by graph theory and machine learning. It completely bypasses the need for inverting the covariance matrix, a major source of instability in MVO, and instead uses the hierarchical relationships discovered through clustering to build a diversified portfolio.46

The core idea of HRP is to allocate risk not among individual assets, but among the hierarchical clusters of assets. This provides diversification both _across_ different clusters and _within_ each cluster, leading to demonstrably more stable and robust portfolios compared to traditional optimizers.48

#### The HRP Algorithm Explained

The HRP algorithm consists of three distinct steps 44:

1. **Hierarchical Tree Clustering:** The algorithm begins by using hierarchical clustering to organize the assets based on their correlation structure. The correlation matrix is transformed into a distance matrix, and a linkage criterion (e.g., single or Ward's) is used to build a dendrogram that groups similar assets together.50 This step provides the structural backbone for the entire allocation process.
    
2. **Quasi-Diagonalization (Matrix Seriation):** The covariance matrix is reordered according to the structure of the dendrogram obtained in the first step. This process, known as seriation, arranges the rows and columns so that similar assets are placed next to each other. The effect is that the covariance matrix is transformed into a **quasi-diagonal** form, where the largest values (highest covariances) are concentrated along the main diagonal in block-like structures.44 This step is crucial because it organizes the risk information in a way that aligns with the market's hierarchical structure.
    
3. **Recursive Bisection:** This is the allocation step. The algorithm recursively traverses the hierarchical tree from the top down.
    
    - It starts with the full portfolio and bisects it into two sub-clusters based on the top-level split in the dendrogram.
        
    - The total variance of each of these two sub-clusters is calculated. A simple inverse-variance portfolio is typically used to compute the variance _within_ each cluster.
        
    - Capital is then allocated between the two sub-clusters in inverse proportion to their respective variances. The sub-cluster with lower risk receives a larger share of the capital.
        
    - This process is repeated for each sub-cluster. Each is split in two, and the capital allocated to it is further divided between its children based on their inverse variance.
        
    - The recursion continues until it reaches the individual assets at the leaves of the tree, at which point each asset has been assigned a final weight.48
        

This top-down, risk-splitting approach ensures that HRP respects the learned market structure. Unlike MVO, which is a "blind" quadratic optimizer that can produce extreme weights, HRP is a graph-based algorithm that allocates capital in a more intuitive and stable manner, leading to superior diversification.48 HRP fundamentally redefines diversification: it is no longer just about low pairwise correlations but about achieving a balance of risk across the

_structural blocks_ of the market. This makes it inherently more robust to market regime changes, where all correlations might increase simultaneously but the underlying hierarchy often remains more stable.

#### Code Example: Building an HRP Portfolio

The `PyPortfolioOpt` library provides a convenient implementation of HRP, making it straightforward to apply.



```Python
from pypfopt import HRPOpt

# We will use the same returns data from our previous examples
# returns = pd.DataFrame(...)

# Step 1 & 2: Initialize the HRPOpt object. 
# It takes returns or a covariance matrix.
# The clustering and quasi-diagonalization happen internally.
hrp = HRPOpt(returns)

# Step 3: Optimize to get the weights.
# We can specify the linkage method, 'single' is the default from the original paper.
hrp_weights = hrp.optimize(linkage_method='ward')

# Clean the weights (remove noise, round) and display
hrp_weights_cleaned = hrp.clean_weights()
print("Hierarchical Risk Parity Portfolio Weights:")
print(hrp_weights_cleaned)

# Visualize the allocation
pd.Series(hrp_weights_cleaned).plot.pie(figsize=(10, 10),
                                        title='HRP Portfolio Allocation',
                                        autopct='%1.1f%%')
plt.ylabel('') # remove the 'None' ylabel
plt.show()
```

The resulting pie chart will typically show a much more balanced and diversified allocation compared to what a standard MVO would produce, demonstrating HRP's practical benefits.

### 3.2 A Hybrid Approach: Combining PCA and Clustering

An even more advanced strategy involves combining PCA and clustering in a two-step process. This hybrid approach leverages the strengths of both techniques to create highly robust and economically meaningful asset groups for portfolio construction.9

#### The Rationale

The logic behind this approach is compelling. Raw asset returns are noisy. By first applying PCA, we can filter out this noise and reduce the dimensionality of the problem. Instead of working with dozens or hundreds of correlated asset returns, we can work with a handful of uncorrelated principal components (the eigen-portfolios), which represent the most significant, systematic risk factors driving the market.55

When we then perform clustering on these principal components, we are no longer grouping assets based on their noisy day-to-day price movements. Instead, we are grouping them based on their **shared exposures to the fundamental, underlying risk factors**. This leads to clusters that are more stable over time and have a clearer economic interpretation.10 For example, a cluster might emerge that contains stocks from different sectors (e.g., a high-end retailer, a luxury car maker, and a premium travel company) that are all highly sensitive to a "consumer sentiment" factor captured by one of the principal components.

#### The Workflow

The hybrid workflow is as follows 54:

1. **Perform PCA:** Apply PCA to the standardized asset returns matrix.
    
2. **Select Components:** Determine the number of principal components (N) to retain, typically by choosing enough to explain a significant portion of the total variance (e.g., 80% or more), as seen in the scree plot.56
    
3. **Extract Loadings:** For each asset, obtain its "loadings" on these N principal components. The loadings are simply the coefficients of the assets in the eigenvectors. This creates a new, lower-dimensional feature set for each asset.
    
4. **Cluster on Components:** Use the matrix of asset loadings as the input for a clustering algorithm like K-Means. This will group the assets based on their factor exposure profiles.
    
5. **Allocate Across Clusters:** Once the clusters are formed, a final portfolio can be constructed. A common approach is to first allocate capital _across_ the clusters (e.g., equal weight per cluster) and then allocate capital _within_ each cluster using a risk-based method like inverse-volatility or even a mini-HRP allocation.
    

#### Code Example: PCA followed by K-Means for Cluster-Based Allocation

This code demonstrates the full hybrid workflow.



```Python
# We use the scaled_returns and pca object from the PCA section

# Step 1 & 2: We have already performed PCA. Let's decide to keep the top 5 components.
n_components = 5
pca_loadings = pd.DataFrame(pca.components_[:n_components, :].T, 
                            index=returns.columns, 
                            columns=[f'PC{i+1}' for i in range(n_components)])

print("Asset Loadings on Principal Components:")
print(pca_loadings.head())

# Step 3: Cluster on the component loadings using K-Means
# We can use the Elbow method or Silhouette analysis on pca_loadings to find the optimal k.
# Let's assume we found k=4 to be optimal.
kmeans_pca = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
asset_clusters = kmeans_pca.fit_predict(pca_loadings)

# Create a DataFrame to view the clusters
clustered_assets = pd.DataFrame({'Ticker': returns.columns, 'Cluster': asset_clusters})
print("\nAssets grouped by PCA-based clusters:")
print(clustered_assets.sort_values('Cluster'))

# Step 4: Allocate across clusters
# A simple strategy: equal weight to each cluster.
# Then, within each cluster, use inverse-volatility weighting.

final_weights = {}
num_clusters = len(clustered_assets['Cluster'].unique())
weight_per_cluster = 1.0 / num_clusters

for i in range(num_clusters):
    cluster_assets = clustered_assets[clustered_assets['Cluster'] == i].tolist()
    
    # Calculate inverse volatility for assets within the cluster
    cluster_vol = returns[cluster_assets].std()
    inv_vol_weights = 1 / cluster_vol
    normalized_weights = inv_vol_weights / inv_vol_weights.sum()
    
    # Allocate the cluster's weight among its assets
    for ticker, weight in normalized_weights.items():
        final_weights[ticker] = weight * weight_per_cluster

# Display final portfolio weights
final_weights_series = pd.Series(final_weights).sort_index()
print("\nFinal Portfolio Weights from PCA+K-Means Hybrid Strategy:")
print(final_weights_series)

# Visualize the allocation
final_weights_series.plot.pie(figsize=(10, 10),
                              title='PCA + K-Means Hybrid Portfolio Allocation',
                              autopct='%1.1f%%')
plt.ylabel('')
plt.show()
```

This hybrid approach represents a state-of-the-art method in quantitative portfolio management. It combines the denoising and factor-discovery power of PCA with the segmentation capabilities of clustering to produce portfolios that are not only diversified in terms of assets but also in terms of their exposure to the underlying, systematic drivers of market risk.

## Part 4: Capstone Project: Designing and Evaluating Unsupervised Portfolio Strategies

Theory and individual code examples are essential, but true understanding comes from application. This capstone project will guide you through the end-to-end process of constructing, backtesting, and comparing three distinct portfolio strategies using real-world data. By completing this project, you will solidify your understanding of the concepts covered in this chapter and gain practical experience in modern portfolio construction.

### 4.1 Project Brief

**Objective:** To construct and evaluate the performance of three different portfolio allocation strategies over a specified historical period. The asset universe will consist of the components of the S&P 100 index, representing a diverse set of large-cap U.S. equities.

**Strategies to Compare:**

1. **Classical Benchmark:** The Global Minimum Variance (GMV) portfolio, a variant of MVO that solely minimizes risk.
    
2. **Structural Approach:** The Hierarchical Risk Parity (HRP) portfolio.
    
3. **Hybrid ML Approach:** A portfolio constructed by first using PCA for dimensionality reduction, then K-Means for clustering, followed by a risk-based allocation across and within clusters.
    

**Evaluation Period:**

- **Training/Lookback Period:** A rolling 3-year window will be used to calculate statistics (covariance, correlations, etc.).
    
- **Testing/Rebalancing Period:** Portfolios will be rebalanced quarterly from the start of 2018 to the end of 2023.
    

### 4.2 Step 1: Data Acquisition and Preparation (Question & Answer)

**_Question 1: How do you programmatically download historical price data for a large universe of stocks (e.g., S&P 100 components) and preprocess it for analysis?_**

**Answer:**

The first step in any quantitative project is to acquire and clean the necessary data. For a large universe like the S&P 100, we need an efficient way to download daily price data for all constituents. The `yfinance` library is an excellent tool for this purpose.59

The process involves three main stages:

1. **Obtaining the Ticker List:** We first need a list of the stock tickers that make up the S&P 100. For this project, we can define this list manually or source it from a reliable online source. It's important to note that index components change over time; for a rigorous backtest, one would use point-in-time constituent lists, but for this educational project, using the current list is sufficient.
    
2. **Downloading the Data:** The `yfinance.download()` function is highly efficient for fetching data for multiple tickers at once. We request the 'Adj Close' price, which is adjusted for dividends and stock splits, providing a true representation of total return performance. We will download data for a period that covers our entire backtest plus the initial 3-year lookback window (e.g., from 2015 to 2023).61
    
3. **Preprocessing:** Once we have the price data, we must convert it into a format suitable for our models.
    
    - **Calculating Returns:** Portfolio optimization models operate on returns, not prices. We will calculate daily returns using the `pct_change()` method in `pandas`.
        
    - **Handling Missing Data:** It's common for data to have missing values, especially for a large set of tickers over a long period (e.g., due to IPOs, delistings, or data errors). A simple and common approach is to use `dropna()` to remove any dates where any of our assets have missing price data, ensuring our return calculations are consistent across the entire universe.53
        

The following Python code implements this entire process.



```Python
# For this example, we'll use a smaller, representative subset of the S&P 100
# In a full project, you would replace this with the full list of ~100 tickers.
sp100_tickers =

# Define the full date range needed (backtest period + initial lookback)
start_date = '2015-01-01'
end_date = '2023-12-31'

# Download adjusted close prices
print("Downloading historical data...")
prices_df = yf.download(sp100_tickers, start=start_date, end=end_date)['Adj Close']

# Forward-fill any missing values, then back-fill, a common way to handle temporary gaps
prices_df.ffill(inplace=True)
prices_df.bfill(inplace=True)

# Drop any stocks that still have NaN values (e.g., not public for the whole period)
prices_df.dropna(axis=1, inplace=True)

print(f"Data downloaded for {prices_df.shape} stocks.")

# Calculate daily returns
returns_df = prices_df.pct_change().dropna()

print("\nSample of preprocessed daily returns data:")
print(returns_df.head())
```

### 4.3 Step 2: Portfolio Construction (Question & Answer)

**_Question 2: What is the process and Python code for constructing a Global Minimum Variance (GMV) portfolio using `PyPortfolioOpt`?_**

**Answer:**

The Global Minimum Variance (GMV) portfolio is a classic benchmark in portfolio optimization. It is a special case of Mean-Variance Optimization that completely ignores expected returns and focuses on a single objective: finding the combination of assets that results in the lowest possible portfolio volatility (risk).64 This makes it a useful benchmark because it is less sensitive to the highly unstable estimates of expected returns.

The `PyPortfolioOpt` library simplifies the construction of this portfolio. The process is as follows:

1. **Calculate the Covariance Matrix:** The only required input for GMV is the sample covariance matrix of asset returns. The library can calculate this directly from a returns DataFrame.
    
2. **Instantiate `EfficientFrontier`:** We create an instance of the `EfficientFrontier` class, passing the expected returns (which will be ignored but are required by the class structure) and the covariance matrix.
    
3. **Call `min_volatility()`:** This method runs the quadratic optimizer to find the set of weights that minimizes the portfolio variance.
    
4. **Extract Weights:** The resulting optimal weights are then extracted for use in the backtest.
    

Here is the Python function to construct the GMV portfolio for a given period of returns:



```Python
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def get_gmv_portfolio(historical_returns):
    """
    Calculates the Global Minimum Variance portfolio weights.
    """
    # Calculate sample covariance matrix
    S = risk_models.sample_cov(historical_returns)
    
    # We don't need expected returns for GMV, but the object requires it.
    # We can pass a zero vector.
    mu = np.zeros(len(historical_returns.columns))
    
    # Instantiate the optimizer
    ef = EfficientFrontier(mu, S)
    
    # Find the minimum volatility portfolio
    weights = ef.min_volatility()
    
    # Clean weights to remove noise
    cleaned_weights = ef.clean_weights()
    return cleaned_weights
```

**_Question 3: How do you implement the Hierarchical Risk Parity strategy in Python?_**

**Answer:**

As discussed previously, Hierarchical Risk Parity (HRP) is a modern approach that uses the market's correlation structure to build diversified portfolios. It avoids many of the pitfalls of MVO by not requiring matrix inversion and by being insensitive to expected return estimates.

The `PyPortfolioOpt` library provides a dedicated `HRPOpt` class that implements the entire three-step HRP algorithm (clustering, quasi-diagonalization, and recursive bisection) internally.46 The implementation is remarkably straightforward:

1. **Instantiate `HRPOpt`:** Create an instance of the `HRPOpt` class, passing it the historical returns DataFrame.
    
2. **Call `optimize()`:** This method runs the full HRP algorithm. We can specify the linkage method for the clustering step; 'ward' is a robust choice that tends to create balanced clusters.
    
3. **Extract Weights:** The optimal weights are then retrieved from the object.
    

Here is the Python function for the HRP strategy:



```Python
from pypfopt.hierarchical_portfolio import HRPOpt

def get_hrp_portfolio(historical_returns):
    """
    Calculates the Hierarchical Risk Parity portfolio weights.
    """
    # Instantiate the optimizer
    hrp = HRPOpt(historical_returns)
    
    # Optimize using 'ward' linkage for clustering
    weights = hrp.optimize(linkage_method='ward')
    
    # Clean weights
    cleaned_weights = hrp.clean_weights()
    return cleaned_weights
```

**_Question 4: How can you leverage PCA and K-Means to form asset clusters and build a diversified portfolio based on these clusters?_**

**Answer:**

This hybrid strategy represents a sophisticated, multi-stage machine learning approach to portfolio construction. The goal is to group assets based on their exposure to underlying risk factors (discovered by PCA) and then build a diversified portfolio on top of these discovered clusters.56

The workflow is as follows:

1. **Dimensionality Reduction with PCA:** We first apply PCA to the historical returns to identify the main drivers of variance (the eigen-portfolios).
    
2. **Feature Extraction:** We select the top `N` principal components that capture a significant amount of the total risk. The "loadings" of each asset on these components become our new feature set.
    
3. **Clustering with K-Means:** We use K-Means to cluster the assets based on their PCA loadings. This groups assets with similar factor exposures. We determine the optimal number of clusters, k, using the silhouette score.
    
4. **Intra- and Inter-Cluster Allocation:** We design a two-level allocation scheme:
    
    - **Inter-Cluster (Across Clusters):** Allocate capital to each of the k clusters. A simple and robust choice is equal weight (i.e., 1/k of the capital to each cluster).
        
    - **Intra-Cluster (Within Clusters):** Allocate the capital assigned to a cluster among its constituent assets. A risk-based approach like inverse-volatility weighting is suitable here, as it gives more weight to less risky assets within the same factor-exposure group.
        

This function encapsulates the entire hybrid strategy:



```Python
from sklearn.metrics import silhouette_score

def get_pca_kmeans_portfolio(historical_returns, max_clusters=10):
    """
    Calculates portfolio weights using a hybrid PCA + K-Means strategy.
    """
    # --- PCA Step ---
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(historical_returns)
    pca = PCA()
    pca.fit(scaled_returns)
    
    # Select components explaining ~80% of variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.where(cumulative_variance >= 0.80) + 1
    
    pca_loadings = pd.DataFrame(pca.components_[:n_components, :].T, 
                                index=historical_returns.columns, 
                                columns=[f'PC{i+1}' for i in range(n_components)])

    # --- K-Means Step with Optimal K ---
    best_k = 2
    best_score = -1
    
    # Find optimal k using silhouette score
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = kmeans.fit_predict(pca_loadings)
        score = silhouette_score(pca_loadings, labels)
        if score > best_score:
            best_score = score
            best_k = k

    kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_loadings)
    
    clustered_assets = pd.DataFrame({'Ticker': historical_returns.columns, 'Cluster': clusters})
    
    # --- Allocation Step ---
    final_weights = {}
    weight_per_cluster = 1.0 / best_k

    for i in range(best_k):
        cluster_tickers = clustered_assets[clustered_assets['Cluster'] == i].tolist()
        if not cluster_tickers:
            continue
            
        cluster_returns = historical_returns[cluster_tickers]
        cluster_vols = cluster_returns.std()
        
        # Inverse volatility weighting within the cluster
        inv_vols = 1 / cluster_vols
        weights_in_cluster = inv_vols / inv_vols.sum()
        
        for ticker, weight in weights_in_cluster.items():
            final_weights[ticker] = weight * weight_per_cluster
            
    return final_weights
```

### 4.4 Step 3: Backtesting and Performance Analysis (Question & Answer)

**_Question 5: How can we implement a simple rolling-window backtest to rebalance these portfolios periodically, and what are the key performance metrics?_**

**Answer:**

Backtesting is the process of simulating a trading strategy on historical data to assess its viability. For portfolio strategies, this typically involves a **rolling-window backtest**. While sophisticated libraries like `Zipline` or `backtrader` offer powerful event-driven engines, they can be complex to set up. For our purpose of comparing portfolio allocations, a simple `for` loop in `pandas` is more transparent and educational.65 The

`backtesting.py` library is excellent but is primarily designed for single-asset trading strategies, making it less suitable for our multi-asset portfolio rebalancing task.66

The logic of our rolling backtest is as follows:

1. **Define Time Periods:** We'll set a rebalancing frequency (e.g., quarterly) and a lookback window for calculations (e.g., 3 years).
    
2. **Iterate Through Time:** We create a loop that steps through our historical data one rebalancing period at a time.
    
3. **Slice Data:** In each iteration, we select the historical data corresponding to the lookback window.
    
4. **Construct Portfolios:** We use this slice of data to run our three allocation functions (`get_gmv_portfolio`, `get_hrp_portfolio`, `get_pca_kmeans_portfolio`) and determine the target weights for the upcoming period.
    
5. **Calculate Performance:** We apply these weights to the _actual_ returns of the next period (the out-of-sample data) to calculate the portfolio's performance.
    
6. **Store Results:** We store the weights and periodic returns for each strategy to build a complete performance history.
    

After the loop completes, we can use the full history of portfolio returns to calculate key performance metrics 68:

- **Annualized Return:** The geometric average return per year.
    
- **Annualized Volatility:** The standard deviation of returns, scaled to a year. A measure of risk.
    
- **Sharpe Ratio:** The ratio of excess return (over a risk-free rate) to volatility. The most common measure of risk-adjusted return.
    
- **Maximum Drawdown (MDD):** The largest peak-to-trough percentage drop in the portfolio's value. A measure of tail risk.
    
- **Calmar Ratio:** The ratio of annualized return to the maximum drawdown. It measures return per unit of extreme risk.
    
- **Weight Concentration:** The sum of weights in the top 5 or 10 holdings, to measure diversification.
    

The Python code below outlines this backtesting loop.



```Python
# --- Backtesting Engine ---
rebalance_dates = pd.date_range(start='2018-01-01', end=end_date, freq='QS') # Quarterly Start
lookback_years = 3

portfolio_returns = {}
for strategy_func in [get_gmv_portfolio, get_hrp_portfolio, get_pca_kmeans_portfolio]:
    strategy_name = strategy_func.__name__.replace('get_', '').replace('_portfolio', '').upper()
    print(f"\nBacktesting strategy: {strategy_name}...")
    
    strategy_history =
    
    for i in range(len(rebalance_dates)):
        rebal_date = rebalance_dates[i]
        
        # Define lookback period
        start_lookback = rebal_date - pd.DateOffset(years=lookback_years)
        
        # Slice historical data for training
        train_data = returns_df.loc[start_lookback:rebal_date]
        
        if train_data.empty:
            continue
        
        # Get portfolio weights
        weights = strategy_func(train_data)
        weights = pd.Series(weights)
        
        # Define testing period (next quarter)
        if i + 1 < len(rebalance_dates):
            end_period = rebalance_dates[i+1]
        else:
            end_period = returns_df.index[-1]
            
        test_data = returns_df.loc[rebal_date:end_period]
        
        # Calculate portfolio returns for the period
        period_returns = (test_data[weights.index] * weights).sum(axis=1)
        strategy_history.append(period_returns)
        
    portfolio_returns[strategy_name] = pd.concat(strategy_history)

# --- Performance Metrics Calculation ---
results =
for name, rets in portfolio_returns.items():
    cum_rets = (1 + rets).cumprod()
    
    # Calculate metrics
    total_return = cum_rets.iloc[-1] - 1
    n_years = len(rets) / 252
    annual_return = (1 + total_return) ** (1/n_years) - 1
    annual_vol = rets.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol # Assuming 0 risk-free rate
    
    # Max drawdown
    rolling_max = cum_rets.cummax()
    daily_drawdown = cum_rets / rolling_max - 1
    max_drawdown = daily_drawdown.min()
    
    calmar_ratio = annual_return / abs(max_drawdown)
    
    results.append({
        'Strategy': name,
        'Annualized Return': f"{annual_return:.2%}",
        'Annualized Volatility': f"{annual_vol:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Maximum Drawdown': f"{max_drawdown:.2%}",
        'Calmar Ratio': f"{calmar_ratio:.2f}"
    })

results_df = pd.DataFrame(results)
print("\n--- Backtesting Results ---")
print(results_df)

# Plot cumulative returns
plt.figure(figsize=(14, 8))
for name, rets in portfolio_returns.items():
    (1 + rets).cumprod().plot(label=name)
plt.title('Cumulative Portfolio Performance')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()
```

|Table 4.4.1: Capstone Project Backtesting Results|
|---|
|**Metric**|
|Annualized Return|
|Annualized Volatility|
|Sharpe Ratio|
|Maximum Drawdown|
|Calmar Ratio|

**_Question 6: Based on the empirical results, what are the relative strengths and weaknesses of each strategy in terms of risk-adjusted performance and diversification?_**

**Answer:**

Analyzing the (hypothetical) results from our backtest allows us to draw powerful conclusions about the practical trade-offs of each portfolio construction method. The quantitative metrics in Table 4.4.1, combined with our theoretical understanding, paint a clear picture.

- **Global Minimum Variance (GMV):** We would expect the GMV strategy to successfully achieve its primary goal: exhibiting the lowest **Annualized Volatility** among the three. However, this often comes at a cost. Because GMV is "unconstrained," it might place very large bets on a few, historically low-volatility assets or sectors (like utilities or consumer staples), leading to high weight concentration and poor diversification. This can result in a lower **Annualized Return** and, consequently, a mediocre **Sharpe Ratio**. Its performance during market downturns (reflected in **Maximum Drawdown**) might be good, but its recovery and overall growth potential could be limited.69
    
- **Hierarchical Risk Parity (HRP):** The HRP strategy is expected to be the star performer in terms of diversification and risk-adjusted returns. Its **Annualized Volatility** might be slightly higher than GMV's, but its **Annualized Return** should be substantially better, leading to a superior **Sharpe Ratio** and **Calmar Ratio**.48 The key strength of HRP lies in its structural approach. By allocating risk across clusters, it avoids the concentration issues of GMV. We would see this in a much lower weight concentration for its top holdings. This inherent diversification makes it more robust and adaptable to different market regimes, protecting it better during stress periods and allowing it to participate more broadly in rallies.49
    
- **PCA + K-Means Hybrid:** This strategy's performance is the most fascinating to analyze. Its results depend heavily on whether the data-driven clusters it identifies correspond to real, persistent economic factors. If successful, this strategy could achieve a unique risk-return profile. For example, it might identify a "high-momentum tech" cluster and a "stable dividend" cluster. By allocating between these, it could potentially achieve a high **Annualized Return** while managing risk through factor diversification. Its **Sharpe Ratio** could be competitive with, or even exceed, HRP's if the discovered factors provide a genuine diversification benefit. Its weakness is that it is more complex, and its performance relies on the stability of the factor structure discovered by PCA. If the underlying market structure shifts dramatically, the clusters may become less meaningful, requiring more frequent re-evaluation.
    

In conclusion, the backtest would likely demonstrate a clear progression: GMV provides a simple risk-reduction benchmark but is often impractical. HRP offers a massive improvement, delivering robust, well-diversified, and high-performing portfolios by respecting the market's hierarchical structure. The PCA+K-Means hybrid represents the frontier of this approach, offering the potential for even more tailored risk management by building portfolios based on data-driven economic factors rather than just individual assets.

## Conclusion: The New Frontier of Data-Driven Asset Allocation

This chapter has charted a course from the foundational, yet fragile, principles of classical portfolio optimization to the robust, data-driven frontier of unsupervised machine learning. We began by acknowledging the practical limitations of Mean-Variance Optimization, particularly its instability in the face of estimation error and the curse of dimensionality. This set the stage for a necessary evolution in thinking—a shift from attempting to predict the future with precision to learning the underlying structure of the present with robustness.

Through Principal Component Analysis, we discovered a powerful method to distill the complex web of asset correlations into a small number of uncorrelated risk factors, or "eigen-portfolios." This not only tames dimensionality but also provides a profound economic lens, revealing the market's primary drivers of risk, from the broad market factor to nuanced, long-short style factors.

With clustering algorithms like K-Means and Hierarchical Clustering, we developed a taxonomy for the market, grouping assets into meaningful segments based on their risk-return profiles or their nested correlation structures. The dendrogram from hierarchical clustering, in particular, provided a rich map of the market's internal relationships.

Finally, we synthesized these techniques into modern portfolio construction frameworks. Hierarchical Risk Parity emerged as a superior alternative to MVO, leveraging the market's hierarchy to build naturally diversified and stable portfolios. The hybrid PCA-plus-clustering approach pushed this concept further, creating portfolios diversified not just across assets, but across fundamental, data-driven risk factors.

The journey through this chapter demonstrates that unsupervised learning is not merely a collection of algorithms; it is a transformative approach to asset allocation. It empowers the quantitative analyst to move beyond rigid assumptions and let the data reveal its own structure, leading to portfolios that are more intuitive, more robust, and better suited for the complexities of real-world financial markets. This is, however, only the beginning. The concepts explored here serve as the foundation for even more advanced techniques, such as using non-linear dimensionality reduction with autoencoders or employing deep learning architectures to model the intricate dynamics of financial data, opening up new and exciting frontiers in the quest for optimal investment strategies.49

## References
**

1. Machine Learning Optimization Algorithms & Portfolio Allocation - Amundi Research Center, acessado em junho 30, 2025, [https://research-center.amundi.com/article/machine-learning-optimization-algorithms-portfolio-allocation](https://research-center.amundi.com/article/machine-learning-optimization-algorithms-portfolio-allocation)
    
2. Machine Learning and Portfolio Optimization - LBS Research Online, acessado em junho 30, 2025, [https://lbsresearch.london.edu/545/1/Bahn_GY_Machine_Learning_Portfolio_Optimization_Mgt_Sci_2016.pdf](https://lbsresearch.london.edu/545/1/Bahn_GY_Machine_Learning_Portfolio_Optimization_Mgt_Sci_2016.pdf)
    
3. [2409.09684] Anatomy of Machines for Markowitz: Decision-Focused Learning for Mean-Variance Portfolio Optimization - arXiv, acessado em junho 30, 2025, [https://arxiv.org/abs/2409.09684](https://arxiv.org/abs/2409.09684)
    
4. Markowitz Mean-Variance Portfolio Optimization with Predictive Stock Selection Using Machine Learning - MDPI, acessado em junho 30, 2025, [https://www.mdpi.com/2227-7072/10/3/64](https://www.mdpi.com/2227-7072/10/3/64)
    
5. What is Curse of Dimensionality? A Complete Guide - Built In, acessado em junho 30, 2025, [https://builtin.com/data-science/curse-dimensionality](https://builtin.com/data-science/curse-dimensionality)
    
6. Curse of dimensionality - Wikipedia, acessado em junho 30, 2025, [https://en.wikipedia.org/wiki/Curse_of_dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
    
7. Curse of Dimensionality | Deepgram, acessado em junho 30, 2025, [https://deepgram.com/ai-glossary/curse-of-dimensionality](https://deepgram.com/ai-glossary/curse-of-dimensionality)
    
8. The Curse of Dimensionality - Domino Data Lab, acessado em junho 30, 2025, [https://domino.ai/blog/the-curse-of-dimensionality](https://domino.ai/blog/the-curse-of-dimensionality)
    
9. Machine Learning and Factor-Based Portfolio Optimization - University College Dublin, acessado em junho 30, 2025, [https://www.ucd.ie/geary/static/publications/workingpapers/gearywp202111.pdf](https://www.ucd.ie/geary/static/publications/workingpapers/gearywp202111.pdf)
    
10. Principal Component Analysis for Clustering Stock Portfolios - Journals at the University of Arizona, acessado em junho 30, 2025, [https://journals.librarypublishing.arizona.edu/azjis/article/id/2384/print/](https://journals.librarypublishing.arizona.edu/azjis/article/id/2384/print/)
    
11. Principal Component Analysis (PCA) in Python Tutorial - DataCamp, acessado em junho 30, 2025, [https://www.datacamp.com/tutorial/principal-component-analysis-in-python](https://www.datacamp.com/tutorial/principal-component-analysis-in-python)
    
12. Principal Component Analysis Made Easy & How To Python Tutorial With Scikit-Learn, acessado em junho 30, 2025, [https://spotintelligence.com/2023/08/25/principal-component-analysis/](https://spotintelligence.com/2023/08/25/principal-component-analysis/)
    
13. In this post, I share my Python implementations of Principal Component Analysis (PCA) from scratch. - Alireza Bagheri, acessado em junho 30, 2025, [https://bagheri365.github.io/blog/Principal-Component-Analysis-from-Scratch/](https://bagheri365.github.io/blog/Principal-Component-Analysis-from-Scratch/)
    
14. PCA Algorithm Tutorial in Python - Accel.AI, acessado em junho 30, 2025, [https://www.accel.ai/anthology/2022/4/5/pca-algorithm-tutorial-innbsppython](https://www.accel.ai/anthology/2022/4/5/pca-algorithm-tutorial-innbsppython)
    
15. Principal Component Analysis (PCA) - Analytics Vidhya, acessado em junho 30, 2025, [https://www.analyticsvidhya.com/blog/2016/03/pca-practical-guide-principal-component-analysis-python/](https://www.analyticsvidhya.com/blog/2016/03/pca-practical-guide-principal-component-analysis-python/)
    
16. Unlocking PCA in Finance - Number Analytics, acessado em junho 30, 2025, [https://www.numberanalytics.com/blog/ultimate-guide-pca-linear-algebra-finance](https://www.numberanalytics.com/blog/ultimate-guide-pca-linear-algebra-finance)
    
17. Mastering PCA: Eigenvectors, Eigenvalues, and Covariance Matrix Explained - CodeSignal, acessado em junho 30, 2025, [https://codesignal.com/learn/courses/navigating-data-simplification-with-pca/lessons/mastering-pca-eigenvectors-eigenvalues-and-covariance-matrix-explained](https://codesignal.com/learn/courses/navigating-data-simplification-with-pca/lessons/mastering-pca-eigenvectors-eigenvalues-and-covariance-matrix-explained)
    
18. Principal Component Analysis Made Easy: A Step-by-Step Tutorial - Medium, acessado em junho 30, 2025, [https://medium.com/data-science/principal-component-analysis-made-easy-a-step-by-step-tutorial-184f295e97fe](https://medium.com/data-science/principal-component-analysis-made-easy-a-step-by-step-tutorial-184f295e97fe)
    
19. Eigenvalues and Eigenvectors in Quantitative Finance | by DeVillar | Renata Villar - Medium, acessado em junho 30, 2025, [https://medium.com/@devillar/eigenvalues-and-eigenvectors-in-quantitative-finance-d040b1517d05](https://medium.com/@devillar/eigenvalues-and-eigenvectors-in-quantitative-finance-d040b1517d05)
    
20. Eigen-vesting I. Linear Algebra Can Help You Choose Your Stock Portfolio - Scott Rome, acessado em junho 30, 2025, [https://srome.github.io/Eigenvesting-I-Linear-Algebra-Can-Help-You-Choose-Your-Stock-Portfolio/](https://srome.github.io/Eigenvesting-I-Linear-Algebra-Can-Help-You-Choose-Your-Stock-Portfolio/)
    
21. The roles of Eigenvector and Eigenvalue in Layman's terms… - Finance Tutoring, acessado em junho 30, 2025, [https://www.finance-tutoring.fr/the-roles-of-eigenvector-and-eigenvalue-in-layman%E2%80%99s-terms%E2%80%A6?mobile=1](https://www.finance-tutoring.fr/the-roles-of-eigenvector-and-eigenvalue-in-layman%E2%80%99s-terms%E2%80%A6?mobile=1)
    
22. An Analysis of Eigenvectors of a Stock Market Cross-Correlation Matrix - ResearchGate, acessado em junho 30, 2025, [https://www.researchgate.net/publication/321947886_An_Analysis_of_Eigenvectors_of_a_Stock_Market_Cross-Correlation_Matrix](https://www.researchgate.net/publication/321947886_An_Analysis_of_Eigenvectors_of_a_Stock_Market_Cross-Correlation_Matrix)
    
23. Statistical Arbitrage Using Eigenportfolios | Swetava Ganguli, acessado em junho 30, 2025, [https://swetava.wordpress.com/wp-content/uploads/2015/08/swetava-ganguli-statistical-arbitrage-of-eigenportfolios.pdf](https://swetava.wordpress.com/wp-content/uploads/2015/08/swetava-ganguli-statistical-arbitrage-of-eigenportfolios.pdf)
    
24. K-Means Clustering Algorithm - Analytics Vidhya, acessado em junho 30, 2025, [https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/)
    
25. k-means clustering - Wikipedia, acessado em junho 30, 2025, [https://en.wikipedia.org/wiki/K-means_clustering](https://en.wikipedia.org/wiki/K-means_clustering)
    
26. K-Means Clustering Explained - neptune.ai, acessado em junho 30, 2025, [https://neptune.ai/blog/k-means-clustering](https://neptune.ai/blog/k-means-clustering)
    
27. Mathematics behind K-Mean Clustering algorithm - Muthukrishnan, acessado em junho 30, 2025, [https://muthu.co/mathematics-behind-k-mean-clustering-algorithm/](https://muthu.co/mathematics-behind-k-mean-clustering-algorithm/)
    
28. Getting started with K-means clustering in Python - Domino Data Lab, acessado em junho 30, 2025, [https://domino.ai/blog/getting-started-with-k-means-clustering-in-python](https://domino.ai/blog/getting-started-with-k-means-clustering-in-python)
    
29. What is k-means clustering? - IBM, acessado em junho 30, 2025, [https://www.ibm.com/think/topics/k-means-clustering](https://www.ibm.com/think/topics/k-means-clustering)
    
30. Elbow Method for optimal value of k in KMeans - GeeksforGeeks, acessado em junho 30, 2025, [https://www.geeksforgeeks.org/machine-learning/elbow-method-for-optimal-value-of-k-in-kmeans/](https://www.geeksforgeeks.org/machine-learning/elbow-method-for-optimal-value-of-k-in-kmeans/)
    
31. Elbow Method — Yellowbrick v1.5 documentation, acessado em junho 30, 2025, [https://www.scikit-yb.org/en/latest/api/cluster/elbow.html](https://www.scikit-yb.org/en/latest/api/cluster/elbow.html)
    
32. Elbow Method in K-Means Clustering: Definition, Drawbacks, vs. Silhouette Score - Built In, acessado em junho 30, 2025, [https://builtin.com/data-science/elbow-method](https://builtin.com/data-science/elbow-method)
    
33. K-Means: Getting the Optimal Number of Clusters - Analytics Vidhya, acessado em junho 30, 2025, [https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/](https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/)
    
34. Selecting the number of clusters with silhouette analysis on KMeans clustering - Scikit-learn, acessado em junho 30, 2025, [https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
    
35. How to Perform Cluster Analysis of Stocks Using K-Means? - YouTube, acessado em junho 30, 2025, [https://www.youtube.com/watch?v=Axi9rBCZ7GA](https://www.youtube.com/watch?v=Axi9rBCZ7GA)
    
36. K-Means Clustering in Banking: Applications & Examples - Datrics AI, acessado em junho 30, 2025, [https://www.datrics.ai/articles/how-k-means-clustering-is-transforming-the-banking-sector](https://www.datrics.ai/articles/how-k-means-clustering-is-transforming-the-banking-sector)
    
37. Hierarchical Clustering in Python: A Comprehensive Implementation Guide – Part IV, acessado em junho 30, 2025, [https://www.interactivebrokers.com/campus/ibkr-quant-news/hierarchical-clustering-in-python-a-comprehensive-implementation-guide-part-iv/](https://www.interactivebrokers.com/campus/ibkr-quant-news/hierarchical-clustering-in-python-a-comprehensive-implementation-guide-part-iv/)
    
38. Hierarchical Clustering Comprehensive & Practical How To Guide In Python, acessado em junho 30, 2025, [https://spotintelligence.com/2023/09/12/hierarchical-clustering-comprehensive-practical-how-to-guide-in-python/](https://spotintelligence.com/2023/09/12/hierarchical-clustering-comprehensive-practical-how-to-guide-in-python/)
    
39. Mastering Hierarchical Clustering : From Basic to Advanced | by Sachinsoni | Medium, acessado em junho 30, 2025, [https://medium.com/@sachinsoni600517/mastering-hierarchical-clustering-from-basic-to-advanced-5e770260bf93](https://medium.com/@sachinsoni600517/mastering-hierarchical-clustering-from-basic-to-advanced-5e770260bf93)
    
40. Hierarchical Clustering in Python: A Comprehensive Implementation ..., acessado em junho 30, 2025, [https://blog.quantinsti.com/hierarchical-clustering-python/](https://blog.quantinsti.com/hierarchical-clustering-python/)
    
41. Types of Linkages in Hierarchical Clustering - GeeksforGeeks, acessado em junho 30, 2025, [https://www.geeksforgeeks.org/machine-learning/ml-types-of-linkages-in-clustering/](https://www.geeksforgeeks.org/machine-learning/ml-types-of-linkages-in-clustering/)
    
42. 14.4 - Agglomerative Hierarchical Clustering | STAT 505, acessado em junho 30, 2025, [https://online.stat.psu.edu/stat505/lesson/14/14.4](https://online.stat.psu.edu/stat505/lesson/14/14.4)
    
43. Different Linkage Methods used in Hierarchical Clustering - Medium, acessado em junho 30, 2025, [https://medium.com/@iqra.bismi/different-linkage-methods-used-in-hierarchical-clustering-627bde3787e8](https://medium.com/@iqra.bismi/different-linkage-methods-used-in-hierarchical-clustering-627bde3787e8)
    
44. Hierarchical Risk Parity: Introducing Graph Theory and Machine Learning in Portfolio Optimizer, acessado em junho 30, 2025, [https://portfoliooptimizer.io/blog/hierarchical-risk-parity-introducing-graph-theory-and-machine-learning-in-portfolio-optimizer/](https://portfoliooptimizer.io/blog/hierarchical-risk-parity-introducing-graph-theory-and-machine-learning-in-portfolio-optimizer/)
    
45. What is Hierarchical Clustering? - IBM, acessado em junho 30, 2025, [https://www.ibm.com/think/topics/hierarchical-clustering](https://www.ibm.com/think/topics/hierarchical-clustering)
    
46. pypfopt.hierarchical_portfolio — PyPortfolioOpt 1.4.1 documentation, acessado em junho 30, 2025, [https://pyportfolioopt.readthedocs.io/en/stable/_modules/pypfopt/hierarchical_portfolio.html](https://pyportfolioopt.readthedocs.io/en/stable/_modules/pypfopt/hierarchical_portfolio.html)
    
47. PyPortfolioOpt/cookbook/5-Hierarchical-Risk-Parity.ipynb at master - GitHub, acessado em junho 30, 2025, [https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/5-Hierarchical-Risk-Parity.ipynb](https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/5-Hierarchical-Risk-Parity.ipynb)
    
48. Hierarchical Risk Parity - Asset Allocation - YouTube, acessado em junho 30, 2025, [https://www.youtube.com/watch?v=e21MfMe5vtU](https://www.youtube.com/watch?v=e21MfMe5vtU)
    
49. Portfolio Optimization – A Comparative Study - arXiv, acessado em junho 30, 2025, [https://arxiv.org/abs/2307.05048](https://arxiv.org/abs/2307.05048)
    
50. Hierarchical Risk Parity | Python | Riskfolio-Lib - Medium, acessado em junho 30, 2025, [https://medium.com/@orenji.eirl/hierarchical-risk-parity-with-python-and-riskfolio-lib-c0e60b94252e](https://medium.com/@orenji.eirl/hierarchical-risk-parity-with-python-and-riskfolio-lib-c0e60b94252e)
    
51. The Hierarchical Risk Parity Algorithm: An Introduction - Hudson & Thames, acessado em junho 30, 2025, [https://hudsonthames.org/an-introduction-to-the-hierarchical-risk-parity-algorithm/](https://hudsonthames.org/an-introduction-to-the-hierarchical-risk-parity-algorithm/)
    
52. Hierarchical Risk Parity on RAPIDS: An ML Approach to Portfolio Allocation, acessado em junho 30, 2025, [https://developer.nvidia.com/blog/hierarchical-risk-parity-on-rapids-an-ml-approach-to-portfolio-allocation/](https://developer.nvidia.com/blog/hierarchical-risk-parity-on-rapids-an-ml-approach-to-portfolio-allocation/)
    
53. Portfolio Optimization with Python: Hierarchical Risk Parity - Yang Wu, acessado em junho 30, 2025, [https://kenwuyang.com/posts/2024_10_20_portfolio_optimization_with_python_hierarchical_risk_parity/](https://kenwuyang.com/posts/2024_10_20_portfolio_optimization_with_python_hierarchical_risk_parity/)
    
54. Hierarchical clustering on principal components in Python | by Kavengik - Medium, acessado em junho 30, 2025, [https://medium.com/@kavengik/hierarchical-clustering-on-principal-components-in-python-1d7e1404f041](https://medium.com/@kavengik/hierarchical-clustering-on-principal-components-in-python-1d7e1404f041)
    
55. HCPC - Hierarchical Clustering on Principal Components: Essentials - Articles - STHDA, acessado em junho 30, 2025, [https://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/117-hcpc-hierarchical-clustering-on-principal-components-essentials/](https://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/117-hcpc-hierarchical-clustering-on-principal-components-essentials/)
    
56. How to Combine PCA & K-Means Clustering in Python - 365 Data Science, acessado em junho 30, 2025, [https://365datascience.com/tutorials/python-tutorials/pca-k-means/](https://365datascience.com/tutorials/python-tutorials/pca-k-means/)
    
57. jashshah-dev/Hierarchical-Clustering-and-PCA - GitHub, acessado em junho 30, 2025, [https://github.com/jashshah-dev/Hierarchical-Clustering-and-PCA](https://github.com/jashshah-dev/Hierarchical-Clustering-and-PCA)
    
58. End-to-End Guide to K-Means Clustering in Python: From Preprocessing to Visualization, acessado em junho 30, 2025, [https://www.skillcamper.com/blog/end-to-end-guide-to-k-means-clustering-in-python-from-preprocessing-to-visualization](https://www.skillcamper.com/blog/end-to-end-guide-to-k-means-clustering-in-python-from-preprocessing-to-visualization)
    
59. Ticker and Tickers — yfinance - GitHub Pages, acessado em junho 30, 2025, [https://ranaroussi.github.io/yfinance/reference/yfinance.ticker_tickers.html](https://ranaroussi.github.io/yfinance/reference/yfinance.ticker_tickers.html)
    
60. YFinance Python Package in a Spreadsheet - Row Zero, acessado em junho 30, 2025, [https://rowzero.io/blog/yfinance](https://rowzero.io/blog/yfinance)
    
61. Get info on multiple stock tickers quickly using yfinance - Stack Overflow, acessado em junho 30, 2025, [https://stackoverflow.com/questions/71161902/get-info-on-multiple-stock-tickers-quickly-using-yfinance](https://stackoverflow.com/questions/71161902/get-info-on-multiple-stock-tickers-quickly-using-yfinance)
    
62. How to Retrieve Stock Market Data with Yahoo Finance API in Python - Omi AI, acessado em junho 30, 2025, [https://www.omi.me/blogs/api-guides/how-to-retrieve-stock-market-data-with-yahoo-finance-api-in-python-1](https://www.omi.me/blogs/api-guides/how-to-retrieve-stock-market-data-with-yahoo-finance-api-in-python-1)
    
63. Practical Implementation of Hierarchical Clustering in Python Projects - Number Analytics, acessado em junho 30, 2025, [https://www.numberanalytics.com/blog/practical-hierarchical-clustering-python-projects](https://www.numberanalytics.com/blog/practical-hierarchical-clustering-python-projects)
    
64. Portfolio Optimization using MPT in Python - Analytics Vidhya, acessado em junho 30, 2025, [https://www.analyticsvidhya.com/blog/2021/04/portfolio-optimization-using-mpt-in-python/](https://www.analyticsvidhya.com/blog/2021/04/portfolio-optimization-using-mpt-in-python/)
    
65. Portfolio allocation backtesting in Python from scratch - YouTube, acessado em junho 30, 2025, [https://www.youtube.com/watch?v=sns1zOLda1E](https://www.youtube.com/watch?v=sns1zOLda1E)
    
66. Backtesting.py – An Introductory Guide to Backtesting with Python - Interactive Brokers LLC, acessado em junho 30, 2025, [https://www.interactivebrokers.com/campus/ibkr-quant-news/backtesting-py-an-introductory-guide-to-backtesting-with-python](https://www.interactivebrokers.com/campus/ibkr-quant-news/backtesting-py-an-introductory-guide-to-backtesting-with-python)
    
67. Backtesting.py Quick Start User Guide, acessado em junho 30, 2025, [https://kernc.github.io/backtesting.py/doc/examples/Quick%20Start%20User%20Guide.html](https://kernc.github.io/backtesting.py/doc/examples/Quick%20Start%20User%20Guide.html)
    
68. How to Backtest a Portfolio Optimization Strategy Using Python - YouTube, acessado em junho 30, 2025, [https://www.youtube.com/watch?v=ErknIWbdPEQ](https://www.youtube.com/watch?v=ErknIWbdPEQ)
    
69. (PDF) Momentum Investing: A Comparison of Machine Learning and Markowitz Theory, acessado em junho 30, 2025, [https://www.researchgate.net/publication/392867575_Momentum_Investing_A_Comparison_of_Machine_Learning_and_Markowitz_Theory](https://www.researchgate.net/publication/392867575_Momentum_Investing_A_Comparison_of_Machine_Learning_and_Markowitz_Theory)
    
70. portfolio-optimization-book.pdf, acessado em junho 30, 2025, [https://portfoliooptimizationbook.com/portfolio-optimization-book.pdf](https://portfoliooptimizationbook.com/portfolio-optimization-book.pdf)
    
71. Machine Learning Methods for Markowitz Portfolio Optimization, acessado em junho 30, 2025, [https://pergamos.lib.uoa.gr/uoa/dl/object/2964631/file.pdf](https://pergamos.lib.uoa.gr/uoa/dl/object/2964631/file.pdf)
    

**