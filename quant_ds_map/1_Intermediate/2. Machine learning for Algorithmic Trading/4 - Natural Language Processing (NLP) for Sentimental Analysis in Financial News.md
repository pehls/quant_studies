## Introduction: Finding Alpha in the Alphabet

The prediction of stock market movements is a notoriously challenging endeavor. Financial markets are complex systems, characterized by a high degree of noise, volatility, and sensitivity to a multitude of stochastic factors, ranging from macroeconomic announcements and political events to the subtle shifts in collective investor psychology.1 For decades, quantitative analysts have sought to model these dynamics using structured numerical data, primarily historical prices, trading volumes, and corporate financial statements. While these approaches form the bedrock of modern finance, they often overlook a vast and rich source of information: unstructured text.

Every day, an immense volume of text is generated in the form of news articles, social media posts, corporate press releases, and regulatory filings.2 This textual data contains crucial information about corporate events, economic conditions, and, perhaps most importantly, the sentiment and emotional responses of market participants. The fundamental premise of using Natural Language Processing (NLP) in finance is that this information is not always instantaneously and efficiently priced into the market. There exists a measurable latency between the emergence of a narrative in the public sphere and its full impact on an asset's price. This latency represents a potential source of "alpha," or excess returns.

Sentiment analysis, a key subfield of NLP, provides the tools to systematically bridge the gap between this qualitative, human-generated information and quantitative, machine-executable trading strategies.4 By analyzing text to determine whether the underlying tone is positive (bullish), negative (bearish), or neutral, algorithms can quantify market psychology.2 This is not merely about classifying headlines. It is about understanding the human psyche behind market movements and making informed decisions that drive success.4 The core hypothesis is that significant shifts in sentiment can be a leading indicator of future stock price movements.2 For instance, a surge of positive sentiment following a company's strong earnings report can presage a wave of buying activity, driving the stock price upward.2 Research from institutions like the University of Michigan has lent empirical support to this idea, suggesting that the integration of public sentiment data can enhance the accuracy of stock price prediction models by as much as 20%.4 Furthermore, studies indicate that negative sentiment often has a more pronounced relationship with market movements and volatility than positive sentiment does.6

In the world of algorithmic trading, where decisions are made in microseconds, the ability to process and act on this information at scale is a distinct competitive advantage.4 An automated system can analyze thousands of news articles in real-time, decoding market sentiment far faster than any human trader.1 This capability allows for the development of sophisticated trading strategies that do not rely on price data alone. Instead, they can create a more complete state representation of the market by integrating sentiment scores with traditional factors like historical prices and technical indicators, feeding this enriched data into advanced decision-making frameworks like Markov Decision Processes (MDPs).1 Ultimately, sentiment analysis is a powerful form of feature engineering, transforming the unstructured "noise" of public discourse into a structured, predictive signal that can inform and enhance automated trading decisions.

## The End-to-End NLP Pipeline for Financial Sentiment

Constructing a trading strategy based on sentiment analysis requires a systematic, multi-stage pipeline. This process transforms raw, unstructured text from disparate sources into a quantifiable sentiment score that can be used to generate trading signals. Each stage, from data acquisition to feature extraction, presents unique challenges and requires careful, domain-specific considerations.

### Data Acquisition: Sourcing the News Flow

The first step in any sentiment analysis project is to gather relevant textual data. For financial applications, the primary sources include real-time news articles from financial media, social media platforms like X (formerly Twitter), official corporate communications such as press releases and SEC filings (e.g., 10-K annual reports, 8-K current event reports, and earnings call transcripts), and analyst reports.2

While web scraping can be used to collect this data, it is often brittle, legally ambiguous, and requires significant maintenance. A more robust and scalable approach is to use Application Programming Interfaces (APIs) provided by specialized data vendors. APIs offer structured, reliable, and often real-time access to a vast array of financial news and data, saving developers the effort of building and maintaining complex scraping infrastructure.8

Numerous APIs are available, each with different features, data coverage, and pricing models. The choice of API depends on the specific needs of the project, such as the required data latency, breadth of sources, and budget. Free tiers are often available for development and small-scale projects, while paid plans offer higher request limits, real-time data, and more extensive historical archives.9 Table 6.1 provides a comparative overview of several popular news APIs.

**Table 6.1: Comparison of Financial News APIs**

|API Name|Key Features|Free Tier Details|Paid Tiers|Data Coverage|
|---|---|---|---|---|
|**MarketAux**|Global news, includes sentiment analysis, extensive entity tracking.13|100 requests/day, 3 articles/request.12|Starts at $29/month for 2,500 requests/day.12|Stocks, ETFs, Indices, Crypto, Currencies, Mutual Funds, Futures.13|
|**NewsAPI**|Broad news coverage from 150,000+ sources, advanced search filters.14|100 requests/day, 24-hour delay, development use only.10|Starts at $449/month for commercial use, real-time access.10|General, Business, Technology news; not finance-specific but can be filtered.|
|**Alpha Vantage**|Real-time & historical market data, 60+ indicators, news & sentiment API.15|Free tier with 5 calls/minute limit.11|Premium plans start from $29.99/month.11|Stocks, ETFs, Forex, Crypto, Economic Indicators.15|
|**Finnhub**|Real-time trades, news, fundamentals, earnings calendars, REST API.16|60 requests/minute on free plan.11|Pro plans start from $50/month.11|Stocks, Forex, Crypto.16|
|**Financial Modeling Prep (FMP)**|Stock news, press releases, financial statements, historical data.17|250 calls/day on free plan.19|Individual plans start at $19/month.19|Stocks, ETFs, Crypto, Forex, Indices.18|

For demonstration purposes, the following Python code shows how to use the `requests` library to fetch news for Apple Inc. (ticker: AAPL) using the MarketAux API. An API key is required, which can be obtained for free from their website.13



```Python
import requests
import pandas as pd

# Replace 'YOUR_API_TOKEN' with your actual MarketAux API token
API_TOKEN = 'YOUR_API_TOKEN'
TICKER = 'AAPL'

# Construct the API request URL
url = f'https://api.marketaux.com/v1/news/all?symbols={TICKER}&filter_entities=true&language=en&api_token={API_TOKEN}'

try:
    # Make the API request
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

    # Parse the JSON response
    news_data = response.json()

    # Convert the 'data' part of the response to a pandas DataFrame
    if 'data' in news_data and news_data['data']:
        df_news = pd.DataFrame(news_data['data'])
        print(f"Successfully fetched {len(df_news)} news articles for {TICKER}.")
        # Display the first few articles
        print(df_news[['published_at', 'title', 'snippet']].head())
    else:
        print(f"No news articles found for {TICKER}.")

except requests.exceptions.RequestException as e:
    print(f"An error occurred during the API request: {e}")
except KeyError as e:
    print(f"Error parsing the API response. Missing key: {e}")

```

### Text Preprocessing: From Raw Text to Clean Tokens

Once the raw text data is acquired, it must be cleaned and normalized. This preprocessing step is critical for reducing noise and preparing the text for feature extraction. While standard NLP preprocessing pipelines exist, financial text requires a domain-specific approach. Generic cleaning methods can inadvertently remove words that carry significant financial meaning, thereby destroying the very signal one aims to capture.5

The key steps in a financial text preprocessing pipeline are as follows 22:

1. **Lowercasing:** Converting all text to a single case (typically lowercase) is a simple yet effective normalization technique that prevents the model from treating the same word with different capitalization (e.g., "Apple" and "apple") as distinct tokens.
    
2. **Punctuation and Number Removal:** Punctuation marks (e.g., `!`, `?`, `,`) and numerical digits generally do not contribute to the semantic sentiment of a sentence and can be removed to reduce the complexity of the vocabulary.
    
3. **Tokenization:** This is the process of breaking down the cleaned text into a list of individual words or "tokens." This is a foundational step for most subsequent NLP tasks.
    
4. **Stop-Word Removal:** Stop words are common words (e.g., "the," "is," "in") that appear frequently but typically carry little semantic weight. While removing them is standard practice, it is crucial to use a stop-word list tailored for finance. Standard NLP libraries like NLTK contain lists where words like "up," "down," "short," "put," and "call" are considered stop words. Removing these would be catastrophic for financial analysis. Specialized resources like the Loughran-McDonald Master Dictionary provide lists of stop words appropriate for the financial domain.26 The goal is to reduce noise without eliminating the signal.
    
5. **Lemmatization:** This process reduces words to their base or dictionary form, known as the "lemma." For example, "gains," "gained," and "gaining" would all be converted to "gain." Lemmatization is generally preferred over the cruder method of "stemming" because it uses lexical knowledge to get the correct base forms, preserving more meaning. However, it's worth noting that for some lexicon-based sentiment analysis approaches, lemmatization might not be necessary if the dictionary already contains various inflected forms of words.21
    

The following Python function demonstrates a financial text preprocessing pipeline using the `nltk` library.



```Python
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK data is downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_financial_text(text):
    """
    Cleans and preprocesses a single string of financial text.
    
    Args:
        text (str): The raw text of a news article or headline.
        
    Returns:
        list: A list of cleaned and lemmatized tokens.
    """
    if not isinstance(text, str):
        return

    # 1. Lowercasing
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 4. Tokenization
    tokens = word_tokenize(text)

    # 5. Stop-word removal (with a custom financial list)
    # Start with the standard English stop words list
    stop_words = set(stopwords.words('english'))
    # Define financial-specific words that should NOT be removed
    financial_keywords_to_keep = {'up', 'down', 'gain', 'loss', 'profit', 'short', 'long', 'buy', 'sell'}
    # Remove these keywords from the stop words set
    custom_stop_words = stop_words - financial_keywords_to_keep
    
    filtered_tokens = [word for word in tokens if word not in custom_stop_words]

    # 6. Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return lemmatized_tokens

# --- Example Usage ---
news_headline = "Apple's profits are expected to gain by 15% in the next quarter, causing the stock to go up."
processed_tokens = preprocess_financial_text(news_headline)
print(f"Original: {news_headline}")
print(f"Processed: {processed_tokens}")

```

### Feature Extraction: Quantifying Language

After preprocessing, the cleaned tokens must be converted into numerical vectors, a process known as feature extraction or text vectorization. Machine learning models cannot operate on raw text; they require numerical inputs. The choice of feature extraction method involves a trade-off between model complexity, interpretability, and performance.28 Table 6.2 outlines the primary approaches.

**Table 6.2: A Comparative Analysis of Sentiment Feature Extraction Techniques**

|Technique|How it Works|Pros|Cons|Best For|
|---|---|---|---|---|
|**Lexicon-Based**|Counts words from predefined lists of positive/negative terms (e.g., Loughran-McDonald dictionary).26|Simple, fast, highly interpretable, no training required. Domain-specific.28|May not capture nuance, context, or new jargon. Limited vocabulary.|Quick baselines, applications requiring transparency, analyzing formal documents like 10-Ks.|
|**ML with TF-IDF**|Creates a document-term matrix where each word's value is weighted by its importance in the document and rarity across the corpus.29|Captures word importance better than simple counts. Can learn from data.|Ignores word order and context. Can be computationally intensive. Requires labeled data for training a classifier.30|Building custom classifiers when labeled data is available. A good balance between performance and complexity.|
|**Transformers (FinBERT)**|Uses a deep learning model pre-trained on a massive financial text corpus to understand word sequence and context.31|State-of-the-art accuracy. Understands context, sarcasm, and nuance.|Computationally expensive ("black box" model). Requires significant resources for training/fine-tuning.28|Production-grade systems where maximum accuracy is paramount and computational cost is manageable.|

#### Lexicon-Based Methods: The Loughran-McDonald Dictionary

A straightforward and effective baseline for financial sentiment analysis is the lexicon-based approach. This method uses a pre-compiled dictionary of words that have been assigned sentiment scores. The most widely respected dictionary in finance is the Loughran-McDonald (LM) Master Dictionary.26 Developed by analyzing the text of tens of thousands of corporate 10-K filings, the LM dictionary provides lists of words classified as

`positive`, `negative`, `uncertainty`, `litigious`, and more. The sentiment of a document is calculated by simply counting the occurrences of words from these lists.27

The `pysentiment2` library provides a convenient Python interface for using the LM dictionary.32



```Python
import pysentiment2 as ps

# Initialize the Loughran-McDonald dictionary
lm = ps.LM()

# Sample financial text
text = "The company reported a significant loss and faces litigation risk, but future earnings are uncertain."

# First, tokenize the text using the library's tokenizer
tokens = lm.tokenize(text)

# Then, get the sentiment score
score = lm.get_score(tokens)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Sentiment Score: {score}")
```

The output dictionary shows the raw counts of words from each category, along with a calculated `Polarity` score.

#### Machine Learning with Bag-of-Words (BoW) and TF-IDF

A more sophisticated approach involves training a machine learning classifier (e.g., Naive Bayes, Logistic Regression) on a labeled dataset of financial news. To do this, the text must first be converted into numerical feature vectors.

**Bag-of-Words (BoW):** The BoW model is a simple way to represent a document as a vector of word counts. It creates a vocabulary of all unique words in the corpus and, for each document, counts the frequency of each word. This approach disregards grammar and word order, treating the text as a "bag" of its words.30 The result is a document-term matrix where rows are documents and columns are words from the vocabulary.34

**Term Frequency-Inverse Document Frequency (TF-IDF):** TF-IDF is an enhancement over BoW. It weights the word counts to reflect how important a word is to a document in a collection of documents (corpus). The intuition is that words that are very frequent in one document but rare across all other documents are more likely to be significant.29 The TF-IDF score is the product of two metrics 35:

1. Term Frequency (TF): Measures how frequently a term appears in a document.
    
    ![[Pasted image 20250701084531.png]]
2. Inverse Document Frequency (IDF): Measures how important a term is by down-weighting common terms.
    
    ![[Pasted image 20250701084543.png]]

The final TF-IDF score for a term t in document d within corpus D is:

![[Pasted image 20250701084553.png]]

Scikit-learn's `TfidfVectorizer` handles this entire process efficiently.



```Python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'company reports record profits',
    'market crashes due to unexpected loss',
    'new technology boosts company profits'
]

# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the corpus and transform it into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(corpus)

# Get the feature names (the vocabulary)
feature_names = vectorizer.get_feature_names_out()

print("Vocabulary:", feature_names)
print("\nTF-IDF Matrix (sparse):\n", tfidf_matrix)
print("\nTF-IDF Matrix (dense):\n", tfidf_matrix.toarray())
```

#### Advanced Methods: Transformers and FinBERT

The state-of-the-art in NLP is dominated by Transformer-based models like BERT (Bidirectional Encoder Representations from Transformers). Unlike BoW or TF-IDF, these models process text by considering the sequence of words and their context, allowing them to understand complex linguistic nuances.3

**FinBERT** is a BERT model that has been specifically pre-trained on a massive corpus of financial text, including analyst reports and news articles. This process, known as transfer learning, fine-tunes the model to understand the specific language and sentiment of the financial domain.31 Instead of just counting words, FinBERT generates rich numerical embeddings that capture the contextual meaning of the text.

The Hugging Face `transformers` library makes it incredibly simple to use pre-trained models like FinBERT.



```Python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load the pre-trained FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Sample financial news headlines
headlines =

# Tokenize the input texts
inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')

# Get model predictions (logits)
outputs = model(**inputs)
logits = outputs.logits

# Convert logits to probabilities using softmax
probabilities = F.softmax(logits, dim=-1)

# Get the labels (positive, negative, neutral)
labels = model.config.id2label
print("Sentiment Analysis with FinBERT:\n")
for i, headline in enumerate(headlines):
    print(f"Headline: {headline}")
    for j, label in enumerate(labels.values()):
        print(f"  - {label.capitalize()}: {probabilities[i][j].item():.4f}")
    print("-" * 30)
```

This code directly outputs the probabilities for each sentiment class, providing a nuanced and context-aware sentiment score that is far more powerful than simple word counting.

## From Sentiment to Signals: Building and Backtesting a Trading Strategy

Generating a sentiment score is only half the battle. To be useful in algorithmic trading, this score must be systematically converted into concrete trading actions (signals) and then rigorously tested against historical data to validate its effectiveness. This process involves two key stages: signal generation and backtesting.

### Signal Generation: From Scores to Actions

A raw sentiment score, such as a polarity value from -1 to 1, must be translated into a discrete trading signal: BUY, SELL, or HOLD.1 The simplest method for this is a threshold-based approach. For example, a strategy might define rules as follows 37:

- If the daily aggregated sentiment score is strongly positive (e.g., > 0.5), generate a BUY signal.
    
- If the score is strongly negative (e.g., < -0.5), generate a SELL signal.
    
- Otherwise, if the sentiment is neutral or weak, generate a HOLD signal (i.e., take no action).
    

A critical and often overlooked step in this process is the careful alignment of sentiment data with price data. The sentiment signal must be based on information that was publicly available _before_ the price movement you are trying to predict. This means ensuring that the timestamp of the news article precedes the timestamp of the market data (e.g., the opening price) used for the trade decision.37 Failure to do so can introduce lookahead bias, leading to unrealistically optimistic backtest results.

The choice of thresholds is not a fixed rule but rather a set of hyperparameters that can be tuned and optimized. A simple static threshold may not be optimal. More advanced strategies might look for changes in sentiment over time, such as using a rolling average of sentiment scores to smooth out noise and identify durable trends.28 A strategy could be triggered not by the absolute level of sentiment, but by a short-term moving average of sentiment crossing above a long-term moving average, a concept known as "sentiment momentum." This elevates the signal generation logic from a simple rule to a dynamic, optimizable system.

### Backtesting with backtesting.py: Validating the Strategy

Once a method for generating signals is established, the strategy must be backtested. Backtesting is the process of simulating the strategy on historical data to assess how it would have performed in the past.37 This is a crucial step for evaluating a strategy's potential profitability and risk profile before deploying it with real capital.

The `backtesting.py` library is a powerful, fast, and user-friendly Python framework for this purpose. It is lightweight, compatible with various financial instruments, and produces interactive visualizations of the results.38

A typical backtesting workflow with `backtesting.py` involves:

1. Defining a strategy class that inherits from `backtesting.Strategy`.
    
2. Implementing the `init()` method to set up indicators (in this case, our sentiment signal).
    
3. Implementing the `next()` method to define the trading logic (e.g., `self.buy()` or `self.sell()`) based on the signals.
    
4. Loading historical price data (Open, High, Low, Close, Volume).
    
5. Merging the sentiment signals with the price data, ensuring correct alignment.
    
6. Creating a `Backtest` object, passing in the data, the strategy class, initial capital, and transaction costs (commission).
    
7. Running the backtest and analyzing the output statistics.
    

The output of a backtest can be overwhelming, but several key metrics are essential for a holistic evaluation. Table 6.3 explains some of the most important performance metrics provided by `backtesting.py`.37

**Table 6.3: Key Backtesting Performance Metrics Explained**

|Metric|Definition|What it Measures|What is a "Good" Value?|
|---|---|---|---|
|**Return [%]**|The total percentage return over the entire backtest period.|Overall profitability of the strategy.|Higher is better, but must be compared to a benchmark (e.g., Buy & Hold).|
|**Sharpe Ratio**|The average return earned in excess of the risk-free rate per unit of volatility or total risk.|Risk-adjusted return. It answers: "How much return did I get for the risk I took?"|> 1 is considered acceptable, > 2 is very good, > 3 is excellent.1|
|**Max. Drawdown [%]**|The largest peak-to-trough decline in portfolio value, expressed as a percentage.|The worst-case loss scenario during the backtest period. Measures downside risk.|Lower is better. A large drawdown can be psychologically difficult to endure and may lead to ruin.|
|**Sortino Ratio**|Similar to the Sharpe Ratio, but it only penalizes for downside volatility (harmful risk).|Downside risk-adjusted return. It differentiates between "good" (upward) and "bad" (downward) volatility.|Higher is better. Often preferred over Sharpe for evaluating strategies with asymmetric returns.|
|**Win Rate [%]**|The percentage of trades that were closed with a profit.|The consistency of the strategy in generating winning trades.|Higher is better, but a high win rate does not guarantee profitability if losses on losing trades are large.|
|**Calmar Ratio**|The compound annualized growth rate divided by the absolute value of the maximum drawdown.|Return relative to the worst-case loss. Measures recovery efficiency.|Higher is better. A value > 1 indicates the strategy recovered from its worst drawdown within a year.|

A common mistake for beginners is to focus solely on the total return. A professional quant evaluates a strategy by balancing its return against its risk. A strategy with a 50% return and a 60% maximum drawdown is likely far worse than one with a 20% return and a 10% maximum drawdown. Metrics like the Sharpe and Sortino ratios provide this crucial risk-adjusted perspective.

## Capstone Project: A Sentiment-Driven Trading System for Apple Inc. (AAPL)

This capstone project synthesizes all the concepts from the chapter into a complete, end-to-end system. It is structured as a series of questions and answers, guiding the reader through the process of building, backtesting, and analyzing a real-world sentiment-driven trading strategy for Apple Inc. (ticker: AAPL).

### Question 1: The Data Pipeline

**Prompt:** How can we build a Python script to fetch the last year of financial news for Apple (AAPL) using an API and its corresponding daily OHLCV price data from `yfinance`? How do we align these two datasets by date?

Response:

The first step is to create a unified dataset containing both daily price history and daily news articles. We will use the yfinance library to get the price data and a news API (here, MarketAux) for the news. The key is to process the news data so that it can be merged with the price data on a daily basis.



```Python
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta

# --- Configuration ---
TICKER = 'AAPL'
# IMPORTANT: Replace with your own API token
API_TOKEN = 'YOUR_API_TOKEN' 
# Define the date range for the last year
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# --- 1. Fetch Historical Price Data using yfinance ---
print(f"Fetching {TICKER} price data from {start_date.date()} to {end_date.date()}...")
try:
    price_data = yf.download(TICKER, start=start_date, end=end_date)
    if price_data.empty:
        raise ValueError("No price data fetched. Check ticker and date range.")
    # Ensure the index is just the date part for merging
    price_data.index = price_data.index.date
    print(f"Successfully fetched {len(price_data)} days of price data.")
except Exception as e:
    print(f"Error fetching price data: {e}")
    exit()

# --- 2. Fetch Financial News Data using MarketAux API ---
print(f"Fetching news data for {TICKER}...")
news_list =
# Format dates for the API
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')
url = f'https://api.marketaux.com/v1/news/all?symbols={TICKER}&filter_entities=true&language=en&published_after={start_date_str}T00:00&published_before={end_date_str}T23:59&api_token={API_TOKEN}'

try:
    response = requests.get(url)
    response.raise_for_status()
    news_data = response.json()
    if 'data' in news_data:
        news_list = news_data['data']
    print(f"Successfully fetched {len(news_list)} news articles.")
except Exception as e:
    print(f"Error fetching news data: {e}")
    news_list =

# --- 3. Process and Align Datasets ---
if news_list:
    df_news = pd.DataFrame(news_list)
    # Extract just the date from the 'published_at' timestamp
    df_news['date'] = pd.to_datetime(df_news['published_at']).dt.date
    # Group news by date and aggregate titles into a list
    daily_news = df_news.groupby('date')['title'].apply(list).reset_index()
    daily_news.set_index('date', inplace=True)
    
    # Merge price data with news data based on the date index
    combined_data = price_data.join(daily_news, how='left')
    # Fill days with no news with an empty list
    combined_data['title'] = combined_data['title'].apply(lambda x: x if isinstance(x, list) else)
else:
    # If no news is fetched, create the column with empty lists
    combined_data = price_data
    combined_data['title'] = [ for _ in range(len(combined_data))]

print("\n--- Combined Data Sample ---")
print(combined_data.head())
print("\n--- Sample of a day with news ---")
print(combined_data[combined_data['title'].apply(len) > 0].head())
```

### Question 2: The Sentiment Engine

**Prompt:** How do we process the fetched news articles to generate a single, daily sentiment score for AAPL? We need to handle days with multiple news articles.

Response:

Now we will build the sentiment engine using the pre-trained FinBERT model from Hugging Face. This function will take a list of news headlines for a given day, calculate the sentiment for each, and aggregate them into a single daily score. We will calculate a weighted polarity score where positive sentiment is added and negative sentiment is subtracted, and then average this score across all articles for the day.



```Python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Load FinBERT model and tokenizer once to be efficient
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment_score(texts):
    """Calculates sentiment for a list of texts and returns the average score."""
    if not texts:
        return 0.0
    
    try:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = F.softmax(outputs.logits, dim=-1)
        
        # Sentiment score = P(positive) - P(negative)
        # The model outputs probabilities for [positive, negative, neutral]
        scores = probabilities[:, 0] - probabilities[:, 1]
        return scores.mean().item()
    except Exception as e:
        # Handle potential errors with long inputs or other issues
        print(f"Error processing texts: {e}")
        return 0.0

# Apply the sentiment analysis function to our combined data
# Using tqdm for a progress bar as this can take some time
tqdm.pandas(desc="Analyzing Sentiment")
combined_data['sentiment'] = combined_data['title'].progress_apply(get_sentiment_score)

# Forward-fill the sentiment scores to handle weekends and holidays
combined_data['sentiment'].replace(0, method='ffill', inplace=True)

print("\n--- Data with Sentiment Scores ---")
print(combined_data[['Close', 'title', 'sentiment']].head())
print("\n--- Statistics for Sentiment Scores ---")
print(combined_data['sentiment'].describe())
```

### Question 3: The Strategy Logic

**Prompt:** Using the combined price and sentiment data, how do we define a trading strategy within the `backtesting.py` framework? Let's implement a strategy that buys when the 5-day moving average of sentiment crosses above a positive threshold (e.g., 0.2) and sells when it crosses below a negative threshold (e.g., -0.2).

Response:

We will now define our trading strategy using the backtesting.py library. The strategy will use a 5-day Simple Moving Average (SMA) of the sentiment score to smooth out daily noise. A buy signal is generated when this sentiment SMA crosses above a positive threshold, and a sell signal is generated when it crosses below a negative threshold. We will also close any open position when the opposite signal occurs.



```Python
from backtesting import Strategy
from backtesting.lib import crossover

def sentiment_sma(sentiment_series, n):
    """Helper function to calculate SMA on a pandas Series."""
    return sentiment_series.rolling(n).mean()

class SentimentMomentumStrategy(Strategy):
    # --- Strategy Parameters ---
    sentiment_window = 5
    buy_threshold = 0.2
    sell_threshold = -0.2

    def init(self):
        # The `self.data.Sentiment` attribute will be passed in from the Backtest call
        # We use self.I to wrap our custom function so backtesting.py can plot it
        self.sentiment_ma = self.I(sentiment_sma, self.data.Sentiment, self.sentiment_window)

    def next(self):
        # If we have an open long position, check for a sell signal to close it
        if self.position.is_long:
            if crossover(self.sentiment_ma, self.sell_threshold):
                self.position.close()

        # If we have an open short position, check for a buy signal to close it
        elif self.position.is_short:
             if crossover(self.buy_threshold, self.sentiment_ma):
                self.position.close()

        # If no position is open, check for new entry signals
        else:
            # Buy signal: sentiment MA crosses above the buy threshold
            if crossover(self.sentiment_ma, self.buy_threshold):
                self.buy()
            
            # Sell signal: sentiment MA crosses below the sell threshold
            elif crossover(self.sell_threshold, self.sentiment_ma):
                self.sell()

print("Strategy class 'SentimentMomentumStrategy' defined successfully.")
```

### Question 4: Performance Analysis

**Prompt:** How do we run the backtest for our AAPL sentiment strategy and evaluate its performance against a simple buy-and-hold benchmark?

Response:

Finally, we instantiate the Backtest object, run the simulation, and analyze the results. We will pass our combined data, the SentimentMomentumStrategy, an initial cash amount, and a commission fee. The backtesting.py library requires the column names to be capitalized (Open, High, Low, Close, Volume), and our custom sentiment data must be passed as an additional column named Sentiment.



```Python
from backtesting import Backtest

# Prepare data for backtesting.py: column names must be capitalized.
# Our custom sentiment data must be passed in with a matching capitalized name.
backtest_data = combined_data.copy()
backtest_data.rename(columns={
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Close': 'Close',
    'Volume': 'Volume',
    'sentiment': 'Sentiment'
}, inplace=True)

# --- Run the Backtest ---
bt = Backtest(
    backtest_data,
    SentimentMomentumStrategy,
    cash=100000,
    commission=.002  # 0.2% commission per trade
)

stats = bt.run()

print("\n--- Backtest Performance ---")
print(stats)

# --- Compare to Buy & Hold ---
buy_and_hold_return = (backtest_data['Close'][-1] / backtest_data['Close'] - 1) * 100
print(f"\nBuy & Hold Return: {buy_and_hold_return:.2f}%")
print(f"Strategy Return: {stats']:.2f}%")

if stats'] > buy_and_hold_return:
    print("The sentiment strategy outperformed the Buy & Hold benchmark.")
else:
    print("The sentiment strategy underperformed the Buy & Hold benchmark.")

# --- Plot the Results ---
print("\nGenerating backtest plot...")
bt.plot()
```

The `stats` output will provide a detailed breakdown of the strategy's performance, which can be compared directly to the simple buy-and-hold return. The interactive plot generated by `bt.plot()` will visualize the equity curve, drawdowns, and the exact points on the price chart where buy (blue triangles) and sell (red triangles) trades were executed, along with the sentiment moving average indicator below. This provides a comprehensive and transparent evaluation of whether our sentiment analysis pipeline successfully generated alpha.

## Conclusion and Further Reading

This chapter has charted a comprehensive course from the abstract concept of market sentiment to the concrete implementation of a quantitative trading strategy. We have demonstrated that unstructured text, particularly financial news, contains valuable predictive information that can be systematically extracted using Natural Language Processing. The journey involved constructing a robust pipeline: acquiring data via APIs, performing domain-specific text preprocessing, and converting text into numerical features using methods ranging from simple lexicons to state-of-the-art Transformer models like FinBERT. Finally, we translated these sentiment scores into trading signals and rigorously evaluated their historical performance through backtesting, culminating in a complete capstone project.

However, it is crucial to recognize that sentiment analysis is not a "silver bullet." The models and methods discussed face several challenges. The nuances of human language, such as sarcasm, irony, and complex contextual dependencies, remain difficult for algorithms to master fully.5 The quality and cleanliness of the input data are paramount; noisy or biased data can lead to erroneous conclusions and flawed trading decisions.5 Furthermore, financial markets are adaptive systems. A profitable strategy today may see its edge erode as more participants adopt similar techniques. Therefore, any sentiment-based trading system requires perpetual learning, adaptation, and fine-tuning to remain effective and relevant amidst changing market dynamics and technological advancements.4

The concepts presented here serve as a robust foundation for further exploration. Ambitious practitioners can enhance their models by:

- **Integrating Diverse Data Sources:** Combining sentiment data with other alternative datasets, such as economic indicators, satellite imagery, or corporate filings, can provide a more holistic and robust view of market conditions.5
    
- **Employing Advanced Architectures:** The sentiment scores generated can serve as a powerful input feature for more complex predictive models. For example, Long Short-Term Memory (LSTM) networks can model the time-series nature of sentiment, while Reinforcement Learning (RL) agents can use sentiment as part of their state representation to learn highly adaptive and optimal trading policies over time.1
    
- **Upholding Ethical Standards:** The deployment of any automated trading system carries significant responsibility. Quants must ensure their models are robust, resilient to anomalies, and do not contribute to market instability or create unfair advantages.37 Transparency and fairness should be guiding principles in the development and operation of these powerful technologies.
    

In conclusion, the fusion of NLP and algorithmic trading represents a dynamic and promising frontier in quantitative finance. By systematically decoding the language of the market, we can uncover new sources of alpha and gain a deeper understanding of the forces that drive financial assets.

## References
**

1. Algorithmic Trading using Sentiment Analysis and Reinforcement ..., acessado em julho 1, 2025, [https://cs229.stanford.edu/proj2017/final-reports/5222284.pdf](https://cs229.stanford.edu/proj2017/final-reports/5222284.pdf)
    
2. (PDF) Sentiment analysis in Algorithmic trading - ResearchGate, acessado em julho 1, 2025, [https://www.researchgate.net/publication/375595713_Sentiment_analysis_in_Algorithmic_trading](https://www.researchgate.net/publication/375595713_Sentiment_analysis_in_Algorithmic_trading)
    
3. How is NLP used in financial analysis? - Milvus, acessado em julho 1, 2025, [https://milvus.io/ai-quick-reference/how-is-nlp-used-in-financial-analysis](https://milvus.io/ai-quick-reference/how-is-nlp-used-in-financial-analysis)
    
4. Stock Market: How sentiment analysis transforms algorithmic trading strategies - Mint, acessado em julho 1, 2025, [https://www.livemint.com/market/stock-market-news/stock-market-how-sentiment-analysis-transforms-algorithmic-trading-strategies-investments-nlp-markets-11713942368194.html](https://www.livemint.com/market/stock-market-news/stock-market-how-sentiment-analysis-transforms-algorithmic-trading-strategies-investments-nlp-markets-11713942368194.html)
    
5. Harnessing Sentiment Analysis in Financial Markets - PyQuant News, acessado em julho 1, 2025, [https://www.pyquantnews.com/free-python-resources/harnessing-sentiment-analysis-in-financial-markets](https://www.pyquantnews.com/free-python-resources/harnessing-sentiment-analysis-in-financial-markets)
    
6. Sentiment Analysis of Financial News: Mechanics & Statistics - Dow Jones, acessado em julho 1, 2025, [https://www.dowjones.com/professional/risk/resources/blog/a-primer-for-sentiment-analysis-of-financial-news](https://www.dowjones.com/professional/risk/resources/blog/a-primer-for-sentiment-analysis-of-financial-news)
    
7. Sentiment analysis for financial services use case - NetApp, acessado em julho 1, 2025, [https://www.netapp.com/media/65045-CSS-7217-FSI-Use-Case.pdf](https://www.netapp.com/media/65045-CSS-7217-FSI-Use-Case.pdf)
    
8. How to Scrape Yahoo Finance for Real-Time Stock Data - PromptCloud, acessado em julho 1, 2025, [https://www.promptcloud.com/blog/how-to-scrape-yahoo-finance/](https://www.promptcloud.com/blog/how-to-scrape-yahoo-finance/)
    
9. Finance News API - APILayer, acessado em julho 1, 2025, [https://apilayer.com/marketplace/financelayer-api](https://apilayer.com/marketplace/financelayer-api)
    
10. Pricing - News API, acessado em julho 1, 2025, [https://newsapi.org/pricing](https://newsapi.org/pricing)
    
11. The 7 Best Financial APIs for Investors and Developers in 2025 (In-Depth Analysis and Comparison) | by Kevin Meneses González | Coinmonks - Medium, acessado em julho 1, 2025, [https://medium.com/coinmonks/the-7-best-financial-apis-for-investors-and-developers-in-2025-in-depth-analysis-and-comparison-adbc22024f68](https://medium.com/coinmonks/the-7-best-financial-apis-for-investors-and-developers-in-2025-in-depth-analysis-and-comparison-adbc22024f68)
    
12. Marketaux API Pricing, acessado em julho 1, 2025, [https://www.marketaux.com/pricing](https://www.marketaux.com/pricing)
    
13. marketaux: Free stock market and finance news API, acessado em julho 1, 2025, [https://www.marketaux.com/](https://www.marketaux.com/)
    
14. News API – Search News and Blog Articles on the Web, acessado em julho 1, 2025, [https://newsapi.org/](https://newsapi.org/)
    
15. Alpha Vantage: Free Stock APIs in JSON & Excel, acessado em julho 1, 2025, [https://www.alphavantage.co/](https://www.alphavantage.co/)
    
16. Real-time Market News API - Finnhub, acessado em julho 1, 2025, [https://finnhub.io/docs/api/market-news](https://finnhub.io/docs/api/market-news)
    
17. Stock News API | Financial Modeling Prep, acessado em julho 1, 2025, [https://site.financialmodelingprep.com/developer/docs/stock-news-api](https://site.financialmodelingprep.com/developer/docs/stock-news-api)
    
18. Free Stock Market API and Financial Statements API... | FMP - Financial Modeling Prep, acessado em julho 1, 2025, [https://site.financialmodelingprep.com/developer/docs](https://site.financialmodelingprep.com/developer/docs)
    
19. Pricing | Financial Modeling Prep | FMP, acessado em julho 1, 2025, [https://site.financialmodelingprep.com/developer/docs/pricing](https://site.financialmodelingprep.com/developer/docs/pricing)
    
20. United States Financial Markets News Data API | marketaux, acessado em julho 1, 2025, [https://www.marketaux.com/news/country/us](https://www.marketaux.com/news/country/us)
    
21. cancan-huang/Sentiment-Analysis-for-Financial-Articles ... - GitHub, acessado em julho 1, 2025, [https://github.com/cancan-huang/Sentiment-Analysis-for-Financial-Articles](https://github.com/cancan-huang/Sentiment-Analysis-for-Financial-Articles)
    
22. News-Sentiment-Analysis-in-Python/News_Sentiment_Analysis_PythonCode.ipynb at master - GitHub, acessado em julho 1, 2025, [https://github.com/farooq96/News-Sentiment-Analysis-in-Python/blob/master/News_Sentiment_Analysis_PythonCode.ipynb](https://github.com/farooq96/News-Sentiment-Analysis-in-Python/blob/master/News_Sentiment_Analysis_PythonCode.ipynb)
    
23. Text Preprocessing in natural language processing with Python, acessado em julho 1, 2025, [https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/](https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/)
    
24. Text Classification Financial News - Kaggle, acessado em julho 1, 2025, [https://www.kaggle.com/code/shivamburnwal/text-classification-financial-news](https://www.kaggle.com/code/shivamburnwal/text-classification-financial-news)
    
25. Text Preprocessing in Python: Steps, Tools, and Examples | by Data Monsters - Medium, acessado em julho 1, 2025, [https://medium.com/product-ai/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908](https://medium.com/product-ai/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908)
    
26. Textual Analysis Resources | Software Repository for Accounting ..., acessado em julho 1, 2025, [https://sraf.nd.edu/textual-analysis/](https://sraf.nd.edu/textual-analysis/)
    
27. Loughran-McDonald Master Dictionary w/ Sentiment Word Lists | Software Repository for Accounting and Finance, acessado em julho 1, 2025, [https://sraf.nd.edu/loughranmcdonald-master-dictionary/](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)
    
28. Sentiment Analysis For Algorithmic Trading - DataCamp, acessado em julho 1, 2025, [https://www.datacamp.com/resources/webinars/sentiment-analysis-for-algorithmic-trading](https://www.datacamp.com/resources/webinars/sentiment-analysis-for-algorithmic-trading)
    
29. TF-IDF [Tutorial] - Kaggle, acessado em julho 1, 2025, [https://www.kaggle.com/code/paulrohan2020/tf-idf-tutorial](https://www.kaggle.com/code/paulrohan2020/tf-idf-tutorial)
    
30. How To Implement Bag-Of-Words In Python [2 Ways: scikit-learn & NLTK] - Spot Intelligence, acessado em julho 1, 2025, [https://spotintelligence.com/2022/12/20/bag-of-words-python/](https://spotintelligence.com/2022/12/20/bag-of-words-python/)
    
31. ProsusAI/finbert · Hugging Face, acessado em julho 1, 2025, [https://huggingface.co/ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
    
32. Welcome to Python Sentiment Analysis documentation! - GitHub Pages, acessado em julho 1, 2025, [https://nickderobertis.github.io/pysentiment/index.html](https://nickderobertis.github.io/pysentiment/index.html)
    
33. Python Bag of Words Model: A Complete Guide | DataCamp, acessado em julho 1, 2025, [https://www.datacamp.com/tutorial/python-bag-of-words-model](https://www.datacamp.com/tutorial/python-bag-of-words-model)
    
34. A friendly guide to NLP: Bag-of-Words with Python example - Analytics Vidhya, acessado em julho 1, 2025, [https://www.analyticsvidhya.com/blog/2021/08/a-friendly-guide-to-nlp-bag-of-words-with-python-example/](https://www.analyticsvidhya.com/blog/2021/08/a-friendly-guide-to-nlp-bag-of-words-with-python-example/)
    
35. Introduction to tf-idf - Jake Tae, acessado em julho 1, 2025, [https://jaketae.github.io/study/tf-idf/](https://jaketae.github.io/study/tf-idf/)
    
36. How to implement TF-IDF in Python - Educative.io, acessado em julho 1, 2025, [https://www.educative.io/answers/how-to-implement-tf-idf-in-python](https://www.educative.io/answers/how-to-implement-tf-idf-in-python)
    
37. Sentiment Analysis in Trading: An In-Depth Guide to Implementation - Medium, acessado em julho 1, 2025, [https://medium.com/funny-ai-quant/sentiment-analysis-in-trading-an-in-depth-guide-to-implementation-b212a1df8391](https://medium.com/funny-ai-quant/sentiment-analysis-in-trading-an-in-depth-guide-to-implementation-b212a1df8391)
    
38. Backtesting.py – An Introductory Guide to Backtesting with Python - Interactive Brokers LLC, acessado em julho 1, 2025, [https://www.interactivebrokers.com/campus/ibkr-quant-news/backtesting-py-an-introductory-guide-to-backtesting-with-python](https://www.interactivebrokers.com/campus/ibkr-quant-news/backtesting-py-an-introductory-guide-to-backtesting-with-python)
    
39. Backtesting.py - An Introductory Guide to Backtesting with Python ..., acessado em julho 1, 2025, [https://algotrading101.com/learn/backtesting-py-guide/](https://algotrading101.com/learn/backtesting-py-guide/)
    
40. Backtesting.py - Backtest trading strategies in Python, acessado em julho 1, 2025, [https://kernc.github.io/backtesting.py/](https://kernc.github.io/backtesting.py/)
    

**