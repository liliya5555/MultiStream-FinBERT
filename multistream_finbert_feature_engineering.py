# multistream_finbert_feature_engineering.py
import pandas as pd
import numpy as np

def engineer_accounting_features(df):
    df['cash_ratio'] = df['cash'] / df['total_assets']
    df['working_capital_ratio'] = (df['current_assets'] - df['current_liabilities']) / df['total_assets']
    df['ebitda_interest'] = df['ebitda'] / (df['interest_expense'] + 1e-5)
    df = df.fillna(0)
    return df

def engineer_market_features(df):
    df['momentum'] = df['close'].pct_change(periods=60)
    df['abnormal_volume'] = (df['volume'] - df['volume'].rolling(window=30).mean()) / df['volume'].rolling(window=30).std()
    df['realized_volatility'] = df['close'].rolling(window=60).std()
    df = df.fillna(0)
    return df

def engineer_text_features(docs):
    import re
    features = []
    for doc in docs:
        sentiment = doc.count("risk") - doc.count("opportunity")
        keywords = sum(1 for w in ["default", "concern", "liquidity"] if w in doc.lower())
        features.append([sentiment, keywords])
    return np.array(features)
