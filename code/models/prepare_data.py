import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split

def parse_reactions(s):
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            return d
        else:
            return {}
    except Exception:
        return {}

def prepare_features(df):
    df = df[~df['text'].isnull()].reset_index(drop=True)
    df['hour'] = pd.to_datetime(df['date']).dt.hour
    df['weekday'] = pd.to_datetime(df['date']).dt.weekday
    df['total_reactions'] = df['reactions'].apply(
        lambda x: sum(parse_reactions(x).values()) if isinstance(x, str) and len(x) > 2 else 0
    )
    df['text_length'] = df['text'].apply(len)
    df['num_words'] = df['text'].apply(lambda x: len(x.split()))
    df['log_views'] = np.log1p(df['views'])
    df['log_total_reactions'] = np.log1p(df['total_reactions'])
    return df

def main():
    df = pd.read_csv('data/raw/telegram_data.csv')
    df = prepare_features(df)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

if __name__ == "__main__":
    main()