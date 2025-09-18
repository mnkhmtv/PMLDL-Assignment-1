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

def extract_features(df):
    # 1. Базовые текстовые признаки
    df['text_filled'] = df['text'].fillna('')  # текст без NaN
    df['text_length'] = df['text_filled'].apply(len)
    df['num_words'] = df['text_filled'].apply(lambda x: len(x.split()))
    df['is_ad'] = df['text_filled'].str.contains('Реклама', case=False).astype(int)

    # 2. Даты
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month

    # 3. Reactions
    df['love_reactions'] = df['reactions'].apply(
        lambda x: parse_reactions(x).get("ReactionEmoji(emoticon='❤')", 0)
    )
    df['total_reactions'] = df['reactions'].apply(
        lambda x: sum(parse_reactions(x).values()) if len(x) > 2 else 0
    )
    df['num_unique_reactions'] = df['reactions'].apply(
        lambda x: len(parse_reactions(x))
    )

    # 4. Пропуски
    df['forwards'] = df['forwards'].fillna(0)
    df['replies'] = df['replies'].fillna(0)

    # 5. Отношения
    df['replies_to_views'] = df['replies'] / df['views']
    df['forwards_to_views'] = df['forwards'] / df['views']
    df[['replies_to_views', 'forwards_to_views']] = df[['replies_to_views', 'forwards_to_views']].fillna(0)

    # 6. Логарифм views — ПОКА только для экспериментов
    df['log_views'] = np.log1p(df['views'])

    return df

def main():
    # Загрузка данных
    df = pd.read_csv('data/raw/telegram_data.csv')
    df = df[~df['text'].isnull()].reset_index(drop=True)  # только с текстом
    df = extract_features(df)
    # Сплит
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

if __name__ == "__main__":
    main()