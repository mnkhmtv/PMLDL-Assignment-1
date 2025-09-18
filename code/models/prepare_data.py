import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split

def parse_reactions(s):
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            return {k: v for k, v in d.items()}
        else:
            return {}
    except Exception:
        return {}

def prepare_features(df):
    # Удалим строки без текста
    df = df[~df['text'].isnull()].reset_index(drop=True)
    # Датафичи
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.weekday
    # Реакции (пример — сердечки и общий счёт)
    df['love_reactions'] = df['reactions'].apply(
        lambda x: parse_reactions(x).get("ReactionEmoji(emoticon='❤')", 0)
    )
    df['total_reactions'] = df['reactions'].apply(
        lambda x: sum(parse_reactions(x).values()) if len(x) > 2 else 0
    )
    # Пропуски во forwards/replies – заполняем нулями
    df['forwards'] = df['forwards'].fillna(0)
    df['replies'] = df['replies'].fillna(0)
    return df

def main():
    df = pd.read_csv('data/raw/telegram_data.csv')
    df = prepare_features(df)
    # Сплит на train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    print('Готово! Train:', train_df.shape, 'Test:', test_df.shape)

if __name__ == "__main__":
    main()