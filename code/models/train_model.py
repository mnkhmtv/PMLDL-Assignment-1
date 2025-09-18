# code/models/train_model.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import joblib

# ВАЖНО: обнови список признаков под твой пайплайн!
FEATURES = [
    'hour', 'weekday', 'month', 'text_length', 'num_words', 'is_ad',
    'forwards', 'replies', 'love_reactions', 'total_reactions',
    'num_unique_reactions', 'replies_to_views', 'forwards_to_views'
]

TARGET = 'log_views'  # для эксперимента, можно заменить на 'views'

def main():
    df = pd.read_csv('data/processed/train.csv')

    # Текстовые признаки
    X_text = df['text_filled'].fillna('')
    vectorizer = TfidfVectorizer(max_features=250)
    X_text_vec = vectorizer.fit_transform(X_text)

    # Числовые признаки
    X_num = df[FEATURES].values
    X = np.hstack([X_text_vec.toarray(), X_num])
    y = df[TARGET].values

    # Модель (RF — мощнее линейных)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Save
    joblib.dump(model, 'models/golden_apple_rf.joblib')
    joblib.dump(vectorizer, 'models/golden_apple_tfidf.joblib')
    print("Модель и TF-IDF сохранены!")

if __name__ == "__main__":
    main()