import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import joblib

FEATURES = ['hour', 'weekday', 'forwards', 'replies', 'love_reactions', 'total_reactions']

def load_data():
    train = pd.read_csv('data/processed/train.csv')
    return train

def main():
    
    train = load_data()
    X_text = train['text'].fillna('')
    y = train['views']

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=250)
    X_text_vec = vectorizer.fit_transform(X_text)

    X_num = train[FEATURES].values
    X = np.hstack([X_text_vec.toarray(), X_num])

    model = Ridge()
    model.fit(X, y)

    # Сохраняем модель и векторайзер
    joblib.dump(model, 'models/golden_apple_ridge.joblib')
    joblib.dump(vectorizer, 'models/golden_apple_tfidf.joblib')
    print("Модель и TF-IDF сохранены!")

if __name__ == "__main__":
    main()