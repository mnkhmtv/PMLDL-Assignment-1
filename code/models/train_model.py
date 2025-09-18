import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

FEATURES = ['hour', 'weekday', 'text_length', 'num_words']
TARGETS = ['log_views', 'log_total_reactions']

def main():
    df = pd.read_csv('data/processed/train.csv')
    X_text = df['text'].fillna('')
    X_num = df[FEATURES].values

    vectorizer = TfidfVectorizer(max_features=250)
    X_text_vec = vectorizer.fit_transform(X_text)
    X = np.hstack([X_text_vec.toarray(), X_num])
    y = df[TARGETS].values

    model = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42))
    model.fit(X, y)

    joblib.dump(model, 'models/output_rf.joblib')
    joblib.dump(vectorizer, 'models/output_tfidf.joblib')
    print('Saved model and vectorizer!')

if __name__ == "__main__":
    main()