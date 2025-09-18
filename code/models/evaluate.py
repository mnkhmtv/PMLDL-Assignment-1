import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib

FEATURES = ['hour', 'weekday', 'text_length', 'num_words']
TARGETS = ['log_views', 'log_total_reactions']

def main():
    df = pd.read_csv('data/processed/test.csv')
    model = joblib.load('models/output_rf.joblib')
    vectorizer = joblib.load('models/output_tfidf.joblib')

    X_text = df['text'].fillna('')
    X_num = df[FEATURES].values
    X_text_vec = vectorizer.transform(X_text)
    X = np.hstack([X_text_vec.toarray(), X_num])

    y_true = df[TARGETS].values
    y_pred = model.predict(X)

    for i, name in enumerate(TARGETS):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = mse ** 0.5
        print(f"{name}: RMSE = {rmse:.2f}")

if __name__ == "__main__":
    main()