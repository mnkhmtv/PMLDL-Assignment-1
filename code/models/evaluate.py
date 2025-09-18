import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

FEATURES = ['hour', 'weekday', 'forwards', 'replies', 'love_reactions', 'total_reactions']

def main():
    test = pd.read_csv('data/processed/test.csv')
    model = joblib.load('models/golden_apple_ridge.joblib')
    vectorizer = joblib.load('models/golden_apple_tfidf.joblib')

    X_text = test['text'].fillna('')
    y_true = test['views']

    X_text_vec = vectorizer.transform(X_text)
    X_num = test[FEATURES].values
    X = np.hstack([X_text_vec.toarray(), X_num])
    
    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R²: {r2:.3f}")

if __name__ == "__main__":
    main()