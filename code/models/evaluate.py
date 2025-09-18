# code/models/evaluate.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

FEATURES = [
    'hour', 'weekday', 'month', 'text_length', 'num_words', 'is_ad',
    'forwards', 'replies', 'love_reactions', 'total_reactions',
    'num_unique_reactions', 'replies_to_views', 'forwards_to_views'
]
TARGET = 'log_views'

def main():
    df = pd.read_csv('data/processed/test.csv')
    model = joblib.load('models/golden_apple_rf.joblib')
    vectorizer = joblib.load('models/golden_apple_tfidf.joblib')

    X_text = df['text_filled'].fillna('')
    X_text_vec = vectorizer.transform(X_text)
    X_num = df[FEATURES].values
    X = np.hstack([X_text_vec.toarray(), X_num])

    # Предсказание логарифма
    y_true_real = df['views'].values
    y_true = df[TARGET].values
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)  # из логарифма — в обычные просмотры

    # Метрики
    mse = mean_squared_error(y_true_real, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true_real, y_pred)
    r2 = r2_score(y_true_real, y_pred)
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R²: {r2:.3f}")

    # График
    plt.figure(figsize=(6,6))
    plt.scatter(y_true_real, y_pred, alpha=0.5)
    plt.xlabel("Фактические просмотры")
    plt.ylabel("Предсказанные просмотры")
    plt.title("Scatter: Предсказание vs Факт")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()