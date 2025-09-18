# Practical Machine Learning and Deep Learning  
## Assignment 1: Telegram Post Performance Prediction  
**Author:** Diana Minnakhmetova

---

### 📖 **Project Overview**

This project implements an end-to-end machine learning pipeline for predicting the success of posts in a Telegram channel. The pipeline takes as input the post text, time of publication, and weekday, and predicts both the expected number of views and the total number of user reactions.  
The solution includes:  
- Data collection from Telegram  
- Data preprocessing and feature engineering  
- Model training, validation, and evaluation  
- Building a REST API and an interactive Streamlit web app  
- Full deployment via Docker Compose

---

### 🎯 **Task Statement**

Given a Telegram post (text), the hour and the weekday of publication, predict:  
- The number of views the post will receive  
- The total number of reactions to the post

---

### 🚦 **Step-by-Step Solution**

#### 1. **Data Collection**

- Used [Telethon](https://github.com/LonamiWebs/Telethon) to collect post history from a real [Telegram channe](https://t.me/diana_minnn) (`code/datasets/download_telegram_data.py`).
- Saved structured data to `data/raw/telegram_data.csv`.

#### 2. **Exploratory Data Analysis & Feature Engineering**

- Explored data in `eda.ipynb` (EDA, missing values, distributions, outliers).
- Parsed and aggregated reactions into a single total count.
- Engineered features:  
  - Post text length  
  - Number of words  
  - Hour and weekday of publication

#### 3. **Data Preparation**

- Implemented in `code/models/prepare_data.py`.
- Removed empty posts.
- Converted date to hour and weekday.
- Calculated features and target variables:
    - Log-transformed targets (`log_views`, `log_total_reactions`) for more stable regression.

#### 4. **Modeling**

- Implemented in `code/models/train_model.py`.
- Vectorized text using TF-IDF (250 features).
- Concatenated text features and engineered numerical features.
- Used `HistGradientBoostingRegressor` wrapped in `MultiOutputRegressor` to predict both targets.
- Saved model and vectorizer to `models/`.

#### 5. **Model Evaluation**

- Implemented in `code/models/evaluate.py`.
- Calculated RMSE on log-transformed targets and analyzed model performance.
- Interpreted results both in log space and original scale.

#### 6. **API Development**

- FastAPI server in `code/deployment/api/app.py`
- Receives a request (`text`, `hour`, `weekday`), preprocesses, vectorizes, predicts and returns the expected views and reactions.

#### 7. **Streamlit Web Application**

- User-friendly frontend in `code/deployment/app/streamlit_app.py`.
- User enters post text, hour, and weekday and receives instant performance forecast.

#### 8. **Deployment (Docker Compose)**

- Two separate containers: `api` (FastAPI) and `app` (Streamlit), managed via `docker-compose.yml`.
- Models are mounted as a shared volume for reproducible deployments.

---

### 🛠️ **How to Run**

#### **1. Clone the Repository**

```bash
git clone https://github.com/mnkhmtv/PMLDL-Assignment-1
cd PMLDL-Assignment-1
```

#### **2. Prepare and Process Data**

> *(Optional: only if you want to recollect or reprocess the data from Telegram)*

```bash
python code/datasets/download_telegram_data.py
python code/models/prepare_data.py
```

#### **3. Train & Evaluate the Model**

```bash
python code/models/train_model.py
python code/models/evaluate.py
```

#### **4. Build and Start the Full Application (API + Web UI)**

```bash
cd code/deployment
docker compose up --build
```

- **API:** [http://localhost:8000/docs](http://localhost:8000/docs)  
- **Web App:** [http://localhost:8501](http://localhost:8501)

#### **5. Usage**

- Go to the web app, enter your post text, hour, and day of week.  
- Instantly receive forecasts of post views and total reactions.

---

### 📝 **Stack & Tools**

- Python 3.10
- Telethon (Data Collection)
- Pandas, Numpy (Data Processing & EDA)
- scikit-learn (ML & Feature Engineering)
- FastAPI (REST API)
- Streamlit (Web App)
- Docker, Docker Compose (Deployment)

---

### 💡 **Why These Choices?**

- **Text features:** TF-IDF is fast and effective for short texts like posts.
- **Log-transform:** Reduces impact of outliers and skewness in regression.
- **MultiOutputRegressor:** Enables predicting multiple targets (views, reactions) in a single pipeline.
- **FastAPI + Streamlit + Docker:** Easy to reproduce, test, and demo. Modern MLOps stack.

---

### 📂 **Project Structure**

```
PMLDL-Assignment-1/
│
├── code/
│   ├── datasets/
│   │   ├── download_telegram_data.py
│   │   └── eda.ipynb
│   ├── deployment/
│   │   ├── api/
│   │   │   ├── app.py
│   │   │   ├── Dockerfile
│   │   │   └── requirements.txt
│   │   ├── app/
│   │   │   ├── streamlit_app.py
│   │   │   ├── Dockerfile
│   │   │   ├── requirements.txt
│   │   │   └── docker-compose.yml
│   ├── models/
│   │   ├── prepare_data.py
│   │   ├── train_model.py
│   │   ├── evaluate.py
│   │   └── ...         
│
├── data/
│   ├── raw/
│   │   └── telegram_data.csv
│   └── processed/
│
├── models/
│   ├── output_rf.joblib
│   ├── output_tfidf.joblib
│
└── requirements.txt
```

---

### 💬 **Course**

This repository is an implementation for  
**Assignment 1, Practical Machine Learning and Deep Learning**  
*Innopolis University, 2025*