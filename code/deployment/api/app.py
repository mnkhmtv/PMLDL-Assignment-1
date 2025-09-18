from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

MODEL_PATH = "/app/models/output_rf.joblib"
VECTORIZER_PATH = "/app/models/output_tfidf.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

app = FastAPI()

class PostIn(BaseModel):
    text: str
    hour: int
    weekday: int

@app.post("/predict")
def predict_post(post: PostIn):

    text_length = len(post.text)
    num_words = len(post.text.split())
    X_text = vectorizer.transform([post.text])
    X_other = np.array([[post.hour, post.weekday, text_length, num_words]])
    X = np.hstack([X_text.toarray(), X_other])
    
    y_pred_log = model.predict(X)[0]
    
    views = int(np.expm1(y_pred_log[0]))
    total_reactions = int(np.expm1(y_pred_log[1]))
    
    return {
        "predicted_views": views,
        "predicted_total_reactions": total_reactions
    }

@app.get("/")
def read_root():
    return {"message": "API is working"}