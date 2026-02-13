from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from transformers import pipeline

# Create FastAPI app
app = FastAPI()

# CORS settings
origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for votes
votes_db = []

# Global ML model
classifier = None

# Load model at startup
@app.on_event("startup")
def load_model():
    global classifier
    print("Loading ML model...")
    try:
        classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=True
        )
        print("Model loaded successfully!")
    except Exception as e:
        print("Failed to load model:", e)
        classifier = None

# Request schemas
class NewsInput(BaseModel):
    text: str

class VoteInput(BaseModel):
    prediction_id: str
    ai_label: str
    user_vote: str

# Prediction endpoint (FIXED)
@app.post("/predict")
def predict_news(data: NewsInput):
    if not classifier:
        return {
            "id": str(uuid.uuid4()),
            "fake_probability": 0.0,
            "label": "ERROR"
        }

    try:
        text = data.text.strip()
        if not text:
            raise ValueError("Empty input")

        result = classifier(text)[0]

        negative_score = None
        positive_score = None

        for r in result:
            if r["label"] == "NEGATIVE":
                negative_score = r["score"]
            elif r["label"] == "POSITIVE":
                positive_score = r["score"]

        # If NEGATIVE not returned, infer it
        if negative_score is None and positive_score is not None:
            negative_score = 1 - positive_score

        fake_probability = round(negative_score, 3)
        label = "FAKE" if fake_probability > 0.5 else "REAL"

        return {
            "id": str(uuid.uuid4()),
            "fake_probability": fake_probability,
            "label": label
        }

    except Exception as e:
        print("Prediction error:", e)
        return {
            "id": str(uuid.uuid4()),
            "fake_probability": 0.0,
            "label": "ERROR"
        }

# Vote endpoint
@app.post("/predict")
def predict_news(data: NewsInput):
    if classifier is None:
        return {
            "id": str(uuid.uuid4()),
            "fake_probability": 0.0,
            "label": "ERROR"
        }

    input_text = data.text.strip()
    if not input_text:
        return {
            "id": str(uuid.uuid4()),
            "fake_probability": 0.0,
            "label": "ERROR"
        }

    try:
        outputs = classifier(input_text, truncation=True)

        # outputs = [[{label, score}, {label, score}]]
        scores = outputs[0]

        fake_score = next(
            item["score"] for item in scores if item["label"] == "NEGATIVE"
        )

        label = "FAKE" if fake_score > 0.5 else "REAL"

        return {
            "id": str(uuid.uuid4()),
            "fake_probability": round(fake_score, 3),
            "label": label
        }

    except Exception as e:
        print("🔥 Prediction failed:", e)
        return {
            "id": str(uuid.uuid4()),
            "fake_probability": 0.0,
            "label": "ERROR"
        }

