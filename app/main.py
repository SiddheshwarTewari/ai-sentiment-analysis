import os
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load AI models
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="siebert/sentiment-roberta-large-english",
    return_all_scores=True
)
aspect_model = SentenceTransformer("all-MiniLM-L6-v2")
emotion_analyzer = pipeline(
    "text-classification",
    model="finiteautomata/bertweet-base-emotion-analysis"
)

# Predefined movie aspects and their embeddings
MOVIE_ASPECTS = {
    "dialogue": "conversation between characters",
    "characters": "character development and personality",
    "plot": "story structure and narrative",
    "animation": "visual motion and effects",
    "cinematography": "camera work and visual style"
}
aspect_embeddings = {k: aspect_model.encode(v) for k,v in MOVIE_ASPECTS.items()}

def analyze_review(text):
    # 1. Base sentiment (AI)
    sentiment_results = sentiment_analyzer(text)[0]
    base_score = next(
        (s['score'] for s in sentiment_results if s['label'] == 'POSITIVE'),
        0.5
    ) * 2 - 1  # Convert to -1 to 1 scale
    
    # 2. Emotion detection (AI)
    emotion_results = emotion_analyzer(text)
    emotion_boost = {
        'joy': 0.3, 'happiness': 0.3, 'excitement': 0.2,
        'anger': -0.2, 'disappointment': -0.3
    }.get(emotion_results[0]['label'].lower(), 0)
    
    # 3. Aspect analysis (AI)
    text_embedding = aspect_model.encode(text)
    aspect_scores = {
        aspect: float(np.dot(text_embedding, emb))
        for aspect, emb in aspect_embeddings.items()
    }
    aspect_modifier = max(aspect_scores.values()) * 0.5
    
    # 4. Comparative analysis (AI)
    comparative_keywords = ["like", "similar to", "reminds me of"]
    comparative_boost = 0.2 if any(
        kw in text.lower() for kw in comparative_keywords
    ) else 0
    
    # Combine all scores
    final_score = (
        base_score * 0.6 +
        emotion_boost * 0.2 +
        aspect_modifier * 0.1 +
        comparative_boost * 0.1
    )
    
    # Convert to -3 to 3 scale
    scaled_score = final_score * 3
    
    # Determine sentiment label
    if scaled_score > 1.0:
        return "Positive", round(scaled_score, 2)
    elif scaled_score < -1.0:
        return "Negative", round(scaled_score, 2)
    else:
        return "Neutral", 0.0

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_sentiment(request: Request, review: str = Form(...)):
    sentiment, score = analyze_review(review)
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": sentiment,
        "score": score
    })