import os
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load lightweight models
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # Use CPU
)

def analyze_review(text):
    try:
        # 1. Base sentiment (lightweight AI)
        result = sentiment_analyzer(text)[0]
        score = result['score'] * (1 if result['label'] == 'POSITIVE' else -1)
        
        # 2. Simple aspect detection
        aspects = {
            'dialogue': 0.15,
            'character': 0.2,
            'plot': 0.15,
            'animation': 0.1,
            'cinematography': 0.1
        }
        aspect_boost = sum(
            weight for word, weight in aspects.items() 
            if word in text.lower()
        )
        
        # 3. Emotion indicators
        emotion_words = {
            'happy': 0.3, 'joy': 0.3, 'love': 0.3,
            'angry': -0.3, 'boring': -0.4, 'disappoint': -0.3
        }
        emotion_boost = sum(
            weight for word, weight in emotion_words.items()
            if word in text.lower()
        )
        
        # Combine scores (-1 to 1 scale)
        final_score = (
            score * 0.7 + 
            min(0.3, aspect_boost) * 0.2 + 
            emotion_boost * 0.1
        )
        
        # Convert to -3 to 3 scale
        scaled_score = final_score * 3
        
        if scaled_score > 1.0:
            return "Positive", round(scaled_score, 2)
        elif scaled_score < -1.0:
            return "Negative", round(scaled_score, 2)
        else:
            return "Neutral", 0.0
            
    except Exception as e:
        print(f"Analysis error: {e}")
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