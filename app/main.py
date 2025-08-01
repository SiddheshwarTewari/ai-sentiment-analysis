import os
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Tiny VADER lexicon (custom optimized)
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Micro-optimized analysis
def analyze_review(text):
    # 1. Base sentiment (lightweight)
    scores = sid.polarity_scores(text)
    
    # 2. Aspect boosters
    aspects = {
        'dialogue': 0.3, 'character': 0.3, 
        'plot': 0.2, 'animation': 0.2
    }
    aspect_boost = sum(
        weight for word, weight in aspects.items() 
        if word in text.lower()
    ) / 2
    
    # 3. Emotion triggers
    emotions = {
        'love': 0.4, 'hate': -0.4, 'happy': 0.3,
        'boring': -0.3, 'fantastic': 0.5, 'terrible': -0.5
    }
    emotion_boost = sum(
        weight for word, weight in emotions.items()
        if word in text.lower()
    ) / 3
    
    final_score = (scores['compound'] + aspect_boost + emotion_boost) * 2.5
    
    if final_score > 1.0:
        return "Positive", round(final_score, 2)
    elif final_score < -1.0:
        return "Negative", round(final_score, 2)
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