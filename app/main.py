from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Initialize app
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Download required NLTK data
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    # Enhanced scoring with aspect detection
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    
    # Aspect detection
    aspects = {
        'dialogue': 0.3, 'character': 0.4, 
        'plot': 0.3, 'animation': 0.2,
        'cinematography': 0.2
    }
    
    # Emotion detection
    emotions = {
        'fantastic': 0.5, 'amazing': 0.4, 'love': 0.4,
        'terrible': -0.5, 'boring': -0.4, 'hate': -0.4
    }
    
    # Calculate boosts
    aspect_boost = sum(
        weight for word, weight in aspects.items() 
        if word in text.lower()
    ) / 2
    
    emotion_boost = sum(
        weight for word, weight in emotions.items()
        if word in text.lower()
    ) / 2
    
    # Final score (-3 to 3 scale)
    final_score = (compound + aspect_boost + emotion_boost) * 3
    
    if final_score > 1:
        return "Positive", round(final_score, 2)
    elif final_score < -1:
        return "Negative", round(final_score, 2)
    return "Neutral", 0.0

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_review(request: Request, review: str = Form(...)):
    sentiment, score = analyze_sentiment(review)
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": sentiment,
        "score": score
    })