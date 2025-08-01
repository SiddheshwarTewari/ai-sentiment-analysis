from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os

# Initialize app
app = FastAPI()

# Configure paths relative to current file
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    
    # Simple aspect detection
    aspects = ['dialogue', 'character', 'plot', 'animation']
    aspect_boost = 0.2 if any(word in text.lower() for word in aspects) else 0
    
    # Final score (-3 to 3 scale)
    return (compound + aspect_boost) * 3

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_review(request: Request, review: str = Form(...)):
    score = analyze_sentiment(review)
    sentiment = "Positive" if score > 1 else "Negative" if score < -1 else "Neutral"
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": sentiment,
        "score": round(score, 2)
    })