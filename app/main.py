import os
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
app = FastAPI()

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Initialize and customize VADER
sid = SentimentIntensityAnalyzer()

# Enhanced negative word lexicon
NEGATIVE_BOOSTER_WORDS = {
    'trash': -3.0,
    'garbage': -3.0,
    'awful': -2.7,
    'terrible': -2.7,
    'horrible': -2.7,
    'boring': -2.3,
    'waste': -2.5,
    'bad': -1.8,
    'sucks': -2.5,
    'hate': -2.5,
    'disappointing': -2.3,
    'not good': -2.0,
    'do better': -2.0,
    'kinda trash': -2.8,
    'mediocre': -1.5
}
sid.lexicon.update(NEGATIVE_BOOSTER_WORDS)

def analyze_sentiment_aggressive(text):
    scores = sid.polarity_scores(text)
    
    # Custom scoring rules
    if any(phrase in text.lower() for phrase in ['not good', 'do better', 'kinda trash']):
        return 'Negative', min(-1.0, scores['compound'])
    
    # Stricter thresholds
    if scores['neg'] > scores['pos'] * 1.5:  # Negative words dominate
        return 'Negative', scores['compound'] * 1.5  # Amplify negativity
    elif scores['compound'] < -0.1:  # More sensitive negative threshold
        return 'Negative', scores['compound']
    elif scores['compound'] > 0.3:  # Higher positive threshold
        return 'Positive', scores['compound']
    else:
        return 'Neutral', scores['compound']

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_sentiment(request: Request, review: str = Form(...)):
    sentiment, score = analyze_sentiment_aggressive(review)
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": sentiment,
        "score": round(score, 2)
    })