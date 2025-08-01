import os
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re

nltk.download('vader_lexicon')
app = FastAPI()

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

# Initialize and weaponize VADER
sid = SentimentIntensityAnalyzer()

# Nuclear negative lexicon
NUCLEAR_NEGATIVE = {
    'trash': -4.0,
    'garbage': -4.0,
    'awful': -3.5,
    'terrible': -3.5,
    'horrible': -3.5,
    'boring': -3.0,
    'waste': -3.2,
    'bad': -2.5,
    'sucks': -3.0,
    'hate': -3.0,
    'disappointing': -2.8,
    'not good': -2.5,
    'do better': -2.5,
    'kinda trash': -3.5,
    'mediocre': -2.0,
    'poor': -2.5,
    'lame': -2.7,
    'dumpster fire': -4.0,
    'hot garbage': -4.0,
    'trainwreck': -3.5
}
sid.lexicon.update(NUCLEAR_NEGATIVE)

# Negative phrase patterns
NEGATIVE_PHRASES = [
    r'not\s+good',
    r'kinda\s+trash',
    r'do\s+better',
    r'was\s+bad',
    r'is\s+bad',
    r'pretty\s+bad',
    r'not\s+great',
    r'could\s+be\s+better',
    r'disappointed',
    r'waste\s+of\s+time'
]

def analyze_sentiment_nuclear(text):
    text_lower = text.lower()
    
    # 1. Immediate negative classification for strong phrases
    for phrase in NEGATIVE_PHRASES:
        if re.search(phrase, text_lower):
            return 'Negative', -2.5  # Default strong negative
    
    # 2. Nuclear lexicon check
    scores = sid.polarity_scores(text)
    
    # 3. Ultra-strict rules
    if scores['neg'] > 0:  # ANY negative words present
        if scores['neu'] < 0.8:  # Not mostly neutral words
            return 'Negative', min(-1.0, scores['compound'])
    
    # 4. Only clearly positive gets positive
    if scores['compound'] > 0.5:  # Very high positive threshold
        return 'Positive', scores['compound']
    
    # 5. Everything else neutral
    return 'Neutral', 0.0

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_sentiment(request: Request, review: str = Form(...)):
    sentiment, score = analyze_sentiment_nuclear(review)
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": sentiment,
        "score": round(score, 2)
    })