import os
import re
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

# Initialize analyzer
sid = SentimentIntensityAnalyzer()

# Nuclear negative lexicon
NUCLEAR_NEGATIVE = {
    'aloof': -1.8, 'disconnected': -1.7, 'wooden': -2.0,
    'did not follow': -2.3, 'plotless': -2.5, 'confusing': -1.8,
    'emotionless': -2.0, 'miscast': -2.0, 'poorly executed': -2.5,
    **{word: -3.0 for word in ['trash', 'garbage', 'awful', 'terrible']}
}
sid.lexicon.update(NUCLEAR_NEGATIVE)

# Contextual phrase patterns
NEGATIVE_PATTERNS = [
    (r'\baloof\b.*\bcharacter\b', -1.8),
    (r'did not follow.*plot', -2.0),
    (r'poorly\sdeveloped', -2.0),
    (r'confusing\s.*plot', -2.0),
    (r'disconnected\s.*character', -1.8)
]

POSITIVE_PATTERNS = [
    (r'well\sput\stogether', 1.8),
    (r'followed.*emotions.*well', 1.7),
    (r'compelling\s.*character', 1.5),
    (r'cohesive\s.*plot', 1.5)
]

def analyze_with_context(text):
    text_lower = text.lower()
    total_score = 0
    
    # 1. Check for negative contextual phrases
    for pattern, score in NEGATIVE_PATTERNS:
        if re.search(pattern, text_lower):
            total_score += score
    
    # 2. Check for positive contextual phrases
    for pattern, score in POSITIVE_PATTERNS:
        if re.search(pattern, text_lower):
            total_score += score
    
    # 3. Only use VADER if no strong patterns found
    if abs(total_score) < 1.0:
        scores = sid.polarity_scores(text)
        total_score = scores['compound']
    
    # 4. Final classification with hysteresis
    if total_score < -0.3:  # Strict negative threshold
        return 'Negative', max(-3.0, total_score)
    elif total_score > 0.5:  # High positive threshold
        return 'Positive', min(3.0, total_score)
    else:
        return 'Neutral', 0.0

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_sentiment(request: Request, review: str = Form(...)):
    sentiment, score = analyze_with_context(review)
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": sentiment,
        "score": round(score, 2)
    })