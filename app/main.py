import os
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
app = FastAPI()

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, "templates")

# Setup templates and static files with absolute paths
templates = Jinja2Templates(directory=templates_dir)

# Initialize sentiment analyzer with adjusted thresholds
sid = SentimentIntensityAnalyzer()

# Custom scoring function with adjusted thresholds
def analyze_sentiment_custom(text):
    scores = sid.polarity_scores(text)
    
    # Adjusted thresholds for better negative detection
    if scores['neg'] > 0.5:  # If negative words dominate
        return 'Negative', scores['compound']
    elif scores['compound'] <= -0.2:  # More sensitive negative threshold
        return 'Negative', scores['compound']
    elif scores['compound'] >= 0.2:  # Standard positive threshold
        return 'Positive', scores['compound']
    else:
        return 'Neutral', scores['compound']

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_sentiment(request: Request, review: str = Form(...)):
    sentiment, score = analyze_sentiment_custom(review)
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": sentiment,
        "score": score
    })