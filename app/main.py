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

sid = SentimentIntensityAnalyzer()

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_sentiment(request: Request, review: str = Form(...)):
    scores = sid.polarity_scores(review)
    sentiment = 'Positive' if scores['compound'] > 0.05 else 'Negative' if scores['compound'] < -0.05 else 'Neutral'
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": sentiment,
        "score": scores['compound']
    })