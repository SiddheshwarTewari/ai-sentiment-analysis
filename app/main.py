from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")

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