from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize sentiment analyzer
nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(request: Request, review: str = Form(...)):
    scores = sid.polarity_scores(review)
    sentiment = "Positive" if scores['compound'] > 0.5 else "Negative" if scores['compound'] < -0.5 else "Neutral"
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": sentiment,
        "score": round(scores['compound'] * 3, 2)  # Scale to -3 to 3
    })