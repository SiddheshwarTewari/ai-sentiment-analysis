from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# 1. First initialize app
app = FastAPI()

# 2. Download NLTK data immediately
nltk.download('vader_lexicon')

# 3. Then create analyzer
sid = SentimentIntensityAnalyzer()

# 4. Configure templates (must be after app init)
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(request: Request, review: str = Form(...)):
    scores = sid.polarity_scores(review)
    sentiment = "Positive" if scores['compound'] > 0 else "Negative"
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": sentiment,
        "score": round(scores['compound'], 2)
    })