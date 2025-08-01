from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load lightweight DistilBERT model (~250MB)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # Force CPU usage
)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(request: Request, review: str = Form(...)):
    try:
        result = sentiment_analyzer(review)[0]
        score = result['score'] * (1 if result['label'] == 'POSITIVE' else -1)
        sentiment = "Positive" if score > 0 else "Negative"
        return templates.TemplateResponse("result.html", {
            "request": request,
            "review": review,
            "sentiment": sentiment,
            "score": round(score, 2)
        })
    except:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "review": review,
            "sentiment": "Neutral",
            "score": 0.0
        })