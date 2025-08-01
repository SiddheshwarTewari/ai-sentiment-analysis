from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from transformers import pipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load lightweight model (will download on first run)
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    print(f"Model loading error: {e}")
    sentiment_analyzer = None

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(request: Request, review: str = Form(...)):
    if not sentiment_analyzer:
        return {"error": "Model not loaded"}, 503
    
    result = sentiment_analyzer(review)[0]
    score = result['score'] * (1 if result['label'] == 'POSITIVE' else -1)
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": "Positive" if score > 0 else "Negative",
        "score": round(score, 2)
    })