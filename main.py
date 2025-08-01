from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from happytransformer import HappyTextClassification

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize lightweight model (~150MB)
model = HappyTextClassification(
    model_type="DISTILBERT",
    model_name="distilbert-base-uncased-finetuned-sst-2-english"
)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(request: Request, review: str = Form(...)):
    result = model.classify_text(review)
    score = result.score * (1 if result.label == "POSITIVE" else -1)
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": result.label.title(),
        "score": round(score, 2)
    })