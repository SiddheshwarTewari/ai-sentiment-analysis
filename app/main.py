import os
import re
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import nltk
from collections import defaultdict

nltk.download(['vader_lexicon', 'punkt', 'stopwords', 'averaged_perceptron_tagger'])
app = FastAPI()

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Initialize analyzer with enhanced movie-specific lexicon
sid = SentimentIntensityAnalyzer()

# Domain-specific lexicon enhancements
MOVIE_LEXICON = {
    # Acting related
    'oscar-worthy': 4.0, 'stellar': 3.0, 'wooden': -2.5, 'stiff': -2.0,
    
    # Plot related  
    'gripping': 3.0, 'riveting': 3.0, 'predictable': -2.0, 'cliché': -2.5,
    
    # Technical aspects
    'visually stunning': 3.5, 'cinematography': 2.0, 'poorly edited': -2.5,
    
    # General impact
    'masterpiece': 4.0, 'forgettable': -3.0, 'must-see': 3.5, 'skip it': -3.0
}
sid.lexicon.update(MOVIE_LEXICON)

# Contextual analysis components
stop_words = set(stopwords.words('english'))
intensifiers = {'extremely', 'absolutely', 'utterly', 'completely', 'totally'}
diminishers = {'slightly', 'somewhat', 'marginally', 'partially'}

def analyze_sentiment_deep(text):
    # Preprocessing
    text_lower = text.lower()
    tokens = word_tokenize(text_lower)
    tagged = pos_tag(tokens)
    
    # Initialize scoring
    sentiment_score = 0
    modifier = 1
    negation = False
    
    # Track aspects mentioned
    aspects = defaultdict(float)
    current_aspect = None
    
    # Sentiment analysis with contextual awareness
    for word, tag in tagged:
        if word in stop_words:
            continue
            
        # Handle negations
        if word in {'not', 'never', 'no'}:
            negation = not negation
            continue
            
        # Handle intensifiers/diminishers
        if word in intensifiers:
            modifier = 1.5
            continue
        elif word in diminishers:
            modifier = 0.7
            continue
            
        # Detect aspects (nouns)
        if tag.startswith('NN'):
            current_aspect = word
            
        # Get base sentiment
        word_sentiment = sid.polarity_scores(word)['compound']
        
        # Apply contextual modifiers
        if negation:
            word_sentiment *= -1
        word_sentiment *= modifier
        
        # Update scores
        sentiment_score += word_sentiment
        if current_aspect:
            aspects[current_aspect] += word_sentiment
            
        # Reset modifiers
        modifier = 1
        negation = False
    
    # Normalize score based on length
    if len(tokens) > 0:
        sentiment_score /= len(tokens)
    
    # Apply aspect-based adjustments
    aspect_adjustment = sum(aspects.values()) / (len(aspects) or 1)
    final_score = (sentiment_score * 0.6) + (aspect_adjustment * 0.4)
    
    # Final classification
    if final_score > 0.3:
        return 'Positive', min(3.0, final_score * 3)
    elif final_score < -0.3:
        return 'Negative', max(-3.0, final_score * 3)
    else:
        # Check for strong individual aspects
        if any(v > 1.0 for v in aspects.values()):
            return 'Positive', min(3.0, max(aspects.values()))
        elif any(v < -1.0 for v in aspects.values()):
            return 'Negative', max(-3.0, min(aspects.values()))
        return 'Neutral', 0.0

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_sentiment(request: Request, review: str = Form(...)):
    sentiment, score = analyze_sentiment_deep(review)
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": sentiment,
        "score": round(score, 2)
    })