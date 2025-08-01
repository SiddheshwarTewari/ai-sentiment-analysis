import os
import re
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

nltk.download(['vader_lexicon', 'punkt', 'stopwords', 'averaged_perceptron_tagger'])
app = FastAPI()

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Enhanced analyzer with aspect detection
sid = SentimentIntensityAnalyzer()

# Movie-specific aspects and their weights
MOVIE_ASPECTS = {
    'animation': 1.5, 'animations': 1.5, 'messaging': 1.2, 'acting': 1.3,
    'plot': 1.4, 'cinematography': 1.3, 'characters': 1.2, 'pacing': 1.1,
    'effects': 1.3, 'soundtrack': 1.1, 'dialogue': 1.1, 'voice acting': 1.2, 
    'character development': 1.3, 'world-building': 1.4
}

# Comparative phrases
COMPARATIVE_PHRASES = {
    'like *': 1.2, 'similar to *': 1.1, 'better than *': 1.3,
    'reminds me of *': 1.1, 'as good as *': 1.2, 'unlike *': -1.2, 
    'nowhere near *': -1.3
}

# Enhanced lexicon
sid.lexicon.update({
    'fantastic': 2.5, 'stellar': 2.3, 'superb': 2.3, 'engaging': 1.8,
    'compelling': 1.8, 'immersive': 1.7, 'groundbreaking': 2.0,
    'forgettable': -2.0, 'generic': -1.8, 'disjointed': -1.9,
    'pretty good': 1.5, 'quite good': 1.4, 'really good': 1.6
})

def analyze_movie_review(text):
    text_lower = text.lower()
    tokens = word_tokenize(text_lower)
    tagged = pos_tag(tokens)
    
    # Initialize scoring
    base_score = 0
    aspect_scores = defaultdict(float)
    aspect_counts = defaultdict(int)
    comparative_boost = 1.0
    
    # Detect comparative statements
    for pattern, boost in COMPARATIVE_PHRASES.items():
        if re.search(r'\b' + pattern.replace('*', r'\w+') + r'\b', text_lower):
            comparative_boost *= boost

    # Handle neutral comparisons with "than"
    if ' than ' in f' {text_lower} ' and not any(
        phrase in text_lower for phrase in ['better than', 'worse than', 'best than']
    ):
        comparative_boost *= 0.9  # Slightly reduce for neutral comparisons
    
    # Analyze each word in context
    i = 0
    while i < len(tagged):
        word, tag = tagged[i]
        
        # Handle aspects
        if word in MOVIE_ASPECTS:
            aspect = word
            aspect_weight = MOVIE_ASPECTS[word]
            
            # Look for sentiment words near the aspect
            window = tagged[max(0,i-3):min(len(tagged),i+4)]
            for w, t in window:
                if t.startswith('JJ') or t.startswith('RB'):  # Adjectives/adverbs
                    sentiment = sid.polarity_scores(w)['compound']
                    aspect_scores[aspect] += sentiment * aspect_weight
                    aspect_counts[aspect] += 1
        
        # Handle negation
        elif word in {'not', 'never', 'no'} and i+1 < len(tagged):
            next_word = tagged[i+1][0]
            sentiment = -sid.polarity_scores(next_word)['compound']
            base_score += sentiment
            i += 1  # Skip next word
        
        # Normal sentiment analysis
        else:
            base_score += sid.polarity_scores(word)['compound']
        
        i += 1
    
    # Calculate weighted aspect scores
    aspect_score = 0
    if aspect_scores:
        aspect_score = sum(
            score/max(1, aspect_counts[aspect]) 
            for aspect, score in aspect_scores.items()
        ) * 0.5  # Aspect weight
    
    # Combine scores with comparative boost
    final_score = (base_score / max(1, len(tokens)) * comparative_boost)
    if aspect_scores:
        final_score = (final_score * 0.4) + (aspect_score * 0.6)
    
    # Final classification
    if final_score > 0.2:
        return 'Positive', min(3.0, final_score * 3)
    elif final_score < -0.2:
        return 'Negative', max(-3.0, final_score * 3)
    else:
        # Check for strong aspects
        if any(s > 0.5 for s in aspect_scores.values()):
            return 'Positive', min(3.0, max(s/max(1,c) for s,c in zip(aspect_scores.values(), aspect_counts.values())))
        elif any(s < -0.5 for s in aspect_scores.values()):
            return 'Negative', max(-3.0, min(s/max(1,c) for s,c in zip(aspect_scores.values(), aspect_counts.values())))
        return 'Neutral', 0.0

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_sentiment(request: Request, review: str = Form(...)):
    sentiment, score = analyze_movie_review(review)
    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "sentiment": sentiment,
        "score": round(score, 2)
    })