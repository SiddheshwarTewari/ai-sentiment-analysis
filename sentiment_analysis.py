import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Sample movie reviews
reviews = [
    "This movie was fantastic! The acting was brilliant.",
    "I hated this movie. It was too long and boring.",
    "It was an average film. Nothing special, nothing terrible.",
    "Absolutely loved the cinematography and soundtrack!",
    "Worst movie ever. A complete waste of time.",
    "The plot was okay, but the characters were very dull."
]

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Analyze each review
results = []
for review in reviews:
    scores = sid.polarity_scores(review)
    sentiment = 'Positive' if scores['compound'] > 0.05 else 'Negative' if scores['compound'] < -0.05 else 'Neutral'
    results.append({'Review': review, 'Sentiment': sentiment, 'Score': scores['compound']})

# Create DataFrame
df = pd.DataFrame(results)

# Print results
print(df)

# Plot sentiments
sentiment_counts = df['Sentiment'].value_counts()
sentiment_counts.plot(kind='bar', color=['red', 'green', 'blue'])
plt.title('Sentiment Analysis of Movie Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.show()
