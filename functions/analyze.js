const Vader = require('vader-sentiment');

exports.handler = async (event) => {
  try {
    const { review } = JSON.parse(event.body);
    
    // Using VADER which is excellent for short texts
    const intensity = Vader.SentimentIntensityAnalyzer.polarity_scores(review);
    
    // Convert to -3 to +3 scale (compound ranges from -1 to +1)
    const score = intensity.compound * 3;
    
    return {
      statusCode: 200,
      body: JSON.stringify({
        review,
        sentiment: score > 0.5 ? 'Positive' : score < -0.5 ? 'Negative' : 'Neutral',
        score: parseFloat(score.toFixed(1))
      })
    };
  } catch (error) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: "Analysis failed. Please try again." })
    };
  }
};