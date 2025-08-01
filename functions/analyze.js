const Sentiment = require('sentiment');
const sentiment = new Sentiment();

exports.handler = async (event) => {
  try {
    const { review } = JSON.parse(event.body);
    const result = sentiment.analyze(review);
    
    // Convert score to -3 to +3 scale (matches your original Python output)
    const normalizedScore = Math.min(3, Math.max(-3, result.score / 5)); 
    
    return {
      statusCode: 200,
      body: JSON.stringify({
        review,
        sentiment: result.score > 0 ? "Positive" : result.score < 0 ? "Negative" : "Neutral",
        score: parseFloat(normalizedScore.toFixed(1)) // 1 decimal place
      })
    };
  } catch (error) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: "Analysis failed. Try a different review." })
    };
  }
};