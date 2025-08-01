const Sentiment = require('sentiment');
const sentiment = new Sentiment();

exports.handler = async (event) => {
  // Log the incoming request for debugging
  console.log('Received event:', JSON.stringify(event, null, 2));

  try {
    // Verify the request has a body
    if (!event.body) {
      throw new Error('No request body provided');
    }

    const { review } = JSON.parse(event.body);
    
    if (!review || typeof review !== 'string') {
      throw new Error('Invalid review text');
    }

    const result = sentiment.analyze(review);
    const normalizedScore = Math.min(3, Math.max(-3, result.score / 5));
    
    const response = {
      statusCode: 200,
      body: JSON.stringify({
        review,
        sentiment: result.score > 0 ? "Positive" : result.score < 0 ? "Negative" : "Neutral",
        score: parseFloat(normalizedScore.toFixed(1))
      })
    };

    console.log('Returning response:', response);
    return response;

  } catch (error) {
    console.error('Error:', error.message);
    return {
      statusCode: 500,
      body: JSON.stringify({ 
        error: error.message,
        stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
      })
    };
  }
};