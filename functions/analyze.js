const HappyTextClassification = require('happy-transformer').HappyTextClassification;

const model = new HappyTextClassification("DISTILBERT", "distilbert-base-uncased-finetuned-sst-2-english");

exports.handler = async (event) => {
  try {
    const { review } = JSON.parse(event.body);
    const result = await model.classifyText(review);
    const score = result.score * (result.label === "POSITIVE" ? 1 : -1);

    return {
      statusCode: 200,
      body: JSON.stringify({
        review,
        sentiment: result.label.charAt(0) + result.label.slice(1).toLowerCase(),
        score: Math.round(score * 100) / 100
      })
    };
  } catch (error) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message })
    };
  }
};