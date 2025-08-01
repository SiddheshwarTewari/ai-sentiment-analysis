from happytransformer import HappyTextClassification
import json

model = HappyTextClassification(
    model_type="DISTILBERT",
    model_name="distilbert-base-uncased-finetuned-sst-2-english"
)

def handler(event, context):
    try:
        body = json.loads(event["body"])
        review = body["review"]
        result = model.classify_text(review)
        score = result.score * (1 if result.label == "POSITIVE" else -1)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "review": review,
                "sentiment": result.label.title(),
                "score": round(score, 2)
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }