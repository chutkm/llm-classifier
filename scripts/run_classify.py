import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_json, save_json, logger
from src.classifier import FeedbackClassifier


def main():
    reviews = load_json("data/sample_reviews.json")
    classifier = FeedbackClassifier()
    results = classifier.classify(reviews)
    output_path = classifier.config["output"]["file"]
    save_json(output_path, results)
    logger.info(f"Результат сохранён: {output_path}")


if __name__ == "__main__":
    main()