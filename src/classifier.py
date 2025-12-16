import yaml
from src.utils import logger, safe_parse_json
from src.llm_client import LLMClient


class FeedbackClassifier:
    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        from dotenv import load_dotenv
        load_dotenv()

        self.llm = LLMClient(self.config)
        self.categories = self.config["classification"]["categories"]
        self.sentiments = self.config["classification"]["sentiments"]
        
        prompt_config = self.config.get("prompt", {})
        version = prompt_config.get("version", "base")
        self.prompt_template = prompt_config.get(version, "").strip()

        if not self.prompt_template:
            raise ValueError(f"Промпт '{version}' не найден в config/settings.yaml")

    def classify(self, reviews):
        results = []
        for rev in reviews:
            logger.info(f" Обработка ID {rev['id']}: {rev['text'][:40]}...")
            prompt = self.prompt_template.format(
                categories=", ".join(self.categories),
                sentiments=", ".join(self.sentiments),
                text=rev["text"]
            )
            try:
                raw = self.llm.generate(prompt)
                parsed = safe_parse_json(raw)
                if not parsed:
                    raise ValueError(f"Невалидный JSON: {raw}")

                cat = parsed.get("category", "Unknown")
                sent = parsed.get("sentiment", "Unknown")

                results.append({
                    "id": rev["id"],
                    "text": rev["text"],
                    "category": cat if cat in self.categories else "Unknown",
                    "sentiment": sent if sent in self.sentiments else "Unknown",
                    "raw_response": raw
                })
            except Exception as e:
                logger.error(f"ID {rev['id']}: {e}")
                results.append({
                    "id": rev["id"],
                    "text": rev["text"],
                    "category": "Error",
                    "sentiment": "Error",
                    "raw_response": str(e),
                    "error": True
                })
        return results