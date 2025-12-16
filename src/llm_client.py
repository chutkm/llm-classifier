import os
import time
import requests
from openai import (
    OpenAI,
    AuthenticationError,
    APIError,
    BadRequestError,
    Timeout,
)
from src.utils import logger


class LLMClient:
    def __init__(self, config):
        hf_token = os.getenv("HF_TOKEN", "").strip()
        if not hf_token:
            raise ValueError("HF_TOKEN не задан в .env или пустой")

        resp = requests.get(
            "https://router.huggingface.co/health",
            headers={"Authorization": f"Bearer {hf_token}"}
        )
        if resp.status_code == 401:
            logger.error("401 Unauthorized: HF_TOKEN недействителен или не имеет доступа к Inference Router")
            logger.error("Проверьте:")
            logger.error("   - Токен в .env на корректность")
            raise PermissionError("HF_TOKEN rejected by /health endpoint")

        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token
        )
        self.model = config["llm"]["model"]
        self.timeout = config["llm"]["timeout"]
        self.max_retries = config["llm"]["max_retries"]
        self.retry_delay = config["llm"]["retry_delay"]

    def generate(self, prompt: str) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Ты — строгий классификатор. Отвечай ТОЛЬКО валидным JSON без пояснений. Формат: {\"category\": \"...\", \"sentiment\": \"...\"}"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=100,
                    timeout=self.timeout
                )
                return resp.choices[0].message.content.strip()

            except AuthenticationError:
                logger.error("401 Unauthorized от HF Router. Проверьте HF_TOKEN.")
                raise

            except BadRequestError as e:
                # Например: "The model `xxx` does not exist", "Invalid model name"
                error_msg = str(e)
                if "model" in error_msg.lower() and ("not found" in error_msg.lower() or "does not exist" in error_msg.lower()):
                    logger.error("Модель недоступна")
                else:
                    logger.error(f"Некорректный запрос: {e}")
                raise

            except APIError as e:
                # Обрабатываем 404/5xx от эндпоинта, которые могут означать недоступность модели
                status_code = getattr(e, "status_code", None)
                error_body = getattr(e, "body", {})
                if isinstance(error_body, dict):
                    message = error_body.get("message", "")
                else:
                    message = str(error_body)

                # Проверяем признаки недоступности модели:
                # - 404 Not Found
                # - 503 Service Unavailable ("currently unavailable", "overloaded", etc.)
                # - 500 Internal Server Error + упоминание модели
                if status_code in (404, 503) or (
                    status_code == 500 and ("model" in message.lower() or "unavailable" in message.lower())
                ):
                    logger.error("Модель недоступна")
                else:
                    logger.warning(f"[{attempt}] APIError ({status_code}): {message}")

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise

            except Timeout as e:
                logger.warning(f"[{attempt}] Таймаут запроса: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise

            except Exception as e:
                logger.error(f"Неизвестная ошибка в generate(): {e}")
                raise

        return ""
