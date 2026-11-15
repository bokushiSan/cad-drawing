import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import logging


PROMPT_PATH = (Path(__file__).resolve().parent.parent / "prompt_review.txt")
PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


def encode_image(path: Path) -> str:
    """
    Кодирование изображения в Base64.

    Args:
        path: Путь к изображению.

    Returns:
        Закодированное изображение.
    """

    mime_type = (
        "image/png"
        if path.suffix.lower() == ".png"
        else "image/jpeg"
    )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def parse_response(raw_response: str) -> dict:
    """
    Преобразование ответа модели в словарь.
    
    Args:
        raw_response: Ответ модели.

    Returns:
        Словарь ответа модели.
    """

    text = raw_response.strip()

    if text.startswith("```"):
        text = text.removeprefix("```json")
        text = text.removeprefix("```")
        text = text.removesuffix("```")
        text = text.strip()

    return json.loads(text)


def review_contour(
    original_path: Path,
    contour_path: Path,
    logger: logging.Logger,
    thinking: bool = True,
) -> tuple[str, str]:
    """
    Оценка качества извлеченного контура.
    
    Args:
        original_path: Путь к изображению чертежа.
        contour_path: Путь к изображению контуров.
        thinking: Использовать режим рассуждения модели.

    Returns:
        Вердикт модели и его обоснование.
    """

    load_dotenv()

    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["BASE_URL"],
    )

    model = os.environ["MODEL_NAME"]

    logger.info("VLM")
    logger.info(f"Модель: {model}")
    logger.info(f"Путь к изображению чертежа: {original_path}")
    logger.info(f"Путь к изображению контуров: {contour_path}")
    logger.info(f"Промпт: {PROMPT}")

    content = [
        {"type": "text", "text": PROMPT}
    ]

    for path in [original_path, contour_path]:
        content.append(
            {"type": "image_url", "image_url": {"url": encode_image(path)}
        })

    thinking_modes = (
        [True, True, False] if thinking else [False]
    )

    for attempt, mode in enumerate(
        thinking_modes, start=1
    ):

        logger.info(f"Попытка номер {attempt}")
        logger.info(f"Режим рассуждения: {mode}")

        response = client.chat.completions.create(
            model=os.environ["MODEL_NAME"],
            max_tokens=16384 if mode else 500,
            temperature=0.0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": mode}
            },
            messages=[{"role": "user", "content": content}]
        )

        choice = response.choices[0]
        raw_response = choice.message.content

        logger.info(f"Причина завершения: {choice.finish_reason}")
        logger.info(f"Токены: {response.usage}")
        logger.info(f"Ответ модели:\n{raw_response}")

        if choice.finish_reason == "length":
            logger.warning("Исчерпан лимит токенов")
            continue

        if choice.finish_reason != "stop":
            raise ValueError(f"Генерация завершена: {choice.finish_reason}")

        if not raw_response or not raw_response.strip():
            raise ValueError("Модель вернула пустой ответ")

        break

    else:
        raise ValueError("Исчерпан лимит токенов во всех попытках")

    result = parse_response(raw_response)

    verdict = result["verdict"].strip().lower()
    reason = result["reason"].strip()

    if verdict not in {"green", "yellow", "red"}:
        raise ValueError(f"Неизвестный вердикт: {verdict}")

    if not reason:
        raise ValueError("Отсутствует объяснение вердикта")

    return verdict, reason
