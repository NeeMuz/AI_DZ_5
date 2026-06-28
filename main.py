from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_ID = "cointegrated/rubert-tiny-sentiment-balanced"
LOCAL_MODEL_DIR = Path(__file__).parent / "models" / "rubert"

LABELS = {
    "POSITIVE": "позитивный",
    "NEGATIVE": "негативный",
    "NEUTRAL": "нейтральный",
    "positive": "позитивный",
    "negative": "негативный",
    "neutral": "нейтральный",
}


def download_and_save_model():
    print(f"Загрузка модели: {MODEL_ID}")
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    model.save_pretrained(LOCAL_MODEL_DIR)

    print(f"Модель сохранена: {LOCAL_MODEL_DIR.resolve()}")
    return LOCAL_MODEL_DIR


def run_local_model(model_path):
    print(f"\nЗапуск модели с диска: {model_path}")
    classifier = pipeline(
        "sentiment-analysis",
        model=str(model_path),
        tokenizer=str(model_path),
    )

    texts = [
        "Мне нравится изучать Python и искусственный интеллект!",
        "Это домашнее задание скучное",
        "Сегодня обычный день",
    ]

    print("\nРезультаты:")
    print("-" * 50)
    for text in texts:
        result = classifier(text)[0]
        label = LABELS.get(result["label"], result["label"])
        score = result["score"] * 100
        print(f"Текст: {text}")
        print(f"Ответ: текст {label} ({score:.1f}%)\n")

    print("Введите свой текст (пустая строка — выход):")
    while True:
        user_text = input("> ").strip()
        if not user_text:
            break
        result = classifier(user_text)[0]
        label = LABELS.get(result["label"], result["label"])
        score = result["score"] * 100
        print(f"Ответ: текст {label} ({score:.1f}%)\n")


def main():
    model_path = download_and_save_model()
    run_local_model(model_path)
    print("Готово.")


if __name__ == "__main__":
    main()
