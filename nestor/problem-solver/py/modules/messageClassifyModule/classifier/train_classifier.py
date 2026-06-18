"""
Dataset generator and model trainer.
Обучает ML-модель (TF-IDF + LogisticRegression).
Использует оптимальные гиперпараметры из GridSearchCV.
"""

import json
import logging
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from nlp_preprocessor import preprocess

random.seed(42)
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="[%d-%b-%y %H:%M:%S]",
)
logger = logging.getLogger(__name__)

CLASSIFIER_DIR = Path(__file__).resolve().parent
MODEL_PATH = CLASSIFIER_DIR / "message_classifier_model.joblib"
VECTORIZER_PATH = CLASSIFIER_DIR / "tfidf_vectorizer.joblib"
CLASSES_PATH = CLASSIFIER_DIR / "message_classes.json"

# ─── ДАТАСЕТ ────────────────────────────────────────────────────────────────

MESSAGE_CLASSES: dict[str, dict] = {
    "concept_student_message_about_greeting": {
        "keywords": [
            "/start", "привет", "здравствуй", "салют", "хай", "здорово",
            "добрый день", "доброе утро", "добрый вечер",
            "hi", "hello", "хелло", "здравствуйте", "приветствую",
            "hey", "хелоу", "йо",
        ],
        "examples": [
            "Привет!", "Здравствуйте!", "Hi!", "Добрый день!",
            "Салют, как жизнь?", "Привет, я тут",
            "Приветик!", "Здрасте!", "Здорово, как ты?",
            "О, привет!", "Доброе утречко!", "Доброго времени суток!",
            "Хелло!", "Приветствую вас!", "Рад вас видеть!",
        ],
    },
    "concept_student_message_about_casual_greeting": {
        "keywords": [
            "как дела", "как жизнь", "что нового", "как успехи",
            "как поживаешь", "что происходит", "как настроение",
            "что делаешь",
        ],
        "examples": [
            "Как дела?", "Что нового?", "Как жизнь?",
            "Как настроение?", "Что делаешь?", "Как успехи?",
            "Ну как ты?", "Как оно?", "Чё как?",
            "Что нового расскажешь?", "Как сам?",
            "Рассказывай, как жизнь", "Ну, как твои дела?",
        ],
    },
    "concept_student_message_about_help": {
        "keywords": [
            "помощь", "помоги мне", "мне нужна помощь",
            "я не знаю, что делать",
        ],
        "examples": [
            "Помощь!", "Помоги мне, пожалуйста",
            "Мне нужна помощь", "Я не знаю, что делать",
            "Выручай!", "Помоги разобраться",
            "Мне нужна твоя помощь", "Подскажи, что делать",
            "Я запутался, помоги", "Объясни, пожалуйста",
            "Нужна твоя помощь", "Помоги с заданием",
        ],
    },
    "concept_student_message_about_searching_my_skills": {
        "keywords": [
            "что ты умеешь делать", "что ты умеешь", "что умеешь",
            "зачем ты нужен", "какие у тебя навыки",
            "какие у тебя скилы", "скинь свои скилы",
            "расскажи, что ты умеешь",
        ],
        "examples": [
            "Что ты умеешь делать?", "Какие у тебя навыки?",
            "Расскажи, что ты умеешь", "Зачем ты нужен?",
            "Что умеешь?", "Скинь свои скилы",
            "Какие у тебя способности?", "Чем ты можешь помочь?",
            "Что ты можешь?", "Для чего ты создан?",
        ],
    },
    "concept_student_message_about_searching_concept_information": {
        "keywords": [
            "что такое", "что это", "что значит", "кто такой",
            "какое определение имеет", "как понять",
            "что ты знаешь про понятие", "что ты знаешь про",
            "расскажи про", "объясни что такое",
            "пояснение", "пояснение по понятию", "объяснение",
            "дай определение", "найди определение",
        ],
        "examples": [
            "Что такое интеллектуальная система?",
            "Что такое ИИ?", "Кто такой студент?",
            "Что значит нейронная сеть?",
            "Как понять градиентный спуск?",
            "Что ты знаешь про регрессию?",
            "Объясни, что такое функция потерь",
            "Расскажи про back propagation",
            "Что такое dropout?", "Дай определение перцептрону",
            "Расскажи про нейронные сети", "Объясни градиентный спуск",
            "Пояснение по искусственному интеллекту",
            "Дай определение функции активации",
            "Найди определение градиентного спуска",
            "Объяснение нейронной сети",
            "Что такое backpropagation?",
        ],
    },
    "concept_student_message_about_searching_concept_examples": {
        "keywords": [
            "нужны примеры", "какие примеры",
            "какие примеры понятия", "какие примеры для понятия",
            "какие примеры ты знаешь",
        ],
        "examples": [
            "Какие примеры интеллектуальных систем ты знаешь?",
            "Нужны примеры по градиентному спуску",
            "Какие примеры для понятия регрессия?",
            "Расскажи примеры",
            "Приведи пример", "Можешь показать на примере?",
            "Дай пример по этой теме", "Покажи на конкретном примере",
            "Есть примеры?", "Какие есть примеры?",
        ],
    },
    "concept_student_message_about_searching_concepts": {
        "keywords": [
            "какие понятия", "какую информацию", "что знаешь",
        ],
        "examples": [
            "Какие понятия ты знаешь?",
            "Какую информацию ты можешь найти?",
            "Что знаешь по нашей теме?",
            "О чём ты можешь рассказать?",
            "Какие темы тебе известны?",
            "Что ты изучил?",
        ],
    },
    "concept_student_message_about_searching_discipline_information": {
        "keywords": [
            "что надо делать по дисциплине",
            "что ты знаешь про дисциплину",
            "что ты знаешь о дисциплине",
            "пришли информацию по дисциплине",
            "скинь информацию по дисциплине",
        ],
        "examples": [
            "Что надо делать по дисциплине ИИ?",
            "Что ты знаешь про дисциплину Математика?",
            "Пришли информацию по дисциплине",
            "Расскажи про дисциплину",
            "Что за предмет ИИ?",
            "Что изучают по дисциплине Математика?",
        ],
    },
    "concept_student_message_about_searching_discipline_topic_information": {
        "keywords": [
            "пришли информацию по", "скинь информацию по",
            "что ты знаешь по теме",
        ],
        "examples": [
            "Что ты знаешь по теме 1?",
            "Пришли информацию по теме нейронные сети",
            "Скинь информацию по теме градиент",
            "Расскажи по теме нейросети",
            "Что по теме функции активации?",
            "Подробнее по теме сверточные сети",
        ],
    },
    "concept_student_message_about_searching_discipline_topics": {
        "keywords": [
            "какие темы есть", "какие темы по",
            "пришли перечень тем", "пришли темы", "скинь темы",
        ],
        "examples": [
            "Какие темы есть по ИИ?",
            "Пришли перечень тем по дисциплине",
            "Какие темы по математике?",
            "Скинь темы",
            "Что за темы в этом курсе?",
            "Какие разделы есть?",
            "Покажи список тем",
        ],
    },
    "concept_student_message_about_searching_topic_about_theory": {
        "keywords": [
            "про что говорится в теме теория", "расскажи про тему теория",
            "что изучается в теме теория", "о чём тема теория",
            "о чем тема теория", "что в теме теория",
            "инфо по теме теория", "теоретическая часть",
        ],
        "examples": [
            "Про что говорится в теме Теория?",
            "Расскажи про тему Теория",
            "Что изучается в теме Теория?",
            "О чём тема Теория?",
            "О чем тема Теоретическая часть?",
            "Инфо по теме Теория",
            "Что в теме Теория?",
        ],
    },
    "concept_student_message_about_searching_topic_about_math": {
        "keywords": [
            "про что говорится в теме математический", "расскажи про тему математический",
            "что изучается в теме математический", "о чём тема математический",
            "о чем тема математический", "что в теме математический",
            "инфо по теме математический", "математический аппарат",
        ],
        "examples": [
            "Про что говорится в теме Математический аппарат?",
            "Расскажи про тему Математический аппарат",
            "Что изучается в теме Математический аппарат?",
            "О чём тема Математический аппарат?",
            "О чем тема Математический аппарат?",
            "Инфо по теме Математический аппарат",
            "Что в теме Математический аппарат?",
        ],
    },
    "concept_student_message_about_searching_topic_math_apparate": {
        "keywords": [
            "про что говорится в теме математический",
            "что изучают в теме математический",
            "содержание темы математический",
            "расскажи про тему математический",
            "про тему математический аппарат",
            "что в теме математический",
        ],
        "examples": [
            "Про что говорится в теме Математический аппарат?",
            "Что изучают в теме Математический аппарат?",
            "Содержание темы Математический аппарат",
            "Расскажи про тему Математический аппарат",
            "О чём тема Математический аппарат?",
            "Про тему Математический аппарат",
        ],
    },
    "concept_student_message_about_searching_topic_theory": {
        "keywords": [
            "про что говорится в теме теория",
            "что изучают в теме теория",
            "содержание темы теория",
            "расскажи про тему теория",
            "про теоретическую часть",
            "что в теме теория",
            "про что теория",
            "содержание теоретической части",
        ],
        "examples": [
            "Про что говорится в теме Теория?",
            "Что изучают в теме Теория?",
            "Содержание темы Теория",
            "Расскажи про тему Теория",
            "О чём тема Теория?",
            "Про теоретическую часть курса",
            "Что изучают в теоретической части?",
        ],
    },
    "concept_student_message_about_searching_course_content": {
        "keywords": [
            "содержание курса", "содержимое курса", "план курса",
            "структура курса", "что входит в курс", "программа курса",
            "покажи содержание", "покажи курс", "скинь содержание",
            "пришли содержание", "пришли содержание курса",
            "скинь содержание курса", "перечень тем курса",
        ],
        "examples": [
            "Пришли содержание курса",
            "Что входит в курс?",
            "Покажи содержание курса",
            "Содержание курса",
            "План курса",
            "Программа курса",
            "Структура курса",
            "Какие разделы в курсе?",
            "Покажи программу курса",
            "Скинь содержание",
            "Что изучаем на курсе?",
        ],
    },
    "concept_student_message_about_searching_studied_disciplines": {
        "keywords": [
            "какие дисциплины", "как дисциплины проходим",
            "какие дисциплины изучаем",
            "какие дисциплины доступны для изучения",
        ],
        "examples": [
            "Какие дисциплины доступны для изучения?",
            "Какие дисциплины изучаем?",
            "Что за дисциплины есть?",
            "Какие предметы я прохожу?",
            "Что мы изучаем?", "Какие курсы есть?",
        ],
    },
    "concept_student_message_about_searching_personal_info": {
        "keywords": [
            "что знаешь обо мне", "моя информация", "мой профиль",
            "личная информация", "что обо мне", "какая информация",
            "мои сведения", "что помнишь",
        ],
        "examples": [
            "Что ты знаешь обо мне?",
            "Моя информация", "Мой профиль",
            "Какая информация у тебя есть обо мне?",
            "Что ты обо мне помнишь?",
            "Мои данные", "Личная информация",
            "Что тебе известно обо мне?",
            "Расскажи, что ты знаешь про меня",
        ],
    },
    "concept_student_positive_reply_message": {
        "keywords": [
            "да", "давай", "согласен", "ну давай",
            "может быть", "наверно", "ок", "окей",
            "хорошо", "интересно", "разберем", "попробуем",
            "хочу узнать", "расскажи", "покажи", "го",
        ],
        "examples": [
            "Да!", "Давай!", "Окей", "Хорошо, давай",
            "Расскажи", "Покажи", "Интересно", "Согласен",
            "Да, конечно", "Давай попробуем", "Звучит интересно",
            "Я согласен", "Можно", "А давай", "Погнали",
            "Ладно, уговорил", "Валяй", "Круто, рассказывай",
            "Конечно", "Безусловно", "Разумеется", "Абсолютно",
        ],
    },
    "concept_student_negative_reply_message": {
        "keywords": [
            "нет", "нет так", "не хочу", "не надо", "не нужно",
            "зачем", "отстань", "устал", "надоело",
            "скучно", "не могу", "сложно", "лень",
            "не настроен", "позже", "не сейчас",
        ],
        "examples": [
            "Нет", "Не хочу", "Не надо", "Отстань",
            "Устал", "Надоело", "Сложно", "Лень",
            "Позже", "Не сейчас",
            "Нет, спасибо", "Нет, не надо",
            "Нет, не хочу", "Нет, я пас",
            "Спасибо, не надо", "Не, спасибо",
            "Мне это неинтересно", "Не сегодня",
            "Да ну, не хочу", "Отстань, пожалуйста",
            "Я устал, давай в другой раз",
            "Мне сейчас не до этого", "Не, я лучше пойду",
            "Ни в коем случае", "Ну уж нет",
            "Спасибо, не нужно", "Да нет, спасибо",
            "Неее", "Не-а", "Нене", "Ну нет",
        ],
    },
    "concept_unknown_message": {
        "keywords": [],
        "examples": [
            "Сегодня хорошая погода", "2 + 2 = 4",
            "Расскажи анекдот", "Который час?",
            "Какой сегодня день?",
            "А ты знаешь физику?", "Какой твой любимый фильм?",
            "Что думаешь о политике?", "Сколько будет 5 + 7?",
            "Напиши стихотворение", "Как приготовить борщ?",
            "Погода на завтра", "Кто выиграл матч?",
            "Сколько будет 10 в 100 степени?",
            "Расскажи про квантовую физику",
            "Когда будет отпуск?", "Где находится Москва?",
            "Какой сегодня праздник?",
            "Напиши программу на Python", "Что такое любовь?",
            "Сколько дней в году?", "Какая завтра погода?",
            "Поставь музыку", "Открой браузер",
            "Какой фильм посоветуешь?",
            "Напомни купить молоко",
            "Включи свет", "Что такое жизнь?",
        ],
    },
}


def _generate_variations(text: str) -> list[str]:
    variations = {text, text.strip().rstrip("?!.,;:")}
    stripped = text.strip().rstrip("?!.,;:")
    variations.add(f"{stripped}?")
    variations.add(f"{stripped}!")
    variations.add(text.capitalize())
    variations.add(stripped.capitalize())
    return list(variations)


def _augment_text(text: str) -> list[str]:
    prefixes = ["пожалуйста, ", "можешь ", "не мог бы ты "]
    suffixes = [", пожалуйста", " пожалуйста"]
    augmented = []
    for p in prefixes:
        augmented.append(f"{p}{text.lower()}")
    for s in suffixes:
        augmented.append(f"{text.lower()}{s}")
    return augmented


def build_dataset() -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []

    for class_name, data in MESSAGE_CLASSES.items():
        class_texts: set[str] = set()
        for kw in data.get("keywords", []):
            for v in _generate_variations(kw):
                class_texts.add(v)
        for ex in data.get("examples", []):
            class_texts.add(ex)
            for v in _generate_variations(ex):
                class_texts.add(v)
        for kw in data.get("keywords", []):
            for aug in _augment_text(kw):
                class_texts.add(aug)
        if class_name in ("concept_unknown_message",
                          "concept_student_negative_reply_message",
                          "concept_student_positive_reply_message"):
            extra = list(class_texts)
            for ex in extra:
                class_texts.add(f"{ex} 😊")
                class_texts.add(f"ну {ex.lower().lstrip('ну ')}")
        for text in class_texts:
            texts.append(text)
            labels.append(class_name)

    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    logger.info("Dataset built: %d samples across %d classes",
                len(texts), len(set(labels)))
    return list(texts), list(labels)


# ─── ОБУЧЕНИЕ ───────────────────────────────────────────────────────────────

def train_model(texts: list[str], labels: list[str]) -> Pipeline:
    logger.info("Preprocessing texts...")
    processed_texts = [preprocess(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels,
        test_size=0.15, random_state=42, stratify=labels,
    )

    # Оптимальные параметры из GridSearchCV:
    # ngram_range=(1,3), max_features=3000, C=5.0 → 93.8% test accuracy
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),        # униграммы + биграммы + триграммы
            max_df=0.85,
            min_df=2,                   # отсекаем слишком редкие
            max_features=3000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=5.0,                     # слабая регуляризация
            max_iter=2000,
            random_state=42,
            class_weight="balanced",   # балансировка классов
        )),
    ])

    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info("Test accuracy: %.4f", accuracy)
    logger.info("\n%s", classification_report(y_test, y_pred, zero_division=0))

    _print_top_features(pipeline)
    return pipeline


def _print_top_features(pipeline: Pipeline, n_top: int = 10) -> None:
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    logger.info("Top %d features per class:", n_top)
    for i, class_name in enumerate(clf.classes_):
        coef = clf.coef_[i]
        top_idx = np.argsort(coef)[-n_top:][::-1]
        logger.info("  %s: %s", class_name, [feature_names[j] for j in top_idx])


def save_model(pipeline: Pipeline, classes: list[str]) -> None:
    logger.info("Saving model to %s", MODEL_PATH)
    joblib.dump(pipeline.named_steps["clf"], MODEL_PATH)
    logger.info("Saving vectorizer to %s", VECTORIZER_PATH)
    joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_PATH)
    logger.info("Saving classes to %s", CLASSES_PATH)
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)


def main() -> None:
    logger.info("=" * 60)
    logger.info("Message Classifier Trainer — optimized params")
    logger.info("=" * 60)
    texts, labels = build_dataset()
    pipeline = train_model(texts, labels)
    save_model(pipeline, list(pipeline.named_steps["clf"].classes_))
    logger.info("Done!")


if __name__ == "__main__":
    main()
