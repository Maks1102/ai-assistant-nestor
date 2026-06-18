"""
MessageClassifier — гибридный классификатор сообщений.

Использует два уровня:
1. ML-модель (TF-IDF + LogisticRegression) для определения интента (класса сообщения)
2. Regex-паттерны для извлечения сущностей из сообщения

Если ML-модель не загружена (не обучена), использует fallback на keyword matching.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Добавляем путь к текущей директории для поддержки прямого запуска
import sys
_classifier_dir = str(Path(__file__).resolve().parent)
if _classifier_dir not in sys.path:
    sys.path.insert(0, _classifier_dir)

from nlp_preprocessor import preprocess

logger = logging.getLogger(__name__)

CLASSIFIER_DIR = Path(__file__).resolve().parent
MODEL_PATH = CLASSIFIER_DIR / "message_classifier_model.joblib"
VECTORIZER_PATH = CLASSIFIER_DIR / "tfidf_vectorizer.joblib"
CLASSES_PATH = CLASSIFIER_DIR / "message_classes.json"


class MessageClassifier:
    """
    Класс MessageClassifier предназначен для классификации сообщений от пользователя
    по заранее определённым типам из базы знаний (.scs файлы).

    Использует ML-модель (TF-IDF + LogisticRegression) для определения намерения (intent)
    и regex-паттерны для извлечения сущностей.

    Основная задача — определить класс сообщения, извлечь при необходимости сущности и их классы,
    и вернуть результат в стандартизированной структуре.
    """

    # Порог уверенности ML-модели. Если вероятность ниже — возвращаем unknown
    CONFIDENCE_THRESHOLD: float = 0.35

    def __init__(self):
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._classifier: Optional[LogisticRegression] = None
        self._classes: Optional[list[str]] = None
        self._load_model()

    def _load_model(self) -> None:
        """Загружает обученную ML-модель, векторизатор и список классов."""
        if not all(p.exists() for p in [MODEL_PATH, VECTORIZER_PATH, CLASSES_PATH]):
            logger.warning(
                "ML model not found at %s. Using fallback keyword matching.",
                CLASSIFIER_DIR,
            )
            return

        try:
            self._vectorizer = joblib.load(VECTORIZER_PATH)
            self._classifier = joblib.load(MODEL_PATH)
            with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                self._classes = json.load(f)
            logger.info(
                "ML model loaded: %d classes, %s features",
                len(self._classes),
                self._vectorizer.max_features if hasattr(self._vectorizer, "max_features") else "?",
            )
        except Exception as e:
            logger.error("Failed to load ML model: %s", e)
            self._vectorizer = None
            self._classifier = None
            self._classes = None

    @property
    def _model_available(self) -> bool:
        """Проверяет, доступна ли ML-модель."""
        return all(v is not None for v in [self._vectorizer, self._classifier, self._classes])

    # --- Ключевые слова для fallback (если ML-модель недоступна) ---
    _GREETING_KEYWORDS: list[str] = [
        "/start", "привет", "здравствуй", "салют", "хай", "здорово",
        "добрый день", "доброе утро", "добрый вечер",
        "hi", "hello", "хелло", "здравствуйте", "приветствую",
    ]

    _CASUAL_GREETING_KEYWORDS: list[str] = [
        "как дела", "как жизнь", "что нового", "как успехи",
        "как поживаешь", "что происходит", "как настроение",
        "что делаешь",
    ]

    _HELP_KEYWORDS: list[str] = [
        "помощь", "помоги мне", "мне нужна помощь",
        "я не знаю, что делать",
    ]

    _MY_SKILLS_KEYWORDS: list[str] = [
        "что ты умеешь делать", "что ты умеешь", "что умеешь", "зачем ты нужен",
        "какие у тебя навыки", "какие у тебя скилы",
        "скинь свои скилы", "расскажи, что ты умеешь",
    ]

    _CONCEPT_EXAMPLES_KEYWORDS: list[str] = [
        "нужны примеры", "какие примеры",
    ]

    _CONCEPTS_KEYWORDS: list[str] = [
        "какие понятия", "какую информацию", "что знаешь",
    ]

    _CONCEPT_INFORMATION_KEYWORDS: list[str] = [
        "что такое", "что это", "что значит", "кто такой",
        "какое определение имеет", "как понять",
        "что ты знаешь про понятие", "что ты знаешь про",
        "пояснение", "пояснение по понятию", "объяснение",
        "дай определение", "найди определение",
    ]

    _DISCIPLINE_TOPIC_INFORMATION_KEYWORDS: list[str] = [
        "пришли информацию по", "скинь информацию по",
        "что ты знаешь по теме",
    ]

    _DISCIPLINE_INFORMATION_KEYWORDS: list[str] = [
        "что надо делать по дисциплине",
        "что ты знаешь про дисциплину",
        "что ты знаешь о дисциплине",
        "пришли информацию по дисциплине",
        "скинь информацию по дисциплине",
    ]

    _DISCIPLINE_TOPICS_KEYWORDS: list[str] = [
        "какие темы есть", "какие темы по",
        "пришли перечень тем",
        "пришли темы", "скинь темы",
    ]

    _STUDIED_DISCIPLINES_KEYWORDS: list[str] = [
        "какие дисциплины", "как дисциплины проходим",
    ]

    _TOPIC_ABOUT_THEORY_KEYWORDS: list[str] = [
        "про что говорится в теме теория", "расскажи про тему теория",
        "что изучается в теме теория", "о чём тема теория",
        "о чем тема теория", "что в теме теория",
        "инфо по теме теория", "теоретическая часть",
    ]

    _TOPIC_ABOUT_MATH_KEYWORDS: list[str] = [
        "про что говорится в теме математический", "расскажи про тему математический",
        "что изучается в теме математический", "о чём тема математический",
        "о чем тема математический", "что в теме математический",
        "инфо по теме математический", "математический аппарат",
    ]

    _TOPIC_MATH_APPARATE_KEYWORDS: list[str] = [
        "про что говорится в теме математический", "что изучают в теме математический",
        "содержание темы математический", "про тему математический аппарат",
    ]

    _TOPIC_THEORY_KEYWORDS: list[str] = [
        "про что говорится в теме теория", "что изучают в теме теория",
        "содержание темы теория", "про теоретическую часть", "содержание теоретической части",
    ]

    _COURSE_CONTENT_KEYWORDS: list[str] = [
        "содержание курса", "содержимое курса", "план курса",
        "структура курса", "что входит в курс", "программа курса",
        "покажи содержание", "покажи курс", "скинь содержание",
        "пришли содержание", "перечень тем курса",
    ]

    _PERSONAL_INFO_KEYWORDS: list[str] = [
        "что знаешь обо мне", "моя информация", "мой профиль",
        "личная информация", "что обо мне", "какая информация",
        "мои сведения", "что помнишь",
    ]

    _POSITIVE_REPLY_KEYWORDS: list[str] = [
        "да", "давай", "согласен", "ну давай",
        "может быть", "наверно", "ок", "окей",
        "хорошо", "интересно", "разберем", "попробуем",
        "хочу узнать", "расскажи", "покажи", "го",
    ]

    _NEGATIVE_REPLY_KEYWORDS: list[str] = [
        "нет", "нет так", "не хочу", "не надо", "не нужно",
        "зачем", "отстань", "устал", "надоело",
        "скучно", "не могу", "сложно", "лень",
        "не настроен", "позже", "не сейчас",
    ]

    # Паттерны с regex для извлечения сущностей
    _CONCEPT_INFORMATION_PATTERNS: list[re.Pattern] = [
        re.compile(r"что такое\s+(.+)", re.IGNORECASE),
        re.compile(r"что это\s+(.+)", re.IGNORECASE),
        re.compile(r"что значит\s+(.+)", re.IGNORECASE),
        re.compile(r"кто такой\s+(.+)", re.IGNORECASE),
        re.compile(r"как понять\s+(.+)", re.IGNORECASE),
        re.compile(r"пояснение(?: по понятию)?\s+(.+)", re.IGNORECASE),
        re.compile(r"объяснение\s+(.+)", re.IGNORECASE),
        re.compile(r"объясни\s+(.+)", re.IGNORECASE),
        re.compile(r"дай определение\s+(.+)", re.IGNORECASE),
        re.compile(r"найди определение\s+(.+)", re.IGNORECASE),
    ]

    _DISCIPLINE_INFORMATION_PATTERNS: list[re.Pattern] = [
        re.compile(r"что надо делать по дисциплине\s+(.+)", re.IGNORECASE),
        re.compile(r"что ты знаешь про дисциплину\s+(.+)", re.IGNORECASE),
        re.compile(r"что ты знаешь о дисциплине\s+(.+)", re.IGNORECASE),
        re.compile(r"пришли информацию по дисциплине\s+(.+)", re.IGNORECASE),
    ]

    _DISCIPLINE_TOPIC_INFORMATION_PATTERNS: list[re.Pattern] = [
        re.compile(r"что ты знаешь по теме\s+(.+)", re.IGNORECASE),
        re.compile(r"пришли информацию по\s+(.+)", re.IGNORECASE),
        re.compile(r"скинь информацию по\s+(.+)", re.IGNORECASE),
    ]

    _DISCIPLINE_TOPICS_PATTERNS: list[re.Pattern] = [
        re.compile(r"какие темы есть по\s+(.+)", re.IGNORECASE),
        re.compile(r"какие темы по\s+(.+)", re.IGNORECASE),
        re.compile(r"какие темы есть\s*(.*)", re.IGNORECASE),
        re.compile(r"пришли перечень тем(?:\s+по\s+(.+))?", re.IGNORECASE),
        re.compile(r"(?:пришли|скинь) темы по\s+(.+)", re.IGNORECASE),
        re.compile(r"(?:пришли|скинь) темы\s*(.*)", re.IGNORECASE),
    ]

    _CONCEPT_EXAMPLES_PATTERNS: list[re.Pattern] = [
        re.compile(r"какие примеры\s+(.+)", re.IGNORECASE),
        re.compile(r"нужны примеры\s+(.+)", re.IGNORECASE),
    ]

    def classify(self, message: str, message_author_class: str, message_history: list[str]) -> tuple[str, dict[str, str], set[str]]:
        """
        Классифицирует текстовое сообщение, исходя из его содержания и принадлежности
        отправителя к определённому классу.

        Использует ML-модель для определения интента и regex для извлечения сущностей.

        Parameters
        ----------
        message : str
            Текст сообщения, подлежащий анализу и классификации.
        message_author_class : str
            Класс автора сообщения (например: "concept_student").
        message_history : list[str]
            История предыдущих сообщений пользователя для контекстного анализа.

        Returns
        -------
        tuple[str, dict[str, str], set[str]]
            Кортеж из трёх элементов:
            1. Системный идентификатор класса сообщения (например: "concept_student_message_about_greeting").
            2. Основные идентификаторы сущностей и системные идентификаторы их классов, извлечённых из сообщения.
            3. Системные идентификаторы классов сущностей, извлечённых из контекста сообщения.
        """
        if message_author_class == "concept_student":
            if self._model_available:
                return self._classify_with_ml(message, message_history)
            else:
                logger.debug("ML model unavailable, using fallback")
                return self._classify_student_message_fallback(message, message_history)

        return ("concept_unknown_message", {}, set())

    def _classify_with_ml(self, message: str, message_history: list[str]) -> tuple[str, dict[str, str], set[str]]:
        """
        Классифицирует сообщение с помощью ML-модели.
        """
        # Предобработка текста
        processed = preprocess(message)

        # Предикт
        probs = self._classifier.predict_proba(self._vectorizer.transform([processed]))[0]
        best_idx = probs.argmax()
        confidence = probs[best_idx]
        predicted_class = self._classes[best_idx]

        logger.debug(
            'ML predict: "%s" -> %s (confidence=%.4f)',
            message[:50], predicted_class, confidence,
        )

        # Если уверенность ниже порога — возвращаем unknown
        if confidence < self.CONFIDENCE_THRESHOLD:
            logger.debug(
                "Confidence %.4f below threshold %.2f, returning unknown",
                confidence, self.CONFIDENCE_THRESHOLD,
            )
            return ("concept_unknown_message", {}, set())

        # Извлекаем сущности в зависимости от предсказанного класса
        entities = self._extract_entities(message, predicted_class)

        # Если класс требует сущность, но ни одна не была извлечена —
        # пробуем восстановить её из контекста предыдущих сообщений
        if self._missing_required_entity(predicted_class, entities):
            entities = self._fill_entity_from_context(entities, predicted_class, message_history)
            if not self._missing_required_entity(predicted_class, entities):
                logger.debug(
                    'Entity for "%s" filled from context: %s',
                    predicted_class, entities,
                )
            else:
                logger.debug(
                    'Predicted "%s" but no required entity extracted, '
                    'falling back to unknown (confidence=%.4f)',
                    predicted_class, confidence,
                )
                return ("concept_unknown_message", {}, set())

        context_entities = self._extract_context_entities(message_history, predicted_class)
        return (predicted_class, entities, context_entities)

    def _extract_entities(self, message: str, predicted_class: str) -> dict[str, str]:
        """
        Извлекает сущности из сообщения на основе предсказанного класса.
        """
        entities: dict[str, str] = {}

        if predicted_class == "concept_student_message_about_searching_concept_information":
            entity = self._extract_entity(message, self._CONCEPT_INFORMATION_PATTERNS)
            if entity:
                entities["concept"] = entity

        elif predicted_class == "concept_student_message_about_searching_discipline_information":
            entity = self._extract_entity(message, self._DISCIPLINE_INFORMATION_PATTERNS)
            if entity:
                entities["concept_discipline"] = entity

        elif predicted_class == "concept_student_message_about_searching_discipline_topic_information":
            entity = self._extract_entity(message, self._DISCIPLINE_TOPIC_INFORMATION_PATTERNS)
            if entity:
                entities["concept_discipline_topic"] = entity

        elif predicted_class == "concept_student_message_about_searching_discipline_topics":
            entity = self._extract_entity(message, self._DISCIPLINE_TOPICS_PATTERNS)
            if entity:
                entities["concept_discipline"] = entity

        elif predicted_class == "concept_student_message_about_searching_concept_examples":
            entity = self._extract_entity(message, self._CONCEPT_EXAMPLES_PATTERNS)
            if entity:
                entities["concept"] = entity

        return entities

    def _extract_context_entities(self, message_history: list[str], predicted_class: str) -> set[str]:
        """
        Извлекает классы сущностей из контекста (истории сообщений).
        """
        context_entity_classes: set[str] = set()

        # Если в истории есть запрос информации о понятии, то контекст — это concept
        for hist_message in message_history:
            if self._matches_any(hist_message, self._CONCEPT_INFORMATION_KEYWORDS):
                context_entity_classes.add("concept")
            if self._matches_any(hist_message, self._DISCIPLINE_INFORMATION_KEYWORDS):
                context_entity_classes.add("concept_discipline")
            if self._matches_any(hist_message, self._DISCIPLINE_TOPIC_INFORMATION_KEYWORDS):
                context_entity_classes.add("concept_discipline_topic")

        return context_entity_classes

    # --- Fallback keyword matching (сохранён для обратной совместимости) ---

    def _classify_student_message_fallback(self, message: str, message_history: list[str]) -> tuple[str, dict[str, str], set[str]]:
        """Fallback-классификация через keyword matching (как в оригинале)."""
        context_entities: set[str] = self._extract_context_entities(message_history, "")

        if self._matches_any(message, self._DISCIPLINE_INFORMATION_KEYWORDS):
            entity = self._extract_entity(message, self._DISCIPLINE_INFORMATION_PATTERNS)
            entities = {"concept_discipline": entity} if entity else {}
            if entity or "concept_discipline" in context_entities:
                return ("concept_student_message_about_searching_discipline_information",
                        entities, set())

        if self._matches_any(message, self._DISCIPLINE_TOPIC_INFORMATION_KEYWORDS):
            entity = self._extract_entity(message, self._DISCIPLINE_TOPIC_INFORMATION_PATTERNS)
            entities = {"concept_discipline_topic": entity} if entity else {}
            if entity or "concept_discipline_topic" in context_entities:
                return ("concept_student_message_about_searching_discipline_topic_information",
                        entities, set())

        if self._matches_any(message, self._DISCIPLINE_TOPICS_KEYWORDS):
            entity = self._extract_entity(message, self._DISCIPLINE_TOPICS_PATTERNS)
            entities = {"concept_discipline": entity} if entity else {}
            if entity or "concept_discipline" in context_entities:
                return ("concept_student_message_about_searching_discipline_topics",
                        entities, set())

        if self._matches_any(message, self._CONCEPT_EXAMPLES_KEYWORDS):
            entity = self._extract_entity(message, self._CONCEPT_EXAMPLES_PATTERNS)
            entities = {"concept": entity} if entity else {}
            if entity or "concept" in context_entities:
                return ("concept_student_message_about_searching_concept_examples",
                        entities, set())

        if self._matches_any(message, self._CONCEPTS_KEYWORDS):
            return ("concept_student_message_about_searching_concepts", {}, set())

        if self._matches_any(message, self._CONCEPT_INFORMATION_KEYWORDS):
            entity = self._extract_entity(message, self._CONCEPT_INFORMATION_PATTERNS)
            entities = {}
            if entity:
                entities["concept"] = entity
            else:
                # Если сущность не найдена в сообщении, пробуем из контекста
                entities = self._fill_entity_from_context(
                    entities, "concept_student_message_about_searching_concept_information",
                    message_history,
                )
            return ("concept_student_message_about_searching_concept_information",
                    entities, set())

        if self._matches_any(message, self._TOPIC_ABOUT_THEORY_KEYWORDS):
            return ("concept_student_message_about_searching_topic_about_theory", {}, set())

        if self._matches_any(message, self._TOPIC_ABOUT_MATH_KEYWORDS):
            return ("concept_student_message_about_searching_topic_about_math", {}, set())

        if self._matches_any(message, self._TOPIC_MATH_APPARATE_KEYWORDS):
            return ("concept_student_message_about_searching_topic_math_apparate", {}, set())

        if self._matches_any(message, self._TOPIC_THEORY_KEYWORDS):
            return ("concept_student_message_about_searching_topic_theory", {}, set())

        if self._matches_any(message, self._COURSE_CONTENT_KEYWORDS):
            return ("concept_student_message_about_searching_course_content", {}, set())

        if self._matches_any(message, self._STUDIED_DISCIPLINES_KEYWORDS):
            return ("concept_student_message_about_searching_studied_disciplines", {}, set())

        if self._matches_any(message, self._PERSONAL_INFO_KEYWORDS):
            return ("concept_student_message_about_searching_personal_info", {}, set())

        if self._matches_any(message, self._MY_SKILLS_KEYWORDS):
            return ("concept_student_message_about_searching_my_skills", {}, set())

        if self._matches_any(message, self._HELP_KEYWORDS):
            return ("concept_student_message_about_help", {}, set())

        if self._matches_any(message, self._GREETING_KEYWORDS):
            return ("concept_student_message_about_greeting", {}, set())

        if self._matches_any(message, self._CASUAL_GREETING_KEYWORDS):
            return ("concept_student_message_about_casual_greeting", {}, set())

        if self._matches_any(message, self._POSITIVE_REPLY_KEYWORDS):
            return ("concept_student_positive_reply_message", {}, set())

        if self._matches_any(message, self._NEGATIVE_REPLY_KEYWORDS):
            return ("concept_student_negative_reply_message", {}, set())

        return ("concept_unknown_message", {}, set())

    # Маппинг: класс сообщения -> ключ сущности, которая для него обязательна
    _REQUIRED_ENTITY_CLASSES: dict[str, str] = {
        "concept_student_message_about_searching_concept_information": "concept",
        "concept_student_message_about_searching_concept_examples": "concept",
        "concept_student_message_about_searching_discipline_information": "concept_discipline",
        "concept_student_message_about_searching_discipline_topic_information": "concept_discipline_topic",
        "concept_student_message_about_searching_discipline_topics": "concept_discipline",
    }

    def _missing_required_entity(self, predicted_class: str,
                                 entities: dict[str, str]) -> bool:
        """Проверяет, требует ли предсказанный класс обязательную сущность,
        и была ли она извлечена.

        Некоторые классы (например, поиск информации о понятии) требуют
        обязательного наличия сущности. Если модель предсказала такой класс,
        но сущность не была извлечена — классификацию считаем некорректной.
        """
        required_entity = self._REQUIRED_ENTITY_CLASSES.get(predicted_class)
        if required_entity is None:
            return False  # класс не требует обязательной сущности
        return required_entity not in entities or not entities[required_entity]

    def _fill_entity_from_context(self,
                                   entities: dict[str, str],
                                   predicted_class: str,
                                   message_history: list[str]) -> dict[str, str]:
        """Пытается восстановить недостающую сущность из истории диалога.

        Если текущее сообщение не содержит явного указания на понятие
        (например, пользователь написал просто "пояснение" как follow-up),
        ищем в предыдущих сообщениях сущность подходящего класса.
        """
        required_entity = self._REQUIRED_ENTITY_CLASSES.get(predicted_class)
        if required_entity is None:
            return entities

        # Если сущность уже есть — не трогаем
        if required_entity in entities and entities[required_entity]:
            return entities

        # Определяем, какие паттерны использовать для поиска в истории
        patterns_map: dict[str, list[re.Pattern]] = {
            "concept": self._CONCEPT_INFORMATION_PATTERNS,
            "concept_discipline": self._DISCIPLINE_INFORMATION_PATTERNS,
            "concept_discipline_topic": self._DISCIPLINE_TOPIC_INFORMATION_PATTERNS,
        }
        patterns = patterns_map.get(required_entity, [])

        # Ищем в истории сообщений (идём от самого свежего к старому)
        for hist_message in reversed(message_history):
            entity_value = self._extract_entity(hist_message, patterns)
            if entity_value:
                entities[required_entity] = entity_value
                logger.debug(
                    'Filled required entity "%s" = "%s" from history',
                    required_entity, entity_value,
                )
                return entities

        return entities

    @staticmethod
    def _matches_any(message: str, keywords: list[str]) -> bool:
        """Проверяет, содержит ли сообщение хотя бы одно ключевое слово (без учёта регистра)."""
        message_lower = message.lower()
        return any(kw.lower() in message_lower for kw in keywords)

    @staticmethod
    def _extract_entity(message: str, patterns: list[re.Pattern]) -> str:
        """Извлекает сущность из сообщения по списку regex-паттернов."""
        for pattern in patterns:
            match = pattern.search(message)
            if match and match.group(1) is not None:
                return match.group(1).strip().rstrip("?!.")
        return ""
