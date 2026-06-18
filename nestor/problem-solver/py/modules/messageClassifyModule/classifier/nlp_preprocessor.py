"""
NLP Preprocessor for Russian text classification.

Предоставляет функции предобработки текста:
- Токенизация
- Лемматизация (через pymorphy2)
- Удаление стоп-слов
- Очистка текста
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Глобальный кэш для морфологического анализатора
_morph = None


def _get_morph():
    """Ленивая инициализация pymorphy2.MorphAnalyzer.
    
    pymorphy2 несовместим с Python 3.12+ (использует устаревший inspect.getargspec).
    Возвращаем None для отключения лемматизации, с минимумом логов.
    """
    global _morph
    if _morph is None:
        try:
            import sys
            if sys.version_info >= (3, 11):
                # pymorphy2 не поддерживает Python 3.11+
                raise RuntimeError("pymorphy2 is incompatible with Python 3.11+")
            from pymorphy2 import MorphAnalyzer
            _morph = MorphAnalyzer(lang="ru")
        except Exception:
            _morph = None
    return _morph


# Стоп-слова для русского и английского языков
RUSSIAN_STOP_WORDS: set[str] = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у",
    "же", "вы", "за", "бы", "по", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был",
    "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там",
    "потом", "себя", "ничего", "ей", "может", "они", "тут", "где",
    "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была",
    "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе",
    "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один",
    "почти", "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда",
    "зачем", "всех", "можно", "при", "наконец", "лишь", "два", "другой",
    "между", "перед", "них", "какая", "друг", "об", "слишком",
    "такой", "более", "нельзя", "сквозь", "эти", "через", "эта",
    "целый", "это", "просто", "вполне", "человек", "словно",
    "пусть", "хорошо", "конечно",
}

ENGLISH_STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "up", "about", "into", "over",
    "after", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "shall", "should", "may", "might", "it", "its",
    "you", "your", "yours", "he", "his", "she", "her", "we", "our",
    "they", "them", "their", "this", "that", "these", "those",
    "i", "me", "my", "mine", "myself", "am", "no", "not", "nor",
    "so", "very", "just", "than", "too", "also",
}

ALL_STOP_WORDS = RUSSIAN_STOP_WORDS | ENGLISH_STOP_WORDS


def clean_text(text: str) -> str:
    """
    Очищает текст от нежелательных символов.
    - Удаляет URL, email, спецсимволы
    - Приводит к нижнему регистру
    - Оставляет буквы, цифры, пробелы
    """
    # Удаляем URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Удаляем /start и другие команды с /
    text = re.sub(r"/\w+", "", text)
    # Оставляем только буквы (рус/англ), цифры и пробелы
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]", " ", text)
    # Приводим к нижнему регистру
    text = text.lower().strip()
    # Схлопываем множественные пробелы
    text = re.sub(r"\s+", " ", text)
    return text


def lemmatize(text: str) -> str:
    """
    Лемматизирует слова в тексте с помощью pymorphy2.
    Если pymorphy2 недоступен, возвращает текст как есть.
    """
    morph = _get_morph()
    if morph is None:
        return text

    words = text.split()
    lemmatized_words = []
    for word in words:
        try:
            parsed = morph.parse(word)[0]
            lemmatized_words.append(parsed.normal_form)
        except (IndexError, Exception):
            lemmatized_words.append(word)
    return " ".join(lemmatized_words)


def remove_stop_words(text: str) -> str:
    """
    Удаляет стоп-слова из текста.
    """
    words = text.split()
    return " ".join(word for word in words if word not in ALL_STOP_WORDS and len(word) > 1)


def preprocess(text: str, do_lemmatize: bool = True, do_remove_stopwords: bool = True) -> str:
    """
    Полный пайплайн предобработки текста.

    Parameters
    ----------
    text : str
        Исходный текст
    do_lemmatize : bool
        Выполнять лемматизацию (по умолчанию True)
    do_remove_stopwords : bool
        Удалять стоп-слова (по умолчанию True)

    Returns
    -------
    str
        Предобработанный текст
    """
    text = clean_text(text)
    if do_lemmatize:
        text = lemmatize(text)
    if do_remove_stopwords:
        text = remove_stop_words(text)
    return text
