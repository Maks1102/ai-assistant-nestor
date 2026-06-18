# Навык работы с проектом Nestor и SCS-кодом

## Общая архитектура проекта

Проект **Nestor** — это интеллектуальная диалоговая система (ostis-система), где знания хранятся в виде SCS-файлов (Semantic Code System). Проект разделён на несколько ключевых областей:

```
knowledge-base/system/
├── nestor.scs              — описание самой системы (имя, навыки, способности)
├── common/
│   └── concepts.scs      — базовые понятия: relation, definition, explanation, idtf и т.д.
├── disciplines/
│   ├── concepts.scs      — понятия дисциплин, навыков
│   └── ai/
│       ├── topic_ai.scs  — дисциплина "ИИ" с её темами
│       └── topics/       — темы дисциплины
│           ├── topic-theory/
│           ├── topic-math-apparate/
│           └── topic-neural-network/
├── messages/
│   ├── concept_message.scs           — базовый класс сообщения
│   ├── unknown-user-messages/        — сообщения от неизвестных пользователей
│   ├── known-user-messages/
│   │   ├── concept_known_user_message.scs
│   │   └── student-messages/
│   │       ├── concept_student_message.scs
│   │       └── examples/
│   │           ├── about-common/     — приветствия, положительные/отрицательные ответы
│   │           ├── about-disciplines/— запросы по дисциплинам и темам
│   │           ├── about-materials/  — запросы по понятиям и материалам
│   │           ├── about-help/       — запросы помощи
│   │           └── about-students/   — запросы информации о студенте
│   └── system-messages/              — системные ответы
└── mood/
    └── concept_mood.scs   — настроение системы
```

---

## Структура SCS-файлов

### 1. Базовые понятия (common/concepts.scs)

Определяют фундаментальные строительные блоки:

```scs
nrel_definition
<- sc_node_non_role_relation;
<- concept_non_role_relation;
<- concept_binary_relation;
=> nrel_main_idtf:
    [определение*]
    (*
        <- lang_ru;;
    *);;

concept_definition
<- sc_node_class;
<- concept_class;
<= nrel_inclusion:
    lang_ru;
=> nrel_main_idtf:
    [определение]
    (*
        <- lang_ru;;
    *);;
```

**Правила:**
- Отношения (`nrel_*`) — `sc_node_non_role_relation`, наследуют `concept_binary_relation`
- Классы (`concept_*`) — `sc_node_class`, наследуют `concept_class`
- Язык помечается `<- lang_ru;;` внутри блока `(* ... *)`
- После `=> nrel_main_idtf:` идёт идентификатор в квадратных скобках

### 2. Тема дисциплины (topic-level)

Файл темы (например, `topic_theory.scs`):

```scs
topic_theory
=> nrel_main_idtf:
    [Теория]
    (*
        <- lang_ru;;
    *);
=> nrel_idtf:
    [Теоретическая часть]
    (*
        <- lang_ru;;
    *);
<- concept_discipline_topic;
=> nrel_key_concepts: {
    concept_neurons;
    concept_activation_func;
    ...
};
=> nrel_definition:
    [Определение...]
    (*
        <- concept_definition;;
    *);
=> nrel_explanation:
    [Пояснение...]
    (*
        <- concept_explanation;;
    *);;
```

**Порядок полей (важен):**
1. `nrel_main_idtf` — короткое название на русском
2. `nrel_idtf` — полное название
3. `<- concept_discipline_topic;` — тип сущности
4. `nrel_key_concepts` — список связанных понятий в `{ }`
5. `nrel_definition` — определение с `<- concept_definition;;`
6. `nrel_explanation` — пояснение с `<- concept_explanation;;`
7. Завершается `*);;` (закрытие блока и двойная точка с запятой)

### 3. Понятие темы (concept-level)

Файл понятия (например, `concept_gradient_descent.scs`):

```scs
concept_gradient_descent
=> nrel_main_idtf:
    [градиентный спуск]
    (*
        <- lang_ru;;
    *);
=> nrel_idtf:
    [Градиентный спуск +оптимизаторы]
    (*
        <- lang_ru;;
    *);
<- concept;
=> nrel_definition:
    [Определение...]
    (*
        <- concept_definition;;
    *);
=> nrel_explanation:
    [Пояснение...]
    (*
        <- concept_explanation;;
    *);;
```

**Отличия от темы:** `<- concept;` вместо `<- concept_discipline_topic;` и отсутствует `nrel_key_concepts`.

### 4. Сообщение-обработчик темы (student message)

Файл сообщения (например, `concept_student_message_about_searching_topic_math_apparate.scs`):

```scs
//////////////////////////////////////
// Topic: Математический аппарат
//////////////////////////////////////

@topic_math_apparate_reply_template = [📚 <b>Математический аппарат</b> — это раздел курса...

<b>Определение:</b>
...
<b>Ключевые понятия:</b>
...
<b>Примеры применения:</b>
...

🤔 Хочешь узнать подробнее по какой-то теме?];;

@if_topic_math_apparate_found_production = {
    rrel_reply_template: {
        rrel_message_template: @topic_math_apparate_reply_template;
        rrel_message_class: concept_system_reply_message_with_question;
        rrel_expected_user_reply_message_classes: { ... }
    }
};;
@if_topic_math_apparate_found_production <- nrel_reply_production;;

concept_student_message_about_searching_topic_math_apparate
<- sc_node_class;
=> nrel_main_idtf:
    [класс сообщений учащихся о содержании темы математический аппарат]
    (*
        <- lang_ru;;
    *);
<- concept_message_topic;
<= nrel_inclusion:
    concept_student_message;
=> nrel_corresponding_skill:
    ...
    (*
        => nrel_main_idtf:
            [узнать содержание темы математический аппарат]
            (*
                <- lang_ru;;
            *);;
        <- concept_skill;;
        <- .process_disciplines;;
    *);
=> nrel_example:
    [Про что говорится в теме Математический аппарат?]
    (*
        <- concept_example;;
    *);
=> nrel_reply_production_chain: <
    @if_topic_math_apparate_found_production
>;
=> nrel_message_keywords: [
    про что говорится в теме математический аппарат;
    ...
];
=> nrel_message_patterns: [
];;
```

**Важные нюансы:**
- Строки внутри `[...]` используют **реальные переносы строк** (не `\n`)
- Эмодзи поддерживаются (`📚`, `🤔`, `🧠` — но некоторые версии парсера могут не принимать `🧠`)
- HTML-теги внутри строк работают (`<b>`, `<i>`)
- `@`-идентификаторы имеют файловую область видимости
- Производственные правила (`nrel_reply_production`) содержат шаблоны ответов и ожидаемые классы ответов

---

## Паттерны работы

### Создание новой темы дисциплины

1. **Создать директорию темы**: `knowledge-base/system/disciplines/ai/topics/topic-xxx/`
2. **Создать файл темы**: `topic_xxx.scs` — с полями `nrel_main_idtf`, `nrel_idtf`, `concept_discipline_topic`, `nrel_key_concepts`, `nrel_definition`, `nrel_explanation`
3. **Создать директорию понятий**: `topic-xxx/concepts/`
4. **Создать файлы понятий**: `concept_xxx.scs` — с полями `nrel_main_idtf`, `nrel_idtf`, `concept`, `nrel_definition`, `nrel_explanation`
5. **Добавить тему в дисциплину**: в `topic_ai.scs` добавить `topic_xxx;` в `nrel_decomposition`
6. **Создать файлы сообщений** (2 файла):
   - `concept_student_message_about_searching_topic_xxx.scs` — специфичный запрос
   - `concept_student_message_about_searching_topic_about_xxx.scs` — общий запрос

### Конвертация из старого формата в новый

Старый формат использует `@link_...` и `@edge_...` с явными рёбрами:

```scs
// СТАРЫЙ ФОРМАТ (не использовать)
@link_2669536797312 = [Текст...];;
@edge_2669536843248 = (topic_neural_network => @link_2669536797312);;
@edge_2669536847088 = (nrel_definition -> @edge_2669536843248);;
```

Новый формат (использовать):

```scs
// НОВЫЙ ФОРМАТ (правильный)
topic_xxx
=> nrel_main_idtf: [Название] (* <- lang_ru;; *);
=> nrel_definition: [Текст...] (* <- concept_definition;; *);;
```

### Правила форматирования SCS

- Каждая конструкция заканчивается `;;`
- Вложенные блоки: `( * ... * )` (со звёздочками у скобок)
- После закрытия вложенного блока: `*);;` или `*);` в зависимости от контекста
- Отступы — табуляция/пробелы (проект использует смешанный стиль)
- Комментарии: `//` для однострочных
- Названия: `snake_case` для идентификаторов, русский язык в строках

### Типы отношений

| Отношение | Назначение |
|-----------|-----------|
| `nrel_main_idtf` | Основной идентификатор (короткое название) |
| `nrel_idtf` | Полный идентификатор (подробное название) |
| `nrel_definition` | Определение сущности |
| `nrel_explanation` | Пояснение/примеры |
| `nrel_key_concepts` | Ключевые понятия темы |
| `nrel_inclusion` | Включение (наследование класса) |
| `nrel_corresponding_skill` | Соответствующий навык |
| `nrel_decomposition` | Декомпозиция (составные части) |
| `nrel_example` | Примеры |
| `nrel_message_keywords` | Ключевые слова для поиска сообщения |

---

## Типичные ошибки и их решение

### 1. `token recognition error`
**Причина:** Некорректный символ в строке или неправильный escape.
**Решение:** Использовать реальные переносы строк вместо `\n`, проверить эмодзи.

### 2. Пропущенная связь в `nrel_decomposition`
**Симптом:** Новая тема не отображается в ответах.
**Решение:** Добавить тему в список `nrel_decomposition` в `topic_ai.scs`.

### 3. Неправильный тип сущности
- Темы: `<- concept_discipline_topic;`
- Понятия: `<- concept;`
- Отношения: `<- sc_node_non_role_relation;`

### 4. Отсутствующие файлы сообщений
**Симптом:** Система не отвечает на вопросы по новой теме.
**Решение:** Создать 2 файла student message (специфичный + общий запрос).

---

## Подход к работе

При работе над проектом следует:

1. **Изучить паттерны** — прочитать 2-3 существующих файла перед созданием новых
2. **Соблюдать консистентность** — новый код должен выглядеть как существующий
3. **Проверять связи** — убедиться, что все ссылки (импорты, декомпозиции) согласованы
4. **Валидировать** — запускать парсер SCS после изменений
5. **Создавать оба сообщения** — для каждой темы всегда создаётся 2 файла (специфичный и общий)
6. **Использовать реальные переносы** — в строках `[...]` использовать реальные `\n`, не экранированные
