#Для решения этой задачи мы будем использовать модель [spaCy ↗](https://spacy.io/), которая является одной из самых популярных и быстрых библиотек для обработки естественного языка. Она предоставляет различные модели для разных языков и задач. В этом примере мы будем использовать английскую модель 'en_core_web_sm'. Вы также можете выбрать другую модель, если она лучше подходит для вашей задачи.

#Для начала установите `spaCy` и модель `en_core_web_sm`, если у вас их еще нет:

# bash
pip install spacy
python -m spacy download en_core_web_sm


# Теперь напишем код для извлечения информации из текстовых сообщений. В этом примере мы будем извлекать имена собственные (имена людей, организаций, географические названия и т. д.) из текста.
import spacy

# Загрузка модели
nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    # Применение модели к тексту
    doc = nlp(text)

    # Извлечение сущностей и их типов
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return entities

# Пример текста
sample_text = "Apple is looking at buying a U.K. startup for $1 billion. John Smith is the CEO of the company."

# Использование функции извлечения
entities = extract_entities(sample_text)
print(entities)
```

#Чтобы оценить качество модели, вам понадобится размеченный датасет с примерами текстов и соответствующими сущностями. Вы можете использовать один из открытых датасетов, таких как [CoNLL-2003 ↗](https://www.clips.uantwerpen.be/conll2003/ner/), или создать свой. Ниже приведен код для оценки качества модели на основе метрики F1-score.

```python
from sklearn.metrics import f1_score

def evaluate_model(test_data):
    y_true = []
    y_pred = []

    for text, true_entities in test_data:
        # Получение предсказанных сущностей
        predicted_entities = extract_entities(text)

        # Извлечение типов сущностей
        true_entities_labels = [ent[1] for ent in true_entities]
        predicted_entities_labels = [ent[1] for ent in predicted_entities]

        y_true.append(true_entities_labels)
        y_pred.append(predicted_entities_labels)

    # Вычисление F1-score
    f1 = f1_score(y_true, y_pred, average='weighted')

    return f1

# Загрузка размеченных данных (пример)
test_data = [
    ("Apple is looking at buying a U.K. startup for $1 billion. John Smith is the CEO of the company.", [('Apple', 'ORG'), ('U.K.', 'GPE'), ('$1 billion', 'MONEY'), ('John Smith', 'PERSON')]),
]

# Оценка качества модели
f1 = evaluate_model(test_data)
print("F1-score:", f1)
```

#Это базовое решение для извлечения информации из текстовых сообщений. Давайте его улучшим.

# Давайте улучшим предыдущий код и добавим некоторые новые функции для извлечения информации из текстовых сообщений. В этом примере мы будем использовать модель [Hugging Face Transformers ↗](https://huggingface.co/transformers/) для более точного извлечения сущностей. Мы также добавим возможность извлекать ключевые слова и кратко обобщать текст.

# Для начала установите `transformers` и `sentence-transformers`:

#bash
pip install transformers
pip install sentence-transformers


#Теперь напишем улучшенный код для извлечения информации из текстовых сообщений:


import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

# Загрузка моделей
nlp_spacy = spacy.load("en_core_web_sm")
nlp_transformers = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")
summarization_pipeline = pipeline("summarization", model="t5-base", tokenizer="t5-base")
sentence_transformer = SentenceTransformer("distilbert-base-nli-mean-tokens")

def extract_entities(text, use_transformers=True):
    if use_transformers:
        # Использование модели Transformers для извлечения сущностей
        entities = nlp_transformers(text)
        entities = [(ent["word"], ent["entity"]) for ent in entities]
    else:
        # Использование модели spaCy для извлечения сущностей
        doc = nlp_spacy(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

    return entities

def extract_keywords(text, n_keywords=5):
    # Разделение текста на предложения
    sentences = [sent.text for sent in nlp_spacy(text).sents]

    # Вычисление векторов предложений
    sentence_embeddings = sentence_transformer.encode(sentences)

    # Кластеризация предложений с использованием K-means
    kmeans = KMeans(n_clusters=n_keywords)
    kmeans.fit(sentence_embeddings)

    # Выбор наиболее центрального предложения из каждого кластера
    keywords = []
    for cluster in range(n_keywords):
        center = kmeans.cluster_centers_[cluster]
        idx = min(range(len(sentence_embeddings)), key=lambda i: np.linalg.norm(sentence_embeddings[i] - center))
        keywords.append(sentences[idx])

    return keywords

def summarize_text(text, max_length=100):
    summary = summarization_pipeline(text, max_length=max_length, min_length=max_length // 2, do_sample=False)
    return summary[0]["summary_text"]

def evaluate_model(test_data, use_transformers=True):
    y_true = []
    y_pred = []

    for text, true_entities in test_data:
        # Получение предсказанных сущностей
        predicted_entities = extract_entities(text, use_transformers=use_transformers)

        # Извлечение типов сущностей
        true_entities_labels = [ent[1] for ent in true_entities]
        predicted_entities_labels = [ent[1] for ent in predicted_entities]

        y_true.append(true_entities_labels)
        y_pred.append(predicted_entities_labels)

    # Вычисление F1-score
    f1 = f1_score(y_true, y_pred, average="weighted")

    return f1

# Пример текста
sample_text = "Apple is looking at buying a U.K. startup for $1 billion. John Smith is the CEO of the company."

# Использование функций извлечения
entities = extract_entities(sample_text)
keywords = extract_keywords(sample_text)
summary = summarize_text(sample_text)

print("Entities:", entities)
print("Keywords:", keywords)
print("Summary:", summary)

# Загрузка размеченных данных (пример)
test_data = [
    ("Apple is looking at buying a U.K. startup for $1 billion. John Smith is the CEO of the company.", [("Apple", "ORG"), ("U.K.", "GPE"), ("$1 billion", "MONEY"), ("John Smith", "PERSON")]),
]

# Оценка качества модели
f1 = evaluate_model(test_data)
print("F1-score:", f1)

"""
В этом решении мы добавили следующие функции:

1. Использование модели Hugging Face Transformers для более точного извлечения сущностей. Мы использовали предварительно обученную модель `dbmdz/bert-large-cased-finetuned-conll03-english` для задачи распознавания именованных сущностей (NER).

2. Извлечение ключевых слов из текста с использованием модели `sentence-transformers` и алгоритма кластеризации K-means. Мы разбиваем текст на предложения, вычисляем их векторные представления и затем группируем их с помощью K-means. Затем выбираем наиболее центральное предложение из каждого кластера в качестве ключевого слова.

3. Краткое изложение текста с использованием модели `t5-base` от Hugging Face Transformers. Мы вводим исходный текст и получаем обобщенный текст благодаря модели сжатия.

4. Оценка качества модели для извлечения сущностей по метрике F1-score. Мы сравниваем истинные и предсказанные метки сущностей для тестовых данных и вычисляем F1-score.

Теперь у вас есть улучшенное решение для извлечения информации из текстовых сообщений с возможностью извлекать именованные сущности, ключевые слова и кратко обобщать текст. Также можно дополнительно настроить и улучшить этот код, добавив предобработку данных, оптимизацию гиперпараметров или использование других моделей, которые лучше подходят для нашей задачи.
"""
