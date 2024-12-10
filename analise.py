import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

nltk.download('punkt')

# Список файлов для обработки
file_paths = [
    'промежуточные результаты/tokenized_2019.txt',
    'промежуточные результаты/tokenized_2020.txt',
    'промежуточные результаты/tokenized_2021.txt',
    'промежуточные результаты/tokenized_2022.txt',
    'промежуточные результаты/tokenized_2023.txt',
    'промежуточные результаты/tokenized_2024.txt'
]

# Список для сохранения всех статей и их метаданных
all_articles = []
years = []

# Чтение и сбор текстов из всех файлов
for file_path in file_paths:
    try:
        with open(file_path, encoding='utf-8') as file:
            text = file.read()
            # Разделение текстов на статьи
            articles = [article.strip() for article in text.split('аннотация') if article.strip()]
            # Извлечение года из имени файла
            year = file_path.split('_')[-1].split('.')[0]
            # Добавление статей и годов
            all_articles.extend(articles)
            years.extend([year] * len(articles))
    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Пропускаем...")

if not all_articles:
    print("Нет данных для анализа.")
    exit()

# 1. TF-IDF анализ
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(all_articles)

# 2. Оптимизация числа тем с использованием перплексии
def find_optimal_topics_perplexity(tfidf_matrix, min_topics=2, max_topics=20):
    perplexity_scores = []
    for n_topics in range(min_topics, max_topics + 1):
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=1000)
        lda_model.fit(tfidf_matrix)
        perplexity = lda_model.perplexity(tfidf_matrix)
        perplexity_scores.append((n_topics, perplexity))
        print(f"Число тем: {n_topics}, Перплексия: {perplexity:.2f}")
    return perplexity_scores

# Подбор числа тем
perplexity_scores = find_optimal_topics_perplexity(tfidf_matrix, min_topics=5, max_topics=20)
topics, perplexities = zip(*perplexity_scores)

# График перплексии
plt.figure(figsize=(10, 6))
plt.plot(topics, perplexities, marker='o')
plt.title("Перплексия в зависимости от числа тем")
plt.xlabel("Число тем")
plt.ylabel("Перплексия (чем меньше, тем лучше)")
plt.grid(True)
plt.show()

# Оптимальное число тем
optimal_topics = min(perplexity_scores, key=lambda x: x[1])[0]
print(f"Оптимальное число тем: {optimal_topics}")

# 3. LDA анализ с оптимальным числом тем
lda_model = LatentDirichletAllocation(n_components=optimal_topics, random_state=42, max_iter=2000)
lda_topics = lda_model.fit_transform(tfidf_matrix)

# Получение топ-слов для каждой темы
def get_top_words(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(f"Тема {topic_idx + 1}: {', '.join(top_words)}")
    return topics

lda_topics_words = get_top_words(lda_model, tfidf_vectorizer.get_feature_names_out(), 10)
print("\n=== Темы (LDA) ===")
for topic in lda_topics_words:
    print(topic)

# 4. Оптимизация числа кластеров (K-Means)
def find_optimal_clusters(tfidf_matrix, min_clusters=2, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, labels)
        silhouette_scores.append((n_clusters, score))
        print(f"Число кластеров: {n_clusters}, Силуэтный коэффициент: {score:.3f}")
    return silhouette_scores

silhouette_scores = find_optimal_clusters(tfidf_matrix, min_clusters=2, max_clusters=10)
clusters, silhouettes = zip(*silhouette_scores)

# График силуэтного коэффициента
plt.figure(figsize=(10, 6))
plt.plot(clusters, silhouettes, marker='o')
plt.title("Силуэтный коэффициент в зависимости от числа кластеров")
plt.xlabel("Число кластеров")
plt.ylabel("Силуэтный коэффициент (чем выше, тем лучше)")
plt.grid(True)
plt.show()

# Оптимальное число кластеров
optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
print(f"Оптимальное число кластеров: {optimal_clusters}")

# Кластеризация с оптимальным числом кластеров
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(tfidf_matrix)

# 5. Анализ изменения популярности тем по годам
topic_distribution_by_year = pd.DataFrame(lda_topics, columns=[f"Тема {i+1}" for i in range(optimal_topics)])
topic_distribution_by_year['Год'] = years
topic_trends = topic_distribution_by_year.groupby('Год').mean()

# Визуализация трендов тем
plt.figure(figsize=(10, 6))
for topic in topic_trends.columns[:-1]:
    plt.plot(topic_trends.index, topic_trends[topic], label=topic)
plt.title("Изменение популярности тем по годам")
plt.xlabel("Год")
plt.ylabel("Средняя частотность темы")
plt.legend()
plt.grid(True)
plt.show()

# Сохранение результатов
results = pd.DataFrame({
    'Статья': all_articles,
    'Год': years,
    'Кластер': kmeans_labels,
    **{f"Тема {i+1}": lda_topics[:, i] for i in range(optimal_topics)}
})

output_file = 'optimized_analysis_without_gensim.xlsx'
results.to_excel(output_file, index=False, sheet_name='Анализ статей', engine='openpyxl')
print(f"Файл с анализом сохранен: {output_file}")
