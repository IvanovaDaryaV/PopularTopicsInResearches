from sklearn.feature_extraction.text import TfidfVectorizer
import io
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Пример очищенных текстов
# texts = ""
# with io.open('промежуточные результаты/tokenized_2020.txt', encoding='utf-8') as file:
#     for line in file:
#         texts += line
#
# texts = texts.split('\n')
file_paths = [
    # 'промежуточные результаты/tokenized_2019.txt',
    # # 'промежуточные результаты/tokenized_2020.txt',
    # 'промежуточные результаты/tokenized_2021.txt',
    # # 'промежуточные результаты/tokenized_2022.txt',
    # 'промежуточные результаты/tokenized_2023.txt',
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

# Построение TF-IDF матрицы
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_articles)

# Словарь терминов
terms = vectorizer.get_feature_names_out()
print("Словарь терминов:", terms)
print("TF-IDF матрица:\n", tfidf_matrix.toarray())


# Настройка LDA
# Получение топ слов для каждой темы
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

# Темы
for idx, topic in enumerate(lda.components_):
    print(f"Тема {idx + 1}:")
    print([terms[i] for i in topic.argsort()[-10:][::-1]])  # Топ-10 слов каждой темы

# Облако слов для первой темы
topic_0_words = {terms[i]: lda.components_[0][i] for i in range(len(terms))}
wordcloud = WordCloud(background_color='white').generate_from_frequencies(topic_0_words)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()