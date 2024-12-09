from sklearn.feature_extraction.text import TfidfVectorizer
import io
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Пример очищенных текстов
texts = ""
with io.open('промежуточные результаты/processed_articles.txt', encoding='utf-8') as file:
    for line in file:
        texts += line

texts = texts.split('\n')

# Построение TF-IDF матрицы
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# Словарь терминов
terms = vectorizer.get_feature_names_out()
print("Словарь терминов:", terms)
print("TF-IDF матрица:\n", tfidf_matrix.toarray())


# Настройка LDA
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(tfidf_matrix)

# Темы
for idx, topic in enumerate(lda.components_):
    print(f"Тема {idx + 1}:")
    print([terms[i] for i in topic.argsort()[-5:][::-1]])  # Топ-5 слов каждой темы

# Облако слов для первой темы
topic_0_words = {terms[i]: lda.components_[0][i] for i in range(len(terms))}
wordcloud = WordCloud(background_color='white').generate_from_frequencies(topic_0_words)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
