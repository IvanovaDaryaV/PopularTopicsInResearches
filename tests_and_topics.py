import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import io

def analyze_article(texts, article_num):
    # Построение Count матрицы без удаления стоп-слов
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(texts)
    count_terms = count_vectorizer.get_feature_names_out()

    # Построение TF-IDF матрицы без удаления стоп-слов
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    tfidf_terms = tfidf_vectorizer.get_feature_names_out()

    # Настройка LDA для Count
    lda_count = LatentDirichletAllocation(n_components=1, max_iter=1000, random_state=50)
    lda_count.fit(count_matrix)

    # Настройка LDA для TF-IDF
    lda_tfidf = LatentDirichletAllocation(n_components=1, max_iter=1000, random_state=50)
    lda_tfidf.fit(tfidf_matrix)

    # Настройка NMF для TF-IDF
    nmf_model = NMF(n_components=1, max_iter=1000, random_state=50)
    nmf_topics = nmf_model.fit_transform(tfidf_matrix)

    # Формирование тем
    top_words_topic1 = [count_terms[i] for i in lda_count.components_[0].argsort()[-10:][::-1]]
    top_words_topic2 = [tfidf_terms[i] for i in lda_tfidf.components_[0].argsort()[-10:][::-1]]
    top_words_topic3 = [tfidf_terms[i] for i in nmf_model.components_[0].argsort()[-10:][::-1]]

    # Вывод результатов
    print(f"\n=== Анализ статьи №{article_num + 1} ===\n")

    # CountVectorizer
    print("Словарь терминов (Count):", count_terms)
    print("Count матрица:\n", count_matrix.toarray())
    print("\nТема 1 (CountVectorizer):")
    print(", ".join(top_words_topic1))

    # TF-IDF
    print("\nСловарь терминов (TF-IDF):", tfidf_terms)
    print("TF-IDF матрица:\n", tfidf_matrix.toarray())
    print("\nТема 2 (TfidfVectorizer):")
    print(", ".join(top_words_topic2))

    # NMF
    print("\nТема 3 (NMF):")
    print(", ".join(top_words_topic3))

    print("\n" + "="*50 + "\n")

    return (
        ', '.join(count_terms),
        ', '.join(top_words_topic1),
        ', '.join(top_words_topic2),
        ', '.join(top_words_topic3),
    )


# Считывание файла
file_path = 'промежуточные результаты/tokenized_2019.txt'
try:
    with io.open(file_path, encoding='utf-8') as file:
        text = file.read()
except FileNotFoundError:
    print(f"Файл {file_path} не найден. Загрузите файл и повторите попытку.")
    text = ""

# Разбиение на статьи по маркеру "аннотация"
articles = [article.strip() for article in text.split('аннотация') if article.strip()]

# Сохранение результатов
results = []

# Анализ каждой статьи
for i, article in enumerate(articles):
    terms, topic1_words, topic2_words, topic3_words = analyze_article([article], i)
    results.append(['2019', '', terms, topic1_words, topic2_words, topic3_words])

# Создание DataFrame
df = pd.DataFrame(results, columns=['Год статьи', 'Название статьи', 'Словарь терминов', 'Тема 1', 'Тема 2', 'Тема 3'])

# Сохранение в Excel
output_file = 'ex/analysis_2019.xlsx'
df.to_excel(output_file, index=False, sheet_name='Анализ статей', engine='openpyxl')

print(f"Файл {output_file} успешно создан.")
