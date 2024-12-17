import io
import spacy
import re
from sklearn.feature_extraction.text import CountVectorizer

'''
Предобработка текста, полученного из PDF в getting_text_from_pdf.py
'''

def preprocess_text(text):
    cleaned_text = text.replace('-\n', '') # обработка переносов слов
    cleaned_text = re.sub(r'[^\u0400-\u04FF\s]', '', cleaned_text)  # Убираем нерусские символы (кроме пробела)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Убираем лишние пробелы и заменяем их одним пробелом
    cleaned_text = cleaned_text.strip()  # Убираем пробелы в начале и в конце текста

    # Загрузка модели spaCy для русского языка
    nlp = spacy.load("ru_core_news_sm")

    # Токенизация и лемматизация
    doc = nlp(cleaned_text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_]

    # Удаление слов длиной меньше 3 (с пдф плохо считываются формулы)
    # lemmatized_tokens = [token for token in lemmatized_tokens if len(token) > 3 and "асч" not in token]
    lemmatized_tokens = [token for token in lemmatized_tokens if len(token) > 3]

    return lemmatized_tokens

# Пример использования
if __name__ == "__main__":
    current_year = 2019
    # vectorizer = CountVectorizer(max_df=0.85, min_df=2, stop_words='english')

    for i in range(5):
        print(f"Считывание файла {i}...")
        tmp = ""
        with io.open(f'промежуточные результаты/{current_year}.txt', encoding='utf-8') as file:
            for line in file:
                tmp += line

        texts = tmp.split('СТАТЬЯ\n')

        print('Предобработка текстов...')
        processed_texts = []
        i = 0
        for sample_text in texts:
            print(i)
            result = preprocess_text(sample_text)
            # dtm = vectorizer.fit_transform(result)
            processed_texts.append(result)
            i += 1

        print('Запись токенизированных предложений...')
        with open(f'промежуточные результаты/tokenized_{current_year}.txt', 'w', encoding='utf-8') as file:
            for article in processed_texts:
                file.write(' '.join(article) + '\n')
        print('Файл успешно записан')
        current_year += 1