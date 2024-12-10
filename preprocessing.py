import io
import spacy
import re

def preprocess_text(text):
    cleaned_text = re.sub(r'[^\u0400-\u04FF\s]', '', text)  # Убираем нерусские символы (кроме пробела)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Убираем лишние пробелы и заменяем их одним пробелом
    cleaned_text = cleaned_text.strip()  # Убираем пробелы в начале и в конце текста

    # Загрузка модели spaCy для русского языка
    nlp = spacy.load("ru_core_news_sm")

    # Токенизация и лемматизация
    doc = nlp(cleaned_text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_]

    # Удаление слова "рис"
    lemmatized_tokens = [token for token in lemmatized_tokens if token.lower() != "рис"]

    return lemmatized_tokens

# Пример использования
if __name__ == "__main__":
    print("Считывание файла...")
    tmp = ""
    with io.open('промежуточные результаты/2024.txt', encoding='utf-8') as file:
        for line in file:
            tmp += line

    texts = tmp.split('СТАТЬЯ\n')

    print('Предобработка текстов...')
    processed_texts = []
    for sample_text in texts:
        result = preprocess_text(sample_text)
        processed_texts.append(result)

    print('Запись токенизированных предложений...')
    with open('промежуточные результаты/tokenized_2024                                                                            .txt', 'w', encoding='utf-8') as file:
        for article in processed_texts:
            file.write(' '.join(article) + '\n')
    print('Файл успешно записан')
