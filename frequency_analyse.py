import io
from collections import Counter

'''
Удаление очень частых слов, которые портят статистику
'''

year = 2019

for i in range(5):
    tmp = ""
    with io.open(f'промежуточные результаты/tokenized_{year}.txt', encoding='utf-8') as file:
        for line in file:
            tmp += line
    print('Текст до очистки: ', tmp[:300])
    # texts = tmp.split('\n')

    # word_counts = Counter(" ".join(texts).split())

    word_counts = Counter(tmp)

    # Удаляем слишком частотные слова
    high_freq_words = {word for word, freq in word_counts.items() if freq > 10}
    processed_texts = [" ".join([word for word in text.split() if word not in high_freq_words]) for text in
                        tmp]
    print(f"\nТекст после удаления частых слов: {processed_texts[:300]}\n\n")
    print(f"Частые слова: {high_freq_words}")
    year += 1