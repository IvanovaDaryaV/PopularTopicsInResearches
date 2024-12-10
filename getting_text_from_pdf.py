import fitz  # PyMuPDF
import re
def extract_articles_from_pdf(pdf_path, article_marker="АННОТАЦИЯ"):
    doc = fitz.open(pdf_path)
    all_text = ""

    # чтение текста постранично
    for page_num in range(len(doc)):
        page = doc[page_num]
        all_text += page.get_text()

    articles = re.split(rf"({article_marker})", all_text, flags=re.IGNORECASE)

    # Объединение частей: текст маркера + контент
    articles_combined = []
    for i in range(1, len(articles), 2):  # Маркер и текст идут парами
        marker = articles[i].strip()
        content = articles[i + 1].strip() if i + 1 < len(articles) else ""
        articles_combined.append(f"{marker}\n{content}")

    return articles_combined

def remove_ending(article):
    return "СТАТЬЯ\n" + re.split(rf"({"СПИСОК ЛИТЕРАТУРЫ"})", article, flags=re.IGNORECASE)[0]

if __name__ == "__main__":
    # pdf_path = "сборник_статей.pdf"
    pdf_path = "исходные файлы (сборники)/2024.pdf"

    # разбиение текста по слову "аннотация"
    articles = extract_articles_from_pdf(pdf_path)

    # убрать текст после "список литературы"
    list_of_articles = []
    for article in articles:
        list_of_articles.append(remove_ending(article))

    with open('промежуточные результаты/2024.txt', 'w', encoding='utf-8') as file:
        # for article in list_of_articles:
        file.write(' '.join(list_of_articles) + '\n')

    print(list_of_articles[0])
