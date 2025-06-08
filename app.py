from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re

app = Flask(__name__)

TAGS = [
    "технологии", "авто", "спорт", "мода", "красота",
    "путешествия", "еда", "финансы", "образование", "игры",
    "кино", "музыка", "здоровье", "политика", "бизнес",
    "дети", "домашний_очаг", "наука", "хобби", "животные",
    "электроника", "книги", "юмор", "религия", "экология", "государство"
]

classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device=-1
)


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text[:2000]


def extract_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        for element in soup(["script", "style", "header", "footer", "nav"]):
            element.decompose()

        text = soup.get_text(separator=' ', strip=True)
        return clean_text(text)
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return ""


@app.route('/classify', methods=['POST'])
def classify_url():
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({"error": "URL is required"}), 400

    page_content = extract_content(url)
    if not page_content:
        return jsonify({"error": "Failed to extract content"}), 500

    try:
        results = classifier(
            page_content,
            candidate_labels=TAGS,
            multi_label=True
        )

        tags = []
        for label, score in zip(results['labels'], results['scores']):
            if score > 0.1:
                tags.append({
                    "tag": label,
                    "confidence": round(score, 3)
                })

        tags.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            "url": url,
            "content_sample": page_content[:300] + "...",
            "tags": tags[:5]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)