import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


data = {
    "User-Agent": [
        # Desktop browsers
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.188 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        # Mobile browsers
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Linux; Android 12; Pixel 4 XL Build/SP1A.210812.015) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Mobile Safari/537.36",
        "Mozilla/5.0 (Linux; U; Android 11; en-US; SM-N986U Build/RP1A.200720.012) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Mobile Safari/537.36",
        # Crawlers
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)",
        "Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots)"
    ],
    "Accept-Language": [
        "en-US,en;q=0.9",  # US English
        "en-GB,en;q=0.8",  # British English
        "pt-BR,pt;q=0.8,en-US;q=0.5,en;q=0.3",  # Brazilian Portuguese
        "es-ES,es;q=0.9",  # Spanish
        "fr-FR,fr;q=0.8,en-US;q=0.5,en;q=0.3",  # French
        "zh-CN,zh;q=0.9",  # Simplified Chinese
        "ru-RU,ru;q=0.9",  # Russian
        "de-DE,de;q=0.8,en-US;q=0.6,en;q=0.4",  # German
        "ja-JP,ja;q=0.9"   # Japanese
    ],
    "Category": [
        "desktop",  # Categoria associada
        "desktop",
        "desktop",
        "mobile",
        "mobile",
        "mobile",
        "crawler",
        "crawler",
        "crawler"
    ]
}

df = pd.DataFrame (data)


vectorizer = CountVectorizer()
X_user_agent = vectorizer.fit_transform(df["User-Agent"]).toarray()
X_lang = vectorizer.fit_transform(df["Accept-Language"]).toarray()

X = pd.concat([pd.DataFrame(X_user_agent), pd.DataFrame(X_lang)], axis=1)
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Relatorio de Classificação: \n", classification_report(y_test, y_pred))