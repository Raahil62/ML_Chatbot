import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

with open("data/intents.json", "r") as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    tags,
    test_size=0.2,
    random_state=42,
    stratify=tags
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, zero_division=0))

import random

def chatbot(message):
    message_vector = vectorizer.transform([message])
    intent = model.predict(message_vector)[0]

    for item in data["intents"]:
        if item["tag"] == intent:
            return random.choice(item["responses"])

# print(chatbot("Where is my order?"))
# print(chatbot("Hello"))
# print(chatbot("I want a refund"))

# while True:
#     message = input("You: ")

#     if message.lower() == "bye":
#         print("Bot: Goodbye!")
#         break

#     print("Bot:", chatbot(message))

import joblib

joblib.dump(model, "chatbot_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")

