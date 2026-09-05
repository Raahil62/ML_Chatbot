import json
import joblib

# Load trained model and vectorizer
model = joblib.load("chatbot_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load intents
with open("data/intents.json", "r") as file:
    data = json.load(file)

print("Chatbot loaded successfully!")

def chatbot(message):
    message_vector = vectorizer.transform([message])
    intent = model.predict(message_vector)[0]

    for item in data["intents"]:
        if item["tag"] == intent:
            return item["responses"][0]

# print(chatbot("Where is my order?"))
# print(chatbot("Hello"))
# print(chatbot("I want a refund"))

# while True:
#     message = input("You: ")

#     if message.lower() == "bye":
#         print("Bot: Goodbye!")
#         break

#     print("Bot:", chatbot(message))