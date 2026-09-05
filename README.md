# Customer Support Chatbot 🤖

A simple **machine-learning based Customer Support Chatbot** developed as part of the 
**Slash Mark Data Analyst Internship – Chapter 2: Chatbot Making**.

The chatbot identifies the user's intent from their message and provides a suitable response.

 Project Demo
 A short demonstration of the Customer Support Chatbot.
 
 https://github.com/user-attachments/assets/994530ad-b016-48fe-a2a2-54f602399c1c
 


## Project Overview

The chatbot can handle common customer-support queries related to:

* Greetings and goodbye
* Working hours
* Services
* Pricing
* Order status
* Payment methods
* Refunds
* Contact/support
* Thank-you messages

For example:

**User:** `Where is my order?`
**Chatbot:** `Please provide your order ID so we can check the status.`

## Technologies Used

* Python
* Scikit-learn
* Flask
* Joblib
* JSON
* HTML
* JavaScript

## Machine Learning Approach

The chatbot uses:

**TF-IDF Vectorizer** → Converts text into numerical features.

**Logistic Regression** → Predicts the intent of the user's message.

### Workflow

```text
User Message
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Intent Prediction
     ↓
Matching Response
```

## Dataset

The training data is stored in:

```text
data/intents.json
```

Each intent contains:

* **Tag** – Intent category
* **Patterns** – Example user questions
* **Responses** – Possible chatbot replies

The dataset currently contains **10 intents**.

### Important Files

**`intents.json`**
Contains the chatbot's training patterns and responses.

**`train.py`**
Loads the dataset, performs TF-IDF vectorization, trains the Logistic Regression model, evaluates it, and saves the trained model.

**`chatbot.py`**
Loads the saved model and vectorizer and predicts the intent of new user messages.

**`app.py`**
Creates the Flask application and provides the `/chat` API endpoint.

**`index.html`**
Provides the web interface where users can enter messages and receive chatbot responses.

## Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report

The evaluation results are displayed when `train.py` is executed.

## How to Run

### 1. Activate the virtual environment

```bash
chatbot-env\Scripts\activate
```

### 2. Train the model

```bash
python train.py
```

This generates:

```text
chatbot_model.pkl
vectorizer.pkl
```

### 3. Start the Flask application

```bash
python app.py
```

### 4. Open in browser

```text
http://127.0.0.1:5000
```

## Objective

The objective of this project is to understand the basic implementation of a **machine-learning chatbot**, including:

* Preparing an intent dataset
* Text feature extraction using TF-IDF
* Intent classification using Logistic Regression
* Model evaluation
* Saving and loading trained models
* Connecting the model to a Flask web application

## Limitations

This chatbot is an **intent-based chatbot**. It can respond to queries related to the intents available in `intents.json`. 
It is not a generative AI chatbot and cannot independently answer completely new topics outside its trained intents.


