import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
data.columns = ["label", "message"]

# Convert labels to binary
data["label"] = data["label"].map({"spam": 1, "ham": 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict new message
def predict_spam(message):
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)
    return "Spam 🚫" if prediction[0] == 1 else "Not Spam ✅"

# Example test
test_message = "Congratulations! You won a free lottery ticket"
print("Message:", test_message)
print("Prediction:", predict_spam(test_message))
