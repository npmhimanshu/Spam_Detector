import pickle

# Load model and vectorizer
with open("model/email_spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_spam(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    return "Spam 🚫" if prediction == 1 else "Not Spam ✅"

# Test
if __name__ == "__main__":
    msg = input("Enter a message: ")
    print("Prediction:", predict_spam(msg))
