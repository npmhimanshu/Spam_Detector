import os
import pickle

os.makedirs("model", exist_ok=True)

with open("model/email_spam_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("model/tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model saved in model/email_spam_model.pkl")
