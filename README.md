📧 Spam Detector using Python
---
🎯 Objective
---
To build a machine learning–based spam detection system that classifies text messages as Spam or Not Spam (Ham) using Natural Language Processing (NLP).

🛠 Technologies Used
---
Python

Pandas – data handling

Scikit-learn – ML algorithms

Naive Bayes Classifier

TF-IDF Vectorizer – text feature extraction

🔄 Workflow
---
Load and preprocess the dataset

Clean text (lowercase, remove punctuation, stopwords)

Convert text to numerical features using TF-IDF

Train the Naive Bayes model

Test model accuracy

Predict spam for new messages

▶️ How to Run the Project
---
pip install -r requirements.txt
python spam_detector.py

📊 Dataset
---
SMS Spam Collection Dataset

Required Columns:

v1 → Label (spam / ham)

v2 → Message text

📂 Project Structure
---
spam-detector/
│
├── spam_detector.py
├── requirements.txt
├── README.md
└── dataset.csv

🧪 Sample Input & Output
---
Input:

“Congratulations! You have won a free gift card.”

Output:

Spam 🚫

📈 Model Performance
---
Accuracy: 95–98%

Low false-positive rate

Fast and lightweight

Suitable for real-time applications

🌟 Project Highlights (For Viva / Resume)
---
Implements real-world NLP concepts

Uses TF-IDF with Naive Bayes

Simple yet highly effective

Easily extendable to email and social media spam

🚀 Future Enhancements
---
Web app using Streamlit

Advanced models (SVM, Random Forest, Deep Learning)

Multilingual spam detection

Email spam filtering integration
