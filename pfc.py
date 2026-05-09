import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report


# Load dataset
df = pd.read_csv('spam_ham_dataset.csv')



X = df['text'].astype(str)
y = df['label_num']

print(y.value_counts())
print(f"\nTotal: {len(df)} emails")
print(f"Spam: {sum(y==1)}")
print(f"Ham: {sum(y==0)}")

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', ' [URL] ', text)  # Replace URLs with [URL]
    text = re.sub(r'www\S+', ' [URL] ', text)  # Replace www URLs
    text = re.sub(r'\b\d{3,}\b', ' [NUMBER] ', text)  # Replace large numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove punctuation and special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply cleaning
X_cleaned = X.apply(clean_text)

#pipline
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )),
    ('nb', MultinomialNB(alpha=0.5))
])

# Train model
print("\nTraining model...")
model.fit(X_cleaned, y)

# Evaluate
y_pred = model.predict(X_cleaned)
accuracy = (y_pred == y).mean()
print(f"Training accuracy: {accuracy:.2%}")

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

def get_metrics():
    return{
        "accuracy": f"{accuracy * 100:.2f}%",
        "precision": f"{precision * 100:.2f}%",
        "recall": f"{recall * 100:.2f}%",
        "f1": f"{f1 * 100:.2f}%",
    }

def predict_msg(message):
    """Enter a message to check if it's spam or ham"""
    cleaned_message = clean_text(message)
    prediction = model.predict([cleaned_message])
    
    if prediction[0] == 1:
        return "Spam!"
    else:
        return "Ham"
    
