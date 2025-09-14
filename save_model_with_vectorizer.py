import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load and prepare the training data
print("Loading training data...")
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['polarity', 'text']
df = df[df.polarity != 2]
df['polarity'] = df['polarity'].map({0: 0, 4: 1})
df['clean_text'] = df['text'].str.lower()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['polarity'], test_size=0.2, random_state=42
)

# Fit vectorizer on training data
print("Fitting vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train logistic regression model
print("Training model...")
logreg = LogisticRegression(max_iter=100)
logreg.fit(X_train_tfidf, y_train)

# Save both model and vectorizer together
model_data = {
    'model': logreg,
    'vectorizer': vectorizer
}

with open('model_with_vectorizer.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model and vectorizer saved to 'model_with_vectorizer.pkl'")
print("Now you can use this file without loading training data!")
