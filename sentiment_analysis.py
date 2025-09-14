import pickle
import os

# Load the combined model and vectorizer
if os.path.exists('model_with_vectorizer.pkl'):
    print("Loading model and vectorizer from combined file...")
    with open('model_with_vectorizer.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        vectorizer = model_data['vectorizer']
    print("Model and vectorizer loaded successfully!")
else:
    print("Combined model file not found. Please run 'save_model_with_vectorizer.py' first.")
    exit(1)

print("Sentiment Analysis - Enter text to predict sentiment (type 'quit' to exit)")

while True:
    text = input("\nEnter text: ").strip()
    
    if text.lower() == 'quit':
        print("Goodbye!")
        break
    
    if not text:
        print("Please enter some text.")
        continue
    
    # Clean and vectorize text
    clean_text = text.lower()
    text_vector = vectorizer.transform([clean_text])
    
    # Predict sentiment
    prediction = model.predict(text_vector)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    print(f"Sentiment: {sentiment}")
