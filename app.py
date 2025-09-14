from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None

def load_model_and_vectorizer():
    """Load the pre-trained model and vectorizer"""
    global model, vectorizer
    
    # Try to load the combined model and vectorizer file first
    if os.path.exists('model_with_vectorizer.pkl'):
        print("Loading model and vectorizer from combined file...")
        with open('model_with_vectorizer.pkl', 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            vectorizer = model_data['vectorizer']
        print("Model and vectorizer loaded successfully!")
    else:
        # Fallback to old method if combined file doesn't exist
        print("Combined model file not found. Please run 'save_model_with_vectorizer.py' first.")
        raise FileNotFoundError("Please run 'save_model_with_vectorizer.py' to create the combined model file.")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment of the input text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text'}), 400
        
        # Clean and vectorize text
        clean_text = text.lower()
        text_vector = vectorizer.transform([clean_text])
        
        # Predict sentiment
        prediction = model.predict(text_vector)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        # Get confidence score
        confidence = model.predict_proba(text_vector)[0]
        max_confidence = max(confidence)
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': round(max_confidence, 3),
            'text': text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model and vectorizer before starting the app
    load_model_and_vectorizer()
    app.run(debug=True, host='0.0.0.0', port=5000)
