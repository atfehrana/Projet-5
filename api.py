from flask import Flask, request, jsonify
import spacy
import joblib
import os

# Initialize the Flask app
app = Flask(__name__)

# Load SpaCy model (make sure 'en_core_web_sm' is installed)
try:
    nlp = spacy.load('en_core_web_sm')
    print("SpaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Load the trained model and vectorizer (replace these with actual paths)
model_local_path = "model.joblib"
vectorizer_local_path = "vectorizer.joblib"

# Load pre-trained model and vectorizer
model = joblib.load(model_local_path)
vectorizer = joblib.load(vectorizer_local_path)
print("Model and vectorizer loaded successfully.")

# Define the /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)

        # Ensure the 'text' key is in the data
        if 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400

        # Process input text with SpaCy
        input_text = data['text']
        doc = nlp(input_text)

        # Vectorize the text using the loaded vectorizer
        vectorized_text = vectorizer.transform([input_text])

        # Make prediction using the loaded model
        prediction = model.predict(vectorized_text)

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("API_PORT", 5000))
    app.run(host='0.0.0.0', port=port)
