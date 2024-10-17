from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Load your trained translation model
model = tf.keras.models.load_model('/python/english_to_santali_translation_model.h5')

# Load tokenizer or other necessary preprocessing functions
# Assuming you have tokenizers or any other preprocessing steps
import pickle

with open('/python/english_tokenizer.pkl', 'rb') as f:
    tokenizer_eng = pickle.load(f)

with open('/python/santali_tokenizer.pkl', 'rb') as f:
    tokenizer_santali = pickle.load(f)

# Function to preprocess input text
def preprocess_text(input_text):
    # Convert input text to sequences based on tokenizer
    sequences = tokenizer_eng.texts_to_sequences([input_text])
    # Pad the sequence for a fixed input length
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=20, padding='post')
    return padded_sequences

# Function to translate the text
def translate(input_text):
    # Preprocess the input
    processed_input = preprocess_text(input_text)
    # Predict the translation
    prediction = model.predict(processed_input)
    # Convert prediction to text
    translated_sequence = np.argmax(prediction, axis=-1)
    translated_text = tokenizer_santali.sequences_to_texts(translated_sequence)
    return translated_text[0] if translated_text else "Translation not available"

# Initialize Flask application
app = Flask(__name__)

# Define route for translation
@app.route('/translate', methods=['POST'])
def translate_text():
    # Get JSON data from the request
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    # Get the text to translate
    input_text = data['text']

    # Translate the text
    translated_text = translate(input_text)

    # Return the translated text
    return jsonify({'translated_text': translated_text})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
