# app.py

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import logging

# Load your trained translation model
model = tf.keras.models.load_model('/model/english_to_santali_translation_model.h5')

# Load tokenizers
with open('/tokenizers/english_tokenizer.pkl', 'rb') as f:
    tokenizer_eng = pickle.load(f)

with open('/tokenizers/santali_tokenizer.pkl', 'rb') as f:
    tokenizer_santali = pickle.load(f)

# Function to preprocess input text
def preprocess_text(input_text):
    sequences = tokenizer_eng.texts_to_sequences([input_text])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=20, padding='post')
    return padded_sequences

# Function to translate the text
def translate(input_text):
    processed_input = preprocess_text(input_text)
    prediction = model.predict(processed_input)
    translated_sequence = np.argmax(prediction, axis=-1)
    translated_text = tokenizer_santali.sequences_to_texts(translated_sequence)
    return translated_text[0] if translated_text else "Translation not available"

# Initialize Flask application
app = Flask(__name__)

# Setup logging
logging.basicConfig(filename='logs/access.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')

# Define route for translation
@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    input_text = data['text']
    translated_text = translate(input_text)
    app.logger.info(f'Translated "{input_text}" to "{translated_text}"')
    return jsonify({'translated_text': translated_text})

# Error handling route
@app.errorhandler(404)
def not_found_error(error):
    app.logger.error(f'404 error: {error}')
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'500 error: {error}')
    return jsonify({'error': 'Internal server error'}), 500
