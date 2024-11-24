# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
from nltk.stem import PorterStemmer

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

try:
    with open('vectorizer.pkl', 'rb') as file:
        # vector = pickle.load(file)
        vector = pickle.load(file, encoding='latin1')  # or 'utf-8'
        print("Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading vectorizer: {e}")

app = Flask(__name__, static_folder='static')

# Preprocessing tools
pt = PorterStemmer()  # Initialize PorterStemmer globally
r = re.compile('[^a-zA-Z0-9\s]')  # Compile regex for preprocessing


# Preprocessing function
def preprocessing(x):
    l = []
    text = r.sub('', x.lower())  # Remove non-alphanumeric characters and convert to lowercase
    for i in text.split():
        l.append(pt.stem(i.lower()))  # Apply stemming
    return " ".join(l)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():

    try:
        # Extract the input statement from the form
        statement = request.form.get('Statement')

        #Preprocess
        process = preprocessing(statement)

        # Transform the input using the vectorizer
        transformed_input = vector.transform([process])

        # Make the prediction
        prediction = model.predict(transformed_input)[0]

        # Send the result back to the frontend
        return render_template('index.html', prediction_text=f'Prediction: {prediction}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)