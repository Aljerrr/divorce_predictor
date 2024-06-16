from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model/divorce_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    output = 'Unlikely' if prediction[0] == 1 else 'Possible'
    
    return render_template('index.html', prediction_text=f'Divorce is {output}')

if __name__ == "__main__":
    app.run(debug=True)
