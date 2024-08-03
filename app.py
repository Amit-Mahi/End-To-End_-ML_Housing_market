from flask import Flask, render_template, url_for, request, jsonify, app
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)
## load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)
