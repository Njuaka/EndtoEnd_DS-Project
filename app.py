
import os
import json
import joblib
from src.constants import MODELS_PATH
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

## Load the model
model = joblib.load(open(os.path.join(MODELS_PATH, 'model.joblib'), 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    #data=request.json['data']
    data = request.get_json(force=True)['data']
    print(data)
    new_data = np.array(list(data.values())).reshape(1,-1)
    output=model.predict(new_data)
    print(output[0])
    return jsonify({'prediction': output[0]})
#jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    #data = [float(x) for x in request.form.values()]
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html", prediction_text= f"The crop yield prediction is {output}")



if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))