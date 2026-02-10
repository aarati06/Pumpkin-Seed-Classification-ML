from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['area']),
        float(request.form['perimeter']),
        float(request.form['major_axis_length']),
        float(request.form['minor_axis_length']),
        float(request.form['convex_area']),
        float(request.form['equiv_diameter']),
        float(request.form['eccentricity']),
        float(request.form['solidity']),
        float(request.form['extent']),
        float(request.form['roundness']),
        float(request.form['aspect_ration']),
        float(request.form['compactness'])
    ]

    final_features = np.array([features])
    prediction = model.predict(final_features)[0]

    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
