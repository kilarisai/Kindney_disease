from flask import Flask, render_template, request
import os
import numpy as np
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier

app = Flask(__name__)

# Load model and scaler
model = TabNetClassifier()
model.load_model("tabnet_kidney_model.zip")

scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from form
    values = []

    for key in request.form:
        val = request.form[key]

        # If empty → replace with 0
        if val == "" or val=="?":
            val = 0

        values.append(float(val))

    data = np.array([values])

    # If features < 25 add zeros
    if data.shape[1] < 25:
        missing = 25 - data.shape[1]
        zeros = np.zeros((1, missing))
        data = np.concatenate((data, zeros), axis=1)

    # Scale data
    data = scaler.transform(data)

    # Predict
    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "CKD Detected"
    else:
        result = "No CKD"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)