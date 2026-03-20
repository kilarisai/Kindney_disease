**Kidney Disease Prediction Web App**

A web application to predict Chronic Kidney Disease (CKD) using a machine learning model (TabNet). Users can input patient features and get an instant prediction.

Hosted Link: https://kindney-disease.onrender.com/

**Features**

Predicts CKD based on user input

Handles missing values automatically

Uses a trained TabNetClassifier model

Scales input features before prediction

Simple web interface using Flask

**Technologies Used**

Python 3.x

Flask – web framework

PyTorch & PyTorch-TabNet – machine learning model

Scikit-learn & Joblib – for preprocessing and loading scaler

HTML/CSS – front-end interface

Render – hosting platform

Installation (Local)

Clone the repository:

git clone <your-repo-url>
cd project

Install dependencies:

pip install -r requirements.txt

Run the app:

python app.py

Open your browser at http://127.0.0.1:5000/

**Usage**

Enter patient feature values in the web form

Click Predict

Result will show “CKD Detected” or “No CKD”

**Deployment**

The app is hosted publicly on Render:
https://kindney-disease.onrender.com/

**Start command on Render:**

gunicorn app:app

**Note**: Flask listens on the PORT environment variable assigned by Render.

**License**

This project is for educational purposes.
