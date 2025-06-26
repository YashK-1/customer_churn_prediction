from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained pipeline
MODEL_PATH = os.path.join("model", "final_pipeline.pkl")
with open(MODEL_PATH, "rb") as f:
    model_pipeline = pickle.load(f)

# Home route: renders the HTML form
@app.route("/", methods=["GET"])
def index():
    return render_template("form.html")

# Prediction route: triggered on form submission
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Collect input data from form
        input_data = {
            'gender': request.form['gender'],
            'senior_citizen': int(request.form['senior_citizen']),
            'partner': request.form['partner'],
            'dependents': request.form['dependents'],
            'tenure': float(request.form['tenure']),
            'phone_service': request.form['phone_service'],
            'multiple_lines': request.form['multiple_lines'],
            'internet_service': request.form['internet_service'],
            'online_security': request.form['online_security'],
            'online_backup': request.form['online_backup'],
            'device_protection': request.form['device_protection'],
            'tech_support': request.form['tech_support'],
            'streaming_tv': request.form['streaming_tv'],
            'streaming_movies': request.form['streaming_movies'],
            'contract': request.form['contract'],
            'paperless_billing': request.form['paperless_billing'],
            'payment_method': request.form['payment_method'],
            'monthly_charges': float(request.form['monthly_charges']),
            'total_charges': float(request.form['total_charges'])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model_pipeline.predict(input_df)[0]
        result = "Yes" if prediction == 1 else "No"

        return render_template("form.html", prediction_result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
