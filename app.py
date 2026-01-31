from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    price = None
    if request.method == "POST":
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])

        features = np.array([[area, bedrooms, bathrooms]])
        price = model.predict(features)[0]

    return render_template("index.html", price=price)

if __name__ == "__main__":
    app.run(debug=True)
