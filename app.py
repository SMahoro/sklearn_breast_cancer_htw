from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("svm_model.pkl")
try:
    feature_names = joblib.load("feature_names.pkl")
except Exception:
    feature_names = None

CLASS_LABELS = {
    0: "Malignant",
    1: "Benign",
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    #liest daten, die von json gesendet werden
    if not data or "features" not in data:
        return jsonify({"error": "Bitte JSON mit Key 'features' senden"}), 400

    features = data["features"]
    if len(features) != 30:
        return jsonify({"error": "Es werden genau 30 Features erwartet."}), 400

    X = np.array(features).reshape(1, -1)
    #table
    pred = int(model.predict(X)[0])  # gutartig bösartig
    proba = model.predict_proba(X)[0].tolist()

    class_label = CLASS_LABELS.get(pred, f"Unbekannte Klasse {pred}")

    response = {
        "prediction": pred,
        "class_label": class_label,
        "probabilities": proba,
    }

    if feature_names is not None:
        response["feature_names"] = list(feature_names)

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)
