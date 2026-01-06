from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from models.llm import generate_advice

app = Flask(__name__)

with open("models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/le_dict.pkl", "rb") as f:
    le_dict = pickle.load(f)

cat_map = {
    "diet": "Diet",
    "shower": "How Often Shower",
    "heating": "Heating Energy Source",
    "transport": "Transport",
    "vehicle": "Vehicle Type",
    "social": "Social Activity",
    "waste_size": "Waste Bag Size",
    "energy": "Energy efficiency"
}

feature_order = [
    "Diet", "How Often Shower", "Heating Energy Source", "Transport", "Vehicle Type",
    "Social Activity", "Monthly Grocery Bill", "Frequency of Traveling by Air", "Vehicle Monthly Distance Km",
    "Waste Bag Size", "Waste Bag Weekly Count", "How Long TV PC Daily Hour", 
    "How Many New Clothes Monthly", "How Long Internet Daily Hour", 
    "Energy efficiency", "Recycling_Count", "Cooking_Count"
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = []

    for form_key, csv_col in cat_map.items():
        val = data[form_key]
        le = le_dict[csv_col]
        if val not in le.classes_:
            val = "Unknown"
        features.append(int(le.transform([val])[0]))

    features += [
        data["distance"],
        data["air"],
        data["grocery"],
        data["waste_count"],
        data["clothes"],
        data["internet"],
        data["tv_pc"],
        len(data["recycling"]),
        len(data["cooking"])
    ]

    features_array = np.array(features).reshape(1, -1)
    prediction = rf_model.predict(features_array)[0]

    contributions = rf_model.feature_importances_ * features_array[0]

    top_indices = np.argsort(contributions)[-3:][::-1]
    top_features = [(feature_order[i], contributions[i]) for i in top_indices]

    advice = generate_advice(data, top_features)

    return jsonify({
        "prediction": round(prediction, 2),
        "feature_labels": feature_order,
        "feature_values": contributions.round(2).tolist(),
        "advice": advice
    })

if __name__ == "__main__":
    app.run(debug=True)