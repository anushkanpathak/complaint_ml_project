
# app.py -- supports both form and JSON clients (safe)
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# load artifacts
tfidf = joblib.load("tfidf_vectorizer.joblib")
clf   = joblib.load("complaint_model_nb.joblib")

# recommendations (same as you had)
RECOMMENDATIONS = {
    "network":[ "IT Helpdesk Support Session", "WiFi Troubleshooting Workshop", "Network Complaint Registration Desk (10:00-17:00)" ],
    "academic":[ "Academic Support / Mentor Meeting", "Assignment Clarification Session", "Exam Preparation Workshop" ],
    "mess":[ "Mess Feedback Meeting with Mess Committee", "Healthy Cooking & Food Safety Seminar", "Student Food Quality Survey & Action" ],
    "library":[ "Library Orientation & Resource Guide", "Library Digital Access Training", "Library Helpdesk Session" ],
    "maintenance":[ "Hostel / Campus Maintenance Helpdesk", "Sanitation & Hygiene Inspection Request", "Electrician / Plumber Visit Scheduling" ],
    "hostel":[ "Hostel Safety & Security Awareness Program", "Visitor Pass & Gate Security Meeting", "Hostel Grievance Redressal Session" ],
    "unknown":[ "Student Helpdesk / General Grievance Cell", "Contact Student Affairs Office" ]
}

# helper: get text from either form or JSON
def get_text_from_request(req):
    # form first (HTML submit)
    if req.form:
        text = req.form.get("complaint") or req.form.get("text")
        if text:
            return text.strip()
    # then json body (fetch)
    json_body = req.get_json(silent=True)
    if isinstance(json_body, dict):
        # accept either "complaint" or "text" key
        text = json_body.get("complaint") or json_body.get("text")
        if text:
            return str(text).strip()
    return None

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = get_text_from_request(request)

    if not text:
        # If caller expects JSON, return JSON error; otherwise render page with message.
        if request.accept_mimetypes.accept_json and not request.form:
            return jsonify({"error":"No complaint text found. Send JSON {'text':'...'} or form field named 'complaint'"}), 400
        else:
            return render_template("index.html", error="No complaint text found. Use the textarea and submit."), 400

    # vectorize + predict
    X = tfidf.transform([text])
    probs = clf.predict_proba(X)[0]
    classes = clf.classes_

    top_idx = int(np.argmax(probs))
    max_prob = float(probs[top_idx])
    predicted_label = str(classes[top_idx])

    # top-2
    top_k_idx = probs.argsort()[-2:][::-1]
    top_k = [{"label": str(classes[i]), "prob": float(probs[i])} for i in top_k_idx]

    recs = RECOMMENDATIONS.get(predicted_label, RECOMMENDATIONS["unknown"])

    # If request was JSON (your fetch), return JSON.
    if request.accept_mimetypes.accept_json and not request.form:
        return jsonify({
            "input_text": text,
            "prediction": predicted_label,
            "max_probability": max_prob,
            "top_labels": top_k,
            "recommendations": recs
        })

    # else render template (for normal form submit)
    return render_template(
        "index.html",
        input_text=text,
        prediction=predicted_label,
        confidence=round(max_prob, 3),
        top_labels=[(t["label"], round(t["prob"],3)) for t in top_k],
        rec=recs
    )

if __name__ == "__main__":
    app.run(debug=True)
