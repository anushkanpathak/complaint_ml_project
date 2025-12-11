# src/test_predict.py
import joblib
MODEL = '../models/complaint_pipeline.joblib'
clf = joblib.load(MODEL)

examples = [
    "WiFi not connecting since morning",
    "Assignment marks not uploaded",
    "Food stale in mess today",
    "Room light not working"
]
for ex in examples:
    print(ex, "->", clf.predict([ex])[0], clf.predict_proba([ex]) if hasattr(clf, "predict_proba") else "")
