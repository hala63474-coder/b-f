from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model("deepfake_model.keras")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File is too large"}), 413


@app.route("/")
def home():
    return "Deepfake Video API is working!"


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]

    if not allowed_file(video.filename):
        return jsonify({"error": "Invalid file type. Allowed: mp4, avi, mov, mkv"}), 400

    filename = secure_filename(video.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    video.save(path)

    cap = cv2.VideoCapture(path)
    preds = []
    frame_count = 0
    FRAME_STEP = 5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_STEP == 0:
                try:
                    frame = cv2.resize(frame, (224, 224))
                    frame = frame.astype(np.float32) / 255.0
                    frame = np.expand_dims(frame, axis=0)

                    pred = model.predict(frame, verbose=0)[0][0]
                    preds.append(float(pred))

                except Exception:
                    continue

            frame_count += 1

    finally:
        cap.release()
        if os.path.exists(path):
            os.remove(path)

    if len(preds) < 3:
        return jsonify({"error": "Not enough frames processed"}), 400

    avg = np.mean(preds)

    fake_votes = sum(p < 0.5 for p in preds)
    real_votes = sum(p >= 0.5 for p in preds)

    result = "Fake" if fake_votes > real_votes else "Real"

    return jsonify({
        "result": result,
        "confidence": float(avg),
        "frames_used": len(preds),
        "fake_votes": fake_votes,
        "real_votes": real_votes
    })


if __name__ == "__main__":
    app.run(debug=True, threaded=True)