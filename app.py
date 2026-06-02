"""
app.py — Flask server for Brain Tumor Detection
"""

import os
import uuid
from datetime import datetime
from collections import deque

from flask import Flask, render_template, request, jsonify

from model import predict, generate_gradcam

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}

# Server-side history (in-memory, avoids cookie size limit)
# For a single-user local app this is perfectly fine.
_history = deque(maxlen=20)


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    # --- validate -----------------------------------------------------------
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    # --- read bytes ---------------------------------------------------------
    image_bytes = file.read()

    try:
        # --- inference ------------------------------------------------------
        result = predict(image_bytes)

        # --- Grad-CAM -------------------------------------------------------
        gradcam_b64 = generate_gradcam(image_bytes)
        result["gradcam"] = gradcam_b64

        # --- add to server-side history -------------------------------------
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": file.filename,
            "class": result["class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "gradcam": gradcam_b64,
        }
        _history.appendleft(entry)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/history")
def history():
    """Return prediction history."""
    return jsonify(list(_history))


@app.route("/history/clear", methods=["POST"])
def clear_history():
    _history.clear()
    return jsonify({"status": "cleared"})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Preload model at startup
    print("[*] Loading model ...")
    from model import get_model
    get_model()
    print("[OK] Model loaded. Starting server ...")

    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
