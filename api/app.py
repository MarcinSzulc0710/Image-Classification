
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from celery import Celery
import os
import uuid
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

models = {}
model_status = {}

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    model_id = str(uuid.uuid4())
    model_status[model_id] = {"status": "training", "error": None}
    train_model.delay(model_id)
    return jsonify({"model_id": model_id})

@celery.task
def train_model(model_id):
    try:
        model = ResNet50(weights='imagenet')
        models[model_id] = model
        model_status[model_id]["status"] = "trained"
    except Exception as e:
        model_status[model_id]["status"] = "error"
        model_status[model_id]["error"] = str(e)

@app.route("/models", methods=["GET"])
def list_models():
    return jsonify(list(models.keys()))

@app.route("/status/<model_id>", methods=["GET"])
def get_status(model_id):
    return jsonify(model_status.get(model_id, {"status": "not_found"}))

@app.route("/predict/<model_id>", methods=["POST"])
def predict(model_id):
    if model_id not in models:
        return jsonify({"error": "model_not_found"}), 404
    if 'image' not in request.files:
        return jsonify({"error": "no_image_provided"}), 400
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    img = load_img(filepath, target_size=(224, 224))
    x = img_to_array(img)
    x = preprocess_input(x.reshape((1,) + x.shape))
    preds = models[model_id].predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    result = [{"label": label, "description": desc, "probability": float(prob)} for (label, desc, prob) in decoded]
    return jsonify(result)

@app.route("/test/<model_id>", methods=["GET"])
def test(model_id):
    if model_id not in models:
        return jsonify({"error": "model_not_found"}), 404
    return jsonify({"test_accuracy": "placeholder"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
