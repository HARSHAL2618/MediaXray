from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64
import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import librosa
from tensorflow.keras.applications import MobileNetV2
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization

# app = Flask(__name__, template_folder="UI/templates", static_folder="UI/static")
app = Flask(
    __name__,
    template_folder=r"UI/templates",
    static_folder=r"UI/static"
)

# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGE_WEIGHTS     = "Image_Detector/models/model_weights.weights.h5"
MULTICLASS_WEIGHTS= "Image_Detector/models/multiclass_weights.weights.h5"
AUDIO_WEIGHTS     = "Audio_Detector/models/audio_weights.weights.h5"
VIDEO_WEIGHTS     = "Video_Detector/models/video_weights.weights.h5"
TEXT_MODEL_PATH   = "Text_Detector/models/text_model"
UPLOAD_FOLDER     = "UI/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Class labels ───────────────────────────────────────────────────────────────
MULTICLASS_LABELS = {0:"DALLE3", 1:"MIDJOURNEY", 2:"SD21", 3:"SD3", 4:"SDXL"}
IMAGE_THRESHOLD   = 0.25

# ── Build MobileNetV2 base ─────────────────────────────────────────────────────
def build_binary_model():
    base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
    x    = GlobalAveragePooling2D()(base.output)
    x    = BatchNormalization()(x)
    x    = Dense(256, activation="relu")(x)
    x    = Dropout(0.4)(x)
    x    = Dense(128, activation="relu")(x)
    x    = Dropout(0.3)(x)
    out  = Dense(1, activation="sigmoid")(x)
    return Model(inputs=base.input, outputs=out)

def build_multiclass_model():
    base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
    x    = GlobalAveragePooling2D()(base.output)
    x    = BatchNormalization()(x)
    x    = Dense(512, activation="relu")(x)
    x    = Dropout(0.4)(x)
    x    = Dense(256, activation="relu")(x)
    x    = Dropout(0.3)(x)
    out  = Dense(6, activation="softmax")(x)
    return Model(inputs=base.input, outputs=out)

# ── Load all models ────────────────────────────────────────────────────────────
print("Loading models...")

image_model = build_binary_model()
image_model.load_weights(IMAGE_WEIGHTS)
print("✅ Image binary model loaded!")

multiclass_model = build_multiclass_model()
multiclass_model.load_weights(MULTICLASS_WEIGHTS)
print("✅ Multiclass model loaded!")

audio_model = build_binary_model()
audio_model.load_weights(AUDIO_WEIGHTS)
print("✅ Audio model loaded!")

video_model = build_binary_model()
video_model.load_weights(VIDEO_WEIGHTS)
print("✅ Video model loaded!")

# device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# text_model = BertForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
# text_model = text_model.to(device)
# text_model.eval()
# tokenizer  = BertTokenizer.from_pretrained(TEXT_MODEL_PATH)
# print("✅ Text model loaded!")

device = torch.device("cpu")  # force CPU on Windows
try:
    text_model = BertForSequenceClassification.from_pretrained(
        TEXT_MODEL_PATH,
        local_files_only = True    # ← load from local folder
    )
    text_model = text_model.to(device)
    text_model.eval()
    tokenizer = BertTokenizer.from_pretrained(
        TEXT_MODEL_PATH,
        local_files_only = True    # ← load from local folder
    )
    print("✅ Text model loaded!")
except Exception as e:
    print(f"❌ Text model error: {e}")
    text_model = None
    tokenizer  = None

print("🎉 All models ready!")

# ── Helper: Grad-CAM ───────────────────────────────────────────────────────────
def get_gradcam(model, img_array, last_conv_layer="Conv_1"):
    try:
        grad_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = [model.get_layer(last_conv_layer).output, model.output]
        )
        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            tape.watch(inputs)
            conv_outputs, predictions = grad_model(inputs, training=False)
            loss = predictions[:, 0]
        grads   = tf.abs(tape.gradient(loss, conv_outputs))
        pooled  = tf.reduce_mean(grads, axis=(0,1,2))
        heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except:
        return np.zeros((7, 7))

def overlay_gradcam(img_array, heatmap):
    img         = np.uint8(img_array[0] * 255)
    heatmap_r   = cv2.resize(heatmap, (224, 224))
    heatmap_c   = cv2.applyColorMap(np.uint8(255 * heatmap_r), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap_c, 0.4, 0)
    _, buffer    = cv2.imencode('.jpg', cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

# ── Image prediction ───────────────────────────────────────────────────────────
def predict_image(img_path):
    img     = tf.keras.utils.load_img(img_path, target_size=(224,224))
    arr     = tf.keras.utils.img_to_array(img) / 255.0
    arr_exp = np.expand_dims(arr, axis=0)

    # Binary prediction
    prob       = float(image_model.predict(arr_exp, verbose=0)[0][0])
    label      = "FAKE" if prob <= IMAGE_THRESHOLD else "REAL"
    confidence = (1 - prob) if label == "FAKE" else prob

    # Multiclass prediction
    mc_probs  = multiclass_model.predict(arr_exp, verbose=0)[0]
    mc_idx    = int(np.argmax(mc_probs))
    mc_label  = MULTICLASS_LABELS[mc_idx]
    mc_conf   = float(mc_probs[mc_idx])

    # Grad-CAM
    heatmap  = get_gradcam(image_model, arr_exp)
    gradcam  = overlay_gradcam(arr_exp, heatmap)
    _, buf   = cv2.imencode('.jpg', cv2.cvtColor(np.uint8(arr*255), cv2.COLOR_RGB2BGR))
    original = base64.b64encode(buf).decode('utf-8')

    # Tool breakdown
    tool_breakdown = {
        MULTICLASS_LABELS[i]: round(float(mc_probs[i])*100, 2)
        for i in range(5)
    }

    return {
        "label"          : label,
        "confidence"     : round(confidence * 100, 2),
        "raw_prob"       : round(prob, 4),
        "tool"           : mc_label if label == "FAKE" else "N/A",
        "tool_confidence": round(mc_conf * 100, 2),
        "tool_breakdown" : tool_breakdown,
        "gradcam"        : gradcam,
        "original"       : original
    }


#updated image function
# def predict_image(img_path):
#     img     = tf.keras.utils.load_img(img_path, target_size=(224,224))
#     arr     = tf.keras.utils.img_to_array(img) / 255.0
#     arr_exp = np.expand_dims(arr, axis=0)

#     # Binary prediction
#     prob       = float(image_model.predict(arr_exp, verbose=0)[0][0])
#     label      = "FAKE" if prob <= IMAGE_THRESHOLD else "REAL"
#     confidence = (1 - prob) if label == "FAKE" else prob

#     # Multiclass prediction
#     mc_probs  = multiclass_model.predict(arr_exp, verbose=0)[0]
#     mc_idx    = int(np.argmax(mc_probs))
#     mc_label  = MULTICLASS_LABELS[mc_idx]
#     mc_conf   = float(mc_probs[mc_idx])

#     # Grad-CAM
#     heatmap  = get_gradcam(image_model, arr_exp)
#     gradcam  = overlay_gradcam(arr_exp, heatmap)
#     _, buf   = cv2.imencode('.jpg', cv2.cvtColor(np.uint8(arr*255), cv2.COLOR_RGB2BGR))
#     original = base64.b64encode(buf).decode('utf-8')

#     # Tool breakdown
#     tool_breakdown = {
#         MULTICLASS_LABELS[i]: round(float(mc_probs[i])*100, 2)
#         for i in range(6)
#     }

#     # Determine tool (only if FAKE and multiclass predicts an AI tool, not REAL)
#     if label == "FAKE" and mc_label != "REAL":
#         tool = mc_label
#         tool_conf = mc_conf
#     else:
#         tool = "N/A"
#         tool_conf = 0.0

#     return {
#         "label"          : label,
#         "confidence"     : round(confidence * 100, 2),
#         "raw_prob"       : round(prob, 4),
#         "tool"           : tool,
#         "tool_confidence": round(tool_conf * 100, 2),
#         "tool_breakdown" : tool_breakdown,
#         "gradcam"        : gradcam,
#         "original"       : original
#     }


# ── Text prediction ────────────────────────────────────────────────────────────
def predict_text(text):
    encoding = tokenizer(
        text,
        truncation     = True,
        padding        = True,
        max_length     = 128,
        return_tensors = "pt"
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = text_model(**encoding)
        probs   = torch.softmax(outputs.logits, dim=1)[0]
        pred    = torch.argmax(probs).item()

    label      = "AI Generated" if pred == 1 else "Human Written"
    confidence = float(probs[pred]) * 100

    return {
        "label"      : label,
        "confidence" : round(confidence, 2),
        "human_prob" : round(float(probs[0]) * 100, 2),
        "ai_prob"    : round(float(probs[1]) * 100, 2)
    }

# ── Audio prediction ───────────────────────────────────────────────────────────
def predict_audio(audio_path):
    SR       = 16000
    DURATION = 3

    audio, sr = librosa.load(audio_path, sr=SR, duration=DURATION)
    target    = SR * DURATION
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))

    mel    = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_r  = tf.image.resize(mel_db[:,:,np.newaxis], [128,128]).numpy()
    mel_r  = np.repeat(mel_r, 3, axis=-1)
    arr    = np.expand_dims(mel_r, axis=0)

    prob       = float(audio_model.predict(arr, verbose=0)[0][0])
    label      = "FAKE" if prob <= 0.5 else "REAL"
    confidence = (1 - prob) if label == "FAKE" else prob

    return {
        "label"      : label,
        "confidence" : round(confidence * 100, 2),
        "raw_prob"   : round(prob, 4)
    }

# ── Video prediction ───────────────────────────────────────────────────────────
def predict_video(video_path):
    cap     = cv2.VideoCapture(video_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, 10, dtype=int)
    preds   = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame   = cv2.resize(frame, (224, 224))
            frame   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            arr     = np.expand_dims(frame / 255.0, axis=0)
            prob    = float(video_model.predict(arr, verbose=0)[0][0])
            preds.append(prob)

    cap.release()

    avg_prob   = np.mean(preds)
    label      = "FAKE" if avg_prob <= 0.5 else "REAL"
    confidence = (1 - avg_prob) if label == "FAKE" else avg_prob

    return {
        "label"        : label,
        "confidence"   : round(float(confidence) * 100, 2),
        "raw_prob"     : round(float(avg_prob), 4),
        "frames_analyzed": len(preds),
        "fake_frames"  : sum(1 for p in preds if p <= 0.5),
        "real_frames"  : sum(1 for p in preds if p > 0.5)
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict/image", methods=["POST"])
def predict_image_route():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    # path = os.path.join(UPLOAD_FOLDER, file.filename)
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    try:
        return jsonify(predict_image(path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/text", methods=["POST"])
def predict_text_route():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text"}), 400
    try:
        return jsonify(predict_text(data["text"]))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/audio", methods=["POST"])
def predict_audio_route():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    # path = os.path.join(UPLOAD_FOLDER, file.filename)
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    try:
        return jsonify(predict_audio(path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/video", methods=["POST"])
def predict_video_route():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    # path = os.path.join(UPLOAD_FOLDER, file.filename)
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    try:
        return jsonify(predict_video(path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)