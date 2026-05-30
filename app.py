import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense as TfDense
from tensorflow.keras.layers import Embedding as TfEmbedding
from tensorflow.keras.layers import InputLayer as TfInputLayer
from tensorflow.keras.layers import LSTM as TfLSTM

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "Fake_job_detection.h5")

tokenizer = None
model = None

class _KerasTokenizerUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "keras.src.legacy.preprocessing.text":
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                module = "keras.preprocessing.text"
        return super().find_class(module, name)


class _PatchedInputLayer(TfInputLayer):
    def __init__(self, *args, **kwargs):
        if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
            kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
        kwargs.pop("optional", None)
        super().__init__(*args, **kwargs)


class _PatchedEmbedding(TfEmbedding):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


class _PatchedDense(TfDense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


class _PatchedLSTM(TfLSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

print("Loading tokenizer...")
try:
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = _KerasTokenizerUnpickler(f).load()
    print("Tokenizer loaded!")
except FileNotFoundError:
    print("ERROR: tokenizer.pkl not found. Please make sure it is in the same folder.")

print("Loading Keras model...")
try:
    model = load_model(
        MODEL_PATH,
        custom_objects={
            "InputLayer": _PatchedInputLayer,
            "Embedding": _PatchedEmbedding,
            "Dense": _PatchedDense,
            "LSTM": _PatchedLSTM,
        },
    )
    print("Keras model loaded!")
except OSError:
    print("ERROR: 'Fake_job_detection.h5' not found. Check the filename.")

MAX_SEQUENCE_LENGTH = 200

def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, dtype="float32")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("home"))
    if tokenizer is None or model is None:
        return render_template(
            "index.html",
            prediction="Model files are missing. Ensure tokenizer.pkl and Fake_job_detection.h5 are present.",
            probability=0,
            fraud_chance=0,
            reason="Model assets are unavailable, so risk scoring cannot run.",
            result="Unavailable",
        )

    combined_text = request.form.get("combined_text")
    print("[predict] Received input length:", len(combined_text) if combined_text else 0)

    if not combined_text:
        return render_template(
            "index.html",
            prediction="Please enter the job description.",
            probability=0,
            fraud_chance=0,
            reason="No job description provided. Submit text to generate a risk score.",
            result="Unavailable",
        )

    input_data = preprocess_text(combined_text)

    try:
        prediction = model.predict(input_data)[0][0]
        probability = float(prediction)
        print("[predict] model prediction:", probability)
    except Exception as exc:
        print("[predict] prediction error:", exc)
        return render_template(
            "index.html",
            prediction="Prediction failed due to a server error.",
            probability=0,
            fraud_chance=0,
            reason="The server failed to score this input. Please try again shortly.",
            result="Unavailable",
        )
    threshold = 0.7

    # Detailed insights based on EDA and model training
    if probability > threshold:
        result = "Fraudulent"
        reason = (
            "The model detected patterns commonly associated with scam or fraudulent job postings. "
            "Common fraud indicators include: vague or missing job requirements, marketing or unspecified job functions, "
            "lack of company profile or logo, requests for personal information, and offers that seem too good to be true. "
            "Jobs with no specified experience or education, or those in marketing roles, are statistically more likely to be fraudulent."
        )
    else:
        result = "Legitimate"
        reason = (
            "The job description does not match known fraud patterns. "
            "Legitimate jobs often provide clear requirements, company details, and realistic offers. "
            "Full-time roles with specified experience, education, and a company profile are less likely to be scams."
        )

    percent = round(probability * 100, 1)
    fraud_chance = percent if result == "Fraudulent" else 100 - percent
    fraud_chance = round(fraud_chance, 1)

    return render_template(
        "index.html",
        prediction=f"The job post is {result}",
        probability=percent,
        fraud_chance=fraud_chance,
        reason=reason,
        result=result
    )

if __name__ == "__main__":
    app.run(debug=True)