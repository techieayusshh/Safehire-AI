# SafeHire AI — Fraudulent Job Detection with Real-Time Risk Scoring

**Engineering-grade fraud detection pipeline for job postings, combining a production Flask service with a TensorFlow LSTM model and a deterministic inference stack.**

![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![ML](https://img.shields.io/badge/ML-TensorFlow%20%2F%20Keras-orange)
![WSGI](https://img.shields.io/badge/WSGI-Gunicorn-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

Live demo (Render): **https://safehire-ai.onrender.com**

---

## ✅ Key Architectural Features

- **Model portability with backward-compatible Keras deserialization** via a custom `_KerasTokenizerUnpickler` that resolves legacy tokenizer modules and patched layer wrappers for `InputLayer`, `Embedding`, `Dense`, and `LSTM`.
- **Deterministic inference pipeline** with fixed `MAX_SEQUENCE_LENGTH=200`, explicit preprocessing (`tokenizer.texts_to_sequences` + `pad_sequences`), and a calibrated fraud threshold of `0.7`.
- **Deployment-safe asset loading** using absolute paths derived from `BASE_DIR` to locate `tokenizer.pkl` and `Fake_job_detection.h5` reliably across environments.
- **Predictive UX loop** with server-rendered Jinja templates feeding a JS-driven risk gauge (dynamic arc + needle rotation) for immediate visual feedback.
- **Failure-aware runtime** with graceful messaging when model assets are missing, preventing 500s at startup.

---

## ⚡ Engineering & Performance Highlights

- **Startup-time model loading** keeps per-request latency low by reusing a single in-memory model instance and tokenizer.
- **TensorFlow runtime noise suppression** using `TF_CPP_MIN_LOG_LEVEL=3`, and OneDNN opt-out (`TF_ENABLE_ONEDNN_OPTS=0`) to stabilize inference on CPU.
- **Strong separation of concerns**: UI presentation in `templates/` and `static/`, inference logic in `app.py`, and training artifacts isolated as `.h5` and `.pkl`.
- **Robust preprocessing contract** ensures the model always receives consistent input shapes and dtypes, minimizing runtime shape errors.

---

## 🧰 Tech Stack & System Design

| Category | Technologies |
| --- | --- |
| Languages | Python 3.11, JavaScript, HTML5, CSS3 |
| Backend | Flask, Gunicorn |
| ML/DL | TensorFlow 2.16.1, Keras |
| Data | NumPy, pandas, scikit-learn, Pickle |
| Deployment | Render, runtime.txt |

**System flow:** User submits a job description → Flask routes `/predict` → text is tokenized and padded to length 200 → LSTM model outputs probability → server computes fraud label and explanation → Jinja template renders prediction and JS animates the risk gauge.

---

## 🧭 Directory Topology

```
fakejob/
├── app.py                # Flask app + inference pipeline + model loading
├── Fake_job_detection.h5 # Trained LSTM model weights
├── tokenizer.pkl         # Serialized tokenizer for consistent preprocessing
├── requirements.txt      # Runtime dependencies
├── runtime.txt           # Python version pin for Render
├── static/
│   └── style.css         # Dashboard UI styling
├── templates/
│   └── index.html        # Jinja template + risk gauge UI
├── EDA_job_data.ipynb    # Exploratory analysis notebook
├── model_training.ipynb  # Model training workflow
└── fake_job_postings.csv # Dataset used for training
```

---

## 🖥️ Installation & Local Deployment

**Prerequisites**
- Python 3.11.x
- Pip 23+

**Steps**
```bash
git clone https://github.com/techieayusshh/Safehire-AI.git
cd Safehire-AI

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Open: http://127.0.0.1:5000/

**Required assets:** `Fake_job_detection.h5` and `tokenizer.pkl` must be in the project root.

---

## 🚀 Production Notes

- **Gunicorn entrypoint:** `gunicorn app:app`
- **Python runtime pin:** `runtime.txt` (python-3.11.8)
- **Model files included** for deterministic inference in production environments.

---

## 🧪 Testing & Validation

No automated tests are currently defined. Model quality is validated via notebook workflows in `model_training.ipynb` and EDA insights in `EDA_job_data.ipynb`. Recommended next step is to add unit tests for preprocessing and prediction endpoints.

---

## ⚠️ Disclaimer

This tool provides probabilistic predictions based on patterns in historical data. Always use independent judgment before sharing personal information or accepting an offer.

---

## 👤 Author

**Ayush Raj**

AI & Data Science Enthusiast

Connect: https://www.linkedin.com/

