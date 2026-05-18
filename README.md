# 🛡️ SafeHire AI — Fake Job & Internship Detection
> **Protect yourself from fraudulent job postings with AI-powered detection.**
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-green)
![TensorFlow](https://img.shields.io/badge/ML-TensorFlow%20%2F%20Keras-orange)
![Gunicorn](https://img.shields.io/badge/WSGI-Gunicorn-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)
![Status](https://img.shields.io/badge/Status-Production-success)
## 🌐 Live Demo
Try the app live: [SafeHire AI on Render](https://safehire-ai.onrender.com)
## 🛠️ Tech Stack
- **Frontend:** HTML5, CSS3, JavaScript
- **Backend:** Flask (Python)
- **Machine Learning:** TensorFlow, Keras
- **Data Processing:** NumPy, scikit-learn, pandas, Pickle
- **Deployment:** Render, Gunicorn
## 📁 Project Structure
```
fakejob/
├── app.py                # Main Flask app
├── EDA_job_data.ipynb    # Data analysis notebook
├── Fake_job_detection.h5 # Trained LSTM model
├── tokenizer.pkl         # Tokenizer for preprocessing
├── requirements.txt      # Python dependencies
├── fake_job_postings.csv # Training data
├── Procfile              # Gunicorn/Render config
├── runtime.txt           # Python version pin for Render
├── static/
│   └── style.css         # App styling
└── templates/
	└── index.html        # Main UI
## 🧠 How It Works
1. **Input:** User submits a job description (title, company, description, requirements).
2. **Preprocessing:**
	- Text is cleaned and tokenized using a pre-trained tokenizer.
	- Sequence is padded to length 200 for model input.
3. **Prediction:**
	- The LSTM model outputs a probability score (0–1).
	- Score > 0.7 = Fraudulent; ≤ 0.7 = Legitimate.
4. **Output:**
	- Shows “Fraud Probability” as a percentage with a color-coded animated progress bar and risk level.
	- Provides a short explanation of risk factors (if enabled).
## 🖥️ Local Setup
1. **Clone the repository:**
	```bash
	git clone https://github.com/yourusername/safehire-ai.git
	cd safehire-ai
	```
2. **Create a virtual environment (recommended):**
	```bash
	python -m venv .venv
	# Windows:
	.venv\Scripts\activate
	# Mac/Linux:
	source .venv/bin/activate
3. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```
4. **Download or place the model files:**
	- `Fake_job_detection.h5` and `tokenizer.pkl` must be present in the project root. (If not public, contact the author.)
5. **Run the app:**
	```bash
	python app.py
	```
6. **Open in browser:**
	[http://127.0.0.1:5000/](http://127.0.0.1:5000/)
## 📸 Screenshots
<!-- Add your own screenshots here -->
<img width="900" alt="SafeHire AI Screenshot" src="https://github.com/user-attachments/assets/d01ec1b8-b91e-447b-8b7f-33d51e06d232" />
## 🏗️ Deployment (Render)
1. Push your code to GitHub.
2. Connect your repo to [Render.com](https://render.com/).
3. Set build & start commands:
	- **Build Command:** `pip install -r requirements.txt`
	- **Start Command:** `gunicorn app:app`
4. Add a `runtime.txt` with `python-3.11.8` to pin Python version.
5. Ensure model files (`Fake_job_detection.h5`, `tokenizer.pkl`) are in the repo and under 100MB each.
6. (Optional) Add a `Procfile` with `web: gunicorn app:app` for explicit Render config.
7. (Optional) Set environment variables in Render dashboard if your app uses any secrets.
## 🔮 Future Improvements
- Chrome Extension for job portals
- Explainable AI: highlight risky words
- Real-time scraping of job posts
- User authentication & dashboard
- REST API for programmatic access
- Docker support for easy deployment
## ⚠️ Disclaimer
This tool provides predictions based on patterns in training data. Always use your own judgment before applying for jobs or sharing personal information.
## 👤 Author
**Ayush Raj**

Role: AI & Data Science Enthusiast

Focus: Machine Learning, Data Science, and AI Projects

Connect: [LinkedIn](https://www.linkedin.com/) <!-- Add your LinkedIn or contact info if desired -->

