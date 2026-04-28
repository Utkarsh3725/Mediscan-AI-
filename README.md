# 🩺 MediScan AI

> **Smart Diagnostics for a Healthier Tomorrow**

MediScan AI is a web-based medical diagnostic tool powered by Machine Learning. It helps predict the likelihood of **Heart Disease**, **Diabetes**, and **Parkinson's Disease** based on patient input data — built to assist in early detection and better healthcare decisions.

---

## 🚀 Features

- 🫀 **Heart Disease Prediction** — Analyze 13 clinical parameters to detect heart disease risk
- 🩸 **Diabetes Prediction** — Predict diabetes based on glucose, BMI, insulin, and more
- 🧠 **Parkinson's Disease Prediction** — Detect Parkinson's using voice measurement features
- 📱 **Responsive Design** — Works on mobile and desktop
- ⚡ **Real-time Results** — Instant prediction via Flask backend API

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| ML Models | Scikit-learn, Joblib |
| Data Processing | NumPy |

---

## 📁 Project Structure

```
Mediscan-AI/
│
├── Model/
│   └── app.py               # Flask backend with prediction routes
│
├── index.html               # Home page
├── Heart.html               # Heart disease form
├── Diabetes.html            # Diabetes form
├── Parkinson.html           # Parkinson's form
│
├── style.css                # Main stylesheet
├── heart.css                # Heart page styles
├── diabetes.css             # Diabetes page styles
├── parkinson.css            # Parkinson page styles
│
├── requirements.txt         # Python dependencies
└── run.sh                   # Shell script to run the app
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Mediscan-AI.git
cd Mediscan-AI
```

### 2. Create a Virtual Environment
```bash
python -m venv env
source env/bin/activate        # On Windows: env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
bash run.sh
```
Or manually:
```bash
cd Model
python app.py
```

### 5. Open in Browser
```
http://localhost:5000
```

---

## 📦 Requirements

```
Flask>=3.1,<4
joblib>=1.4
numpy>=2.0
scikit-learn==1.5.2
```

---

## 🔬 How It Works

1. User visits the homepage and selects a disease category
2. Fills in the diagnostic form with medical parameters
3. Data is sent to the Flask backend via a POST request
4. The ML model processes the input and returns a prediction
5. Result is displayed instantly on the page

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict-heart` | POST | Heart disease prediction |
| `/predict-diabetes` | POST | Diabetes prediction |
| `/predict-parkinson` | POST | Parkinson's prediction |

---

## 👨‍💻 Author

**Utkarsh**

---

## ⚠️ Disclaimer

> This tool is intended for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

⭐ If you found this project helpful, consider giving it a star!
