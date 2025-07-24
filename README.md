# 🤖 AI Interview System

An intelligent interview assistant built with **Streamlit**, designed to simulate real-world interviews using voice prompts and AI-powered analysis. This system helps assess candidates for various customer-facing job roles by analyzing their responses, tone, and communication skills using machine learning models.

> 🎯 Project Goal: To simulate a professional interview environment using voice interaction, enabling candidates to practice and receive feedback in real-time.

---

## 🌐 Live Demo
👉 [Launch AI Interview System on Streamlit](https://YOUR-STREAMLIT-APP-LINK-HERE)

---

## 🛠️ Tech Stack

| Area | Technologies |
|------|--------------|
| 💻 Frontend | [Streamlit](https://streamlit.io/) |
| 🧠 AI/NLP | OpenAI (for analysis & insights) |
| 🔊 Audio Processing | Python `wave`, `pydub`, etc. |
| ☁️ Cloud & Integration | **Google Cloud Platform (GCP)** <br> (Used: Cloud Run, Pub/Sub as Kafka replacement, Service Accounts) |
| 🔒 Secrets Management | `.env`, GCP Service Account (ignored in git) |
| 🐳 Deployment (local) | Docker (optional setup) |
| 🔁 CI/CD | Streamlit Cloud |

---

## 📂 Project Structure

```

interview-system/
│
├── backend/                     # Handles all backend logic
│   ├── app.py                   # Main backend logic (FastAPI or internal logic)
│   ├── main.py                  # Entry point or utility script
│   ├── questions.json           # Interview questions by role
│   └── gcp\_creds\_application.json  # GCP service account (not pushed)
│
├── frontend/                    # Streamlit frontend
│   ├── app.py                   # Streamlit app entry point
│   └── static\_job\_roles/        # Job role folders with WAV files
│       └── ServiceAmbassador/
│           ├── Q1.wav
│           ├── Q2.wav
│           └── Q3.wav
│
├── requirements.txt             # Dependencies
├── .env                         # API keys and secrets (ignored)
└── .gitignore                   # Git ignore config

````

---

## 🧪 Features

- 🎙️ **Audio-Based Questions** for each job role
- 🧠 **AI-Powered Analysis** using OpenAI models
- 📊 **Real-Time Feedback** on candidate responses
- 📁 **Role-Based Question Sets** (Customer Service, Branch Officer, etc.)
- 🔐 Secure API/Cloud Integration using GCP and `.env`
- 🌐 Deployable to Streamlit Cloud with ease

---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/Aliharis007/AI-Interview-System.git
cd AI-Interview-System
````

### 2. Set up the Environment

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
```

Download your GCP `gcp_creds_application.json` and keep it in the backend folder (DO NOT commit it).

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit App

```bash
cd frontend
streamlit run app.py
```

---

## 🧠 Google Cloud Platform Integration

This project uses several GCP services:

* ✅ **Cloud Run** for deploying scalable backend logic (optional)
* ✅ **Pub/Sub** as a message broker (Kafka replacement)
* ✅ **Service Account** for authentication and secure API access

> GCP creds are stored in `gcp_creds_application.json` and accessed securely during backend interaction. *This file is excluded from GitHub using `.gitignore`.*

---

## ⚙️ Deployment on Streamlit Cloud (Coming next...)

Stay tuned below for deployment instructions 👇

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

Made with ❤️ by **Ali Haris**
🔗 [GitHub](https://github.com/Aliharis007)

```
