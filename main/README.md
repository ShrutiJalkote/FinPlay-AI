
# 🎮📈 FinPlay AI – Gamified Financial Literacy & Stock Prediction Platform

FinPlay AI is an **AI-powered, voice-enabled, gamified financial literacy platform** combined with **LSTM-based stock market prediction**. It helps **Students and Farmers** learn how to save, invest, budget, and protect money using simulations, real-life scenarios, and deep learning forecasting.

---

## 🎯 Overview

This project integrates **stock price prediction using LSTM neural networks** with a **behavior-driven financial learning system**. Users upload stock CSV data, train models, visualize trends, predict future prices, and simultaneously improve financial habits through a wallet simulator, XP, badges, and monthly scenarios.

The platform supports **Marathi, Hindi, and English**, making it rural-ready and low-literacy friendly with **voice guidance**.

---

## ✨ Features

### 📊 Stock Market Module
- CSV upload with validation  
- Automatic date parsing  
- Candlestick and volume charts  
- LSTM training with hyperparameters  
- Train/Test split (80/20)  
- Performance metrics (RMSE, MAE, R²)  
- Multi-day future prediction  
- Plotly interactive visualizations  

### 🎮 Gamified Financial Learning
- Virtual wallet simulator  
- Save / Spend / Invest decisions  
- XP and Level system  
- Badge achievements  
- Behaviour-based learning  

### 📅 Monthly Scenarios
- Emergency expenses  
- Scam alerts  
- Investment opportunities  
- Seasonal income cases  

### 🧑‍🌾👩‍🎓 User Modes
- Farmer Mode (seasonal income, risk, savings)  
- Student Mode (budgeting, spending, scam safety)  

### 🗣️ Voice & Language Support
- Text-to-Speech guidance  
- Marathi / Hindi / English modes  
- Rural-ready UI  

### 🎨 Dashboard
- Streamlit professional layout  
- Sidebar configuration  
- Tabs for Overview, Training, Prediction, Performance, Simulator  

---

## 🛠️ Technology Stack

| Layer | Tools |
|------|------|
| Frontend | Streamlit |
| AI / ML | TensorFlow, Keras |
| Data | Pandas, NumPy |
| Visualization | Plotly |
| ML Utils | Scikit-learn |
| Voice | gTTS |
| Model | LSTM (RNN) |

---

## 📋 Requirements

- Python 3.8+  
- pip  

### requirements.txt

```text
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
tensorflow==2.13.0
plotly==5.16.1
gTTS==2.5.1
```

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/finplay-ai.git
cd finplay-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📊 CSV Format

Required columns:

```csv
Date,Open,High,Low,Close,Volume
```

Date formats supported automatically.

You can use data from:
- Yahoo Finance  
- NSE India  
- Kaggle  

---

## 🧠 LSTM Architecture

```
Input → LSTM → Dropout → LSTM → Dropout → Dense → Output
```

---

## 📈 Metrics

- RMSE  
- MAE  
- R²  

Lower RMSE and MAE with higher R² indicate better performance.

---

## 🎯 Alignment

✔ Behaviour-based learning  
✔ Gamification  
✔ Voice guidance  
✔ Rural-ready UI  
✔ Multi-theme finance (Savings, Budgeting, Investment, Fraud Prevention)  
✔ Simulation and decision making  

---

## 🎯 Use Cases

- Financial literacy training  
- Students money management  
- Farmers seasonal planning  
- AI in finance research  
- Hackathon demos  

---

## 🔮 Future Scope

- Real-time stock API  
- Portfolio tracker  
- Scam detection AI  
- Insurance planner  
- Mobile app  

---

## ⚠️ Disclaimer

This project is for educational purposes only. Stock predictions are uncertain. Do not use this for real investments.

---

##  Built with Python, TensorFlow & Streamlit

⭐ Star the repository if you like the project!
