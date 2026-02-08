import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from gtts import gTTS
import base64, io, random

st.set_page_config(layout="wide")

# ---------------- GAMIFICATION STATE ----------------
if 'wallet' not in st.session_state: st.session_state.wallet = 50000.0  # float
if 'xp' not in st.session_state: st.session_state.xp = 0
if 'level' not in st.session_state: st.session_state.level = 1
if 'badges' not in st.session_state: st.session_state.badges = []
if 'language' not in st.session_state: st.session_state.language = "English"
if 'user_mode' not in st.session_state: st.session_state.user_mode = "Student"

# Initialize model session state
for key in ["model", "scaler", "X_test", "y_test", "scaled"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------- HELPERS ----------------
def speak(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        st.markdown(f"<audio autoplay src='data:audio/mp3;base64,{b64}'></audio>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Audio Error: {e}")

def t(en, hi, mr):
    if st.session_state.language == "Hindi": return hi
    if st.session_state.language == "Marathi": return mr
    return en

def process_df(df):
    df.columns = [c.strip() for c in df.columns]

    # Detect Date column
    for col in df.columns:
        if col.lower() == "date":
            df.rename(columns={col: "Date"}, inplace=True)

    # Parse Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")

    # Detect Volume column safely
    volume_col = None
    for col in df.columns:
        if "volume" in col.lower() or "shares" in col.lower():
            volume_col = col
            break
    if volume_col:
        df.rename(columns={volume_col: "Volume"}, inplace=True)
    else:
        df["Volume"] = 0
    return df

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Settings")
st.session_state.language = st.sidebar.selectbox("Language", ["English","Hindi","Marathi"])
st.session_state.user_mode = st.sidebar.selectbox("User Mode", ["Student","Farmer"])

# ---------------- TITLE ----------------
st.title(t(
    "🎮 FinPlay AI – Gamified Financial Literacy Platform",
    "🎮 फिनप्ले एआई – वित्तीय साक्षरता मंच",
    "🎮 फिनप्ले एआय – आर्थिक साक्षरता मंच"
))

uploaded = st.file_uploader("Upload Stock CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df = process_df(df)

    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "📊 Data",
        "🤖 Train",
        "🔮 Predict",
        "📈 Performance",
        "🎮 Simulator"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.subheader("Stock Overview")
        # Check required columns
        for col in ["Open","High","Low","Close"]:
            if col not in df.columns:
                st.error(f"CSV missing required column: {col}")
                st.stop()

        fig = go.Figure(data=[
            go.Candlestick(x=df["Date"],
                           open=df["Open"], high=df["High"],
                           low=df["Low"], close=df["Close"])
        ])
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure(go.Bar(x=df["Date"], y=df["Volume"]))
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- TAB 2 ----------------
    with tab2:
        st.subheader("Train LSTM")

        epochs = st.slider("Epochs", 5, 50, 20)
        lookback = st.slider("Lookback Days", 10, 60, 30)

        data = df["Close"].values.reshape(-1,1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)

        X,y = [],[]
        for i in range(lookback,len(scaled)):
            X.append(scaled[i-lookback:i,0])
            y.append(scaled[i,0])
        X,y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0],X.shape[1],1)

        split = int(len(X)*0.8)
        X_train,X_test = X[:split],X[split:]
        y_train,y_test = y[:split],y[split:]

        if st.button("Train Model"):
            model = Sequential()
            model.add(LSTM(50,return_sequences=True,input_shape=(X.shape[1],1)))
            model.add(Dropout(0.2))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer="adam",loss="mse")
            model.fit(X_train,y_train,epochs=epochs,verbose=0)

            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.scaled = scaled

            speak("Model trained successfully")
            if "🏆 Model Trainer" not in st.session_state.badges:
                st.session_state.badges.append("🏆 Model Trainer")

            st.success("Model Trained!")

    # ---------------- TAB 3 ----------------
    with tab3:
        if st.session_state.model:
            days = st.slider("Days to Predict",1,10,3)
            last = st.session_state.scaled[-lookback:]
            preds = []
            for _ in range(days):
                p = st.session_state.model.predict(last.reshape(1,lookback,1),verbose=0)
                preds.append(p[0,0])
                last = np.append(last[1:],p,axis=0)
            prices = st.session_state.scaler.inverse_transform(np.array(preds).reshape(-1,1))
            st.write(prices)
            speak("Prediction completed")
            if "🔮 Predictor" not in st.session_state.badges:
                st.session_state.badges.append("🔮 Predictor")

    # ---------------- TAB 4 ----------------
    with tab4:
        if st.session_state.model:
            pred = st.session_state.model.predict(st.session_state.X_test)
            y_test = st.session_state.y_test

            rmse = np.sqrt(mean_squared_error(y_test,pred))
            mae = mean_absolute_error(y_test,pred)
            r2 = r2_score(y_test,pred)

            st.metric("RMSE",rmse)
            st.metric("MAE",mae)
            st.metric("R2",r2)

    # ---------------- TAB 5 ----------------
    with tab5:
        st.subheader("🎮 Financial Simulator")

        c1,c2,c3 = st.columns(3)
        c1.metric("Wallet",f"₹{st.session_state.wallet:.2f}")
        c2.metric("XP",st.session_state.xp)
        c3.metric("Level",st.session_state.level)

        # ---------------- Float-safe sliders ----------------
        wallet = float(st.session_state.wallet)

        spend_default = min(1000.0, wallet)
        spend = st.slider("Spend", 0.0, wallet, spend_default, step=100.0)

        save_max = wallet - spend
        save_default = min(1000.0, save_max)
        save = st.slider("Save", 0.0, save_max, save_default, step=100.0)

        invest_max = wallet - spend - save
        invest_default = min(1000.0, invest_max)
        invest = st.slider("Invest", 0.0, invest_max, invest_default, step=100.0)

        if st.button("Apply"):
            gain = 0.1 if st.session_state.user_mode=="Farmer" else 0.07
            st.session_state.wallet -= spend
            st.session_state.wallet += save*0.03
            st.session_state.wallet += invest*gain
            st.session_state.xp += 20

            if st.session_state.xp >= 100:
                st.session_state.level += 1
                st.session_state.xp = 0

            speak("Good financial decision")

        if st.session_state.wallet > 80000 and "Saver 🏅" not in st.session_state.badges:
            st.session_state.badges.append("Saver 🏅")

        st.write("🏆 Badges:", st.session_state.badges)

        # Random financial scenarios
        scenario, effect = random.choice([
            ("Insurance payout +5000",5000.0),
            ("Medical bill -4000",-4000.0),
            ("Side income +6000",6000.0)
        ])

        if st.button("Run Scenario"):
            st.session_state.wallet += effect
            st.info(scenario)
            speak(scenario)

else:
    st.info("Upload CSV to begin.")
