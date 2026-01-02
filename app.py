import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from PIL import Image


SEQ_LEN = 30
CAP_RUL = 125
MODELS_DIR = "models"
DATA_DIR = "outputs"

def get_alert_level(rul):
    if rul <= 10:
        return "CRITICAL 🔴"
    elif rul <= 30:
        return "WARNING 🟠"
    else:
        return "NORMAL 🟢"

def build_test_sequences(df, sensor_cols):
    X, engine_ids = [], []

    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit].sort_values("cycle")
        data = unit_df[sensor_cols].values

        if len(data) < SEQ_LEN:
            pad = np.repeat(data[0:1], SEQ_LEN - len(data), axis=0)
            data = np.vstack([pad, data])

        X.append(data[-SEQ_LEN:])
        engine_ids.append(unit)

    return np.array(X), engine_ids


st.set_page_config(
    layout="wide",
    page_title="PrognosAI",
    page_icon="app/assets/favicon.png"
)

logo = Image.open("app/assets/logo.png")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    

    st.markdown(
    """
    <div style="text-align: center;">
        <img src="data:image/png;base64,{}" width="220">
        <h3>🛠 Predictive Maintenance Dashboard⚙️</h3>
    </div>
    """.format(
        __import__("base64").b64encode(
            open("app/assets/logo.png", "rb").read()
        ).decode()
    ),
    unsafe_allow_html=True
)



fd_id = st.selectbox("Select Dataset", ["FD001", "FD002", "FD003", "FD004"])

if st.button("Run Prediction"):

    model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, f"gru_{fd_id}_best.h5"),
        compile=False
    )

    scaler = joblib.load(os.path.join(MODELS_DIR, f"scaler_{fd_id}.joblib"))
    sensor_cols = joblib.load(os.path.join(MODELS_DIR, f"sensor_cols_{fd_id}.joblib"))

    test_df = pd.read_csv(os.path.join(DATA_DIR, fd_id, "cleaned_test.csv"))

    st.subheader("📊 Test data columns")
    st.code(list(test_df.columns))

    missing = [c for c in sensor_cols if c not in test_df.columns]
    if missing:
        st.error(f"❌ Missing required sensors for {fd_id}: {missing}")
        st.stop()

    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])

    X_test, engine_ids = build_test_sequences(test_df, sensor_cols)

    preds = model.predict(X_test).ravel()
    preds = np.clip(preds, 0, CAP_RUL)

    results = pd.DataFrame({
        "Engine ID": engine_ids,
            "Predicted RUL": preds.round(2),
            "Alert Level": [get_alert_level(r) for r in preds]
    })
    st.subheader("🚦 Fleet Health Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        " 🛑 Critical",
        int((results["Alert Level"] == "CRITICAL 🔴").sum())
        )

    col2.metric(
        "⚠️ Warning",
        int((results["Alert Level"] == "WARNING 🟠").sum())
        )

    col3.metric(
        "🛡️ Normal",
        int((results["Alert Level"] == "NORMAL 🟢").sum())
        )
    

    st.subheader("⚙️ Prediction Results 🔔")
    def highlight_alert(row):
        if row["Alert Level"] == "CRITICAL 🔴":
            return ["background-color: #ffcccc"] * len(row)
        elif row["Alert Level"] == "WARNING 🟠":
            return ["background-color: #fff3cd"] * len(row)
        else:
            return ["background-color: #d4edda"] * len(row)

    st.dataframe(
    results.style.apply(highlight_alert, axis=1),
    use_container_width=True
    )
    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
    label="📥 Download Predictions (CSV)",
    data=csv,
    file_name=f"PrognosAI_{fd_id}_predictions.csv",
    mime="text/csv"
)


    st.subheader("📈 Predicted RUL Distribution")

    st.bar_chart(
    results.set_index("Engine ID")["Predicted RUL"]
)


    st.subheader("🚨 Alert Summary")
    st.write(results["Alert Level"].value_counts())
    
    st.markdown("---")
    st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "PrognosAI © 2025 | Predictive Maintenance of Machines using AI"
    "</p>",
    unsafe_allow_html=True
    )

