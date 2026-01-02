import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import math

# ============================
# CONFIG
# ============================
SEQ_LEN = 30
CAP_RUL = 125

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


# ============================
# ALERT LOGIC (MILESTONE 3)
# ============================
def get_alert_level(rul):
    if rul <= 10:
        return "CRITICAL 🔴"
    elif rul <= 30:
        return "WARNING 🟠"
    else:
        return "NORMAL 🟢"


# ============================
# LOAD MODEL, SCALER, FEATURES
# ============================
def load_artifacts(fd_id):
    model_path = os.path.join(MODELS_DIR, f"gru_{fd_id}_best.h5")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{fd_id}.joblib")
    sensor_cols_path = os.path.join(MODELS_DIR, f"sensor_cols_{fd_id}.joblib")

    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    sensor_cols = joblib.load(sensor_cols_path)

    print("✔ Model, scaler, sensor columns loaded")
    return model, scaler, sensor_cols


# ============================
# BUILD TEST SEQUENCES
# ============================
def build_test_sequences(df, sensor_cols):
    X = []

    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit].sort_values("cycle")
        data = unit_df[sensor_cols].values

        if len(data) < SEQ_LEN:
            pad = np.repeat(data[0][None, :], SEQ_LEN - len(data), axis=0)
            data = np.vstack([pad, data])

        X.append(data[-SEQ_LEN:])

    return np.array(X)


# ============================
# PREDICTION PIPELINE
# ============================
def predict_for_fd(fd_id):
    print(f"\n🔵 Running prediction for {fd_id}")

    model, scaler, sensor_cols = load_artifacts(fd_id)

    test_path = os.path.join(OUTPUTS_DIR, fd_id, "cleaned_test.csv")
    rul_path = os.path.join(OUTPUTS_DIR, fd_id, "RUL_test.txt")

    test_df = pd.read_csv(test_path)
    print("✔ Test data loaded")

    # Ensure all training features exist
    for col in sensor_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    test_df = test_df[["unit", "cycle"] + sensor_cols]

    # Scale
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])

    # Sequences
    X_test = build_test_sequences(test_df, sensor_cols)
    print("✔ Test sequences:", X_test.shape)

    # Predict
    y_pred = model.predict(X_test).ravel()
    y_pred = np.clip(y_pred, 0, CAP_RUL)

    print("\n🔔 ALERT STATUS\n")
    for i, rul in enumerate(y_pred):
        alert = get_alert_level(rul)
        print(f"Engine {i+1}: RUL = {rul:.1f} → {alert}")

    # RMSE if true RUL exists
    if os.path.exists(rul_path):
        y_true = np.loadtxt(rul_path)
        y_true = np.clip(y_true, 0, CAP_RUL)

        rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
        print(f"\n📌 RMSE ({fd_id}): {rmse:.3f}")

        # Plot
        plt.figure()
        plt.scatter(y_true, y_pred)
        plt.plot([0, CAP_RUL], [0, CAP_RUL], "--")
        plt.xlabel("True RUL")
        plt.ylabel("Predicted RUL")
        plt.title(f"{fd_id}: Prediction vs True RUL")
        plt.show()


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    print("\nSelect FD dataset:")
    print("1 → FD001")
    print("2 → FD002")
    print("3 → FD003")
    print("4 → FD004")

    choice = input("Enter choice (1-4): ")

    fd_map = {
        "1": "FD001",
        "2": "FD002",
        "3": "FD003",
        "4": "FD004"
    }

    if choice in fd_map:
        predict_for_fd(fd_map[choice])
    else:
        print("❌ Invalid choice")

