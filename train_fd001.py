import os
import numpy as np
import math
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =============================
# CONFIG
# =============================
FD_ID = "FD001"
SEQ_LEN = 30

DATA_PATH = f"outputs/{FD_ID}"
MODEL_PATH = "models"

os.makedirs(MODEL_PATH, exist_ok=True)

# =============================
# LOAD DATA
# =============================
X = np.load(f"{DATA_PATH}/sequences_train.npy")
y = np.load(f"{DATA_PATH}/labels_train.npy")

print(f"Loaded X: {X.shape}")
print(f"Loaded y: {y.shape}")

num_features = X.shape[2]

# =============================
# SAVE SENSOR COLUMN ORDER
# =============================
# Load sensor column names from cleaned CSV
import pandas as pd

clean_df = pd.read_csv(f"{DATA_PATH}/cleaned_train.csv")

sensor_cols = [
    c for c in clean_df.columns
    if c.startswith("sensor_") or c.startswith("op")
]

joblib.dump(sensor_cols, f"{MODEL_PATH}/sensor_cols_{FD_ID}.joblib")
print("✅ Sensor columns saved:", sensor_cols)



# =============================
# SCALING
# =============================
scaler = StandardScaler()

X_2d = X.reshape(-1, num_features)
X_scaled = scaler.fit_transform(X_2d)
X = X_scaled.reshape(-1, SEQ_LEN, num_features)

joblib.dump(scaler, f"{MODEL_PATH}/scaler_{FD_ID}.joblib")
print("Scaler saved.")

# =============================
# TRAIN / VAL SPLIT
# =============================
split = int(0.85 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)

# =============================
# BUILD MODEL
# =============================
model = Sequential([
    GRU(128, input_shape=(SEQ_LEN, num_features)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# =============================
# TRAIN
# =============================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        f"{MODEL_PATH}/gru_{FD_ID}_best.h5",
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks
)

model.save(f"{MODEL_PATH}/gru_{FD_ID}_final.h5")
print("Model saved.")

# =============================
# EVALUATION
# =============================
y_pred = model.predict(X_val).ravel()
rmse = math.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.2f}")

# =============================
# PLOT LOSS
# =============================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title(f"Training Curve ({FD_ID})")
plt.show()
