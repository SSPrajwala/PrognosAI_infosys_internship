# 🛠 PrognosAI — Predictive Maintenance System

PrognosAI is an end-to-end **Predictive Maintenance Dashboard** built using **Deep Learning (GRU)** to estimate the **Remaining Useful Life (RUL)** of aircraft engines using NASA’s **C-MAPSS dataset**.  
The system helps identify engines at risk and generates actionable alerts for maintenance planning.

---

## 🚀 Project Objectives

- Predict Remaining Useful Life (RUL) of engines
- Prevent unexpected failures using early warnings
- Visualize fleet health using an interactive dashboard
- Support multiple datasets (FD001–FD004)

---

## 🧠 Technologies Used

| Component | Technology |
|--------|-----------|
| Programming | Python |
| Deep Learning | TensorFlow / Keras (GRU) |
| Dashboard | Streamlit |
| Data Handling | Pandas, NumPy |
| Scaling | Scikit-learn (StandardScaler) |
| Model Storage | Joblib |
| Deployment | Streamlit Community Cloud |
| Version Control | GitHub |

---

## 📊 Dataset Description

**NASA C-MAPSS Turbofan Engine Dataset**

- FD001 – Single operating condition, single fault
- FD002 – Multiple operating conditions
- FD003 – Single condition, multiple faults
- FD004 – Multiple conditions & faults

Each dataset contains:
- Engine Unit ID
- Operational settings (`op1`, `op2`)
- Sensor readings (`sensor_1` to `sensor_21`)
- Cycle count
- RUL (Remaining Useful Life)

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to:
- Verify sensor availability across datasets
- Identify constant / non-informative sensors
- Observe degradation trends over cycles
- Validate RUL distribution and capping

Scripts used:
- `eda_check_output.py`
- `check_output.py`

---

## ⚙️ Feature Engineering & Preprocessing

- Selected only meaningful sensor columns
- Renamed sensor columns consistently
- Scaled features using `StandardScaler`
- Capped RUL values at **125 cycles** to reduce noise
- Generated rolling sequences of **30 cycles**

Scripts:
- `preprocess.py`
- `fix_sensor_names.py`
- `create_sequences.py`

---

## 🔄 Sequence Creation

For each engine:
- Data is grouped by unit
- Sorted by cycle
- Converted into overlapping sequences of length 30
- Used as input to GRU network

---

## 🧠 Model Architecture

**GRU-based Recurrent Neural Network**

- Input: `(30 timesteps × N sensors)`
- GRU Layers capture temporal degradation patterns
- Output: Single continuous RUL value

Models trained separately for:
- FD001
- FD002
- FD003
- FD004

Scripts:
- `train_fd001.py`
- `train_fd002.py`
- `train_fd003.py`
- `train_fd004.py`

---

## 📈 Model Evaluation

- Loss function: Mean Squared Error (MSE)
- Metric: RMSE
- Early stopping to prevent overfitting
- Best models saved based on validation loss

---

## 🚦 Alert System Logic

Based on **Predicted RUL**:

| RUL Range | Alert Level |
|--------|------------|
| ≤ 10 | 🔴 Critical |
| 11 – 30 | 🟠 Warning |
| > 30 | 🟢 Normal |

This enables maintenance prioritization.

---

## 🖥️ Streamlit Dashboard Features

- Dataset selection (FD001–FD004)
- Real-time prediction
- Engine-wise RUL table
- Fleet health metrics
- Alert summary
- Downloadable results

- 
Run locally:
```bash
streamlit run app/app.py

📁 Project Structure
PrognosAI/
│
├── app/
│   └── app.py
│
├── src/
│   ├── preprocess.py
│   ├── create_sequences.py
│   ├── train_fd001.py
│   ├── train_fd002.py
│   ├── train_fd003.py
│   ├── train_fd004.py
│   ├── predict.py
│   ├── utils.py
│   └── run_all_fd.py
│
├── outputs/
│   └── FD00X/cleaned_test.csv
│
├── models/
│   └── sensor_cols_FD00X.joblib
│
├── requirements.txt
├── .gitignore
└── README.md

⚠️ Note on Large Files
Large training artifacts (.npy) are excluded from GitHub due to size limits.
They can be regenerated using training scripts.

👩‍💻 Author
Kaluvala Sri Sai Prajwala
Infosys Internship Project
PrognosAI
