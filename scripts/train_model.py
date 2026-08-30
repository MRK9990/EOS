import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ============================================================
# EOS PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "eos_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "eos_model_v1.pkl"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# 1️⃣ Завантаження даних
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# 2️⃣ Визначаємо targets
target_columns = [
    "white_balance",
    "red_gain",
    "blue_gain",
    "brightness",
    "contrast",
    "exposure_ev"
]

# 3️⃣ Features = все інше
feature_columns = [col for col in df.columns if col not in target_columns]

X = df[feature_columns]
y = df[target_columns]

# 4️⃣ Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Нормалізація (тільки для X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ Модель
base_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model = MultiOutputRegressor(base_model)
model.fit(X_train_scaled, y_train)

# 7️⃣ Оцінка
y_pred = model.predict(X_test_scaled)

print("\nEvaluation (Mean Absolute Error):")

for i, col in enumerate(target_columns):
    mae = mean_absolute_error(y_test[col], y_pred[:, i])
    print(f"{col}: {mae:.4f}")

# 8️⃣ Збереження моделі
joblib.dump({
    "model": model,
    "scaler": scaler,
    "feature_columns": feature_columns,
    "target_columns": target_columns
}, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)