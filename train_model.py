# ==========================================
# University Admission Prediction System
# AI Model Training
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------
# Load Dataset
# -------------------------------

data = pd.read_csv("dataset.csv")

# -------------------------------
# Input Features (X)
# -------------------------------

X = data.drop("Chance", axis=1)

# -------------------------------
# Output (Y)
# -------------------------------

y = data["Chance"]

# -------------------------------
# Split Dataset
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Feature Scaling
# -------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Create AI Model
# -------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# -------------------------------
# Train Model
# -------------------------------

model.fit(X_train, y_train)

# -------------------------------
# Evaluate Model
# -------------------------------

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# -------------------------------
# Save Model
# -------------------------------

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("=" * 40)
print(" AI Model Trained Successfully ")
print("=" * 40)
print(f"R^2 Score (closer to 1 is better): {r2:.4f}")
print(f"Mean Absolute Error (closer to 0 is better): {mae:.4f}")
print("-" * 40)
print("model.pkl saved")
print("scaler.pkl saved")