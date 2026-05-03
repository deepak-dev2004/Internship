import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("churn.csv")

st.title("📉 Customer Churn Prediction")

# Drop ID
df = df.drop("CustomerID", axis=1)

# Encode target
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Split features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Save columns for prediction
columns = X.columns

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"{acc * 100:.2f}%")

# -----------------------------
# 🔮 Prediction UI
# -----------------------------
st.subheader("Predict New Customer")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
tenure = st.slider("Tenure", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# Convert input → dataframe
input_dict = {
    "SeniorCitizen": senior,
    "Tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Gender_Male": 1 if gender == "Male" else 0,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0
}

input_df = pd.DataFrame([input_dict])

# Match training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    result = "⚠️ Churn" if pred == 1 else "✅ No Churn"
    st.success(f"Prediction: {result}")