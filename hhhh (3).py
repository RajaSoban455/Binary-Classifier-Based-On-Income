import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("Income Prediction App (Regression)")

uploaded_file = st.file_uploader("Upload the 'adult.csv' file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data", df.head())

    # Replace '?' with NaN and drop rows with missing values
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Drop rows where income is not numeric if needed (for safety)
    if not np.issubdtype(df['income'].dtype, np.number):
        df['income'] = pd.to_numeric(df['income'], errors='coerce')
        df.dropna(subset=['income'], inplace=True)

    # Split into features and target
    X = df.drop('income', axis=1)
    y = df['income']

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train models
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train, y_train)

    # Model evaluation
    lin_preds = lin_reg.predict(X_test)
    rf_preds = rf_reg.predict(X_test)

    st.subheader("Linear Regression Performance")
    st.write(f"MSE: {mean_squared_error(y_test, lin_preds):.4f}")
    st.write(f"R^2 Score: {r2_score(y_test, lin_preds):.4f}")

    st.subheader("Random Forest Regressor Performance")
    st.write(f"MSE: {mean_squared_error(y_test, rf_preds):.4f}")
    st.write(f"R^2 Score: {r2_score(y_test, rf_preds):.4f}")

    # Ask for user input for prediction
    st.subheader("Make a Prediction")
    user_input = {}
    for col in X.columns:
        if col in label_encoders:
            options = label_encoders[col].classes_
            user_input[col] = st.selectbox(f"{col}", options)
        else:
            user_input[col] = st.number_input(f"{col}", format="%.2f")

    if st.button("Predict Income"):
        input_df = pd.DataFrame([user_input])

        # Encode categorical fields
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Scale input features
        input_scaled = scaler.transform(input_df)

        lin_prediction = lin_reg.predict(input_scaled)[0]
        rf_prediction = rf_reg.predict(input_scaled)[0]

        st.write(f"**Linear Regression Prediction:** Estimated income = ${lin_prediction:.2f}")
        st.write(f"**Random Forest Prediction:** Estimated income = ${rf_prediction:.2f}")

else:
    st.info("Please upload the 'adult.csv' file to get started.")
