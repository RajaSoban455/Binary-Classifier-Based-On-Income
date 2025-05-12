import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Load the dataset (after downloading from Kaggle and placing it locally)
df = pd.read_csv('adult.csv')  # Ensure the path matches where you save the file
df.head()
df.info()
df.describe()
df.isnull().sum()

# Replace '?' with NaN for imputation
df.replace('?', np.nan, inplace=True)
df.isnull().sum()
# Impute missing values (mode for categorical)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])


# Encode categorical variables
cat_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature Scaling
scaler = StandardScaler()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = scaler.fit_transform(df[num_cols])

# Split into features and target
X = df.drop('income', axis=1)
y = df['income']
import joblib
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

print("Logistic Regression Report:\n", classification_report(y_test, log_preds))
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("Random Forest Report:\n", classification_report(y_test, rf_preds))
# Confusion Matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, log_preds), annot=True, fmt='d', ax=ax[0], cmap='Blues')
ax[0].set_title('Logistic Regression Confusion Matrix')
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', ax=ax[1], cmap='Greens')
ax[1].set_title('Random Forest Confusion Matrix')
plt.show()

# ROC Curve
log_probs = log_model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]
log_fpr, log_tpr, _ = roc_curve(y_test, log_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

plt.figure(figsize=(10, 6))
plt.plot(log_fpr, log_tpr, label='Logistic Regression')
plt.plot(rf_fpr, rf_tpr, label='Random Forest')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, None]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
import joblib
# Save the trained model
joblib.dump(rf_model, "random_forest_model.pkl")
# Save the scaler
joblib.dump(scaler, "scaler.pkl")
# Save label encoders for categorical variables
joblib.dump(label_encoders, "label_encoders.pkl")

import streamlit as st
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

# Load the saved model and encoders
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Income Classification App")

# Initialize session state to store history
if 'history' not in st.session_state:
    st.session_state['history'] = []

st.write("### Please enter the following details:")

user_input = {}
for col in feature_columns:
    if col in label_encoders:
        options = label_encoders[col].classes_
        user_input[col] = st.selectbox(f"{col}", options)
    else:
        user_input[col] = st.number_input(f"{col}", format="%.4f")

# Predict button
if st.button("Predict Income Category"):
    input_df = pd.DataFrame([user_input])

    # Encode categorical features
    for col in input_df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])

    # Scale numerical features
    input_df[feature_columns] = scaler.transform(input_df[feature_columns])

    # Prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    # Decode the prediction
    income_label = label_encoders['income'].inverse_transform([prediction])[0]

    # Display result
    st.success(f"Predicted Income Category: **{income_label}**")
    st.write(f"Probability of >50K: **{prediction_proba:.2f}**")

    # Save history
    record = user_input.copy()
    record['Predicted Income'] = income_label
    record['Probability >50K'] = round(prediction_proba, 2)
    st.session_state['history'].append(record)

# Show history
if st.session_state['history']:
    st.write("### Prediction History")
    st.dataframe(pd.DataFrame(st.session_state['history']))
