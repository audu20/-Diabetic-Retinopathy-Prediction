import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

# Model file paths
model_paths = {
    "SVM Model": r"C:\Users\admin\Downloads\Data_Science_Project_ExcelR\___Diabetic_retinopathy_prediction_in_patients\svm_model.pkl",
    "Random Forest": r"C:\Users\admin\Downloads\Data_Science_Project_ExcelR\___Diabetic_retinopathy_prediction_in_patients\random_forest_model.pkl",
    "Logistic Regression": r"C:\Users\admin\Downloads\Data_Science_Project_ExcelR\___Diabetic_retinopathy_prediction_in_patients\logistic_regression_model.pkl",
    "KNN": r"C:\Users\admin\Downloads\Data_Science_Project_ExcelR\___Diabetic_retinopathy_prediction_in_patients\knn_model.pkl",
}

# Features and label mapping
features = ['age', 'systolic_bp', 'diastolic_bp', 'cholesterol']
label_mapping = {0: "no_retinopathy", 1: "retinopathy"}

st.title("🩺 Diabetic Retinopathy Prediction")

# Model selection
selected_model_name = st.selectbox("📌 Select model:", list(model_paths.keys()))
model = joblib.load(model_paths[selected_model_name])

# Dataset path input
file_path = st.text_input("📂 Enter full path to dataset CSV file:")

if file_path and os.path.exists(file_path):
    try:
        df = pd.read_csv(file_path, sep=';')
        if 'ID' in df.columns:
            df.drop(columns=['ID'], inplace=True)

        # Full predictions for slider display
        X = df[features]
        all_preds = [label_mapping.get(pred, str(pred)) for pred in model.predict(X)]
        results_df = pd.DataFrame({"Prediction": all_preds})

        # Slider to preview predictions
        max_display = st.slider("Number of predictions to display", 10, 1000, 100, 10)
        st.dataframe(results_df.head(max_display))

        # Download button
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download All Predictions", csv, f"predictions_{selected_model_name}.csv", "text/csv")

        # Accuracy on test split only (matches training evaluation)
        if 'prognosis' in df.columns:
            y = df['prognosis'].astype(str)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            test_preds = [label_mapping.get(pred, str(pred)) for pred in model.predict(X_test)]
            acc = accuracy_score(y_test, np.array(test_preds).astype(str))
            st.subheader("✅ Test Set Accuracy")
            st.write(f"{acc * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
elif file_path:
    st.error("❌ File not found. Please check the path.")
