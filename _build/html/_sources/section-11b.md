# Section 11b: Machine Learning Projects in Chemical Engineering Research

Machine learning also has applications in industrial uses and professional research. Below is an example of machine learning used by chemical engineers to model grid power so that communities can more effeciently understand their power use.

This Streamlit-based Python application is designed to predict energy demand for hybrid microgrid systems using machine learning (ML) models. The app allows users to upload an Excel file containing historical energy demand data with features like temperature, humidity, season, and irradiance. It processes the dataset by removing outliers, selecting relevant features, and training multiple regression models—Random Forest, Gradient Boosting, Linear Regression, and Support Vector Regression (SVR). After training, the model with the highest R² score is selected as the best predictor, and users can input new values to estimate future energy demand. The application also includes visualizations like correlation heatmaps and feature importance analysis using SHAP values to enhance interpretability.  

In the context of chemical engineering, this application is particularly relevant for energy systems optimization, process control, and sustainability efforts. Chemical plants and industrial facilities often rely on hybrid microgrid systems to balance power sources such as solar, wind, and conventional grid electricity. By accurately forecasting energy demand, engineers can optimize power distribution, reduce operational costs, and minimize environmental impact. The use of ML models helps capture complex nonlinear relationships between environmental variables and energy consumption, which traditional modeling approaches might struggle with.  

Furthermore, the methodology used in this application—data preprocessing, feature selection, and ML model evaluation—is widely applicable in chemical engineering research. For instance, similar techniques can be employed to predict reactor performance, optimize heat exchanger operations, or model chemical process efficiencies. The integration of ML in these areas enables engineers to make data-driven decisions, improving both process efficiency and sustainability. This aligns with broader efforts in food and energy sustainability, where predictive modeling can enhance resource management and waste reduction.  

Overall, this project showcases how ML techniques can be leveraged to solve real-world engineering challenges, particularly in energy and process industries. By applying statistical learning to large datasets, chemical engineers can develop smarter, more efficient systems that align with global sustainability goals.

```{code-cell} ipython3
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Streamlit App Title and Introduction
st.title("Energy Demand Forecasting for Hybrid Microgrid Systems")
st.markdown("""
This application allows you to upload an Excel file with data on energy demand forecasting for hybrid microgrid systems.
The model will estimate the total energy demand based on input features such as temperature, humidity, season, and irradiance.
You can upload your dataset, remove outliers, select relevant features, and train different regression models.
The app will then predict the energy demand based on your input values.

This app is based on the work done in:

Energy Demand Forecasting for Hybrid Microgrid Systems Using Machine Learning Models

Tahir Aja Zarma 1,1
Emmanuel Ali 2
Ahmadu Adamu Galadima 1
Tologon Karataev 1
Suleiman Usman Hussein 1,3
Adekunle Akanni Adeleke 4

1 Department of Electrical Electronics Engineering, Nile University of Nigeria, Abuja, Nigeria

2 Department of Computer Engineering, Nile University of Nigeria, Abuja, Nigeria

3 National Space Research and Development Agency, Abuja, Nigeria

4 Department of Mechanical Engineering, Nile University of Nigeria, Abuja, Nigeria

DOI: https://doi.org/10.46604/peti.2024.14098

### Steps:
1. Upload your Excel file.
2. Select the sheet containing the data.
3. Choose the features to train the model.
4. Make predictions using the trained model.


### Format of Data File:
Date	Day	Hour	Month	Temp	Humidity	Irr	Rain	Grid (MWh)	Gen (MWh)	PV (MWh)	Total (MWh)

""")

# File Upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file:
    # Load the uploaded Excel file
    xls = pd.ExcelFile(uploaded_file)

    # Ask user to select sheet
    sheet_names = xls.sheet_names
    selected_sheet = st.selectbox("Select Sheet", sheet_names)

    # Read the selected sheet into a DataFrame
    df = pd.read_excel(xls, sheet_name=selected_sheet)

    st.write("### Preview of Uploaded Data")
    st.write(df.head())

    # Remove columns containing MWh (except the target 'Total (MWh)')
    mwh_cols = [col for col in df.columns if "MWh" in col and col != "Total (MWh)"]
    df.drop(columns=mwh_cols, inplace=True)

    # Remove Outliers (Using IQR Method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns  # Filter only numeric columns
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    st.write("### Data After Outlier Removal")
    st.write(df.head())

    # Set Target Variable as 'Total (MWh)' and auto-select it
    target = "Total (MWh)"

    # Display Correlation Heatmap
    st.write("### Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])  # Ensure only numeric columns are used
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Features Selection
    features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target])

    if features and target:
        X = df[features]
        y = df[target]

        # Normalize Data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = pd.DataFrame(scaler_X.fit_transform(X), columns=features)
        y = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "LinearRegression": LinearRegression(),
            "SVR": SVR()
        }

        # Train and evaluate models
        best_model = None
        best_r2 = float('-inf')

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(scaler_y.inverse_transform(y_test.reshape(-1, 1)), scaler_y.inverse_transform(y_pred.reshape(-1, 1)))

            if r2 > best_r2:
                best_r2 = r2
                best_model = model

        # Save Best Model
        joblib.dump(best_model, "energy_demand_best_model.pkl")

        # Model Performance
        y_pred_best = best_model.predict(X_test)
        mae = mean_absolute_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), scaler_y.inverse_transform(y_pred_best.reshape(-1, 1)))

        st.write("### Best Model Performance")
        st.write(f"Selected Model: {type(best_model).__name__}")
        st.write(f"Mean Absolute Error: {mae:.4f}")
        st.write(f"R² Score: {best_r2:.4f}")

        # Shapley Values (Feature Importance)
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_train)

        st.write("### Feature Importance (Shapley Values)")
        shap.summary_plot(shap_values, X_train)

        # Future Predictions
        st.write("### Make Predictions")
        input_data = {}
        for feature in features:
            input_data[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))

        if st.button("Predict Energy Demand"):
            input_df = pd.DataFrame([input_data])
            input_df = pd.DataFrame(scaler_X.transform(input_df), columns=features)  # Normalize input data
            prediction = best_model.predict(input_df)
            prediction_denormalized = scaler_y.inverse_transform(prediction.reshape(-1, 1))
            st.write(f"### Estimated Energy Demand: {prediction_denormalized[0][0]:.4f} MWh")
```

# Acknowledgements

The code examples in this book were developed with the assistance of OpenAI.

The code in the student projects section was written by the following individuals during the FA2024 Data Science in Chemical Engineering class at Bucknell University.
1. Trevor Nugent
2. Nga Vu
3. Brandon Roman
4. Maya Fetzer
5. Kayla Yi
6. Frank Onwudinjo
7. Anh Le