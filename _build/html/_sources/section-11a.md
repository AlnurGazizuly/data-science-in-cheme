# Section 11a: Student Machine Learning Projects in Chemical Engineering

Machine learning can be used across a large variety of applications related to chemical engineering. Explore the below projects to see how stuents are using a combination of chemical engineering and machine learning to solve real world problems.

Machine learning is increasingly being used by chemical engineering students to tackle real-world challenges, from process optimization to material property prediction. This section highlights innovative student projects that apply machine learning techniques to solve chemical engineering problems. By exploring these case studies, you'll see how data-driven approaches can enhance process efficiency, sustainability, and decision-making. These projects serve as inspiration for integrating machine learning into your own work, demonstrating the potential of AI-driven solutions in chemical engineering.

#### Project 1: Time Series $CO_2$ Forecasting

##### ReadMe
---

**Machine-Learning-Project:**

CO2 Forecasting utilizing the ARIMA and SARIMAX ML Models. Time Series CO2 Forecasting

**By:**

Author: Trevor Nugent

Authors of Raw Data Collection: Independent Statistics and Analysis U.S. Energy Information Administration 2023 Survey

Instructor: Dr. Jude Okolie

University: Bucknell University

Department: Chemical Engineering

**Problem Statement:**

CO2 emissions are one of the biggest causes of global warming, and their consequences are getting serious over time. The objectives of this study is to determine, using an ARIMA model over the time series of the CO2 emission from electricity generation.

**Input and Output Variables:**

**Inputs:**

Source, Date (MM/YYYY), Emissions (Million Metric Tons).

**Targets:**

Forecasted Emissions (Million Metric Tons)

**Machine Learning Algorithm(s):**

ARIMA and SARIMA models are designed for analyzing and forecasting univariate time series data. ARIMA was used when CO₂ emissions exhibit a trend but no seasonality. SARIMA was used when emissions display seasonal patterns (e.g., winter spikes). These models are used as they are reliable, interpretable, and effective for short-to-medium-term forecasting, making them ideal for time series data like CO₂ emissions. ARIMA and SARIMA models are widely used for time series forecasting due to their robustness and simplicity.

**Ethics Considerations:**

**Ethics Checklist and Data Card can be found in the Rpository Graphical User Interface (GUI):** NA

**Dataset:** Raw dataset .csv file can be found in the repository

Carbon emissions from electicity production A Monthly/Annual CO2 emissions from electricity generation from the Energy Information Administration. Data is broken down by fuel type. Preprocessing and EDA were applied.

##### Machine Learning Development




---

```python
import numpy as np
import pandas as pd
import matplotlib.pylab
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 20
import warnings
import itertools
warnings.filterwarnings("ignore")
```

```python
df = pd.read_csv("MER_T12_06.csv")
df.head()
```

```python
df.info()
```

```python
# Changing Month from int to Date
dateparse = lambda x: pd.to_datetime(x, format='%Y%m', errors = 'coerce') #coerce to have a NaN field in empty or corrupted
df = pd.read_csv("MER_T12_06.csv", parse_dates=['YYYYMM'], index_col='YYYYMM', date_parser=dateparse)
df.head()
```

```python
#drop rows with null index
#ts = time serie
ts = df[pd.Series(pd.to_datetime(df.index, errors='coerce')).notnull().values]
ts.head(15)
```

```python
ts = df[pd.Series(pd.to_datetime(df.index, errors='coerce')).notnull().values]
ts.head(15)
```

```python
ts['Value'] = pd.to_numeric(ts['Value'] , errors='coerce')
ts.head()
```

```python
ts.dropna(inplace = True)
```

```python
# sources = CO2 sources
sources = ts.groupby('Description')
ts.head(15)
```

```python
fig, ax = plt.subplots()
for desc, grp in sources:
    grp.plot( y='Value', label=desc,ax = ax, title='Carbon Emissions per Source', fontsize = 20)
    ax.set_xlabel('Time : Month')
    ax.set_ylabel('Carbon Emissions in Million Metric Tons')
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.legend(fontsize = 16)
```

```python
#Plot per Emission source
fig, axes = plt.subplots(3,3, figsize = (40, 40))
for (desc, group), ax in zip(sources, axes.flatten()):
    group.plot(y='Value',ax = ax, title=desc, fontsize = 20)
    ax.set_xlabel('Time : Month')
    ax.set_ylabel('Carbon Emissions in Million Metric Tons')
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
```

```python
QuantityPerSource = ts.groupby('Description')['Value'].sum().sort_values()
src = ['Geothermal Energy', 'Non-Biomass Waste', 'Petroleum Coke','Distillate Fuel ',
        'Residual Fuel Oil', 'Petroleum', 'Natural Gas', 'Coal', 'Full Emissions']
fig = plt.figure(figsize = (16,9))
x_label = src
x_tick = np.arange(len(src))
plt.bar(x_tick, QuantityPerSource, align = 'center', alpha = 0.7)
fig.suptitle("CO2 Emissions by Sector", fontsize= 25)
plt.xticks(x_tick, x_label, rotation = 80, fontsize = 30 )
plt.yticks(fontsize = 30)
plt.xlabel('Carbon Emissions in Million Metric Tons', fontsize = 30)
plt.show()
```

```python
# ems = emissions
ems = ts.drop(['MSN','Column_Order','Unit'], axis=1)  # dropping columns and letting only value and description column
# total emissions (pte)
pte = ems[ems['Description'] =="Total Energy Electric Power Sector CO2 Emissions"]
pte = pte.drop(['Description'], axis=1)
pte
```

```python
plt.plot(pte)
# same result
```

```python
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
def TestStationaryPlt(pte):
    rolling_mean = pte.rolling(window = 12, center = False).mean()
    rolling_std = pte.rolling(window = 12, center = False).std()
    plt.plot(pte, color = 'blue',label = 'Original Data')
    plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
    plt.plot(rolling_std, color ='black', label = 'Rolling Standard Deviation')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel('Time in Years', fontsize = 20)
    plt.ylabel('Total Emissions', fontsize = 20)
    plt.legend(loc='best', fontsize = 20)
    plt.title('Rolling Mean & Rolling Standard Deviation', fontsize = 20)
    plt.show()
TestStationaryPlt(pte)
```

```python
def Adf_test(pte):
    result = adfuller(pte['Value'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
Adf_test(pte)
```

```python
moving_avg = pte.rolling(12).mean()
pte_trans = pte - moving_avg
pte_trans.head(13)
```

```python
pte_trans.dropna(inplace=True)
TestStationaryPlt(pte_trans)
```

```python
Adf_test(pte_trans)
```

```python
# - Find optimal parameters and build an ARIMA model
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(pte_trans.iloc[13:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(pte_trans.iloc[13:], lags=40, ax=ax2)
```

```python
mod = sm.tsa.statespace.SARIMAX(pte,
                                order=(1,1,1),
                                seasonal_order=(0,1,1,12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
```

```python
results.plot_diagnostics(figsize=(15, 12))
plt.show()
```

```python
# gaussian residual error
results.resid.plot(kind='kde')
```

```python
pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
pred_ci = pred.conf_int()
pred_ci.head()
```

```python
ax = pte['1973':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.5)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Time : Years')
ax.set_ylabel('CO2 Emissions')
plt.legend()

plt.show()
```

##### GUI App Deployment


---

```python
import pandas as pd
from datetime import datetime

# Assuming 'results' is the fitted model object
# Ask the user for a prediction start date
user_date = input("Enter the start date for prediction (YYYY-MM-DD): ")

# Validate and parse the user input
try:
    start_date = pd.to_datetime(user_date, format='%Y-%m-%d')
except ValueError:
    print("Invalid date format. Please use YYYY-MM-DD.")
    raise

# Generate predictions starting from the user-provided date
pred = results.get_prediction(start=start_date, dynamic=False)
pred_ci = pred.conf_int()

# Rename the confidence interval columns for clarity
pred_ci.columns = [
    'Lower Limit (Million Metric Tons CO₂)',
    'Upper Limit (Million Metric Tons CO₂)'
]

# Display the labeled confidence intervals
print(pred_ci.head())
```

##### Ethics Analysis
---

**ETHICS CHECKLIST**

---



**Data Collection and Consent:**

*Was informed consent obtained from individuals whose data is being used?*

Yes. These values are reported directly to US EIA

*Are individuals aware of how their data will be used and the scope of data collection?*

Yes. Individuals know that these reports are necesseary to monitor CO2 emissions

*Is there an option for users to opt-out of data collection?*

No.

**Privacy and Confidentiality:**

*Are measures in place to protect the privacy and confidentiality of individuals' data?*

This data is public.

*Are there safeguars against unauthorized access or data breaches?*

There is no unauthorized access.

**Transparency and Accountability:**

*Are the methods and purpose of the project transparent to stakeholders?*

Yes. The purpose of this is to clearly forecast CO2 emissions

*Is there a clear accountability structure for ethical concerns*

No, there is no clear accountability structure from the US EIA website

*Can the results of the analysis be audited for ethical and methodological rigor*

Yes, the findings are reliable, valid, and credible

**Fairness and Non-discrimination:**
  
*Does the data or algorithm ingtroduce any bias that could lead to discrimination?*

No.

*Are the outcomes fair and unbiased across different demographic groups*

Yes. This is a National data collection that spans many demographics.

**Purpose and Justifiability:**
  
*Is the purpose of the data analysis justifiable, and does it align with etghical goals?*

Yes. The purpose is to monitor CO2 emissions in hopes to begin reducing emissions for a more sustainable existence.

*Could this project cause harm or benefit society?*

It may benefit society by beginning the process to a healthier Earth with less climate change from CO2 emissions

**Safety and Security:**

*Are measures in place to ensure the project idoes not put individuals or society at risk?*

No. However, this project will not put individuals or society at risk

*Are there clear steps for mitigating risks if they arise?*

No. There are no risks to worry about for a model that simply forecasts CO2 emissions

**Impact on Stakeholders:**

*Who would be impacted by the project, both positively and negatively?*

Stakeholders would be positively impacted by the project as this model could be used widespread to monitor CO2 emissions. This could negatively impact companies that already emit too much CO2.

*Are vulnerable populations considered in the project design and analysis?*

Yes. Since vulnerable populations are impacted more harshly by climate change, this project aims to mitigate these impacts by monitoring CO2 emissions

**ETHICS DATA CAD**


---


**Project Purpose and Scope:**

*What is the primary purpose of this project, how does it benefit stakeholders?*

To forecast/monitor CO2 emissions. This will benefit stakeholders as this monitoring system will be used widespread to mitigate CO2 emissions in America.

*Is the project scope clear and justifiable in terms of ethical impact?*

Yes.

**Data Sources and Reliability:**

*What are the sources of data, and are they credible and ethical?*

https://www.eia.gov/totalenergy/.

US Energy Information Administration. Yes. It is credible.

*Are potential biases in the data sources acknowledged or mitigated?*

There are no biases in the data source to be mitigated.

**Stakeholder Transparency:**

*Are stakeholders informed about the project's aims, data use, and potential outcomes?*

Yes.

*Is there clear documentation accissible to stakeholders for full transparancy?*

Yes. The complete raw dataset is on the website

**Bias and Fairness Check:**
  
*Has the project undergone rigorous checks to identify and mitigate biases?*

Yes, no biases are present in the reports to the US EIA.

*How does the project ensure fairness in its outputs, especially for marginalized groups?*

This is objective data reported with no regard to demographics. This data is unbiased and represents teh entirety of America and every demographic that lives there.

**Data Protection and Anonymization:**

*Are data protection measures like anonymization in place?*

No. It is not needed because no personal information is input by the user.

*How does the projet secure data to prevent unauthorized access?*

There is no unauthorized access to the public data.

**Ethical Impact Assessment:**

*Are there mechanisms to assess and track the ethical impact of the project over time?*
    
No. These should be implemented in order to determine the results of CO2 emission mitigation to see if strategies put in place are actually affective.

*Have potential risks been identified, and are there mitigation strategies in place?*

There are no potential risks in monitoring CO2 emissions with the long-term goal of reducing these emissions.

**Ongoing Monitoring and Accountabilty:**

*Is there a plan for ongoing monitoring of the project's ethical impact?*

Yes. Continuous monitoring and forecasting will allow users to see the sustainability results and impact of strategies put in place to mitigate CO2 emissions.

*Are roles and responsibilities clearly defined to ensure accountability?*

No. This website does not mention an accountability hierarchy involved in the collection of this data or the reasoning behind it.

#### Project 2: Seed-Assisted Zeolite Synthesis Prediction

##### ReadMe
---

**Seed-Assisted Zeolite Synthesis Prediction Project**

**Author**

Author: Nga Vu

Instructor: Dr. Jude Okolie

University: Bucknell University

Department: Chemical Engineering


**Project Overview**

This project focuses on developing machine learning models to predict the success of seed-assisted zeolite synthesis experiments. The goal is to create a predictive model that can help determine whether a synthesis experiment will result in a pure zeolite phase (success) or undesired phases (failure).

**Dataset Description**

The dataset contains 385 historical records of seed-assisted zeolite synthesis experiments with the following features:

**Input Features**
1. Seed Properties:
   - Seed type
   - Seed amount (normalized to SiO2 weight = 1)
   - Seed framework density (FD) in T/Å3
   - Seed Si/Al molar ratio (measured using ICP-AES)

2. Gel Composition:
   - SiO2 (normalized to 1)
   - NaOH/SiO2 molar ratio
   - B2O3/SiO2 molar ratio
   - H2O/SiO2 molar ratio
   - OTMAC/SiO2 molar ratio (SDA)

3. Crystallization Conditions:
   - Crystallization temperature (°C)
   - Crystallization time (days)

**Target Variable**
- Class "0": Failed experiments (amorphous, mixed, dense, or layered phases)
- Class "1": Successful experiments (pure zeolite phase)

A**nalysis Steps**

1. Data Preprocessing:
   - Handling outliers
   - Feature scaling (StandardScaler for numerical features)
   - One-hot encoding for categorical variables

2. Correlation Analysis:
   - Created correlation heatmaps
   - Analyzed feature relationships with the target variable

3. Model Development:
   - Implemented multiple classification models:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Neural Network

4. Model Evaluation:
   - Performance metrics used:
     - Accuracy
     - Precision
     - Recall
     - F1 Score
     - AUC-ROC

5. Feature Importance Analysis:
   - Feature importance plots for tree-based models
   - Partial Dependence Plots (PDP) for understanding feature impacts

**Files Description**
- `data_preprocessing.py`: Code for data cleaning and preprocessing
- `model_training.py`: Implementation of machine learning models
- `evaluation.py`: Model evaluation and visualization code
- `requirements.txt`: List of required Python packages

**Results**

(Specific model performances and key findings can be added here)

**Requirements**

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

**Usage**

```
# Example code for running the analysis
python model_training.py
```

##### GUI App Deployment




---

```python
# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

def load_model():
    """Load the trained model and its metadata"""
    with open('zeolite_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def main():
    st.set_page_config(page_title="Zeolite Synthesis Predictor", layout="wide")

    # Add title and description
    st.title("🔮 Zeolite Synthesis Prediction")
    st.markdown("""
    This app predicts the success of zeolite synthesis experiments based on input parameters.

    **Prediction Classes:**
    - **Class 0**: Failed experiment (amorphous, mixed, dense, or layered phases)
    - **Class 1**: Successful experiment (pure zeolite phase)
    """)

    try:
        # Load model and metadata
        model_data = load_model()
        pipeline = model_data['pipeline']
        numerical_columns = model_data['numerical_columns']
        categorical_columns = model_data['categorical_columns']

        # Create two columns for input parameters
        col1, col2 = st.columns(2)

        # Dictionary to store user inputs
        input_data = {}

        with col1:
            st.subheader("Composition Parameters")
            input_data['SiO2'] = st.number_input(
                'SiO2 Amount',
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                help='Silicon dioxide content'
            )

            input_data['NaOH'] = st.number_input(
                'NaOH Amount',
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                help='Sodium hydroxide content'
            )

            input_data['SDA'] = st.number_input(
                'SDA Amount',
                min_value=0.0,
                max_value=5.0,
                value=0.2,
                help='Structure-Directing Agent content'
            )

            input_data['B2O3'] = st.number_input(
                'B2O3 Amount',
                min_value=0.0,
                max_value=2.0,
                value=0.1,
                help='Boron oxide content'
            )

            input_data['H2O'] = st.number_input(
                'H2O Amount',
                min_value=10.0,
                max_value=100.0,
                value=30.0,
                help='Water content'
            )

        with col2:
            st.subheader("Process Parameters")
            input_data['temperature\n(°C)'] = st.slider(
                'Temperature (°C)',
                min_value=100,
                max_value=200,
                value=150,
                help='Synthesis temperature'
            )

            input_data['seed \namount'] = st.number_input(
                'Seed Amount',
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                help='Amount of seed crystals'
            )

            input_data['fd'] = st.number_input(
                'Framework Density (FD)',
                min_value=10.0,
                max_value=25.0,
                value=18.0,
                help='Framework density of the target zeolite'
            )

            input_data['seed'] = st.selectbox(
                'Seed Type',
                options=['Type A', 'Type B', 'None'],
                help='Type of seed crystals used'
            )

            input_data['si/al\n(ICP-AES)'] = st.selectbox(
                'Si/Al Ratio',
                options=['infy', '10', '30', '29', '15'],
                help='Silicon to Aluminum ratio (infy represents infinite ratio)'
            )

        # Add a predict button
        if st.button('Predict Synthesis Outcome'):
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])

            # Make prediction
            prediction = pipeline.predict(input_df)
            probability = pipeline.predict_proba(input_df)

            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")

            # Create three columns for the results
            result_col1, result_col2, result_col3 = st.columns([2,2,1])

            with result_col1:
                if prediction[0] == 1:
                    st.success("🎯 Predicted Outcome: SUCCESSFUL")
                    st.write("The synthesis is predicted to result in a pure zeolite phase.")
                else:
                    st.error("❌ Predicted Outcome: FAILED")
                    st.write("The synthesis is predicted to result in amorphous, mixed, dense, or layered phases.")

            with result_col2:
                st.write("### Prediction Confidence")
                confidence = probability[0][prediction[0]]
                st.progress(confidence)
                st.write(f"Confidence: {confidence:.2%}")

            with result_col3:
                st.write("### Probabilities")
                st.write("Failure (0):", f"{probability[0][0]:.2%}")
                st.write("Success (1):", f"{probability[0][1]:.2%}")

    except FileNotFoundError:
        st.error("Error: Model file 'zeolite_model.pkl' not found!")
        st.write("Please ensure the model file is in the same directory as this application.")

    # Add footer with additional information
    st.markdown("---")
    st.markdown("""
    **Note:** This is a machine learning model prediction and should be used as a guide only.
    The actual synthesis outcome may vary based on other experimental conditions and factors
    not included in this prediction model.
    """)

if __name__ == '__main__':
    main()
```

##### Machine Learning Development




---

```python
#This script showing all the colume name, some example of the value in the column and type of the value in the colunn (numerical or categorical value)

import pandas as pd

# Replace 'your_file.xlsx' with the path to your Excel file
file_path = 'Dataset_3_zeolites.xlsx'

# Read the Excel file
df = pd.read_excel(file_path)

# Print the first 5 rows of the dataset
print(df.head())
print("Column names and data types:")
for column, dtype in df.dtypes.items():
    print(f"{column}: {dtype}")
```

```python
import numpy as np

print("Missing data information:")
missing_data = df.isnull().sum()
missing_percent = 100 * df.isnull().sum() / len(df)
missing_table = pd.concat([missing_data, missing_percent], axis=1, keys=['Total Missing', 'Percent Missing'])
total_rows = len(df)

for column, missing_count in missing_data.items():
    print(f"{column}: {missing_count} missing values ({missing_count/total_rows:.2%})")
if missing_data.sum() == 0:
    print("No missing data found in the dataset.")
else:
    print(f"\nTotal number of missing values: {missing_data.sum()}")
    print(f"Percentage of missing data: {missing_data.sum() / np.product(df.shape):.2%}")
```

No missing data deteched

```python
# Check for duplicates in the entire dataset
duplicates = df.duplicated()
# If there are any duplicates, the 'duplicates' variable will contain True for those rows
if duplicates.any():
    # Get the rows with duplicates
    duplicate_rows = df[duplicates]
else:
    print("No duplicates found in the dataset.")

duplicate_rows
```

Found duplicate row => delete

```python
# drop duplicates
df_clean = df.drop_duplicates()
```

```python
import scipy.stats as stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# This script collect al the rows with outliers data
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
print(numeric_columns)

rows_to_remove = set()

for column in numeric_columns:
    print(f"\nAnalyzing outliers in {column}:")

    # Calculate Z-score
    z_scores = np.abs(stats.zscore(df_clean[column]))

    # Identify outliers (Z-score > 3)
    outliers = df_clean[z_scores > 3]

    print(f"Number of outliers: {len(outliers)}")
    if len(outliers) > 0:
        print("Outlier values:")
        print(outliers[column])
        rows_to_remove.update(outliers.index)

#Box plot
for column in numeric_columns:

    plt.figure(figsize=(8,6))
    sns.boxplot(x=df_clean[column], color='cyan')
    plt.title(f"Boxplot of {column}")
    plt.xlabel('value')
    plt.show()
```

```python
# Your existing code remains the same, then add:

total_outliers = sum(len(df_clean[np.abs(stats.zscore(df_clean[column])) > 3]) for column in numeric_columns)
print(f"\nTotal number of outliers across all columns: {total_outliers}")
print(f"Number of rows containing at least one outlier: {len(rows_to_remove)}")
```

```python
# Create new dataframe without outliers
df_clean_2 = df_clean.drop(index=rows_to_remove)

# Print shape information to verify the removal
print(f"Original dataframe shape: {df_clean.shape}")
print(f"New dataframe shape (without outliers): {df_clean_2.shape}")
print(f"Number of rows removed: {len(rows_to_remove)}")
```

```python
# Method 1: Find the exact column name first
si_al_column = df_clean_2.columns[df_clean_2.columns.str.contains('si/al', case=False, na=False)][0]
print("Column name containing 'si/al':", si_al_column)
print("\nUnique values:", df_clean_2[si_al_column].unique())

# Or Method 2: If you know the exact column index
print("\nAll column names:", df_clean_2.columns)
# Then use the index, for example if it's the 9th column (index 8):
print("\nUnique values using index:", df_clean_2.iloc[:, 8].unique())
```

```python
# Import required libraries if not already imported
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Calculate correlations
correlation_matrix = df_clean.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix,
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme: red for positive, blue for negative correlations
            center=0,  # Center the colormap at 0
            fmt='.2f',  # Show 2 decimal places
            square=True,  # Make the plot square-shaped
            linewidths=0.5)  # Add lines between cells

plt.title('Correlation Heatmap of Features')
plt.tight_layout()
plt.show()

# If you want to see the strongest correlations, you can print them:
print("\nStrongest Correlations:")
# Get the upper triangle of correlations
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
# Find strongest correlations
strongest_correlations = upper.unstack()
sorted_correlations = strongest_correlations.sort_values(key=abs, ascending=False)
print(sorted_correlations.head(10))  # Print top 10 strongest correlations
```

```python
# Create a copy of your dataframe to avoid modifying the original
df_corr = df_clean_2.copy()

# Identify categorical columns (object dtype)
categorical_columns = df_corr.select_dtypes(include=['object']).columns

# One-hot encode all categorical columns at once
df_corr = pd.get_dummies(df_corr, columns=categorical_columns)

# Create correlation matrix and plot
plt.figure(figsize=(15, 12))
sns.heatmap(df_corr.corr(),
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            square=True,
            linewidths=0.5)

plt.title('Correlation Heatmap (with all encoded categorical variables)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Print strongest correlations
corr_matrix = df_corr.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
strongest_correlations = upper.unstack()
print("\nStrongest Correlations:")
print(strongest_correlations.sort_values(key=abs, ascending=False).head(10))
```

Do a correlation heat map with respect to class - the thing we are trying to predict

```python
# Create a copy of your dataframe to avoid modifying the original
df_corr = df_clean_2.copy()

# Identify categorical columns (object dtype)
categorical_columns = df_corr.select_dtypes(include=['object']).columns

# One-hot encode all categorical columns at once
df_corr = pd.get_dummies(df_corr, columns=categorical_columns)

# Calculate correlations
correlations = df_corr.corr()['class'].sort_values(ascending=False)

# Remove class's correlation with itself
correlations = correlations.drop('class')

# Create a figure
plt.figure(figsize=(10, 8))

# Create heatmap with only class correlations
sns.heatmap(correlations.to_frame(),
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            linewidths=0.5)

plt.title('Feature Correlations with Class')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()

# Print correlations
print("\nCorrelations with Class (sorted):")
print(correlations)
```

```python
df.describe()
```

```python
df_clean_2['si/al\n(ICP-AES)'].unique()
```

```python
# Import necessary libraries for classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# First convert categorical columns to string type
df_clean_2 = df_clean_2.copy()
for col in categorical_columns:
    df_clean_2[col] = df_clean_2[col].astype(str)

# Define your feature columns based on the actual column names in your dataframe
numerical_columns = ['SiO2', 'NaOH', 'SDA', 'B2O3', 'H2O', 'temperature\n(°C)', 'seed \namount', 'fd']  # Adjusted column names

categorical_columns = ['seed', 'si/al\n(ICP-AES)']  # Adjusted column names
target_column = 'class'

def train_classification_models(df, numerical_columns, categorical_columns, target_column):
    # Create preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Rest of the code stays the same...
    X = df[numerical_columns + categorical_columns]
    y = df[target_column]

    # Initialize classification models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
    }

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # Calculate metrics for training set
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_auc = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])

        # Calculate metrics for test set
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

        # Print results
        print(f"\n{name} Results:")
        print("Training Metrics:")
        print(f"  Accuracy: {train_accuracy:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  Recall: {train_recall:.4f}")
        print(f"  F1 Score: {train_f1:.4f}")
        print(f"  AUC-ROC: {train_auc:.4f}")

        print("\nTest Metrics:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")
        print(f"  AUC-ROC: {test_auc:.4f}")

        # Feature Importance plots (if available)
        if hasattr(pipeline['classifier'], 'feature_importances_'):
            feature_importance = pipeline['classifier'].feature_importances_
            feature_names = pipeline['preprocessor'].get_feature_names_out()

            plt.figure(figsize=(10, 6))
            sorted_idx = feature_importance.argsort()
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title(f'{name} - Feature Importance')
            plt.tight_layout()
            plt.show()

    return results

# Run the function
results = train_classification_models(df_clean_2, numerical_columns, categorical_columns, target_column)
```

PDP plot for numerical value

```python
# Import necessary libraries for classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence, plot_partial_dependence
import matplotlib.pyplot as plt
import numpy as np

# First convert categorical columns to string type
df_clean_2 = df_clean_2.copy()
for col in categorical_columns:
    df_clean_2[col] = df_clean_2[col].astype(str)

def create_pdp_plots(pipeline, X, feature_names, model_name, numerical_columns, categorical_columns):
    """
    Create partial dependence plots for all features, handling both numerical and categorical features.

    Parameters:
    -----------
    pipeline : sklearn.Pipeline
        Fitted pipeline containing preprocessor and classifier
    X : pandas.DataFrame
        Feature dataset
    feature_names : list
        List of feature names
    model_name : str
        Name of the model for plot titles
    numerical_columns : list
        List of numerical column names
    categorical_columns : list
        List of categorical column names
    """
    # Calculate number of rows needed for subplots
    n_features = len(numerical_columns) + len(categorical_columns)
    n_rows = (n_features + 1) // 2  # 2 plots per row, rounded up

    # Create PDP plots
    fig, axes = plt.subplots(nrows=n_rows, ncols=2,
                            figsize=(15, 5 * n_rows))
    axes = axes.ravel()

    plot_idx = 0

    # Handle numerical features
    for col in numerical_columns:
        feature_idx = [i for i, name in enumerate(feature_names) if f'num__{col}' in name][0]
        try:
            pdp = partial_dependence(pipeline, X, [feature_idx])

            axes[plot_idx].plot(pdp[1][0], pdp[0][0])
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel('Partial dependence')
            axes[plot_idx].grid(True)
            axes[plot_idx].tick_params(axis='x', rotation=45)
            plot_idx += 1
        except Exception as e:
            print(f"Warning: Could not create PDP plot for numerical feature {col}: {str(e)}")

    # Handle categorical features
    for col in categorical_columns:
        # Get all feature indices for this categorical variable (one-hot encoded columns)
        cat_feature_indices = [i for i, name in enumerate(feature_names) if f'cat__{col}' in name]
        try:
            # Create separate lines for each category
            cat_values = [name.split('_')[-1] for name in feature_names if f'cat__{col}' in name]
            pdp_values = []

            for idx in cat_feature_indices:
                pdp = partial_dependence(pipeline, X, [idx])
                pdp_values.append(pdp[0][0][0])  # Get the PDP value for this category

            # Create bar plot for categorical feature
            axes[plot_idx].bar(range(len(cat_values)), pdp_values)
            axes[plot_idx].set_xticks(range(len(cat_values)))
            axes[plot_idx].set_xticklabels(cat_values, rotation=45)
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel('Partial dependence')
            axes[plot_idx].grid(True, axis='y')
            plot_idx += 1

        except Exception as e:
            print(f"Warning: Could not create PDP plot for categorical feature {col}: {str(e)}")

    # Remove any empty subplots
    for idx in range(plot_idx, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f'Partial Dependence Plots for {model_name}')
    plt.tight_layout()
    plt.show()

def train_classification_models(df, numerical_columns, categorical_columns, target_column):
    # Create preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    X = df[numerical_columns + categorical_columns]
    y = df[target_column]

    # Initialize classification models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
    }

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # Calculate metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'auc': roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])
        }

        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'auc': roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        }

        # Store results
        results[name] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'pipeline': pipeline
        }

        # Print results
        print(f"\n{name} Results:")
        print("Training Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")

        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")

        # Feature Importance plots (if available)
        if hasattr(pipeline['classifier'], 'feature_importances_'):
            feature_importance = pipeline['classifier'].feature_importances_
            feature_names = pipeline['preprocessor'].get_feature_names_out()

            plt.figure(figsize=(10, 6))
            sorted_idx = feature_importance.argsort()
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title(f'{name} - Feature Importance')
            plt.tight_layout()
            plt.show()

        # Create Partial Dependence Plots
        feature_names = pipeline['preprocessor'].get_feature_names_out()
        create_pdp_plots(pipeline, X_train, feature_names, name, numerical_columns, categorical_columns)

    return results

# Run the function
results = train_classification_models(df_clean_2, numerical_columns, categorical_columns, target_column)
```

PDP plot for best 5

```python
# Import necessary libraries for classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence, plot_partial_dependence
import matplotlib.pyplot as plt
import numpy as np

# First convert categorical columns to string type
df_clean_2 = df_clean_2.copy()
for col in categorical_columns:
    df_clean_2[col] = df_clean_2[col].astype(str)

def create_pdp_plots(pipeline, X, feature_names, model_name, num_features_to_plot=5):
    """
    Create partial dependence plots for the most important features.

    Parameters:
    -----------
    pipeline : sklearn.Pipeline
        Fitted pipeline containing preprocessor and classifier
    X : pandas.DataFrame
        Feature dataset
    feature_names : list
        List of feature names
    model_name : str
        Name of the model for plot titles
    num_features_to_plot : int
        Number of top features to create PDP plots for
    """
    # Get feature importance if available
    if hasattr(pipeline['classifier'], 'feature_importances_'):
        importance = pipeline['classifier'].feature_importances_
        # Get indices of top features
        top_features_idx = np.argsort(importance)[-num_features_to_plot:]
        top_features = [feature_names[i] for i in top_features_idx]
    else:
        # If feature importance is not available, use first num_features_to_plot features
        top_features = feature_names[:num_features_to_plot]

    # Create PDP plots
    fig, axes = plt.subplots(nrows=((len(top_features) + 1) // 2), ncols=2,
                            figsize=(12, 4 * ((len(top_features) + 1) // 2)))
    axes = axes.ravel()

    for idx, feature in enumerate(top_features):
        # Calculate partial dependence
        pdp = partial_dependence(pipeline, X, [feature_names.tolist().index(feature)])

        # Plot
        axes[idx].plot(pdp[1][0], pdp[0][0])
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Partial dependence')
        axes[idx].grid(True)

    # Remove any empty subplots
    for idx in range(len(top_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f'Partial Dependence Plots for {model_name}')
    plt.tight_layout()
    plt.show()

def train_classification_models(df, numerical_columns, categorical_columns, target_column):
    # Create preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    X = df[numerical_columns + categorical_columns]
    y = df[target_column]

    # Initialize classification models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
    }

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # Calculate metrics for training set
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_auc = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])

        # Calculate metrics for test set
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

        # Print results
        print(f"\n{name} Results:")
        print("Training Metrics:")
        print(f"  Accuracy: {train_accuracy:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  Recall: {train_recall:.4f}")
        print(f"  F1 Score: {train_f1:.4f}")
        print(f"  AUC-ROC: {train_auc:.4f}")

        print("\nTest Metrics:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")
        print(f"  AUC-ROC: {test_auc:.4f}")

        # Feature Importance plots (if available)
        if hasattr(pipeline['classifier'], 'feature_importances_'):
            feature_importance = pipeline['classifier'].feature_importances_
            feature_names = pipeline['preprocessor'].get_feature_names_out()

            plt.figure(figsize=(10, 6))
            sorted_idx = feature_importance.argsort()
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title(f'{name} - Feature Importance')
            plt.tight_layout()
            plt.show()

        # Create Partial Dependence Plots
        feature_names = pipeline['preprocessor'].get_feature_names_out()
        create_pdp_plots(pipeline, X_train, feature_names, name)

    return results

# Run the function
results = train_classification_models(df_clean_2, numerical_columns, categorical_columns, target_column)
```

```python
# Import necessary libraries for classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import pandas as pd

# Define your feature columns
numerical_columns = ['SiO2', 'NaOH', 'SDA', 'B2O3', 'H2O', 'temperature\n(°C)', 'seed \namount', 'fd']
categorical_columns = ['seed', 'si/al\n(ICP-AES)']
target_column = 'class'

def train_classification_models(df, numerical_columns, categorical_columns, target_column):
    # First convert categorical columns to string type
    df = df.copy()
    for col in categorical_columns:
        df[col] = df[col].astype(str)

    # Create preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    X = df[numerical_columns + categorical_columns]
    y = df[target_column]

    # Initialize classification models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
    }

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Track best model
    best_f1 = 0
    best_model = None
    best_model_name = None
    best_model_metrics = None

    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # Calculate metrics for training set
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_auc = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])

        # Calculate metrics for test set
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

        # Update best model if current one is better
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model = pipeline
            best_model_name = name
            best_model_metrics = {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1,
                'auc_roc': test_auc
            }

        # Print results
        print(f"\n{name} Results:")
        print("Training Metrics:")
        print(f"  Accuracy: {train_accuracy:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  Recall: {train_recall:.4f}")
        print(f"  F1 Score: {train_f1:.4f}")
        print(f"  AUC-ROC: {train_auc:.4f}")

        print("\nTest Metrics:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")
        print(f"  AUC-ROC: {test_auc:.4f}")

        # Feature Importance plots (if available)
        if hasattr(pipeline['classifier'], 'feature_importances_'):
            feature_importance = pipeline['classifier'].feature_importances_
            feature_names = pipeline['preprocessor'].get_feature_names_out()

            plt.figure(figsize=(10, 6))
            sorted_idx = feature_importance.argsort()
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title(f'{name} - Feature Importance')
            plt.tight_layout()
            plt.show()

    # Print best model details
    print(f"\nBest performing model: {best_model_name}")
    print("Best model metrics:")
    for metric, value in best_model_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save model information
    model_info = {
        'pipeline': best_model,
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns,
        'model_name': best_model_name,
        'metrics': best_model_metrics
    }

    # Export model
    with open('zeolite_model.pkl', 'wb') as f:
        pickle.dump(model_info, f)

    print("\nBest model exported as 'zeolite_model.pkl'")

    return best_model

# Train models and export best one
best_model = train_classification_models(df_clean_2, numerical_columns, categorical_columns, target_column)
```

##### Ethics Analysis




---

#### Project 3: Removing Pharmaceutical Pollutants

##### ReadMe
---

Author: Brandon Roman

Professor/Supervisor: Prof. Jude Okolie

This project was completed while in attendance at Bucknell University

This is the Machine Learning Project for my Data Analysis Class

This project handled the prediction of the effectiveness of different organic compounds at removing pharmaceutical pollutants

The features are the type of pharmaceutical tat we wish to remove, and the composition of the biomaterials that are used to remove them. The target for this model is the removal capacity of the biomaterial.

The data was obtained from https://github.com/J4RELY/IGUIDEpharmEDA/tree/main. A project that my professor has worked on in the past.

AI was used to write some of the code and solve errors.

GUI:
The GUI for the prediction was developed using Huggingface and is accessible at the link below.

Prediction Interface: https://huggingface.co/spaces/asafuM/Pollution-Removal-Prediction

Ethics:
The ethics analysis for the project are in the following checklists

DEON Ethics Checklist: https://huggingface.co/spaces/asafuM/Deon-Ethics-for-Pollution

Ethics DataCard: https://huggingface.co/spaces/asafuM/DataCard-for-Pollution

##### GUI App Deployment




---

```python
import subprocess
import sys

# Force install scikit-learn if not found
try:
    import sklearn
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import sklearn  # Import again after installation

import gradio as gr
import pandas as pd
import pickle

# Load the pre-trained model
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict_forest_fire(temperature, time, ps, bet, pv, carbon, hydrogen, nitrogen, oxygen, tp):
    # Creating input DataFrame for the model
    input_data = pd.DataFrame({
        'TemP (K)': [temperature],
        'Time (min)': [time],
        'PS (mm)': [ps],
        'BET (m2/g)': [bet],
        'PV (cm3)': [pv],
        'C (wt.%)': [carbon],
        'H  (wt.%)': [hydrogen],
        'N  (wt.%)': [nitrogen],
        'O  (wt.%)': [oxygen],
        'TP': [tp]
    })

    # One-hot encode the input data (ensure it matches the training data)
    input_encoded = pd.get_dummies(input_data)

    # Align columns with the training data (required columns)
    required_columns = model.feature_names_in_  # Get the feature columns from the model
    for col in required_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[required_columns]

    # Make the prediction
    prediction = model.predict(input_encoded)[0]

    return prediction

# Gradio Interface using components
interface = gr.Interface(
    fn=predict_forest_fire,
    inputs=[
        gr.Slider(minimum=300, maximum=800, step=5, label="TemP (K)"),
        gr.Slider(minimum=0.0, maximum=100.0, step=0.5, label="Time (min)"),
        gr.Slider(minimum=0.0, maximum=300, step=1, label="BET (m2/g)"),
        gr.Slider(minimum=0.0, maximum=100.0, step=0.5, label="PV (cm3)"),
        gr.Slider(minimum=0.0, maximum=100.0, step=0.5, label="C (wt.%)"),
        gr.Slider(minimum=0.0, maximum=100.0, step=0.5, label="N  (wt.%)"),
        gr.Slider(minimum=0.0, maximum=100.0, step=0.5, label="O  (wt.%)"),
        gr.Dropdown(['Benzocaine', 'Ciprofloxacin', 'Citalopram', 'Diclofenac', 'Dimetridazole', 'Floxentine',
                     'Metronidazole', 'Nitroimidazole', 'Norflozacin', 'Oxytetracycline', 'Salicylic acid',
                     'Sulfadiazine', 'Sulfamethoxazole', 'Tetracycline', 'Triclosan', 'ibuprofen', 'Sulfamethazine'], label="TP")
    ],
    outputs=gr.Textbox(label="Pharmaceutical Pollution Removal Prediction"),
    title="Pharmaceutical Pollution Removal Prediction"
)

if __name__ == "__main__":
    interface.launch()
```

##### Machine Learning Development




---

```python
# Import necessary libraries
!pip install shap -q

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
import joblib
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Import filters to remove unnecessary warnings
from warnings import simplefilter
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Import filters to remove unnecessary warnings
from warnings import simplefilter
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error
import shap

#Load the dataset
actual_path = '/content/updated_DATASET (1).xlsx'
synth_path = '/content/synthetic_data.csv'

# Read the files into a Pandas DataFrame
act = pd.read_excel(actual_path)
synth = pd.read_csv(synth_path)

#Combine the datasets
df = pd.concat([act, synth], ignore_index=True)

print(df.head())  # Print the first few rows of the DataFrame
```

From previous iterations of this project, will drop H (%) and PS (mm) due to their low impact on Qm (mg/g)

```python
df = df.drop(['H  (wt.%)', 'PS (mm)'], axis = 1)

print(df.head())
```

Missing Data

```python
#Check for missing data
print(df.isna().sum())

#Check for the number of rows and column
num_rows, num_cols = df.shape

print('Number of Rows = ', num_rows)
print('Number of Columns = ', num_cols)
```

No Missing Data so moving on to outliers

Excluding All TP Columns from the outlier search

```python
outlier_cols = ['TemP (K)', 'Time (min)', 'BET (m2/g)', 'PV (cm3)', 'C (wt.%)', 'N  (wt.%)', 'O  (wt.%)', 'Qm (mg/g)']
df[outlier_cols] = df[outlier_cols].astype('float32')
```

Handling Outliers

```python
#Looking for outliers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the interquartile range (IQR) for each column
Q1 = df[outlier_cols].quantile(0.25)
Q3 = df[outlier_cols].quantile(0.75)
IQR = Q3 - Q1

# Define the outlier threshold
outlier_threshold = 1.5 * IQR

# Identify outliers
outliers = (df[outlier_cols] < (Q1 - outlier_threshold)) | (df[outlier_cols] > (Q3 + outlier_threshold))

# Create a figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# Create the boxplot
sns.boxplot(data=df[outlier_cols], ax=ax)

# Set labels and title
ax.set_title('Boxplots for Selected Columns')
ax.set_xlabel('Columns')
ax.set_ylabel('Values')

plt.show()

# Check if outliers exist
if outliers.any().any():
    print("Outliers detected in the following columns:")
    print(outliers.sum())
else:
    print("No outliers detected.")
```

Drop rows with outliers

```python
# Calculate the interquartile range (IQR) for each column
Q1 = df[outlier_cols].quantile(0.25)
Q3 = df[outlier_cols].quantile(0.75)
IQR = Q3 - Q1

# Define the outlier threshold
outlier_threshold = 2 * IQR

# Identify outliers
outliers = (df[outlier_cols] < (Q1 - outlier_threshold)) | (df[outlier_cols] > (Q3 + outlier_threshold))

# Drop rows containing outliers
df = df[~outliers.any(axis=1)]

#Re-identifying the outliers for display purposes
outliers = (df[outlier_cols] < (Q1 - outlier_threshold)) | (df[outlier_cols] > (Q3 + outlier_threshold))

# Check if outliers exist
if outliers.any().any():
    print("Outliers detected in the following columns:")
    print(outliers.sum())
else:
    print("No outliers detected.")
```

Check for duplicates

```python
# Check for duplicates in the entire dataset
duplicates = df.duplicated()
# If there are any duplicates, the 'duplicates' variable will contain True for those rows
if duplicates.any():
    # Get the rows with duplicates
    duplicate_rows = df[duplicates]
else:
    print("No duplicates found in the dataset.")
```

Make a heatmap

```python
#Redefining cause I have to
outlier_cols = ['TemP (K)', 'Time (min)', 'BET (m2/g)', 'PV (cm3)', 'C (wt.%)', 'N  (wt.%)', 'O  (wt.%)', 'Qm (mg/g)']
df[outlier_cols] = df[outlier_cols].astype('float32')
corr_df = df[outlier_cols]

#Make the correlation matrix
correlation_matrix = corr_df.corr()

#Making the heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap="viridis")

# Set the title and labels
plt.title("Correlation Heatmap")
plt.xlabel("Variables")
plt.ylabel("Variables")

# Show the plot
plt.show()
```

No variable appears so correlated that any of the data is redundant

Training the models

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Split the dataset into features and target
X = df.drop('Qm (mg/g)', axis=1)
Y = df['Qm (mg/g)']

# Define the number of folds for K-Fold cross-validation
n_folds = 6

# Initialize empty lists to store evaluation metrics
rmse_scores = []
mae_scores = []
r2_test_scores = []
r2_train_scores = []

# Define the models dictionary
models = {
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'MLP': MLPRegressor(),
    'SVR': SVR(),
    'Linear Regression': LinearRegression(),
}

# K-Fold cross-validation loop
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
for name, model in models.items():
    # Loop through each fold
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        # Train the model on the training data for this fold
        model.fit(X_train, y_train)

        # Predict on the testing data for this fold
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        rmse = mean_squared_error(y_test, y_pred, squared=False)  # Calculate RMSE directly
        mae = mean_absolute_error(y_test, y_pred)
        r2_test = r2_score(y_test, y_pred)

        # Additionally, calculate R-squared on the training data for each fold (optional)
        y_train_pred = model.predict(X_train)
        r2_train = r2_score(y_train, y_train_pred)

        # Calculate R^2 difference
        r2_diff = r2_train - r2_test

        # Append the scores to the lists
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_test_scores.append(r2_test)
        r2_train_scores.append(r2_train)

    # Print average scores after all folds for each model
    print(f"{name}:")
    print(f"  Average RMSE: {np.mean(rmse_scores):.3f}")
    print(f"  Average MAE: {np.mean(mae_scores):.3f}")
    print(f"  Average R² Test Score: {np.mean(r2_test_scores):.3f}")
    print(f"  Average R² Train Score: {np.mean(r2_train_scores):.3f}")
    print(' ')

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Model': ['Random Forest', 'Random Forest', 'Random Forest', 'Random Forest', 'Random Forest', 'Random Forest',
              'Gradient Boosting', 'Gradient Boosting', 'Gradient Boosting', 'Gradient Boosting', 'Gradient Boosting', 'Gradient Boosting',
              'MLP', 'MLP', 'MLP', 'MLP', 'MLP', 'MLP',
              'SVR', 'SVR', 'SVR', 'SVR', 'SVR', 'SVR',
              'Linear Regression', 'Linear Regression', 'Linear Regression', 'Linear Regression', 'Linear Regression', 'Linear Regression'],
    'R^2 Test': r2_test_scores,
    'R^2 Train': r2_train_scores,
})

# Melt the DataFrame for easier plotting
melted_df = results_df.melt(id_vars='Model', var_name='Metric', value_name='R^2')

# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='R^2', hue='Metric', data=melted_df)
plt.title('R^2 Scores for Different Models')
plt.xlabel('Model')
plt.ylabel('R^2')
plt.legend(title='Metric')
plt.show()
```

Based on the data I am going to choose to optimize the parameters of the Random Forest Model

Hyperparameter tuning using Bayesian Optimization

```python
!pip install scikit-optimize -q
from skopt import BayesSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Define the hyperparameter search space
param_dist = {
    'n_estimators': np.arange(50, 201, 50),
    'learning_rate': np.linspace(0.01, 0.3, 10),
    'max_depth': np.arange(3, 11, 2),
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 11),
    'subsample': np.linspace(0.6, 1.0, 5)
}

# Create a Bayesian Optimization object
opt = BayesSearchCV(
    GradientBoostingRegressor(),
    param_grid,
    n_iter=10,  # Number of iterations is lower for the sake of time
    cv=6,  # Cross-validation folds
    random_state=42
)

# Fit the optimizer to your data
opt.fit(X_train, y_train)

# Print the best hyperparameters and score
print("Best parameters:", opt.best_params_)
print("Best score:", opt.best_score_)
```

Hyperparameter Tuning using Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define the parameter grid
param_dist = {
    'n_estimators': np.arange(50, 201, 50),
    'learning_rate': np.linspace(0.01, 0.3, 10),
    'max_depth': np.arange(3, 11, 2),
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 11),
    'subsample': np.linspace(0.6, 1.0, 5)
}

# Create a Grid Search object
grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=5)

# Fit the Grid Search object to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

Hyperparameter tuning using Random Search

```python
from sklearn.model_selection import RandomizedSearchCV

# Define the parameter distribution
param_dist = {
    'n_estimators': np.arange(50, 201, 50),
    'learning_rate': np.linspace(0.01, 0.3, 10),
    'max_depth': np.arange(3, 11, 2),
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 11),
    'subsample': np.linspace(0.6, 1.0, 5)
}

# Create a Random Search object
random_search = RandomizedSearchCV(GradientBoostingRegressor(), param_grid, n_iter=30, cv=5)

# Fit the Random Search object to the data
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
```

Using the optimal parameters given by tunning

```python
# Define the number of folds for K-Fold cross-validation
n_folds = 10  # You can adjust this value based on your needs

# K-Fold cross-validation loop
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Assuming you've already performed hyperparameter tuning and obtained the best parameters
best_params = {
    'n_estimators': 100,
    'learning_rate': 0.3,
    'max_depth': 3,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'subsample': 0.9
}

# Create a Random Forest Regressor with the best parameters
gb_model = GradientBoostingRegressor(**best_params)

# Loop through each fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Train the model on the training data for this fold
    gb_model.fit(X_train, y_train)

    # Predict on the testing data for this fold
    y_pred = gb_model.predict(X_test)

    # Calculate evaluation metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Calculate RMSE directly
    mae = mean_absolute_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)

    # Additionally, calculate R-squared on the training data for each fold (optional)
    y_train_pred = gb_model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)

    # Append the scores to the lists
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_test_scores.append(r2_test)
    r2_train_scores.append(r2_train)

# Print average scores after all folds for each model
print(f"Gradient Boost after Tuning:")
print(f"  Average RMSE: {np.mean(rmse_scores):.3f}")
print(f"  Average MAE: {np.mean(mae_scores):.3f}")
print(f"  Average R² Test Score: {np.mean(r2_test_scores):.3f}")
print(f"  Average R² Train Score: {np.mean(r2_train_scores):.3f}")
print(' ')
```

Saving the model

```python
import pickle

# Save the best model as a pickel
with open('best_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)
```

Interpretability plot

```python
def plot_shap(data, target_column, model):
    x = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    # Train and evaluate the model
    model.fit(X_train, y_train)  # Train with scaled data
    # SHAP summary plot
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train,
                    plot_size= (10,7))

# define model to use
plot_shap(df,'Qm (mg/g)',gb_model)
```

##### Ethics Analysis




---

```python
# Customized Deon Ethics Checklist for Forest Wildfire Prediction Model
import gradio as gr

md_content = """
# Ethics Checklist for Pharmaceutical Pollution Removal Prediction Model
## 1. Data Collection
- Are the data sources properly licensed and legally available?
**A large portion of the data is machine generated, but the parts of the data that is not is available on GitHub and came from literature**
- Has any sensitive information, such as private property or personal location data, been anonymized?
**No personal data was collected for this project and as such this checklist item does not apply **
- Have you obtained consent for data collected from private or proprietary sources, such as satellite imagery or drone footage?
**As the data is publically available, this part is unnecessary**
## 2. Fairness & Justice
- Does the fata or algorithm introduce any biases that could lead to discrimination?
**The dataset has disproportionate amounts of data on each pharmaceutical type, so it may be better at predicting removal for specific drugs**
- Are the outcomes fair and unbiased across different demographic groups?
**As far as I am aware, the outcomes are fair and unbiased across demographics
## 3. Transparency
- How will you ensure transparency about the data sources, algorithms, and decision-making process of the model?
**While I will not display my code, I will explain the weaknesses of the model, as well as explain the methodology behind the model**
- How will you explain false positives and false negatives to the affected communities or stakeholders?
**I would explain ahead of time that the model can never be perfect, but is still accurate, so false positives and negatives aren't expected but don't come out of the blue.**
## 4. Privacy
- Are measures in place to protect the privacy and confidentiality of individuals' data?
**No individual's data has been collected for this project**
- Are there safeguards against unauthorized access or data breaches?
**The data is publically available, so this is not a concern**
## 5. Accountability
- Is there a clear accountability structure for ethics concerns?
**I will take full accountability if any arise since I don't believe there are any ethicss concernes in the data**
- Can the results of the analysis be audited for ethical and methodological rigor?
**Yes**
## 6. Purpose
- Is the purpose of the data analysis justifiable, and does it align with ethical goals?
**Yes, increasing the effectiveness of pollution removal benefits society**
- Could this project harm or benefit society
**This project will certainly benefit society**
## 7. Impact on Stakeholders
- Who could be impacted by the project, both positively and negatively?
**I worry that this could cause some pharmaceutical companies to become complacent with their disposal methods if the removal is predicted to be nearly perfect, however removing the pollution will benefit many impacted communities**
- Are vulnerable populations considered in the project design and analysis?
**Yes since the model can be applied regardless of demographic, they will be able to benefit from the data**
"""

def display_markdown():
    return md_content

# Create a Gradio interface
iface = gr.Interface(fn=display_markdown, inputs=[], outputs="markdown")
iface.launch()
```

```python
import gradio as gr

# Ethics DataCard content for forest wildfire prediction model
datacard_content = """
# Ethics DataCard for Forest Wildfire Prediction Model
## Data Collection Process
- Data is collected from public and licensed sources, ensuring proper consent and anonymization of any private information (e.g., personal property location).
## Fairness & Justice
- The model has been trained to predict the removal of pharmaceutical drugs whic is useful to all demographics.
- Special attention is given to reducing false positives (overpredicting removal) and false negatives (failure to predict sufficient removal), balancing the risk for all stakeholders.
## Privacy and Security
- No personal data was used in the project
- All data is public
## Bias and Fairness Checks
- There is potential bias in the data between the models effectiveness at removing certain drugs.
- The model's outputs analyze pollution removal, which can be applied to all groups
## Sustainability and Environmental Impact
- The model aims to assist in cleaning up pharmaceutical waste.
- It supports long-term environmental sustainability by informing decisions around waste treatment for pharmacetuical companies and environmenal agencies.
## Model Limitations
- The model's accuracy may vary depending on the drug being removed and the quality of data available.
- There are limitations to predicting removal of drugs with insufficient historical data, leading to potential inaccuracies.
- The model is regularly updated to incorporate new drug removal data.
## Accountability and Transparency
- The development team will monitor the model for performance over time, ensuring that it adapts to new data and environmental shifts.
- Stakeholderswill be informed of the model’s limitations, ensuring proper interpretation of the predictions.
- False predictions will be communicated to stakeholders, with a process in place for continuous feedback and model improvement.
## Societal Impact
- The model is designed to protect both human lives and the environment by enabling better planning and response to pollution incidents.
- It has the potential to inform policy changes in waste management for pharmaceutics
"""

# Function to display the DataCard
def display_datacard():
    return datacard_content

# Gradio interface to display the ethics DataCard
iface = gr.Interface(fn=display_datacard, inputs=[], outputs="markdown")

# Launch the Gradio interface
iface.launch()
```

#### Project 4: Coffee Freshness Prediction

##### ReadMe
---

**Freshness ML Model**

A repository used to hold a ML model made to analyze the freshness of coffee at it ages.

**Research Overview and Purpose**

**Purpose:**

As coffee ages, it loses key physical and chemical characteristics that are responsible for how it tastes fresh. The trend of how coffee ages (model for a rate of decay), as well as
what specifically changes when coffee ages, is currently unknown. The goal of this project is to create a machine learning platform that can analyze the data collected of the chemical
and physical traits of coffee and predict the age of the sample. This model can be used by roasters to help predict how old their coffee can be before it needs to be sold. The goal
of this model is to reduce the environmental and economic impacts of the coffee industry by reducing the waste produced by prematurely disposing of the coffee beans.

**Input and output variable:**

**Input:**
1. Date
2. Triplicate_#
3. Roast Level
4. FTIR Lipid Absorbance
5. FTIR Carboxylic Acid Absorbance
6. FTIR Peak Ratio
7. GCMS 2-Methylfuran Area (%)
8. GCMS Methanethiol Area (%)
9. GCMS Peak Ratio
10. TGA air onset deg (C)
11. TGA N2 residual wt at 700C
12. TGA/DSC Air exo pk (C)
13. TGA/DSC DH 250-360, 200-360	DSC endo pk (C)
14. DSC DH 40-180C (J/g)
15. Density (g/cm^3)
16. Moisture
17. Brew pH
18. Brew TDS
19. Brew Color L
20. Brew Color a
21. Brew Color b

**Outputs:**

Sample Age (days)

**Machine Learning Algorithms**

There are a large variety of machine learning models. Without trying a couple of the models, we have no way of knowing which one is correct. So, this project tested five different types of models.

1. Linear Regression (Linear Model)
2. k-Nearest Neighbors (k-NN) (Instance-Based Learning)
3. Decision Tree (Tree-Based Model)
4. Support Vector Machine (SVM) (Support Vector Model)
5. Gradient Boosting Machine (XGBoost) (Ensemble Model)

XGBoost was chosen because it consistently had the highest R2, RSME, and MAE values. Further, it was hyperparameter tuned using a Bayasian optimization, which was chosen because it resulted in the most improvement of the accuracy of the model in comparison to grid search and random search.

**Dataset used**

The dataset used is called Coffee_Freshness_Dataset_FA2024.xlsx and is provided in this repo. Before any machine learning was done, the dataset was cleaned and normalized. This allowed for the machine learning process to be more accurate. Also, various columns in the dataset were dropped because they either weren't collected, didn't have physical meaning, or were represented more clearly by another test that was done that is in the dataset.

All preprocessing and machine learning choices made using this dataset can be found in the .ipynb file attached in this repo.

**Public App**

https://huggingface.co/spaces/mayafetzer/CoffeeApp

**Ethics Analysis**

Deon's Ethics Checklist: https://huggingface.co/spaces/mayafetzer/CoffeeDeonsChecklist

Ethics DataCAD: https://huggingface.co/spaces/mayafetzer/CoffeeEthicsDataCAD

**Authors**

Maya Fetzer and Dr. Kat Wakabayashi

Bucknell University Department of Chemical Engineering

**Acknowledgements**

This research was only possible due to the help of the following people, companies, and organizations:

Bucknell University Department of Engineering

Bucknell Robert R. Rooke Professorship

Monica Hoover - Bucknell University Department of Environmental Engineering

Diane Hall - Bucknell University Department of Chemical Engineering

Dr. Jude Okolie - Bucknell University Departmment of Chemical Engineering

Ryan Koes - Bucknell University Department of Computer Science and Engineering

Sean O'Connor  - Bucknell University Department of Computer Science and Engineering

Brewista

Acaia

**Code acknowledgement:**
The code present in this repository was written using the help of GPTs, including OpenAI ChatGPT-4o and Google Gemini.

**Files in this repository**

Coffee_Freshness_Dataset_FA2024.xlsx - the dataset used to train the ML model

coffee_ethics_checklist.md - the Deon's Ethics Checklist for this ML model

CHEG472_Machine_Learning_Project.ibynb - Google Colab file that shows the steps to developing the ML model

app.py - gradio app to launch public app in Hugging Face

best_model_bayes.pkl - ML model

scaler.pkl - scaler model

requirements.txt - requirements for Hugging Face app

**Prerequisites**

**Python**

Ensure you have Python 3.10 or later installed.

**Libraries**

Install the following libraries using pip:

```
pip install gradio
pip install pandas
pip install numpy
pip install sklearn
pip install openpyxl
pip install pickle5
```

**Explanation**

- **Streamlit**: Provides a simple way to create interactive web applications with Python.
- **Matplotlib**: Used for creating visualizations like plots and charts.
- **Pandas**: Offers data structures and analysis tools for working with tabular data.
- **NumPy**: Provides efficient numerical operations and arrays.
- **Sklearn**: Machine learning library.
- **Gradio:** A user-friendly library for building and sharing interactive web interfaces for machine learning models and data science projects.
- **Openpyxl:** A library for reading and writing Excel files in the XLSX format, making it easy to work with spreadsheets directly from Python.
- **Pickle5:** An enhanced version of Python's pickle module for object serialization and

##### GUI App Deployment




---

```python
import gradio as gr
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load models
with open('best_model_bayes.pkl', 'rb') as f:
    best_model_bayes = pickle.load(f)

# Load the pre-fitted scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the actual output range (replace with the true range of your data)
output_min, output_max = 0, 100  # Replace with your actual denormalization values

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

def predict(Moisture, CO2, Aroma, Acidity, Oxidation, Oils, Roast_Level):

    # Encode the Roast Level as numerical value
    roast_level_encoded = 0 if Roast_Level == "13.5%" else 1

    # Prepare the input array with the new variables
    input_array = np.array([[Moisture, CO2, Aroma, Acidity, Oxidation, Oils, roast_level_encoded]])

    # Scale input values using the loaded, pre-fitted scaler
    input_scaled = scaler.transform(input_array)

    # Make prediction using the best model
    normalized_output = best_model_bayes.predict(input_scaled)[0]

    # Denormalize the prediction to reflect time
    time_output = denormalize(normalized_output, output_min, output_max)

    return round(time_output, 2)  # Rounded for readability

# Create the Gradio interface with sliders for each input variable
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.components.Slider(0, 1, step=0.01, label="Moisture (%)"),
        gr.components.Slider(0, 1, step=0.01, label="CO2 (ppm)"),
        gr.components.Slider(0, 1, step=0.01, label="Aroma"),
        gr.components.Slider(0, 1, step=0.01, label="Acidity"),
        gr.components.Slider(0, 1, step=0.01, label="Oxidation"),
        gr.components.Slider(0, 1, step=0.01, label="Oil Expression"),
        gr.components.Dropdown(choices=["13.5%", "20.7%"], label="Roast Level"),
    ],
    outputs=gr.components.Label(label="Time to reach target freshness (days)"),
    title="Coffee Freshness Prediction Model",
    description="Adjust the sliders and select the roast level to predict time. A value closer to 0 means more fresh while a value closer to 1 means less fresh.",
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
```

##### Machine Learning Development




---

The goal of this project is to create a machine learning model that will predict the freshness of coffee based on how old is it. The model will be trained on a sample dataset of experimental data that I have collected as part of my individual research project with Dr. Kat Wakabayashi.

Step 1: Download and preprocess the dataset

In order to make a machine learning model, you need a dataset in order to train the model. For this project, I will be using a dataset that I have collected as a part of my independant research.

```python
import pandas as pd

# Replace '/path/to/your/file.xlsx' with the actual file path
file_path = '/content/Coffee_Freshness_Dataset_FA2024.xlsx'

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(file_path)

# Print the first few rows of the DataFrame
print("The first 5 rows of the dataset:")
print(df.head())

# Print the description of the DataFrame
print("The descriptive statistics of the dataset")
print(df.describe())

# Get the data types of all columns
data_types = df.dtypes
# Print the datatypes of each column
print("The data types of each column:")
print(data_types)

# Get the numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
# Get the categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
# Print the results
print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)
```

```python
# Remove the redundant data columns that are not relevant to the dataset

# For this dataset, I am going to remove the actual date. This does not matter,
# only the number of days since the roast date. Next, I am going to drop the hardness
# column because it only says "no", which is because we weren't able to
# collect this data. I am also going to the triplicate
# number, as this was only meant to help organize the samples while collecting them.

# Finally, I am going to remove the TGA/DSC DH values, because these ended up
# not being something that we could map back to aspects of coffee freshness over
# time in the literature
df = df.drop(columns=['Date', 'Triplicate_#','TGA/DSC DH 250-360, 200-360','Hardness'])
```

Before I can use a machine learning model, I need to make sure that the dataset has no factors that would skew the model. So, I will be looking at removing redundancies, outliers, missing data, etc. and doing some dimensional reduction and normalization of the variables in order to make the dataset ideal for modeling.

```python
import pandas as pd
import numpy as np

# Step 1: Check for missing data
missing_data = df.isnull().sum()
print("Total missing values for each column before filling:")
print(missing_data[missing_data > 0])

# There is no missing data in this dataset, so we don't have to worry about handling
# missing values.
```

```python
import matplotlib.pyplot as plt
import numpy as np

# Calculate outliers for each column
outliers_dict = {}

# Only check the columns that are numerical and would contain outliers
columns_to_check = df.select_dtypes(include=['int64', 'float64']).columns

# Create boxplots for specified columns
df[columns_to_check].boxplot()
plt.title("Boxplots for Numerical Columns")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.show()

# Check for outliers and plot the boxplot
for col in columns_to_check:
   q1 = np.quantile(df[col], 0.25)
   q3 = np.quantile(df[col], 0.75)
   iqr = q3 - q1
   lower_bound = q1 - 1.5 * iqr
   upper_bound = q3 + 1.5 * iqr
   outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
   outliers_dict[col] = outliers.tolist()

# Print outliers for each column
for col, outliers in outliers_dict.items():
   if outliers:
       print(f"Outliers in column '{col}': {outliers}")
   else:
       print(f"No outliers found in column '{col}'")

# This dataset does have outliers, but I am going to choose to not remove them.
# At this point, the dataset is very small, and so removing the outliers
# removes some important information about the trends of the data that
# I think will be important to keep going forward.
```

```python
# To better understand the trends of the data, I want to then look at the
# pairplots to better understand if there are variables that relate strongly
# with one another. This will be helpful when I need to go through and
# reduce the dataset to make it more fit for machine learning analysis.

import seaborn as sns
sns.pairplot(df)
```

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Print the correlation matrix
print(correlation_matrix)

# Create a heatmap to visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Analyze significant correlations
significant_correlations = correlation_matrix[(correlation_matrix > 0.7) | (correlation_matrix < -0.7)]
significant_correlations = significant_correlations.stack().reset_index()
significant_correlations.columns = ['feature1', 'feature2', 'correlation']
print(significant_correlations)
```

```python
# Now I will remove anything will a low correlation to time to avoid overfitting.
df.drop(['GCMS 2-Methylfuran Area (%)','GCMS Methanethiol Area (%)', 'Brew pH','Brew TDS','Brew Color L','Brew Color a','Brew Color b','Density (g/cm^3)'],axis=1, inplace=True)

# Calculate the new correlation matrix
correlation_matrix = df.corr()

# Create a heatmap to visualize new correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

```python
# Next, we need to deal with multicollinearity within the dataset. This is
# when two or more variables are highly correlated with one another. If you
# leave these as seperate terms, it can skew the model and lead to distortion.

# Terms with interaction:
# Acidity: FTIR Carboxylic Acid Absorbance + FTIR Lipid Absorbance
# Aroma: GCMS Ratio
df['Aroma'] = df['GCMS Peak Ratio']
df['Acidity'] = df['FTIR Carboxylic Acid Absorbance'] * df['FTIR Lipid Absorbance']
df['Oxidation'] = df['DSC endo pk (C)'] * df['TGA N2 residual wt at 700C']

# Now we drop everything that was combined together
df.drop(['FTIR Carboxylic Acid Absorbance', 'FTIR Lipid Absorbance','DSC endo pk (C)','TGA N2 residual wt at 700C','TGA/DSC Air exo pk (C)'], axis=1, inplace=True)
```

```python
# Because we removed the multicollinearity because the GCMS and the FTIR peaks,
# I am also going to remove the FTIR peak ratio sections of this datasheet.
# This way we won't have duplicates of the data.
df.drop(['FTIR Peak Ratio','GCMS Peak Ratio'], axis=1, inplace=True)

# To avoid overfitting, we can drop relationships that are not strongly correlated
# with time.

df.drop(['DSC DH 40-180C (J/g)'],axis=1, inplace=True)

# I am also going to rename "TGA air onset deg (C)" because that name is, while
# representative of what we measured, not very clear in how it relates to the
# freshness of the coffee.

df['Oil Expression'] = df['TGA air onset deg (C)']
df.drop(['TGA air onset deg (C)'],axis=1, inplace=True)
```

```python
# We can now relook at the heat map and the significant correlations to make sure that
# there are no correlations between the features that are going to be close
# to 1.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print(df.head())

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Print the correlation matrix
print(correlation_matrix)

# Create a heatmap to visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Analyze significant correlations
significant_correlations = correlation_matrix[(correlation_matrix > 0.7) | (correlation_matrix < -0.7)]
significant_correlations = significant_correlations.stack().reset_index()
significant_correlations.columns = ['feature1', 'feature2', 'correlation']
print(significant_correlations)
```

```python
# Finally, we can normalize the dataframe. There are a lot of different
# numerical categories in this dataframe, and making sure that they are normalized
# will improve the accuracy of the model and prevent it from being skewed by numbers
# that are very different in magnitude.

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Assuming 'df' is your DataFrame and you want to normalize all columns
scaler = MinMaxScaler()

# Normalize the DataFrame
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Display the normalized DataFrame
print(df)
```

```python
# Before moving on, we are going to look at the dataframe again and make sure that
# all of the data is still present.

df.info()
```

Step 2: Pick and validate a machine learning model

There are a large variety of machine learning models. Without trying a couple of the models, we have no way of knowing which one is correct. So, I will be trying five different machine learning models and seeing which one is the most accurate.

There are models that are better and worse at certain things. I am going to be choosing models that are better at working with numerical data and working with small datasets. The five models that I have chosen to examine are as follows:

1. Linear Regression (Linear Model)
2. k-Nearest Neighbors (k-NN) (Instance-Based Learning)
3. Decision Tree (Tree-Based Model)
4. Support Vector Machine (SVM) (Support Vector Model)
5. Gradient Boosting Machine (XGBoost) (Ensemble Model)

```python
# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Replace 'df' with your actual DataFrame name and specify your feature (X) and target (y) columns
X = df.drop(columns='Sample Age (days)')  # Replace 'target' with the name of your target column
y = df['Sample Age (days)']  # Replace 'target' with the name of your target column

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Define models
models = {
    'Linear Regression': LinearRegression(),
    'k-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Support Vector Machine': SVR(kernel='rbf'),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Initialize results list
results = []

# Train and evaluate each model
for model_name, model in models.items():
    # Fit the model
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)

    # Store results
    results.append({
        'Model': model_name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'RMSE': rmse,
        'MAE': mae
    })

# Create a DataFrame to display results
results_df = pd.DataFrame(results)
print(results_df)

# Select the best model based on Test R², or lowest RMSE if R² is identical
best_model_row = results_df.loc[results_df['Test R²'].idxmax()]

print("\nBest model based on Test R² performance:")
print(best_model_row)

# Assuming 'results_df' is the DataFrame containing the model performance metrics
models = results_df['Model']
train_r2 = results_df['Train R²']
test_r2 = results_df['Test R²']

# Set up bar width and positions for each group
bar_width = 0.35
index = np.arange(len(models))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Bars for Train R²
train_bars = ax.bar(index, train_r2, bar_width, label='Train R²', color='skyblue')

# Bars for Test R²
test_bars = ax.bar(index + bar_width, test_r2, bar_width, label='Test R²', color='salmon')

# Labels and Titles
ax.set_xlabel('Models')
ax.set_ylabel('R² Score')
ax.set_title('Train and Test R² Scores by Model')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(models, rotation=45)
ax.legend()

# Display plot
plt.tight_layout()
plt.show()
```

According to the above testing of the 5 models, I am going to continue
forward with the Gradient Boosting model. This model had the best outcomes in terms
of accuracy, so we want to continue with it.

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Define the Gradient Boosting model
gb_model = GradientBoostingRegressor()

# Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store R², RMSE, and MAE scores for each fold
train_r2_scores = []
test_r2_scores = []
train_rmse_scores = []
test_rmse_scores = []
train_mae_scores = []
test_mae_scores = []

# Cross-validation loop
for train_index, test_index in kf.split(X):
    # Split data into train and test for each fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training data
    gb_model.fit(X_train, y_train)

    # Predict and calculate R², RMSE, and MAE for train and test sets
    train_r2_scores.append(r2_score(y_train, gb_model.predict(X_train)))
    test_r2_scores.append(r2_score(y_test, gb_model.predict(X_test)))

    train_rmse_scores.append(np.sqrt(mean_squared_error(y_train, gb_model.predict(X_train))))
    test_rmse_scores.append(np.sqrt(mean_squared_error(y_test, gb_model.predict(X_test))))

    train_mae_scores.append(mean_absolute_error(y_train, gb_model.predict(X_train)))
    test_mae_scores.append(mean_absolute_error(y_test, gb_model.predict(X_test)))

# Display R², RMSE, and MAE for each fold
print("Train R² scores for each fold:", train_r2_scores)
print("Test R² scores for each fold:", test_r2_scores)
print("Train RMSE scores for each fold:", train_rmse_scores)
print("Test RMSE scores for each fold:", test_rmse_scores)
print("Train MAE scores for each fold:", train_mae_scores)
print("Test MAE scores for each fold:", test_mae_scores)

# Plotting the R² scores for each fold
folds = np.arange(1, kf.get_n_splits() + 1)

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25

# Bars for Train R²
train_bars = ax.bar(folds - bar_width, train_r2_scores, bar_width, label='Train R²', color='skyblue')

# Bars for Test R²
test_bars = ax.bar(folds, test_r2_scores, bar_width, label='Test R²', color='salmon')

# Labels and Titles
ax.set_xlabel('Fold')
ax.set_ylabel('R² Score')
ax.set_title('Train and Test R² Scores for Each Fold (Gradient Boosting)')
ax.set_xticks(folds)
ax.legend()

# Display plot
plt.tight_layout()
plt.show()

# Plotting the RMSE scores for each fold
fig, ax = plt.subplots(figsize=(10, 6))

# Bars for Train RMSE
train_rmse_bars = ax.bar(folds - bar_width, train_rmse_scores, bar_width, label='Train RMSE', color='lightgreen')

# Bars for Test RMSE
test_rmse_bars = ax.bar(folds, test_rmse_scores, bar_width, label='Test RMSE', color='orange')

# Labels and Titles
ax.set_xlabel('Fold')
ax.set_ylabel('RMSE')
ax.set_title('Train and Test RMSE Scores for Each Fold (Gradient Boosting)')
ax.set_xticks(folds)
ax.legend()

# Display plot
plt.tight_layout()
plt.show()

# Plotting the MAE scores for each fold
fig, ax = plt.subplots(figsize=(10, 6))

# Bars for Train MAE
train_mae_bars = ax.bar(folds - bar_width, train_mae_scores, bar_width, label='Train MAE', color='lightcoral')

# Bars for Test MAE
test_mae_bars = ax.bar(folds, test_mae_scores, bar_width, label='Test MAE', color='gold')

# Labels and Titles
ax.set_xlabel('Fold')
ax.set_ylabel('MAE')
ax.set_title('Train and Test MAE Scores for Each Fold (Gradient Boosting)')
ax.set_xticks(folds)
ax.legend()

# Display plot
plt.tight_layout()
plt.show()

# Make predictions on the entire test set from the last fold
y_pred = gb_model.predict(X_test)

# Plotting predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Line for y=x

# Labels and Titles
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values for Gradient Boosting Model')

plt.show()
```

Step 3: Hyperparameter training

The R2 here is ok, but it could be improved in order to make a more accurate model. Something that we can do in order to make the model more accurate is that we can use hyperparameter training.

To do this, I am going to be looking at 3 different types of hyperparameter training models: grid search, random search, and Bayasian optimization. The best of these three models can then be used going forward.

```python
pip install scikit-optimize
```

```python
# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from skopt import BayesSearchCV  # Import for Bayesian Optimization

# Create a function to plot learning curves
def plot_learning_curve(model, X, y, cv=5, n_jobs=-1):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # Calculate the mean and standard deviation for training scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    # Calculate the mean and standard deviation for test scores
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

    # Plot the std deviation as a shaded area
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')

    plt.title('Learning Curves')
    plt.xlabel('Training Size')
    plt.ylabel('R² Score')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# Load your dataset here
# df = pd.read_csv('your_data.csv')  # Example: load your data

# Specify your features (X) and target (y)
X = df.drop(columns='Sample Age (days)')  # Adjust as needed
y = df['Sample Age (days)']  # Adjust as needed

# Create a pipeline for scaling and modeling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gb', GradientBoostingRegressor())
])

# Expanded parameter grid for Gradient Boosting
param_grid = {
    'gb__n_estimators': [50, 100, 150, 200],
    'gb__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'gb__max_depth': [3, 5, 7, 9],
    'gb__subsample': [0.8, 1.0],
    'gb__min_samples_split': [2, 5, 10]
}

# Set up Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X, y)

print("Best parameters from Grid Search:", grid_search.best_params_)
best_model_grid = grid_search.best_estimator_

# Set up Randomized Search with more iterations
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=50, cv=5, scoring='r2', random_state=42)
random_search.fit(X, y)

print("Best parameters from Randomized Search:", random_search.best_params_)
best_model_random = random_search.best_estimator_

# Set up Bayesian Optimization with more iterations
bayes_search = BayesSearchCV(pipeline, param_grid, n_iter=50, cv=5, scoring='r2', random_state=42)
bayes_search.fit(X, y)

print("Best parameters from Bayesian Optimization:", bayes_search.best_params_)
best_model_bayes = bayes_search.best_estimator_

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the best model from Grid Search on the training data
best_model_grid.fit(X_train, y_train)
y_train_pred_grid = best_model_grid.predict(X_train)
y_test_pred_grid = best_model_grid.predict(X_test)

# Calculate R² scores for Grid Search model
train_r2_grid = round(r2_score(y_train, y_train_pred_grid), 3)
test_r2_grid = round(r2_score(y_test, y_test_pred_grid), 3)
print("Grid Search Training R²:", train_r2_grid)
print("Grid Search Test R²:", test_r2_grid)

# Fit the best model from Randomized Search on the training data
best_model_random.fit(X_train, y_train)
y_train_pred_random = best_model_random.predict(X_train)
y_test_pred_random = best_model_random.predict(X_test)

# Calculate R² scores for Randomized Search model
train_r2_random = round(r2_score(y_train, y_train_pred_random), 3)
test_r2_random = round(r2_score(y_test, y_test_pred_random), 3)
print("Randomized Search Training R²:", train_r2_random)
print("Randomized Search Test R²:", test_r2_random)

# Fit the best model from Bayesian Optimization on the training data
best_model_bayes.fit(X_train, y_train)
y_train_pred_bayes = best_model_bayes.predict(X_train)
y_test_pred_bayes = best_model_bayes.predict(X_test)

# Calculate R² scores for Bayesian Optimization model
train_r2_bayes = round(r2_score(y_train, y_train_pred_bayes), 3)
test_r2_bayes = round(r2_score(y_test, y_test_pred_bayes), 3)
print("Bayesian Optimization Training R²:", train_r2_bayes)
print("Bayesian Optimization Test R²:", test_r2_bayes)

# Use the best model from each search to plot learning curves
plot_learning_curve(best_model_random, X_train, y_train)
plot_learning_curve(best_model_grid, X_train, y_train)
plot_learning_curve(best_model_bayes, X_train, y_train)

# Compare predictions
plt.scatter(y_test, y_test_pred_random, label='Randomized Search Predictions', alpha=0.5)
plt.scatter(y_test, y_test_pred_bayes, label='Bayesian Optimization Predictions', alpha=0.5)
plt.scatter(y_test, y_test_pred_grid, label='Grid Search Optimization Predictions', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.title('Comparison of Predictions')
plt.show()
```

Based on the hyperparameter tuning, the best model that came out of this is the Bayasian Optimization predictions. This is able to accurto have a higher test and training R2 value, so I will be using this model going forward.

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Define the Gradient Boosting model as the hypertuned model
gb_model = best_model_bayes

# Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store R², RMSE, and MAE scores for each fold
train_r2_scores = []
test_r2_scores = []
train_rmse_scores = []
test_rmse_scores = []
train_mae_scores = []
test_mae_scores = []

# Cross-validation loop
for train_index, test_index in kf.split(X):
    # Split data into train and test for each fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training data
    gb_model.fit(X_train, y_train)

    # Predict and calculate R², RMSE, and MAE for train and test sets
    train_r2_scores.append(r2_score(y_train, gb_model.predict(X_train)))
    test_r2_scores.append(r2_score(y_test, gb_model.predict(X_test)))

    train_rmse_scores.append(np.sqrt(mean_squared_error(y_train, gb_model.predict(X_train))))
    test_rmse_scores.append(np.sqrt(mean_squared_error(y_test, gb_model.predict(X_test))))

    train_mae_scores.append(mean_absolute_error(y_train, gb_model.predict(X_train)))
    test_mae_scores.append(mean_absolute_error(y_test, gb_model.predict(X_test)))

# Display R², RMSE, and MAE for each fold
print("Train R² scores for each fold:", train_r2_scores)
print("Test R² scores for each fold:", test_r2_scores)
print("Train RMSE scores for each fold:", train_rmse_scores)
print("Test RMSE scores for each fold:", test_rmse_scores)
print("Train MAE scores for each fold:", train_mae_scores)
print("Test MAE scores for each fold:", test_mae_scores)

print("Average Train R²", sum(train_r2_scores)/len(train_r2_scores))
print("Average Test R²", sum(test_r2_scores)/len(test_r2_scores))
print("Average Train RSME", sum(train_rmse_scores)/len(train_rmse_scores))
print("Average Test RSME", sum(test_rmse_scores)/len(test_rmse_scores))
print("Average Train MAE", sum(train_mae_scores)/len(train_mae_scores))
print("Average Test MAE", sum(test_mae_scores)/len(test_mae_scores))

# Plotting the R² scores for each fold
folds = np.arange(1, kf.get_n_splits() + 1)

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25

# Bars for Train R²
train_bars = ax.bar(folds - bar_width, train_r2_scores, bar_width, label='Train R²', color='skyblue')

# Bars for Test R²
test_bars = ax.bar(folds, test_r2_scores, bar_width, label='Test R²', color='salmon')

# Labels and Titles
ax.set_xlabel('Fold')
ax.set_ylabel('R² Score')
ax.set_title('Train and Test R² Scores for Each Fold (Gradient Boosting)')
ax.set_xticks(folds)
ax.legend()

# Display plot
plt.tight_layout()
plt.show()

# Plotting the RMSE scores for each fold
fig, ax = plt.subplots(figsize=(10, 6))

# Bars for Train RMSE
train_rmse_bars = ax.bar(folds - bar_width, train_rmse_scores, bar_width, label='Train RMSE', color='lightgreen')

# Bars for Test RMSE
test_rmse_bars = ax.bar(folds, test_rmse_scores, bar_width, label='Test RMSE', color='orange')

# Labels and Titles
ax.set_xlabel('Fold')
ax.set_ylabel('RMSE')
ax.set_title('Train and Test RMSE Scores for Each Fold (Gradient Boosting)')
ax.set_xticks(folds)
ax.legend()

# Display plot
plt.tight_layout()
plt.show()

# Plotting the MAE scores for each fold
fig, ax = plt.subplots(figsize=(10, 6))

# Bars for Train MAE
train_mae_bars = ax.bar(folds - bar_width, train_mae_scores, bar_width, label='Train MAE', color='lightcoral')

# Bars for Test MAE
test_mae_bars = ax.bar(folds, test_mae_scores, bar_width, label='Test MAE', color='gold')

# Labels and Titles
ax.set_xlabel('Fold')
ax.set_ylabel('MAE')
ax.set_title('Train and Test MAE Scores for Each Fold (Gradient Boosting)')
ax.set_xticks(folds)
ax.legend()

# Display plot
plt.tight_layout()
plt.show()

# Make predictions on the entire test set from the last fold
y_pred = gb_model.predict(X_test)

# Plotting predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Line for y=x

# Labels and Titles
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values for Gradient Boosting Model')

plt.show()
```

Step 4: Increase the accuracy of the model

If the model has a lower accuracy, you can take additional steps to increase teh accuracy of the model. At this point, the model has a decent accuracy and I don't believe that addtional steps, like GAN, would be worth the time that it would require to do them. So, for now, I am going to not do this step.

In the future as this research continues, GAN might become a more useful tool to use as the dataset develops.

Step 5: Interpotable Analysis

```python
pip install matplotlib seaborn scikit-learn
```

```python
# Required Libraries for Partial Dependence Plot
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Fit the best model from Randomized Search on the entire training data
best_model_bayes.fit(X_train, y_train)

# Select the features for which you want to create the PDP
features_to_plot = X.columns.tolist()  # Use all features; modify if you want to limit

# Create Partial Dependence Plots for each feature separately
for feature in features_to_plot:
    plt.figure(figsize=(8, 6))  # Create a new figure for each plot
    display = PartialDependenceDisplay.from_estimator(
        best_model_bayes,
        X_train,
        features=[feature],  # Pass a list with the current feature
        grid_resolution=50
    )

    # Set axis labels and title
    ax.set_xlabel(feature)
    ax.set_ylabel('Partial Dependence')
    ax.set_title(f'Partial Dependence of {feature}')

    plt.suptitle('Partial Dependence Plot for Bayasian Optimization Search Best Model', fontsize=16)
    plt.show()  # Display the plot
```

```python
import shap
import matplotlib.pyplot as plt

# Assuming you have a trained model and test data
model = best_model_bayes  # Your trained model

# Extract the actual model from the pipeline
gradient_boosting_model = model.named_steps['gb']

# Use TreeExplainer for the extracted tree-based model
explainer = shap.TreeExplainer(gradient_boosting_model)

# Calculate SHAP values for the test data
shap_values = explainer(X_test)  # For regression, use explainer(X_test)

# Create the beeswarm plot
shap.plots.beeswarm(shap_values)
plt.show()
```

##### Ethics Analysis




---

**Ethics Checklist for Coffee Freshness Prediction**

**A. Data Collection**
 - **A.1 Informed consent**: If there are human subjects, have they given informed consent, where subjects affirmatively opt-in and have a clear understanding of the data uses to which they consent?

This dataset did not use human subjects, so there was no need for informed consent. The only human interaction with the dataset was the researchers collecting the data. These researchers, including myself
and my research advisor Dr. Wakabayashi, both consented to the use of this data for the machine learning project.

 - **A.2 Collection bias**: Have we considered sources of bias that could be introduced during data collection and survey design and taken steps to mitigate those?

It is possible that the collection methods from the sureying techniques introduced in bias into the dataset. To mitigate any bias or error that was introduced into the dataset
through the collection technqiues of the study, every sample was run in triplicate. The goal of using this method was to reduce any error that could come from one
incorrect measurement.

 - **A.3 Limit PII exposure**: Have we considered ways to minimize exposure of personally identifiable information (PII) for example through anonymization or not collecting information that isn't relevant for analysis?

There is no PPI present in this dataset. So, no steps were taken to limit PPI exposure because it was not relavent to this dataset.

 - **A.4 Downstream bias mitigation**: Have we considered ways to enable testing downstream results for biased outcomes (e.g., collecting data on protected group status like race or gender)?

There was no information that was collected that could have a biases result based on protected group status. However, in order to reduce any bias in the model that was generated,
all samples will always be run in triplicate if not more and any error associated with the measurements will be additionally recorded.

**B. Data Storage**
 - **B.1 Data security**: Do we have a plan to protect and secure data (e.g., encryption at rest and in transit, access controls on internal users and third parties, access logs, and up-to-date software)?

There is a plan to make sure that the dataset and code is not edited to change the dataset. The available copy of the dataset to the research group that is using the ML model has an access and edit
log so that no data is changed after the tests have been conducted. There is no plan to encrypt the data, as the dataset does not contain any private information

 - **B.2 Right to be forgotten**: Do we have a mechanism through which an individual can request their personal information be removed?

There is no personal information in this dataset, so no mechanisms were put into considerations to allow people to remove personal information. In the future, as more data is
collected, this can be re-evaluated. If other people contribute data to this project, a mechanism can be added into the data analysis process that allows people to request for
their datasets to be removed. For now, this is not a feature that was added into the project because it is not relavent.

 - **B.3 Data retention plan**: Is there a schedule or plan to delete the data after it is no longer needed?

There is no plan to delete the data after it is no longer needed. The goal with maintaining the accuracy and transparancy of the model is to keep the dataset, as well as any updates
to the dataset, are open to the public. The data does not have any personal or sensitive information about any people, so there is no worry about anyone needing to remove their information
as time passes. The only instance where data would be removed from the ML model is if new data is obtained in the lab and the old dataset was proven to be inaccurate. However, this is not a
situation that I am currently considering, because we are not close to being able to remove datapoints. This is something to consider in the future to maintain an accurate model.

**C. Analysis**
 - **C.1 Missing perspectives**: Have we sought to address blindspots in the analysis through engagement with relevant stakeholders (e.g., checking assumptions and discussing implications with affected communities and subject matter experts)?

The analysis of this data relies on previous studies that have been conducted on the compounds that were measured in the coffee. It is possible that the freshness and taste characteristics
that were associated with the coffee in this analysis do not accurately reflect the way that the stakeholders taste and process the coffee, resulting in discrepancies between the user of the model
and the actual result of the mdoel. To address this, the different components were confirmed in multiple literature searches that have occurred over the past six months of development of the dataset.

The model has also been discussed with local coffee experts, including professors at Bucknell University who have studied coffee science and local roasters and coffee salespeople from the  Lewisburg and
surrounding areas. The model's results were discussed and the coffee experts helped to shed light into further characteristics of the model that needed to be developed, like additional tests that were added
throughout the collection of the dataset.

 - **C.2 Dataset bias**: Have we examined the data for possible sources of bias and taken steps to mitigate or address these biases (e.g., stereotype perpetuation, confirmation bias, imbalanced classes, or omitted confounding variables)?

The dataset could have bias that occurred during the sampling process. It is possible that the steps of collecting the data skewed the results. In order to adress this, each experimental method was documented and repeated by the same researcher in
triplicate. This ensured that any outliers that could have been caused by the data collection method would be recognized and could be removed if neccessary. Also, the results were confirmed by the advisor of the research group as well as checked with
other studies within the body of coffee literature.

 - **C.3 Honest representation**: Are our visualizations, summary statistics, and reports designed to honestly represent the underlying data?

All visualizations, summary statistics, and reports that were created in the analysis of the dataset were designed to honestly represent the underlying data. Any figure that was generated represents the dataset. No attempts were made throughout
the analysis process to alter the figures or statistics in a way that would misrepresent the trends present in the data.

 - **C.4 Privacy in analysis**: Have we ensured that data with PII are not used or displayed unless necessary for the analysis?

Data with PII was not used in this dataset. So, not measures were taken to ensure that PII was not used or displayed because it was not present.

 - **C.5 Auditability**: Is the process of generating the analysis well documented and reproducible if we discover issues in the future?

The process of generating the analysis of the dataset is present in the Google Colab file that is attached to this Github repository. The analysis and creation of the model is broken down into the steps that
were taken and is also commented to describe why certain choices were made throughout the analysis of the dataset. The goal with formatting the analysis in this way is that this code could be taken and used with an alternative dataset.
This way, another researcher or research group could repeat the process with data that they obtained. Because all decisions were documented, it is also possible to locate issues in the code and change
any parts of the model if errors arise in the future.

**D. Modeling**
 - **D.1 Proxy discrimination**: Have we ensured that the model does not rely on variables or proxies for variables that are unfairly discriminatory?

 In designing the methods of collecting data for this experiment, a variety of different aspects of the coffee were considered so that one variable does not solely dictate the
 results of the ML model. In doing the pre-processing of the dataset, the goal was to make sure that the model does not only rely on one correlation that would unfairly swing the
 results of the predicted freshness. Further, no proxy variables were used because only one specific of origin was used, and the roast levels were represented numerically instead of
 categorically with words like "light" and "dark". As the model continues to be develoepd, the goal is to add more species of origin. The researchers recognize that the reigon where
 the coffee beans are grown could act as a proxy variable, and so only quantitive factors that would affect the coffee (temperature, humidity, altitude) will be considered.

 - **D.2 Fairness across groups**: Have we tested model results for fairness with respect to different affected groups (e.g., tested for disparate error rates)?

The model has not currently been tested for fairness across different groups. The model parameters were chosen and developed from literature searches that spanned decades of research from different countries and research
groups. Because the model has not been beta tested, it is unclear if the model is calibrated in a way that would unfairly discriminate against certain groups of people that will drink the coffee or use the
model. Going forward, as the model continues into further levels of development, it is important to check with a wide variety of people to see how they interpret or are affected by the model.

 - **D.3 Metric selection**: Have we considered the effects of optimizing for our defined metrics and considered additional metrics?

Yes, additional metrics are going to be added as this study is continued and more data is collected. The goal is to find more variables that are strongly correlated with time,
which should make a more accurate model and predict the quality of the coffee in a more complete and thorough manner.

 - **D.4 Explainability**: Can we explain in understandable terms a decision the model made in cases where a justification is needed?

All decisions made in the creation of this model have been commented and documented, so all aspects of the model can be justified. The explainations of the model and the
choices made throughout its development were written in plain English to avoid confusion from any listeners.

 - **D.5 Communicate bias**: Have we communicated the shortcomings, limitations, and biases of the model to relevant stakeholders in ways that can be generally understood?

The model only works for Brazilian Cerrado beans and for the two roast levels that have been analyzed so far. This is stated in the app for the ML model, so that any stakeholders can
understand the limitations of the dataset. Further, the accuracy of the model (R2, RMSE, MAE) were documented in the analysis process so that any stakeholders can read through the
creation of the model and decide if it is accurate enough for their specific use.

**E. Deployment**
 - **E.1 Redress**: Have we discussed with our organization a plan for response if users are harmed by the results (e.g., how does the data science team evaluate these cases and update analysis and models to prevent future harm)?

As a research team, the goal of this project is to be informative. However, the research team recognizes that the model is not perfectly accurate and is just one of many attempts to quantify the freshness of coffee. After deployment,
the goal is to keep in contact with any roasters or brewers who use the model and check in with how the model has been informing their decisions and if they believe that the model has been working properly. The users' feedback will be
prioritized. The goal of the ML platform is to make it easier for people to gauge how old their coffee is. If it is not achieving that goal, the model needs to be updated in order to better reflect the needs of the clients who are using
the platform.

 - **E.2 Roll back**: Is there a way to turn off or roll back the model in production if necessary?

The model is hosted on a Hugging Face app, which can be shut down at any point if deemed necessary. It is also possible to update the model while keeping the platform deployed, so any bugs that are found in
the current version of the model can be corrected. Users will only be able to access the most current version of the ML model, so users will not be able to access a worse version of the model if any updates
have been made.

 - **E.3 Concept drift**: Do we test and monitor for concept drift to ensure the model remains fair over time?

Over time, the model will continuously be trained with newly collected data. The new data will be validated on the new data and any drift from the original accuracy and fairness of the model will be corrected.
The model will be checked to make sure that the results are as consistant and accurate as possible, even as time passes since initial deployment.

 - **E.4 Unintended use**: Have we taken steps to identify and prevent unintended uses and abuse of the model and do we have a plan to monitor these once the model is deployed?

The goal of this model is to create a place where users can be informed of how their coffee ages. Because this model will only be used by local roasters and brewers (at least at initial deployment), it should be
possible to track the use of the model and make sure that roasters are not using it in a negative way (like trying to harm the sales of another roaster). If the model is being abused, it will be taken down, and
additional measures will be created to secure the model so that only those with permission can access it. For now, this seems like a step that is not needed, because the model is not being fully deployed or
recommended for use until the dataset is further developed.

```python
import gradio as gr

# Ethics DataCard content for coffee freshness prediction model
datacard_content = """
# Ethics DataCard for Coffee Freshness Prediction Model
## Dataset Overview
- **Input Variables**: Chemical and physical characteristcs of coffee
- **Output Variables**: Time (days)
## Data Collection Process
- Data was collected over a 21 day period and documented as it was collected.
- All data samples were run in triplicate to reduce error in the dataset.
## Bias Considerations
- Potential Bias: Data only contains a Brazillian coffee, which may not be indicative of all species of coffee.
- Mitigation: For now, the model clearly states this limitation, with future plans to collect more data from different species of origin.
## Fairness & Justice
- The study aims to benefit both large and small coffee producers by providing insights that can reduce waste and improve sustainability for all stakeholders.
- Focuses on equitable outcomes by considering cost-effective methods for measuring and preserving freshness, accessible to producers of varying scales.
## Privacy and Security
- No personal or identifying information is collected; the research focuses on product-level data only.
## Sustainability and Environmental Impact
- The research emphasizes reducing coffee waste by optimizing freshness retention, directly contributing to environmental and economic sustainability.
- Reduces waste from premature disposal of roasted coffee beans.
## Model Limitations
- Model only considers Brazil cerrado coffee at two differnt raost levels.
- Freshness metrics may not capture all sensory attributes valued by diverse consumer groups.
- Results are most applicable to third-wave coffee, with limited direct application to traditional or commercial coffee practices.
## Accountability and Transparency
- Regular reviews and open communication with coffee industry stakeholders ensure the relevance and integrity of the research.
- Findings are shared in a way that promotes transparency, including limitations and potential biases in the dataset.
## Societal Impact
- By reducing waste and improving product quality, the research supports more sustainable coffee production and consumption.
- It has the potential to inform industry-wide practices, benefitting small-scale farmers, roasters, and consumers alike.
"""

# Function to display the DataCard
def display_datacard():
    return datacard_content

# Gradio interface to display the ethics DataCard
iface = gr.Interface(fn=display_datacard, inputs=[], outputs="markdown")

# Launch the Gradio interface
iface.launch()
```

#### Project 5: Effluent Concentration Prediction

##### ReadMe
---

**Effluent Concentration Prediction App**

**Author**

Kayla Yi

**About**

The purpose of this app is to predict effluent electrical conductivity, which is a good measure for indutries to evaluate and improve the quality of water in order to preserve the environment and protect public health. It's important that the wastewater that is returned to the water cycle is free of harmful pollutants and this metric allows us to determine if a treatment is effective. Although measuring electrical conductivity alone does not indicate the exact concentrations of ions in the water, it gives a broad overview on if there are pollutants in the water and if it is safe. Wastewater treatment plants can utilize this tool to determine if their water treatement is effective and can adjust and optimize their treatment plans accordingly.

**Usage**
This app can be run through huggingface (https://huggingface.co/spaces/kaylayi18/EffluentConcentration).

The required libraries are listed in the requirement.txt file. The user inputs various parameters and the model predicts the electrical conductivity of the effluent from these inputs. The user can then compare that prediction with the safe/unsafe EC ranges listed below the prediction to determine if treatment was effective and sufficient. The app also includes Deon's checklist and the ethics datacard that discusses and addresses several ethical considerations and concerns at the bottom of the app.

The user inputs are the following:
- Influent Flowrate: The volume of water that passes through the treatment facility per hour
- Zinc Concentration: Concentration of zinc which contributes to pollution in bodies of water where wastewater is released
- pH: The acidity or alkalinity of the water
- Biochemical Oxygen Demand (BOD): A measure of amount of organic matter in the water
- Chemical Oxygen Demand (COD): A measure of chemicals that can consume oxygen in the water
- Total Suspended Solids (TSS): A measure of the amount of particles suspended in the water
- Volatile Suspended Solids (VSS): A measure of the amount of dissolved organic matter in the water
- Electrical Conductivity (EC): How many dissolved substances, chemicals, and minerals are in the water (can predict how pure water is based on how many ions conduct

**Description**

This repository includes the data preprocessing in google colab, the best/trained machine learning model, and Deon's checklist. The best model identified from training multiple machine learning models is KNN. The datasource for this project is from the Manresa Wastewater Treatment Plant collected from https://github.com/amroohi/Predict-wastewater and https://doi.org/10.1016/j.jenvman.2024.120324, other acknowledgements and references are included in this repository as well.

The ML model that was used to build this app is K-nearest neighbor (KNN). This model was chosen because over three evaluation metrics (RMSE, MAE and R^2), it performed the best. Out of the five ML models tested, it had the lowest RMSE, second to lowest MAE, and highest R^2, so taking all of these into account, KNN would provide the most accurate model. KNN is a supervised ML model uses a data point's proximity to a known set of data points for it's predictions. It's mainly used for tackling classification and regression problems.

Since this model uses data from one water treatment facility, there may be limited representation that does not reflect the diversity of conditions at other wastewater treatment facilities. Certain facilities may have different water treatment plans that are more or less effective than the plant used here and the country where the plant is located may also impose certain regulations that control how certain activities are performed. The size of the plant, climate conditions such as rainfall, and pollutant load may impact influents across plants. While this may provide an accurate prediction for this specific plant, it will not perform as accurately for others. Universal features that are likely to be used at other treatment plants are used in this model, so it can still be used with caution to make loose predictions elsewhere. As a warning, this model is to be used as a prediction only and to consult with professionals before taking any actions.

##### GUI App Deployment




---

```python
import streamlit as st
import pandas as pd
import pickle
import subprocess

subprocess.run(["pip", "install", "numpy"])

# Load the pre-trained model
def load_model():
    with open('best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model
# Title of the app
st.title("Wastewater Effluent Concentration Prediction")
st.write(f"Electrical conductivity is a good measure for indutries to evaluate and improve the quality of water in order to preserve the environment and protect public health. It's important that the wastewater that is returned to the water cycle is free of harmful pollutants and this metric allows us to determine if a treatment is effective.")
st.write(f"The input variables for this model are flowrate, zinc concentration, pH, biochemical oxygen demand (BOD), chemical oxygen demand (COD), total suspended solids (TSS), volitile suspended solids (VSS), and electrical conductivity (EC) of the influent. The output is electrical conductivity of the effluent.")
st.markdown(
"""
Here is what each variable represents:
- Influent Flowrate: The volume of water that passes through the treatment facility per hour
- Zinc Concentration: Concentration of zinc which contributes to pollution in bodies of water where wastewater is released
- pH: The acidity or alkalinity of the water
- Biochemical Oxygen Demand (BOD): A measure of amount of organic matter in the water
- Chemical Oxygen Demand (COD): A measure of chemicals that can consume oxygen in the water
- Total Suspended Solids (TSS): A measure of the amount of particles suspended in the water
- Volatile Suspended Solids (VSS): A measure of the amount of dissolved organic matter in the water
- Electrical Conductivity (EC): How many dissolved substances, chemicals, and minerals are in the water (can predict how pure water is based on how many ions conduct)
"""
)
st.write(f"The model was trained using data collected from a wastewater treatment plant in Barcelona, Spain from the research article https://doi.org/10.1016/j.jenvman.2024.120324")
st.write(f"The ML model that was used to build this app is K-nearest neighbor (KNN). This model was chosen because over three evaluation metrics (RMSE, MAE and R^2), it performed the best. Out of the five ML models tested, it had the lowest RMSE, second to lowest MAE, and highest R^2, so taking all of these into account, KNN would provide the most accurate model. KNN is a supervised ML model uses a data point's proximity to a known set of data points for it's predictions. It's mainly used for tackling classification and regression problems.")
st.write(f"Since this model uses data from one water treatment facility, there may be limited representation that does not reflect the diversity of conditions at other wastewater treatment facilities. Certain facilities may have different water treatment plans that are more or less effective than the plant used here and the country where the plant is located may also impose certain regulations that control how certain activities are performed. The size of the plant, climate conditions such as rainfall, and pollutant load may impact influents across plants. While this may provide an accurate prediction for this specific plant, it will not perform as accurately for others. Universal features that are likely to be used at other treatment plants are used in this model, so it can still be used with caution to make loose predictions elsewhere. As a warning, this model is to be used as a prediction only and to consult with professionals before taking any actions.")
st.subheader(f"Please input values for the following:")

influent_flowrate = st.number_input("Enter the influent Flowrate (L/h):", value=40000)
influent_zinc = st.number_input("Enter concentration of zinc in the influent (%):", min_value=0, max_value=100, value=0)
influent_ph = st.number_input("Enter pH of the influent:", min_value=0, max_value=14, value=7)
influent_BOD = st.number_input("Enter influent Biochemical Oxygen Demand (mg/L):", value=350)
influent_COD = st.number_input("Enter influent Chemical Oxygen Demand (mg/L):", value=350)
influent_TSS = st.number_input("Enter Total Suspended Solids in influent (mg/L)", value=100)
influent_VSS = st.number_input("Enter Volitile Suspended Solids in influent (mg/L):", value=50)
influent_EC = st.number_input("Enter influent Electrical Conductivity (uS/cm):", value=2000)

# Encoding the inputs manually (same encoding as in your training data)
input_data = pd.DataFrame({
    'Q_in': [influent_flowrate],
    'Zn_in': [influent_zinc],
    'pH_in': [influent_ph],
    'BOD_in': [influent_BOD],
    'COD_in': [influent_COD],
    'TSS_in': [influent_TSS],
    'VSS_in': [influent_VSS],
    'EC_in': [influent_EC]

})

# One-hot encode the input data (ensure it matches the training data)
input_encoded = pd.get_dummies(input_data)

# Align columns with the training data (required columns)
model = load_model()
required_columns = model.feature_names_in_  # Get the feature columns from the model
for col in required_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[required_columns]

# Make the prediction
prediction = model.predict(input_encoded)[0]

# Display the prediction
st.subheader(f"Electrical Coductivity of the Effluent: {prediction} uS/cm")
st.write(f"Pure water has low electrical conductivity due to the lack of impurities. Pollutants and contaminants present in wastewater contribute to the electrical conductivity. Although measuring electrical conductivity alone does not indicate the exact concentrations of ions in the water, it gives a broad overview that there are pollutants in the water and if it is safe.")
st.write(f"0-800 uS/cm is considered safe drinking water for humans and good for all livestock.")
st.write(f"800-2500 uS/cm is still safe for consumption, but further treatment if used for irrigation is recommended.")
st.write(f"Above 2500 uS/cm should not be used for human consumption and is not suitable for irrigation and requires further treatment.")

# Deon's Checklist
with open('wastewater_effluent_ethics_checklist.md', 'r') as file:
    checklist_content = file.read()
st.markdown(checklist_content)

# Ethics Data Card
datacard_content = """
# Ethics DataCard for Wastewater Effluent Concentration Prediction Model
## Dataset Overview
- **Input Variables**: Influent Flowrate, zinc, pH, BOD, COD, TSS, VSS, EC
- **Output Variables**: Effluent COD and EC
## Data Collection Process
- Data source collected from Manresa Wastewater Treatment Plant in Barcelona
- Data is collected from public and licensed sources, ensuring proper consent and anonymization of any private information (e.g., personal property location).
## Bias Considerations
- **Potential Bias**: Data source based on one water treatment location so there may be limited representation that does not reflect the diversity of conditions at other wastewater treatment plants.
- **Mitigation**: The model utlizes universal features such as pH and flowrate that are likely consistent across different wastewater treatment plants and is trained to avoid overly specific features that are unique to this specific plant.
## Fairness & Justice
- The model has been trained to predict wastewater effluent concentration based on universal features. Efforts have been made to avoid disproportionate impacts (releasing unsafe water) on vulnerable communities (e.g., rural populations, indigenous lands).
- Special attention is given to reducing false indication of safe electrical conductivity level, balancing the risk for all stakeholders.
## Privacy and Security
- Data collected from Manresa Wastewater Treatment Plant does not contain information that identifies workers.
- No social media or surveillance data is used at all.
- Data source only contains data about influent and effluent wastewater.
## Sustainability and Environmental Impact
- The model aims to evaluate and improve the quality of water in order to preserve the surrounding environment and protect public health.
- It supports long-term environmental sustainability by supporting a circular economy that encourages and optimizes water reuse.
## Model Limitations
- The model's accuracy is limited to the Manresa Wastewater Treatment Plant.
- There are limitations to predicting the effluent concentration of other plants with this model due to plant uniqueness from utilizing different features.
- The model is continuously monitored and tries to utilize universal features in order to be used at other treatment plants.
## Accountability and Transparency
- The development team will monitor the model for performance over time, ensuring that it adapts to new data.
- Stakeholders (e.g., environmental agencies, local governments) will be informed of the model’s limitations, ensuring proper interpretation of the predictions.
- False predictions will be communicated to stakeholders, with a process in place for continuous feedback and model improvement.
## Societal Impact
- The model is designed to protect both human lives and the environment by enabling better resourse allocation to ensure safe water release back into the environment
- It has the potential to inform policy changes in land management, conservation, and public health strategies.
"""
st.markdown(datacard_content)
```

##### Machine Learning Development




---

```python
!pip install scikit-optimize
```

```python
!pip install streamlit
```

```python
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import shap
from skopt import BayesSearchCV, gp_minimize
from skopt.space import Real, Integer
import math
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
```

```python
# Load the dataset
file_name = 'Wastewater Effluent Dataset.xlsx'
df = pd.read_excel(file_name)
df.head()
```

```python
# Check for missing data
missing_data = df.isnull().sum()

# Print the results
print(missing_data)
```

```python
# Fill missing values with the mean of each column
for column in df.columns:
    if df[column].isnull().any():  # Check if column has missing values
        mean_value = df[column].mean()  # Calculate mean of the column
        df[column].fillna(mean_value, inplace=True)  # Fill missing values with the mean

# Verify if there are any missing values left
print(df.isnull().sum())
```

```python
# Calculate the IQR for each column
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

# Remove outliers
df_cleaned = df[~outliers.any(axis=1)]
```

```python
df_cleaned.describe()
```

```python
# Calculate the correlation matrix
correlation_matrix = df_cleaned.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```

```python
# Splitting the dataset into training and test sets
columns_to_drop = [col for col in df_cleaned.columns if 'EC_eff' in col]
X = df_cleaned.drop(columns=columns_to_drop, axis=1)
y = df_cleaned['EC_eff']

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Machine Learning Models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# Training and evaluating the models
best_model = None
best_rmse = float('inf')

for model_name, model in models.items():
    print(f"Training {model_name}...")

    if model_name in ["SVM", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)

    # Calculate evaluation metrics
    rmse_test = math.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Print the metrics for the current model
    print(f"{model_name} Metrics:")
    print(f"  RMSE: {rmse_test:.4f}")
    print(f"  MAE: {mae_test:.4f}")
    print(f"  R^2: {r2_test:.4f}")

    # Save the best model based on RMSE for the test dataset
    if rmse_test < best_rmse:
        best_rmse = rmse_test
        best_model = model
        best_model_name = model_name

    print(f"{model_name} RMSE on Test Data: {rmse_test}")

# Save the best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"Best model: {best_model_name} with RMSE: {best_rmse}")
```

```python
# 1. Grid Search
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

grid_search = GridSearchCV(KNeighborsRegressor(),
                           param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# 2. Random Search
param_dist = {
    'n_neighbors': np.arange(1, 21),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

random_search = RandomizedSearchCV(KNeighborsRegressor(),
                                   param_dist, n_iter=10, scoring='neg_mean_squared_error', cv=5)
random_search.fit(X_train, y_train)

# 3. Bayesian Optimization
search_spaces = {
    'n_neighbors': (1, 20),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

bayes_search = RandomizedSearchCV(KNeighborsRegressor(),
                             search_spaces, n_iter=10, scoring='neg_mean_squared_error', cv=5)
bayes_search.fit(X_train, y_train)

# Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, r2, mae

# Evaluate each method
for model_name, model in [('Grid Search', grid_search.best_estimator_),
                         ('Random Search', random_search.best_estimator_),
                         ('Bayesian Optimization', bayes_search.best_estimator_)]:
    rmse, r2, mae = evaluate_model(model, X_test, y_test)
    print(f"{model_name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R^2: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")

results = {}
for model_name, model in [('Grid Search', grid_search.best_estimator_),
                         ('Random Search', random_search.best_estimator_),
                         ('Bayesian Optimization', bayes_search.best_estimator_)]:
    rmse, r2, mae = evaluate_model(model, X_test, y_test)
    results[model_name] = {'RMSE': rmse, 'R^2': r2, 'MAE': mae}

# Find the best method based on lowest RMSE
best_method = min(results, key=lambda k: results[k]['RMSE'])

# Print the best method
print(f"The best hyperparameter tuning method is: {best_method}")
print(f"with RMSE: {results[best_method]['RMSE']:.4f}, "
      f"R^2: {results[best_method]['R^2']:.4f}, "
      f"and MAE: {results[best_method]['MAE']:.4f}")
```

```python
# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,  # Use all available cores for parallel processing
    return_train_score=True
)

grid_search.fit(X_train, y_train)

results = grid_search.cv_results_

# Create a meshgrid of hyperparameter values
n_estimators_values = param_grid['n_estimators']
max_depth_values = param_grid['max_depth']
N_ESTIMATORS, MAX_DEPTH = np.meshgrid(n_estimators_values, max_depth_values)

# Calculate RMSE for each point on the meshgrid
rmse_values = np.zeros_like(N_ESTIMATORS, dtype=float)
for i, n_estimators in enumerate(n_estimators_values):
    for j, max_depth in enumerate(max_depth_values):
        mask = (results['param_n_estimators'] == n_estimators) & (results['param_max_depth'] == max_depth)
        rmse_values[j, i] = np.sqrt(-results['mean_test_score'][mask][0])

# Create the contour plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(N_ESTIMATORS, MAX_DEPTH, rmse_values, levels=10, cmap='jet')
plt.xlim(100, 300)  # Set x-axis limits
plt.ylim(10, 30)    # Set y-axis limits
plt.colorbar(contour, label='RMSE')
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('Maximum Depth (max_depth)')
plt.title('Contour Plot of RMSE for Grid Search')
plt.show()
```

```python
# Partial Dependence Plot
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Create a SHAP explainer
explainer = shap.KernelExplainer(knn_model.predict, X_train)

# Calculate SHAP values
shap_values = explainer.shap_values(X_train)

# Plot the PDP
shap.partial_dependence_plot(
    "EC_in", knn_model.predict, X_train
)
```

```python
# Beeswarm Plot
model = KNeighborsRegressor()
model.fit(X_train, y_train)

# Create a SHAP explainer
explainer = shap.KernelExplainer(model.predict, X_train[:100])

# Calculate SHAP values for a subset of data
shap_values = explainer.shap_values(X_test[:100])

# Create a beeswarm plot
shap.summary_plot(shap_values, X_test[:100])
```

##### Ethics Analysis




---

Data Science Ethics Checklist

[![Deon badge](https://img.shields.io/badge/ethics%20checklist-deon-brightgreen.svg?style=popout-square)](http://deon.drivendata.org/)

**A. Data Collection**
 - **A.1 Informed consent**: If there are human subjects, have they given informed consent, where subjects affirmatively opt-in and have a clear understanding of the data uses to which they consent?
 -  **A.2 Collection bias**: Have we considered sources of bias that could be introduced during data collection and survey design and taken steps to mitigate those?
 - **A.3 Limit PII exposure**: Have we considered ways to minimize exposure of personally identifiable information (PII) for example through anonymization or not collecting information that isn't relevant for analysis?
 - **A.4 Downstream bias mitigation**: Have we considered ways to enable testing downstream results for biased outcomes (e.g., collecting data on protected group status like race or gender)?

**B. Data Storage**
 - **B.1 Data security**: Do we have a plan to protect and secure data (e.g., encryption at rest and in transit, access controls on internal users and third parties, access logs, and up-to-date software)?
 - **B.2 Right to be forgotten**: Do we have a mechanism through which an individual can request their personal information be removed?
 - **B.3 Data retention plan**: Is there a schedule or plan to delete the data after it is no longer needed?

**C. Analysis**
 - **C.1 Missing perspectives**: Have we sought to address blindspots in the analysis through engagement with relevant stakeholders (e.g., checking assumptions and discussing implications with affected communities and subject matter experts)?
 - **C.2 Dataset bias**: Have we examined the data for possible sources of bias and taken steps to mitigate or address these biases (e.g., stereotype perpetuation, confirmation bias, imbalanced classes, or omitted confounding variables)?
 - **C.3 Honest representation**: Are our visualizations, summary statistics, and reports designed to honestly represent the underlying data?
 - **C.4 Privacy in analysis**: Have we ensured that data with PII are not used or displayed unless necessary for the analysis?
 - **C.5 Auditability**: Is the process of generating the analysis well documented and reproducible if we discover issues in the future?

**D. Modeling**
 - **D.1 Proxy discrimination**: Have we ensured that the model does not rely on variables or proxies for variables that are unfairly discriminatory?
 - **D.2 Fairness across groups**: Have we tested model results for fairness with respect to different affected groups (e.g., tested for disparate error rates)?
 - **D.3 Metric selection**: Have we considered the effects of optimizing for our defined metrics and considered additional metrics?
 - **D.4 Explainability**: Can we explain in understandable terms a decision the model made in cases where a justification is needed?
 - **D.5 Communicate bias**: Have we communicated the shortcomings, limitations, and biases of the model to relevant stakeholders in ways that can be generally understood?

**E. Deployment**
 - **E.1 Redress**: Have we discussed with our organization a plan for response if users are harmed by the results (e.g., how does the data science team evaluate these cases and update analysis and models to prevent future harm)?
 - **E.2 Roll back**: Is there a way to turn off or roll back the model in production if necessary?
 - **E.3 Concept drift**: Do we test and monitor for concept drift to ensure the model remains fair over time?
 - **E.4 Unintended use**: Have we taken steps to identify and prevent unintended uses and abuse of the model and do we have a plan to monitor these once the model is deployed?

*Data Science Ethics Checklist generated with [deon](http://deon.drivendata.org).*

```python
# Ethics Data Card
datacard_content = """
# Ethics DataCard for Wastewater Effluent Concentration Prediction Model

## Dataset Overview
- **Input Variables**: Influent Flowrate, zinc, pH, BOD, COD, TSS, VSS, EC
- **Output Variables**: Effluent COD and EC

## Data Collection Process
- Data source collected from Manresa Wastewater Treatment Plant in Barcelona
- Data is collected from public and licensed sources, ensuring proper consent and anonymization of any private information (e.g., personal property location).

## Bias Considerations
- **Potential Bias**: Data source based on one water treatment location so there may be limited representation that does not reflect the diversity of conditions at other wastewater treatment plants.
- **Mitigation**: The model utlizes universal features such as pH and flowrate that are likely consistent across different wastewater treatment plants and is trained to avoid overly specific features that are unique to this specific plant.

## Fairness & Justice
- The model has been trained to predict wastewater effluent concentration based on universal features. Efforts have been made to avoid disproportionate impacts (releasing unsafe water) on vulnerable communities (e.g., rural populations, indigenous lands).
- Special attention is given to reducing false indication of safe electrical conductivity level, balancing the risk for all stakeholders.

## Privacy and Security
- Data collected from Manresa Wastewater Treatment Plant does not contain information that identifies workers.
- No social media or surveillance data is used at all.
- Data source only contains data about influent and effluent wastewater.

## Sustainability and Environmental Impact
- The model aims to evaluate and improve the quality of water in order to preserve the surrounding environment and protect public health.
- It supports long-term environmental sustainability by supporting a circular economy that encourages and optimizes water reuse.

## Model Limitations
- The model's accuracy is limited to the Manresa Wastewater Treatment Plant.
- There are limitations to predicting the effluent concentration of other plants with this model due to plant uniqueness from utilizing different features.
- The model is continuously monitored and tries to utilize universal features in order to be used at other treatment plants.

## Accountability and Transparency
- The development team will monitor the model for performance over time, ensuring that it adapts to new data.
- Stakeholders (e.g., environmental agencies, local governments) will be informed of the model’s limitations, ensuring proper interpretation of the predictions.
- False predictions will be communicated to stakeholders, with a process in place for continuous feedback and model improvement.

## Societal Impact
- The model is designed to protect both human lives and the environment by enabling better resourse allocation to ensure safe water release back into the environment
- It has the potential to inform policy changes in land management, conservation, and public health strategies.
"""
st.markdown(datacard_content)
```

#### Project 6: Prediction of Catalysis Performance In Dry Methane Reformation

1. Data Integrity & Research Honesty

All data sources used in this project should be accurate, appropriately cited, and free from manipulation to produce misleading or biased results. Researchers must maintain transparency throughout the data collection, preprocessing, and model training processes, ensuring that all methods are clearly documented and reproducible. Any limitations, assumptions, or uncertainties inherent in the predictive model should be explicitly stated to prevent misinterpretation of results.

2. Fairness & Non-Maleficence

The selection and use of data should be carried out in a way that minimizes bias and prevents the model from generating misleading predictions that could disproportionately favor or disadvantage specific catalysts or processes. Researchers must carefully consider the potential negative consequences of incorrect or overly optimistic predictions, especially if the results influence industrial decision-making. The project should not, either directly or indirectly, contribute to the promotion of catalytic technologies that could cause harm to the environment or society.

3. Environmental Responsibility

Given the significant environmental implications of methane reformation, it is essential to evaluate the ecological impact of the catalysts and processes being studied. The project should align with broader efforts to reduce greenhouse gas emissions and minimize resource waste, ensuring that its findings contribute to more sustainable industrial practices. Researchers should be mindful of the long-term environmental consequences of promoting specific catalysts and avoid endorsing solutions that may lead to increased ecological harm over time.

4. Safety & Accountability

The limitations of the predictive model must be communicated clearly to ensure that it is not misused or relied upon in ways that could compromise safety in real-world catalytic applications. Any recommendations derived from the model should prioritize process stability and worker safety, preventing hazardous conditions from arising due to inaccurate predictions. All research steps should be meticulously documented to maintain accountability and allow for independent verification of the results by other researchers.

5. Respect for Intellectual Property & Collaboration Ethics

All contributions, whether from team members, external researchers, or open-source tools, should be properly acknowledged to uphold academic integrity and give credit where it is due. The project should not make unauthorized use of proprietary data or algorithms without obtaining the necessary permissions. Ethical collaboration should be maintained throughout the research process, ensuring fair recognition of all individuals involved and avoiding any potential conflicts of interest that could compromise the integrity of the work.

6. Duty to Society & Scientific Community

The findings of this project should be shared in a responsible and ethical manner, ensuring that they contribute meaningfully to advancements in catalysis research without overstating the model’s capabilities. Any claims made about the predictive accuracy or potential applications of the model should be carefully substantiated to prevent the spread of misinformation. Research communication should be structured in a way that is accessible not only to experts in the field but also to a broader audience, allowing for a wider understanding of its implications and potential benefits.

##### Read Me



---

**Prediction of Catalysis Performance during Dry reforming of Methane**

**Problem Statement**:

The Dry Reforming of Methane (DRM) is a process for converting methane (CH₄) and carbon dioxide (CO₂) into syngas, a mixture of hydrogen (H₂) and carbon monoxide (CO). In the manufacturing industry, this method is used for the production of fuels and chemicals mitigating the impact of greenhouse gases through reduction and utilization. Nonetheless, the sustainability of DRM  largely depends on the catalyst performance which affected by factors like deactivation resulting from coking, particle sintering, and catalyst poisoning. Catalyst performance is affected by operating conditions such as temperature, pressure, feed gas composition, GSHV and many other catalyst properties. To improve the efficiency and sustainability of the DRM process, consistent metrics concerns have to be addressed (e.g. scalability) required for catalysis performance prediction regardless of reaction condition.

The goal of this project is to  develop a predictive model that forecasts the catalytic performance during the Dry Reforming of Methane process. Leveraging machine learning techniques, key performance indicators (KPIs) such as reaction rate, catalyst, coke formation, and syngas yield, based on a set of input parameters, including catalyst type, reaction conditions, and feed composition. By addressing these challenges, this project will contribute to optimizing the DRM process, reducing catalyst deactivation rates, improving syngas production efficiency, and ultimately supporting sustainable industrial practices in methane utilization and carbon dioxide mitigation.

**Input and Output Variables**:
The features (inputs) of the dataset include Reaction Temperature, Ratio of CH₄ in Feed, Ni Loading, Surface Area, GHSV, Reaction Time, Pore Size, Pore Volume, H2-TPR Peak Temperature , Ni Particle Size, Ni Dispersion, Modifier Electronegativity. The targets (output) are  Syngas Ratio, CO₂ Conversion, or CH₄ Conversion

**Machine Learning Algorithm(s)**:
Regression models such as Support Vector Regression (SVR), Random Forest, Linear Regression and K-Nearest Neighbors (KNN) were applied. The classification models used are RandomForestClassifier, Logistic Regression, SVC and KNN Classifier.

**Ethics Considerations**: https://github.com/Chi36/C.-Frank-Onwudinjo/blob/main/C_Frank_Onwudinjo__Catalysis_Performance_Prediction_Ethics_DataCard_and_Deon_Checklist.ipynb

**Graphical User Interface (GUI)**: https://huggingface.co/spaces/Frankie89/Catalysis_Performance_1

**Dataset**: Datasets were sourced from publicly available research journals recommended by the course instructor. Data preprocessing done include checking for missing values, summary statistics and correlation matrix obtained using heatmaps.

**Project Details**:
This predicts catalyst performance based on the available datasets. Install specific versions of libraries, resolve file path and load datasets. Check the columns in the dataframe and remove the first the column. Define features and target columns. Check if all required feature columns exist performing imputation for missing values. Then standardize features and define models to be used.  Separate  training and saving models for each target variable. Perform hyperparameter tuning to determine the best model using GridSearch. Define a unified prediction function based on selected target. Prepare input data for prediction while ensuring that input data has the same structure as X. Now, load the model based on the target variable. Using Gradio, display the Interface with defined inputs and dropdown to select the target variable. Finally, push GUI to Hugging Face!


**The names of the author and the instructor**:                  Chimezie Frank Onwudinjo,  Dr. Jude A. Okolie                                                                                                                     
**Your department and university information**:                  Chemical Engineering Departmnent, Bucknell University

**Course Code**:                                                        CHEG 672 (Data Science in Chemical Engineering)

##### GUI App Development



---

```python
# Install specific versions of libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import gradio as gr

# Dynamically resolve file path using os
file_path = os.path.join(os.getcwd(), "cleaned_data_no_missing_values_numerical.xlsx")

# Load dataset
df = pd.read_excel(file_path)

# Check the columns in the DataFrame
print("Columns in the DataFrame:", df.columns.tolist())

# Remove the first column
df = df.iloc[:, 1:]

# Define the updated feature columns
features = ['Ni Loading', 'Surface Area', 'GHSV', 'Ratio of CH4 in Feed', 'Reaction Temperature',
            'Reaction Time', 'Pore Size', 'Pore Volume', 'H2-TPR Peak Temperature', 'Ni Particle Size',
            'Ni Dispersion', 'Modifier Electronegativity']

# Define multiple target variables
target_variables = ['Syngas_Ratio', 'CO2 Conversion', 'CH4 Conversion']

# Check if all required feature columns exist
missing_columns = [col for col in features if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing required columns: {missing_columns}")

# Prepare the data
X = df[features]  # Only use the features
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)  # Fill any remaining NaN values with 0

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models to be used
models = {
    'RandomForest': RandomForestRegressor(),
    'LinearRegression': LinearRegression(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
}

# Hyperparameter grids for tuning models
param_grids = {
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [5, 10, None]},
    'LinearRegression': {'fit_intercept': [True, False]},
    'SVR': {'C': [1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf']},
    'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
}

# Separate training and saving models for each target variable
best_models = {}
for target_variable in target_variables:
    y = df[target_variable]  # Separate target variable for training

    best_score = -np.inf
    best_model = None
    for model_name, model in models.items():
        try:
            print(f"Training {model_name} for {target_variable}")

            # Hyperparameter tuning using GridSearchCV
            grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='r2')
            grid_search.fit(X_scaled, y)

            # Select the best model and its parameters
            mean_score = grid_search.best_score_
            if mean_score > best_score:
                best_score = mean_score
                best_model = (model_name, grid_search.best_estimator_)

        except Exception as e:
            print(f"Warning: Failed to train {model_name} for {target_variable}: {e}")

    if best_model:
        model_name, model = best_model
        # Save each model for the corresponding target variable separately
        with open(f'best_{model_name}_{target_variable}.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        best_models[target_variable] = (model_name, model)

# Define a unified prediction function based on selected target
def predict_catalysis_metrics(target_variable, Reaction_Temperature, Ratio_of_CH4_in_Feed, Ni_Loading, Surface_Area,
                              GHSV, Reaction_Time, Pore_Size, Pore_Volume, H2_TPR_Peak_Temperature,
                              Ni_Particle_Size, Ni_Dispersion, Modifier_Electronegativity):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Reaction Temperature': [Reaction_Temperature],
        'Ratio of CH4 in Feed': [Ratio_of_CH4_in_Feed],
        'Ni Loading': [Ni_Loading],
        'Surface Area': [Surface_Area],
        'GHSV': [GHSV],
        'Reaction Time': [Reaction_Time],
        'Pore Size': [Pore_Size],
        'Pore Volume': [Pore_Volume],
        'H2-TPR Peak Temperature': [H2_TPR_Peak_Temperature],
        'Ni Particle Size': [Ni_Particle_Size],
        'Ni Dispersion': [Ni_Dispersion],
        'Modifier Electronegativity': [Modifier_Electronegativity],
    })

    # Ensure input data has the same structure as X
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)

    # Load the model based on the target variable
    try:
        model_name, model = best_models[target_variable]
        prediction = model.predict(input_data_scaled)[0]

        if target_variable == 'Syngas_Ratio':
            return {f'{model_name} - {target_variable}': f"{prediction:.2f} (unitless)"}
        elif target_variable == 'CO2 Conversion':
            return {f'{model_name} - {target_variable}': f"{prediction:.2f} (%)"}
        elif target_variable == 'CH4 Conversion':
            return {f'{model_name} - {target_variable}': f"{prediction:.2f} (%)"}
    except KeyError:
        return {"Error": f"No model found for {target_variable}"}

# Gradio Interface: Define the inputs and dropdown to select the target variable
inputs = [
    gr.Dropdown(choices=['Syngas_Ratio', 'CO2 Conversion', 'CH4 Conversion'], label="Select Target Variable", type="value"),
    gr.Number(label='Reaction Temperature (°C)', value=0, precision=2),
    gr.Number(label='Ratio of CH4 in Feed', value=0, precision=2),
    gr.Number(label='Ni Loading (%)', value=0, precision=2),
    gr.Number(label='Surface Area (m²/g)', value=0, precision=2),
    gr.Number(label='GHSV (h⁻¹)', value=0, precision=2),
    gr.Number(label='Reaction Time (hours)', value=0, precision=2),
    gr.Number(label='Pore Size (nm)', value=0, precision=2),
    gr.Number(label='Pore Volume (cm³/g)', value=0, precision=2),
    gr.Number(label='H2-TPR Peak Temperature (°C)', value=0, precision=2),
    gr.Number(label='Ni Particle Size (nm)', value=0, precision=2),
    gr.Number(label='Ni Dispersion (%)', value=0, precision=2),
    gr.Number(label='Modifier Electronegativity', value=0, precision=2),
]

# Output: JSON format for the prediction
outputs = gr.JSON()

# Gradio interface
app = gr.Interface(fn=predict_catalysis_metrics,
                   inputs=inputs,
                   outputs=outputs,
                   title="Catalysis Performance Metrics",
                   description="Select a target variable to predict (Syngas Ratio, CO2 Conversion, or CH4 Conversion) based on catalyst metrics. Units will be displayed with the results.")

# Launch the app
if __name__ == "__main__":
    app.launch()
```

##### Machine Learning Development



---

**Study of catalysis performance during dry methane reforming with machine learning**

An individual project on machine learning for the data science in chemical engineering (Fall 2024).
Name: Chimezie Frank ONWUDINJO

**Acknowledgement page**

I write to acknowledge the support of all authors whose resources were consulted as well as the innovators of essential platforms instrumental to the accomplishing this capstone project.

Also worthy of my gratitude is the course instructor (Dr. Jude Okolie) whose relentless effort geared towards demystifying machine learning concepts.

My appreciation goes to the management of Bucknell University for incorporating this data science course into the chemical engineering program.
Thank you!

**Statement of AI Usage**

I write to affirm that AI tools such as chatgpt and googlegemni were helpful tools especially for adjusting the codes and debugging with relevant prompts.

**Data preprocessing**

Load dataset and display first few rows

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel('/content/Data DRM_Git.xlsx')

# Display the first few rows
print(df.head())
```

**Data overview**

```python
# Basic information about the dataset
print(df.info())
```

**Summary statistics**

```python
# Summary statistics
print(df.describe())
```

**Check for missing values**

```python
import pandas as pd

# Load your dataset into a DataFrame
df = pd.read_excel('/content/Data DRM_Git.xlsx')

# Check for missing values
missing_values = df.isnull().sum()

# Print the count of missing values for each column
print("Missing values per column:")
print(missing_values)

# Check if there are any missing values in the entire DataFrame
has_missing_values = df.isnull().any().any()
print("Has missing values:", has_missing_values)
```

**Imputation for missing values**

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Step 1: Load the data
df = pd.read_excel('/content/Data DRM_Git.xlsx')  # Replace with your actual file path

# Step 2: Identify numerical columns (exclude non-relevant columns like 'Unnamed: 0')
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns  # Numerical columns
numerical_columns = [col for col in numerical_columns if 'Unnamed' not in col]  # Remove 'Unnamed' columns
print("Numerical columns:", numerical_columns)  # Debugging: Check the numerical columns

# Step 3: Check if there are numerical columns
if len(numerical_columns) > 0:
    # Step 4: Impute missing values for numerical columns (using median for numerical columns)
    imputer_num = SimpleImputer(strategy='median')  # Using median for numerical columns

    # Ensure that we have valid numerical columns to impute
    df[numerical_columns] = imputer_num.fit_transform(df[numerical_columns])

# Step 5: Verify the imputation for numerical columns
print("\nAfter numerical imputation, missing data summary:")
print(df.isnull().sum())  # Check if there are still any missing values

# Step 6: Save the cleaned dataset (optional)
df.to_excel('cleaned_data_no_missing_values_numerical.xlsx', index=False)  # Save the cleaned dataset to a new Excel file
```

**Recheck for missing values**

```python
import pandas as pd
import numpy as np

# df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# Step 1: Identify numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Step 2: Generate synthetic data for missing numerical values
for col in numerical_columns:
    # Get the column data
    data = df[col]


# Step 2: Verify the result
print(f"Data shape after imputation: {df.shape}")
print(f"Missing values after imputation:\n{df.isnull().sum()}")
```

**Check for outliers**

```python
import pandas as pd
import numpy as np
import scipy.stats as stats

# df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# Step 1: Identify numerical columns in the DataFrame
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Step 2: Function to detect outliers using Z-score
def detect_outliers_zscore(df, columns, threshold=3):
    outliers = pd.DataFrame()  # Initialize an empty DataFrame to store outlier information
    for col in columns:
        # Calculate the Z-score for the column
        z_scores = stats.zscore(df[col].dropna())  # Drop NaN values before calculating z-score
        # Identify outliers: If Z-score > threshold, consider it an outlier
        outlier_idx = np.where(np.abs(z_scores) > threshold)[0]
        if len(outlier_idx) > 0:
            outliers[col] = df.iloc[outlier_idx][col]
    return outliers

# Step 3: Function to detect outliers using IQR
def detect_outliers_iqr(df, columns):
    outliers = pd.DataFrame()  # Initialize an empty DataFrame to store outlier information
    for col in columns:
        # Calculate the Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers: If data is below lower bound or above upper bound, it's an outlier
        outlier_idx = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        if len(outlier_idx) > 0:
            outliers[col] = df.loc[outlier_idx, col]
    return outliers

# Step 4: Detect outliers using Z-score
zscore_outliers = detect_outliers_zscore(df, numerical_columns)

# Step 5: Detect outliers using IQR
iqr_outliers = detect_outliers_iqr(df, numerical_columns)

# Step 6: Print outliers detected by both methods
print("Outliers detected using Z-score method:")
print(zscore_outliers)

print("\nOutliers detected using IQR method:")
print(iqr_outliers)

# Optionally, you can save the outliers to an Excel file:
# zscore_outliers.to_excel('zscore_outliers.xlsx', index=False)
# iqr_outliers.to_excel('iqr_outliers.xlsx', index=False)
```

**Box plot for each variable**

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (adjust the path as needed)
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')  # Replace with your actual file path

# Ensure column names are clean and consistent (remove extra spaces, lowercase)
df.columns = df.columns.str.strip()  # Strip leading/trailing spaces from column names


# Remove the first column (which might not be 'Catalyst')
df = df.iloc[:, 1:]

Ni_Loading_column = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx', usecols=['Ni Loading'])


# Get the list of columns in the dataframe excluding the first column
columns_to_plot = df.columns.tolist()

# Box plots to show the distribution of chemical components by 'Catalyst Preparation Method'
sns.set_theme(style="whitegrid")

# Determine the number of rows and columns for subplots based on the number of columns to plot
n_cols = 3  # You can adjust this depending on how many columns you want per row
n_rows = (len(columns_to_plot) // n_cols) + (1 if len(columns_to_plot) % n_cols != 0 else 0)

# Create subplots with dynamic rows and columns
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop over the columns to create box plots
for i, col in enumerate(columns_to_plot):
    if col in df.columns:
        sns.boxplot(data=df, x='Ni Loading', y=col, ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_xlabel('Ni Loading')
        axes[i].set_ylabel(f'{col}')
    else:
        print(f"Warning: '{col}' not found in DataFrame columns")

# Hide any unused axes if there are more subplots than columns to plot
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust the layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()
```

**Check for categorical variables**

```python
# Get the data types of each column
data_types = df.dtypes

# Print the data types
print(data_types)

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("\nNumeric columns:", numeric_cols)
print("\nCategorical columns:", categorical_cols)
```

**Heatmaps**

Remove categorical variables before plotting heatmaps

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# Remove categorical variables
data_numerical = data.select_dtypes(include=['float64', 'int64'])

# Display the numerical DataFrame
print("\nNumerical DataFrame (Categorical Variables Removed):")
print(data_numerical.head())

# Calculate the correlation matrix
correlation_matrix = data_numerical.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,            # Annotate cells with correlation coefficients
    fmt=".2f",            # Format for the annotation
    cmap='coolwarm',      # Color map for the heatmap
    square=True,          # Make cells square-shaped
    cbar_kws={"shrink": .8}  # Adjust color bar size
)

plt.title('Heatmap of Correlation Matrix (Numerical Variables Only)')
plt.show()
```

**Feature Engineering**

Splitting dataset into features and targets

Targets (y) = Syngas Ratio, CO2 Conversion, CH4 Conversion

Features (x) = Ratio of CH4 in Feed, Reaction Temperature, Ni Loading (catalyst), Reaction Time, Pore Size, Pore Volume, Surface Area, H2-TPR Peak Temperature, Ni Particle Size, Ni Dispersion, Modifier Electronegativity, GHSV

```python
import pandas as pd

# Load your data (replace 'your_data.xlsx' with your actual file path)
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# Define target variables
target_variables = ['Syngas_Ratio', 'CH4 Conversion', 'CO2 Conversion']

# Split the data into features (X) and target variables (y)
X = df.drop(columns=target_variables)

# Create separate target variables (y) for each target
y_syngas_ratio = df['Syngas_Ratio']
y_ch4_conversion = df['CH4 Conversion']
y_co2_conversion = df['CO2 Conversion']

# Print out the features and target variables for verification
print("Features (X):")
print(X.head())

print("Target Variable (Syngas_Ratio):")
print(y_syngas_ratio.head())

print("Target Variable (CH4_Conversion):")
print(y_ch4_conversion.head())

print("Target Variable (CO2_Conversion):")
print(y_co2_conversion.head())
```

**Histograms of target vs features**

Histogram for CH4 Conversion vs Reaction Temperature

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate random data for the histogram (if intended)
data = np.random.randn(1000)

# Load your dataset
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# If you want to plot data from your DataFrame, replace 'data' with the appropriate column
# Example: data = df['YourColumnName'].values

# Plotting a basic histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Reaction Temperature')
plt.ylabel('CH4 Conversion')
plt.title('Histogram for CH4 Conversion vs Reaction Temperature')

# Display the plot
plt.show()
```

Histogram for CH4 Conversion vs Ni Loading

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate random data for the histogram (if intended)
data = np.random.randn(1000)

# Load your dataset
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# If you want to plot data from your DataFrame, replace 'data' with the appropriate column
# Example: data = df['YourColumnName'].values

# Plotting a basic histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Ni Loading')
plt.ylabel('CH4 Conversion')
plt.title('Histogram for CH4 Conversion vs Ni Loading')

# Display the plot
plt.show()
```

Histogram for CH4 Conversion vs Ratio of CH4 in Feed

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate random data for the histogram (if intended)
data = np.random.randn(1000)

# Load your dataset
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# If you want to plot data from your DataFrame, replace 'data' with the appropriate column
# Example: data = df['YourColumnName'].values

# Plotting a basic histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Ratio of CH4 in Feed')
plt.ylabel('CH4 Conversion')
plt.title('Histogram for CH4 Conversion vs Ratio of CH4 in Feed')

# Display the plot
plt.show()
```

Histogram for CO2 Conversion vs Ni Loading

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate random data for the histogram (if intended)
data = np.random.randn(1000)

# Load your dataset
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# If you want to plot data from your DataFrame, replace 'data' with the appropriate column
# Example: data = df['YourColumnName'].values

# Plotting a basic histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Ratio of CH4 in Feed')
plt.ylabel('CO2 Conversion')
plt.title('Histogram for CO2 Conversion vs Ratio of CH4 in Feed')

# Display the plot
plt.show()
```

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate random data for the histogram (if intended)
data = np.random.randn(1000)

# Load your dataset
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# If you want to plot data from your DataFrame, replace 'data' with the appropriate column
# Example: data = df['YourColumnName'].values

# Plotting a basic histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Ni Loading')
plt.ylabel('CO2 Conversion')
plt.title('Histogram for CO2 Conversion vs Ni Loading')

# Display the plot
plt.show()
```

Histogram for CO2 Conversion vs Reaction Temperature

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate random data for the histogram (if intended)
data = np.random.randn(1000)

# Load your dataset
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# If you want to plot data from your DataFrame, replace 'data' with the appropriate column
# Example: data = df['YourColumnName'].values

# Plotting a basic histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Reaction Temperature')  # Combine labels into a single string if needed
plt.ylabel('CO2 Conversion')
plt.title('Histogram for CO2 Conversion vs Reaction Temperature')

# Display the plot
plt.show()
```

Histogram for 'Syngas_Ratio vs 'Reaction Temperature'

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate random data for the histogram (if intended)
data = np.random.randn(1000)

# Load your dataset
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# Plotting a basic histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Reaction Temperature')  # Combine labels into a single string if needed
plt.ylabel('Syngas_Ratio,')
plt.title('Histogram for Syngas Ratio vs Reaction Temperature')

# Display the plot
plt.show()
```

Histogram for 'Syngas_Ratio' vs 'Ratio of CH4 in Feed'

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate random data for the histogram (if intended)
data = np.random.randn(1000)

# Load your dataset
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# If you want to plot data from your DataFrame, replace 'data' with the appropriate column
# Example: data = df['YourColumnName'].values

# Plotting a basic histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Ratio of CH4 in the Feed')
plt.ylabel('Syngas_Ratio,')
plt.title('Histogram for Syngas_Ratio vs Ratio of CH4 in Feed')

# Display the plot
plt.show()
```

Histogram for Syngas_Ratio vs Ni Loading

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate random data for the histogram (if intended)
data = np.random.randn(1000)

# Load your dataset
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')


# Plotting a basic histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Ni Loading')  # Combine labels into a single string if needed
plt.ylabel('Syngas_Ratio,')
plt.title('Histogram for Syngas_Ratio vs Ni Loading')

# Display the plot
plt.show()
```

**Selection of appropriate and suitable Machine Learning Model**

**Training of datasets and evaluation of models**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Load your dataset
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# Remove the first column (only the first column)
df = df.drop(df.columns[0], axis=1)

# List of feature columns
features = ['Ni Loading', 'Surface Area', 'GHSV', 'Reaction Temperature', 'Ratio of CH4 in Feed',
            'Reaction Time', 'Pore Size', 'Pore Volume', 'H2-TPR Peak Temperature', 'Ni Particle Size',
            'Ni Dispersion', 'Modifier Electronegativity']

# Define multiple target variables
target_variables = ['Syngas_Ratio', 'CO2 Conversion', 'CH4 Conversion']

# Fill NaN values with the mean of each column (for both features and target variables)
df[features + target_variables] = df[features + target_variables].fillna(df.mean())

# Dictionary to hold the performance results
results = {}

# Loop over each target variable to train and evaluate models for each
for target_variable in target_variables:
    # Prepare the feature set (X) and current target variable (y)
    X = df[features]
    y = df[target_variable]

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the models to evaluate
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Support Vector Regressor": SVR(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree Regressor": DecisionTreeRegressor()
    }

    # Evaluate each model
    model_results = {}
    for name, model in models.items():
        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        predictions = model.predict(X_test)

        # Calculate Mean Absolute Error (MAE) and R² for model evaluation
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Store the result in the dictionary
        model_results[name] = {'MAE': mae, 'R²': r2}

        # Print the result for this model
        print(f"{target_variable} - {name}: MAE = {mae:.4f}, R² = {r2:.4f}")

    # Store the results for each target variable
    results[target_variable] = model_results

    # Identify the best model for the current target variable (the model with the lowest MAE and highest R²)
    best_model_name = min(model_results, key=lambda x: model_results[x]['MAE'])
    best_model_mae = model_results[best_model_name]['MAE']
    best_model_r2 = model_results[best_model_name]['R²']

    # Output the best model and its performance for the current target variable
    print(f"\nBest Model for {target_variable}: {best_model_name} with MAE = {best_model_mae:.4f} and R² = {best_model_r2:.4f}\n")
```

**Cross Validation**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Load your dataset
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# Remove the first column (only the first column)
df = df.drop(df.columns[0], axis=1)

# List of feature columns
features = ['Ni Loading', 'Surface Area', 'GHSV', 'Reaction Temperature', 'Ratio of CH4 in Feed',
            'Reaction Time', 'Pore Size', 'Pore Volume', 'H2-TPR Peak Temperature', 'Ni Particle Size',
            'Ni Dispersion', 'Modifier Electronegativity']

# Define multiple target variables
target_variables = ['Syngas_Ratio', 'CO2 Conversion', 'CH4 Conversion']

# Fill NaN values with the mean of each column (for both features and target variables)
df[features + target_variables] = df[features + target_variables].fillna(df.mean())

# Dictionary to hold the performance results
results = {}

# Loop over each target variable to train and evaluate models for each
for target_variable in target_variables:
    # Prepare the feature set (X) and current target variable (y)
    X = df[features]
    y = df[target_variable]

    # Define the models to evaluate
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Support Vector Regressor": SVR(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree Regressor": DecisionTreeRegressor()
    }

    # Initialize a dictionary to store the cross-validation results
    model_results = {}

    for name, model in models.items():
        # Perform cross-validation (using 5-fold cross-validation)

        # MAE using negative mean absolute error for cross-validation (because sklearn's cross_val_score returns the negative of the MAE)
        mae_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')

        # R² scores (using R² for cross-validation)
        r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        # Calculate the average MAE and R² scores
        mean_mae = -mae_scores.mean()  # Convert back to positive MAE
        mean_r2 = r2_scores.mean()

        # Store the results in the dictionary
        model_results[name] = {'MAE': mean_mae, 'R²': mean_r2}

        # Print the result for this model
        print(f"{target_variable} - {name}: Mean MAE = {mean_mae:.4f}, Mean R² = {mean_r2:.4f}")

    # Store the results for each target variable
    results[target_variable] = model_results

    # Identify the best model for the current target variable (the model with the lowest MAE and highest R²)
    best_model_name = min(model_results, key=lambda x: model_results[x]['MAE'])
    best_model_mae = model_results[best_model_name]['MAE']
    best_model_r2 = model_results[best_model_name]['R²']

    # Output the best model and its performance for the current target variable
    print(f"\nBest Model for {target_variable}: {best_model_name} with Mean MAE = {best_model_mae:.4f} and Mean R² = {best_model_r2:.4f}\n")
```

Display a plot of training and testing
𝑅^2 scores for different models

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Load your dataset
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')

# Remove the first column (usually an index or irrelevant column)
df = df.drop(df.columns[0], axis=1)

# List of feature columns
features = ['Ni Loading', 'Surface Area', 'GHSV', 'Reaction Temperature', 'Ratio of CH4 in Feed',
            'Reaction Time', 'Pore Size', 'Pore Volume', 'H2-TPR Peak Temperature', 'Ni Particle Size',
            'Ni Dispersion', 'Modifier Electronegativity']

# Define multiple target variables
target_variables = ['Syngas_Ratio', 'CO2 Conversion', 'CH4 Conversion']

# Fill NaN values with the mean of each column (features + target columns)
df[features + target_variables] = df[features + target_variables].fillna(df.mean())

# Split the data into features (X)
X = df[features]

# Define the models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Regressor": SVR(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor()
}

# Dictionaries to hold the R^2 scores for each target variable
train_r2_scores = {target: {} for target in target_variables}
test_r2_scores = {target: {} for target in target_variables}

# Train and evaluate each model for each target variable
for target_variable in target_variables:
    y = df[target_variable]  # Select the target variable

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Calculate R^2 score for training and testing data
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)

        # Store the R^2 scores for each target variable
        train_r2_scores[target_variable][name] = train_r2
        test_r2_scores[target_variable][name] = test_r2

# Plotting the R^2 scores for training and testing across different target variables and models
for target_variable in target_variables:
    labels = list(models.keys())  # Model names

    # Bar width
    bar_width = 0.35

    # X axis positions for the bars
    r1 = np.arange(len(labels))  # Position for training bars
    r2 = [x + bar_width for x in r1]  # Position for testing bars

    # Create the plot for each target variable
    plt.figure(figsize=(12, 6))

    # Plot training R^2 scores
    plt.bar(r1, train_r2_scores[target_variable].values(), color='b', width=bar_width, edgecolor='grey', label='Training R²')

    # Plot testing R^2 scores
    plt.bar(r2, test_r2_scores[target_variable].values(), color='r', width=bar_width, edgecolor='grey', label='Testing R²')

    # Add labels and title
    plt.xlabel('Model', fontweight='bold', fontsize=14)
    plt.ylabel('R² Score', fontweight='bold', fontsize=14)
    plt.title(f'Training and Testing R² Scores for {target_variable}', fontsize=16)
    plt.xticks([r + bar_width / 2 for r in range(len(labels))], labels, rotation=45)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()
```

**Hyperparameter tuning**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.utils import shuffle

# Load your dataset
df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')  # Adjust the file path as necessary

# Remove the first column (usually an index or irrelevant column)
df = df.drop(df.columns[0], axis=1)

# List of feature columns
features = ['Ni Loading', 'Surface Area', 'GHSV', 'CO2 Conversion', 'CH4 Conversion', 'Reaction Temperature',
            'Reaction Time', 'Pore Size', 'Pore Volume', 'H2-TPR Peak Temperature', 'Ni Particle Size',
            'Ni Dispersion', 'Modifier Electronegativity']

# Define multiple target variables
target_variables = ['Syngas_Ratio', 'CO2 Conversion', 'CH4 Conversion']

# Shuffle the data to ensure no order bias
df = shuffle(df, random_state=42)

# Prepare the feature set (X)
X = df[features]

# Define the Random Forest model
random_forest_model = RandomForestRegressor(random_state=42)

# Hyperparameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': np.arange(10, 200, 10),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': np.arange(2, 20),
    'min_samples_leaf': np.arange(1, 20),
    'bootstrap': [True, False]
}

# Dictionaries to hold the R^2 scores and MAE for each target variable
train_r2_scores = {target: [] for target in target_variables}
test_r2_scores = {target: [] for target in target_variables}
mae_scores = {target: [] for target in target_variables}

# Train and evaluate for each target variable
for target_variable in target_variables:
    y = df[target_variable]  # Select the target variable

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up RandomizedSearchCV with 5-fold cross-validation
    random_search = RandomizedSearchCV(estimator=random_forest_model, param_distributions=param_dist,
                                       n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

    # Perform the random search
    random_search.fit(X_train, y_train)

    # Get the best model from the random search
    best_model = random_search.best_estimator_

    # Make predictions using the best model
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Calculate performance metrics for Random Forest
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    # Store the evaluation results
    train_r2_scores[target_variable] = train_r2
    test_r2_scores[target_variable] = test_r2
    mae_scores[target_variable] = mae

    # Print the evaluation results for Random Forest
    print(f"\nTarget Variable: {target_variable}")
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Training R^2: {train_r2:.4f}")
    print(f"Testing R^2: {test_r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Visualization: Plotting R^2 Scores for Training and Testing
for target_variable in target_variables:
    labels = ['Random Forest']  # Model names

    # Bar width
    bar_width = 0.35  # Width of bars

    # X axis positions for the bars
    r1 = np.arange(len(labels))
    r2 = [x + bar_width for x in r1]

    # Create the plot for each target variable
    plt.figure(figsize=(12, 6))

    # Plot training R^2 scores
    plt.bar(r1, [train_r2_scores[target_variable]], color='b', width=bar_width, edgecolor='grey', label='Training R²')

    # Plot testing R^2 scores
    plt.bar(r2, [test_r2_scores[target_variable]], color='r', width=bar_width, edgecolor='grey', label='Testing R²')

    # Add labels and title
    plt.xlabel('Model', fontweight='bold', fontsize=14)
    plt.ylabel('R² Score', fontweight='bold', fontsize=14)
    plt.title(f'Training and Testing R² Scores for {target_variable} (after Hyperparameter Tuning)', fontsize=16)
    plt.xticks([r + bar_width / 2 for r in range(len(labels))], labels)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

# Visualization: Plotting Mean Absolute Error (MAE)
for target_variable in target_variables:
    plt.figure(figsize=(10, 6))

    # Plot MAE for Random Forest
    plt.bar([target_variable], [mae_scores[target_variable]], color='orange', edgecolor='grey')

    # Add labels and title for MAE
    plt.xlabel('Target Variable', fontweight='bold', fontsize=14)
    plt.ylabel('Mean Absolute Error (MAE)', fontweight='bold', fontsize=14)
    plt.title(f'Mean Absolute Error (MAE) for Random Forest (after Hyperparameter Tuning) for {target_variable}', fontsize=16)

    # Display the plot
    plt.tight_layout()
    plt.show()
```

**Feature importance analysis with SHAP**

Here, there is also the best model estimator.

The procedure is as follows:
- Load your dataset
- Remove the irrelevant columns
- Define the features and target variables
- Handle missing values replacing with the mean of each column (if any)
- Key in the best hyperparameters
- Evaluate the best model
- SHAP analysis
- Display SHAP analysis plot

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import shap

df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')  # Adjust the file path as necessary

df = df.drop(df.columns[0], axis=1)

features = ['Ni Loading', 'Surface Area', 'GHSV', 'Ratio of CH4 in Feed', 'Reaction Temperature',
            'Reaction Time', 'Pore Size', 'Pore Volume', 'H2-TPR Peak Temperature', 'Ni Particle Size',
            'Ni Dispersion', 'Modifier Electronegativity']

target_variables = ['Syngas_Ratio', 'CO2 Conversion', 'CH4 Conversion']

X = df[features]

X = X.fillna(X.mean())

for target in target_variables:
    df[target] = df[target].fillna(df[target].mean())  # Replace missing target values with mean

# Prepare the target variable (y) for the first target (Syngas_Ratio)
y = df[target_variables[0]]  # You can choose one of the target variables, e.g., 'Syngas_Ratio'

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': np.arange(10, 200, 10),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': np.arange(2, 20),
    'min_samples_leaf': np.arange(1, 20),
    'bootstrap': [True, False]
}

# Initialize the Random Forest model
random_forest_model = RandomForestRegressor(random_state=42)

# Set up RandomizedSearchCV with 5-fold cross-validation
random_search = RandomizedSearchCV(estimator=random_forest_model, param_distributions=param_dist,
                                   n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Perform the random search
random_search.fit(X_train, y_train)

# Get the best model from the random search
best_model = random_search.best_estimator_

# Print the best hyperparameters
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Model: {best_model}")

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error on Test Set: {mae}")

# SHAP analysis
# Use the TreeExplainer for the best model
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train)

# Visualize feature importance with SHAP summary plot
# We do not need to index `shap_values[0]`, just pass the shap_values directly
shap.summary_plot(shap_values, X_train, plot_type="bar")
```

The most important feature here is the reaction temperature.

**Interpretable Analysis**

Using partial dependence plots (PDP) such as one-way variable and two-way PDP

- One-way partial dependence plots
- Two-way partial dependence plots

```python
# Install specific versions of libraries
!pip install pandas==2.2.2 numpy==1.23.5 matplotlib seaborn scikit-learn openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from scipy.stats import randint

# Load your dataset
try:
    df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')
except Exception as e:
    print(f"Error loading the dataset: {e}")
    exit()

# Check the dataframe info and ensure the target variable exists
print(df.info())

# List of features and target variables
features = ['Ni Loading', 'Surface Area', 'GHSV', 'Ratio of CH4 in Feed', 'Reaction Temperature',
            'Reaction Time', 'Pore Size', 'Pore Volume', 'H2-TPR Peak Temperature', 'Ni Particle Size',
            'Ni Dispersion', 'Modifier Electronegativity']

target_variables = ['Syngas_Ratio', 'CO2 Conversion', 'CH4 Conversion']

# Remove only the first column
df = df.drop(df.columns[0], axis=1)

# Check if all target columns exist in the dataframe before proceeding
for target in target_variables:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in the dataset.")

# Define features (X) and target variables (y)
X = df[features]
y = df[target_variables]

# Replace missing values with the mean of each column
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the RandomForest model
rf_model = RandomForestRegressor(random_state=42)

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 200),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_depth': [None] + list(range(5, 21)),
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=30, cv=5,
                                   scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)

# Fit the model using RandomizedSearchCV
try:
    random_search.fit(X_train, y_train)
except Exception as e:
    print(f"Error during fitting: {e}")
    exit()

# Output the best parameters found by RandomizedSearchCV
print(f"Best Parameters: {random_search.best_params_}")

# Refit the best model with the optimal parameters
best_rf_model = random_search.best_estimator_

# One-way Partial Dependence Plot for feature importance
features_one_way = ['Ni Particle Size','Ratio of CH4 in Feed','Modifier Electronegativity', 'GHSV', 'Ni Dispersion', 'Surface Area', 'Reaction Time', 'Pore Size', 'Pore Volume', 'H2-TPR Peak Temperature','Reaction Temperature', 'Ni Loading']

# Check if features exist in the DataFrame
for feature in features_one_way:
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in the DataFrame.")

# Create and display the Partial Dependence Plot
fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(best_rf_model, X_train, features=features_one_way, ax=ax, grid_resolution=50)
plt.title('One-Way Partial Dependence Plots')
plt.show()

# Two-way Partial Dependence Plot (Now only pairs of features)
# Define pairs of features for two-way plots (now using two features per plot)
features_two_way = [
    ('Ni Particle Size', 'Ratio of CH4 in Feed'),
    ('Modifier Electronegativity', 'GHSV'),
    ('Ni Dispersion', 'Surface Area'),
    ('Reaction Time', 'Pore Size'),
    ('Pore Volume', 'H2-TPR Peak Temperature'),
    ('Reaction Temperature', 'Ni Loading')
]

# Loop through each pair and create a 2D PDP for it
for feature_pair in features_two_way:
    fig, ax = plt.subplots(figsize=(12, 6))
    PartialDependenceDisplay.from_estimator(best_rf_model, X_train, features=[feature_pair], ax=ax, grid_resolution=50)
    plt.title(f'Two-Way Partial Dependence Plot for {feature_pair}')
    plt.show()

# For multiple target variables, you can repeat the above steps in a loop
for target_variable in target_variables:
    # Check if the target variable exists in the DataFrame
    if target_variable not in df.columns:
        print(f"Target column '{target_variable}' not found. Skipping this target.")
        continue

    # Define features (X) and target variable (y)
    y = df[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model using RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Output the best parameters found by RandomizedSearchCV for the current target variable
    print(f"Best Parameters for {target_variable}: {random_search.best_params_}")

    # Refit the best model with the optimal parameters for the current target variable
    best_rf_model = random_search.best_estimator_

    # One-way Partial Dependence Plot for feature importance for the current target variable
    features_one_way = ['Ni Particle Size','Ratio of CH4 in Feed','Modifier Electronegativity', 'GHSV', 'Ni Dispersion', 'Surface Area', 'Reaction Time', 'Pore Size', 'Pore Volume', 'H2-TPR Peak Temperature','Reaction Temperature', 'Ni Loading']  # Example features to plot

    # Check if features exist in the DataFrame
    for feature in features_one_way:
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found in the DataFrame.")

    # Create and display the Partial Dependence Plot for the current target variable
    fig, ax = plt.subplots(figsize=(12, 6))
    PartialDependenceDisplay.from_estimator(best_rf_model, X_train, features=features_one_way, ax=ax, grid_resolution=50)
    plt.title(f'One-Way Partial Dependence Plots for {target_variable}')
    plt.show()

    # Two-way Partial Dependence Plot for the current target variable
    features_two_way = [
        ('Ni Particle Size', 'Ratio of CH4 in Feed'),
        ('Modifier Electronegativity', 'GHSV'),
        ('Ni Dispersion', 'Surface Area'),
        ('Reaction Time', 'Pore Size'),
        ('Pore Volume', 'H2-TPR Peak Temperature'),
        ('Reaction Temperature', 'Ni Loading')
    ]

    # Loop through each pair and create a 2D PDP for it
    for feature_pair in features_two_way:
        fig, ax = plt.subplots(figsize=(12, 6))
        PartialDependenceDisplay.from_estimator(best_rf_model, X_train, features=[feature_pair], ax=ax, grid_resolution=50)
        plt.title(f'Two-Way Partial Dependence Plot for {target_variable} ({feature_pair})')
        plt.show()
```

Bee-swarm plot

```python
# Install specific versions of libraries to avoid conflicts
!pip install pandas==1.5.3 numpy==1.23.5 matplotlib seaborn scikit-learn openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the dataset
try:
    df = pd.read_excel('/content/cleaned_data_no_missing_values_numerical.xlsx')
except Exception as e:
    print(f"Error loading the dataset: {e}")
    exit()

# List of features and target variables
features = ['Ni Loading', 'Surface Area', 'GHSV', 'Ratio of CH4 in Feed', 'Reaction Temperature',
            'Reaction Time', 'Pore Size', 'Pore Volume', 'H2-TPR Peak Temperature', 'Ni Particle Size',
            'Ni Dispersion', 'Modifier Electronegativity']

target_variables = ['Syngas_Ratio', 'CO2 Conversion', 'CH4 Conversion']

# Ensure all target variables exist in the dataframe
for target_variable in target_variables:
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in the dataframe.")

# Remove the first column (if needed)
df = df.drop(df.columns[0], axis=1)

# Define features (X) and target variables (y)
X = df[features]

# Replace missing values with the mean of each column
X = X.fillna(X.mean())

# Check for remaining missing values
if X.isnull().any().any():
    raise ValueError("There are still missing values in the dataset after imputation.")

# Bee-swarm plots for each target variable against the features
# Adjust figure size based on the number of features and target variables
fig, axes = plt.subplots(len(target_variables), len(features), figsize=(15, 10))

# Loop through each target variable to generate plots
for j, target_variable in enumerate(target_variables):
    df_combined = df[features + [target_variable]]

    # Loop through each feature and plot against the target variable
    for i, feature in enumerate(features):
        # Generate the bee-swarm plot
        sns.swarmplot(x=feature, y=target_variable, data=df_combined, color='blue', alpha=0.6, ax=axes[j, i])

        # Set titles and labels
        axes[j, i].set_title(f'{target_variable} vs {feature}')
        axes[j, i].set_xlabel(feature)
        axes[j, i].set_ylabel(target_variable)

# Adjust layout to fit all subplots
plt.tight_layout()

# Show the plots
plt.show()
```

**Hugging face user interface**

```python
# Install specific versions of libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import pickle
import gradio as gr

# Dynamically resolve file path using os
file_path = os.path.join(os.getcwd(), "cleaned_data_no_missing_values_numerical.xlsx")

# Load dataset
df = pd.read_excel(file_path)

# Check the columns in the DataFrame
print("Columns in the DataFrame:", df.columns.tolist())

# Remove the first column
df = df.iloc[:, 1:]

# Define the updated feature columns
features = ['Ni Loading', 'Surface Area', 'GHSV', 'Ratio of CH4 in Feed', 'Reaction Temperature',
            'Reaction Time', 'Pore Size', 'Pore Volume', 'H2-TPR Peak Temperature', 'Ni Particle Size',
            'Ni Dispersion', 'Modifier Electronegativity']

# Define multiple target variables
target_variables = ['Syngas_Ratio', 'CO2 Conversion', 'CH4 Conversion']

# Check if all required feature columns exist
missing_columns = [col for col in features if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing required columns: {missing_columns}")

# Prepare the data
X = df[features]  # Only use the features
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)  # Fill any remaining NaN values with 0

# Define models to be used
models = {
    'RandomForest': RandomForestRegressor(),
    'LinearRegression': LinearRegression(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
}

# Train models for each target variable and save the best one
best_models = {}
for target_variable in target_variables:
    y = df[target_variable]  # Separate target variable for training

    best_score = -np.inf
    best_model = None
    for model_name, model in models.items():
        try:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_model = (model_name, model)
        except Exception as e:
            print(f"Warning: Failed to train {model_name} for {target_variable}: {e}")

    if best_model:
        model_name, model = best_model
        model.fit(X, y)
        with open(f'best_{model_name}_{target_variable}.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        best_models[target_variable] = (model_name, model)

# Define Gradio app for the updated feature set
def predict_catalyst_metrics(Reaction_Temperature, Ratio_of_CH4_in_Feed, Ni_Loading, Surface_Area,
                              GHSV, Reaction_Time, Pore_Size, Pore_Volume, H2_TPR_Peak_Temperature,
                              Ni_Particle_Size, Ni_Dispersion, Modifier_Electronegativity):
    # Create input data for prediction
    input_data = pd.DataFrame({
        'Reaction Temperature': [Reaction_Temperature],
        'Ratio of CH4 in Feed': [Ratio_of_CH4_in_Feed],
        'Ni Loading': [Ni_Loading],
        'Surface Area': [Surface_Area],
        'GHSV': [GHSV],
        'Reaction Time': [Reaction_Time],
        'Pore Size': [Pore_Size],
        'Pore Volume': [Pore_Volume],
        'H2-TPR Peak Temperature': [H2_TPR_Peak_Temperature],
        'Ni Particle Size': [Ni_Particle_Size],
        'Ni Dispersion': [Ni_Dispersion],
        'Modifier Electronegativity': [Modifier_Electronegativity],
    })

    # Ensure the input data has the same structure as X
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    predictions = {}
    for output_name in target_variables:
        try:
            model_name, model = best_models[output_name]
            prediction = model.predict(input_data)[0]
            # Add the unit to the prediction result
            if output_name == 'Syngas_Ratio':
                predictions[f'{model_name} - {output_name}'] = f"{prediction:.2f} (unitless)"
            elif output_name == 'CO2 Conversion':
                predictions[f'{model_name} - {output_name}'] = f"{prediction:.2f} (%)"
            elif output_name == 'CH4 Conversion':
                predictions[f'{model_name} - {output_name}'] = f"{prediction:.2f} (%)"
        except Exception as e:
            predictions[f'{output_name}'] = f"Error: {e}"
    return predictions

# Gradio Interface with Units
inputs = [
    gr.Number(label='Reaction Temperature (°C)', value=0, precision=2),
    gr.Number(label='Ratio of CH4 in Feed', value=0, precision=2),
    gr.Number(label='Ni Loading (%)', value=0, precision=2),
    gr.Number(label='Surface Area (m²/g)', value=0, precision=2),
    gr.Number(label='GHSV (h⁻¹)', value=0, precision=2),
    gr.Number(label='Reaction Time (hours)', value=0, precision=2),
    gr.Number(label='Pore Size (nm)', value=0, precision=2),
    gr.Number(label='Pore Volume (cm³/g)', value=0, precision=2),
    gr.Number(label='H2-TPR Peak Temperature (°C)', value=0, precision=2),
    gr.Number(label='Ni Particle Size (nm)', value=0, precision=2),
    gr.Number(label='Ni Dispersion (%)', value=0, precision=2),
    gr.Number(label='Modifier Electronegativity', value=0, precision=2),
]

# Output is JSON with the prediction results
outputs = gr.JSON()

# Define Gradio interface
app = gr.Interface(fn=predict_catalyst_metrics,
                   inputs=inputs,
                   outputs=outputs,
                   title='Catalyst Performance Prediction during Dry Reforming of Methane',
                   description="Predict Syngas Ratio, CO2 Conversion, and CH4 Conversion based on catalyst metrics. Units will be displayed with the results.")

# Launch the app
if __name__ == "__main__":
    app.launch()
```

**References**

https://chatgpt.com/

https://gemini.google.com/app

D. Chinenye Divine et. al. (2024). Enhancing biomass pyrolysis: Predictive insights from process simiulation integrated with interpretable machine learning models

J.Roh et al. (2023).Interpretable machine learning framework for catalyst performance prediction and validation with dry reforming of methane. https://doi.org/10.1016/j.apcatb.2023.123454

Muhammad Asif et al (2024)
Machine learning-driven catalyst design, synthesis and performance prediction for CO2 hydrogenation, Journal of Industrial and Engineering Chemistry. https://doi.org/10.1016/j.jiec.2024.09.035

T.S. Gendey et al. (2024). Enhanced Predcitive Optimization of methane dry reforming via ResponseSurface methodology and artificial neural network approaches: Insights using a novel nickel-strontium-zirconium-aluminum catalyst  https://doi.org/10.1016/j.mcat.2024.114216

##### Ethics Analysis



---

#### Project 7: Fake Job Posting Detector

##### Read Me



---

**Fake Job Posting Detector**

A sophisticated Streamlit application that leverages machine learning to detect fraudulent job postings. The app employs multiple ML models, comprehensive data preprocessing, and interactive visualizations to provide accurate fraud detection and insights. The application can be found at https://lecheg472-individual-project-kshfsbmubj4dgnjwkn6xca.streamlit.app/  

**Author**

Anh Le

Bucknell University

**Features**

**Data Processing**

- **US Job Filtering**: Focuses on US job market data
- **Text Cleaning**:
  - Removes URLs, HTML tags, and special characters
  - Normalizes text formatting
  - Handles missing values
- **Feature Engineering**:
  - TF-IDF vectorization with n-gram support
  - Text length analysis
  - Outlier detection and removal

**Machine Learning Models**

- **Multiple Classifiers**:
  - Random Forest
  - XGBoost
  - Logistic Regression
  - Support Vector Machine (SVM)
- **Class Imbalance Handling**:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - Balanced class weights

**Visualization & Analysis**

- **Interactive Charts**:
  - Text length distributions
  - Class distribution analysis
  - Model performance comparisons
- **Statistical Insights**:
  - Missing value analysis
  - Summary statistics
  - Model performance metrics

**Real-time Prediction**

- Interactive job description testing
- Confidence scores for predictions
- Detailed analysis of prediction factors

**Prerequisites**


**Software Requirements**

- Python 3.8 or higher
- Git (for cloning repository)

**Required Python Packages**

```bash
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
xgboost>=1.5.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
```

**Installation**

1. Clone the repository:
```bash
git clone https://github.com/your-username/fake-job-detector.git
cd fake-job-detector
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```python
python -c "import nltk; nltk.download('stopwords')"
```

**Usage**

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Access the application:
- Open your web browser
- Navigate to http://localhost:8501

3. Using the App:
   - Upload a CSV file containing job posting data
   - View data preprocessing insights and visualizations
   - Train and compare multiple ML models
   - Test individual job descriptions for fraud detection

**Data Format**

The input CSV should contain the following columns:
- title
- location
- company_profile
- description
- requirements
- benefits
- fraudulent (0 for legitimate, 1 for fraudulent)

**Model Performance**

Current model performance metrics (as of latest testing):
- SVM: 98.27% accuracy
- XGBoost: 98.10% accuracy
- Logistic Regression: 97.87% accuracy
- Random Forest: 97.62% accuracy

All models show strong performance in detecting both legitimate and fraudulent job postings, with particular emphasis on minimizing false positives.

**Contributing**

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


**Acknowledgments**

- Dataset: Employment Scam Aegean Dataset (EMSCAD)
- NLTK Project for text processing capabilities
- Scikit-learn community for machine learning tools
- Streamlit team for the web application framework


**Future Improvements**

- Add deep learning models
- Implement feature importance analysis
- Add multi-language support
- Enhance visualization capabilities
- Add model explainability features

##### GUI App Deployment




---

```python
# Set the title
st.title("Fake Job Post Detection")

    # Data Loading and Model Comparison section
st.header("Model Comparison")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="csv_uploader")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file, encoding='latin-1')

    # Show data preview
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Model comparison button
    if st.button("Compare Models", key="compare_models_button"):
        with st.spinner("Training and comparing models..."):
            best_model, vectorizer, conf_matrix_fig = compare_models(df)
            st.success("Model comparison completed!")
            st.pyplot(conf_matrix_fig)

            # Save best model and vectorizer for predictions
            st.session_state['best_model'] = best_model
            st.session_state['vectorizer'] = vectorizer

    # Prediction section
    if 'best_model' in st.session_state:
        st.header("Job Prediction with Best Model")
        input_text = st.text_area("Enter job description:")

        if input_text:
            # Transform input text
            input_vector = st.session_state['vectorizer'].transform([input_text])

            # Make prediction
            prediction = st.session_state['best_model'].predict(input_vector)

            if prediction[0] == 1:
                st.warning("This job posting appears to be fraudulent")
            else:
                st.success("This job posting appears to be legitimate")
else:
    st.write("Please upload a CSV file to begin.")
```

##### Machine Learning Development




---

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
import nltk
import re
import missingno as msno
from scipy import stats
```

```python
# Initialize nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def analyze_word_frequencies(df):
    """Analyze and visualize word frequencies in fraudulent and non-fraudulent postings"""
    st.subheader("Word Frequency Analysis")

    # Separate fraudulent and non-fraudulent job postings
    fraudulent_jobs = df[df['fraudulent'] == 1]['text']
    non_fraudulent_jobs = df[df['fraudulent'] == 0]['text']

    def get_word_freq(text_series):
        words = ' '.join(text_series).split()
        return pd.Series(words).value_counts()

    # Plot top words for fraudulent jobs
    fraud_word_freq = get_word_freq(fraudulent_jobs).head(20)
    fig_fraud = px.bar(fraud_word_freq,
                      x=fraud_word_freq.index,
                      y=fraud_word_freq.values,
                      title='Top Words in Fraudulent Job Postings',
                      labels={'index': 'Words', 'y': 'Frequency'},
                      color=fraud_word_freq.values,
                      color_continuous_scale='Reds')
    st.plotly_chart(fig_fraud)

    # Plot top words for non-fraudulent jobs
    non_fraud_word_freq = get_word_freq(non_fraudulent_jobs).head(20)
    fig_non_fraud = px.bar(non_fraud_word_freq,
                          x=non_fraud_word_freq.index,
                          y=non_fraud_word_freq.values,
                          title='Top Words in Non-Fraudulent Job Postings',
                          labels={'index': 'Words', 'y': 'Frequency'},
                          color=non_fraud_word_freq.values,
                          color_continuous_scale='Blues')
    st.plotly_chart(fig_non_fraud)

    # Compare word frequencies
    st.write("\nUnique words in fraudulent vs non-fraudulent postings:")
    fraud_unique = set(fraud_word_freq.index) - set(non_fraud_word_freq.index)
    non_fraud_unique = set(non_fraud_word_freq.index) - set(fraud_word_freq.index)
    st.write("Words unique to fraudulent postings:", list(fraud_unique))
    st.write("Words unique to non-fraudulent postings:", list(non_fraud_unique))

def plot_class_distribution(df):
    """Plot the distribution of fraudulent vs non-fraudulent postings"""
    st.subheader("Class Distribution")

    fig = px.histogram(df,
                      x='fraudulent',
                      title='Distribution of Fraudulent vs Non-Fraudulent Job Postings',
                      labels={'fraudulent': 'Fraudulent'},
                      color='fraudulent',
                      color_discrete_sequence=['#1f77b4', '#ff7f0e'])

    fig.update_layout(
        xaxis_title='Fraudulent',
        yaxis_title='Count',
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=14),
        xaxis=dict(gridcolor='gray'),
        yaxis=dict(gridcolor='gray')
    )

    st.plotly_chart(fig)

    # Add statistics
    fraud_percent = (df['fraudulent'].mean() * 100)
    st.write(f"Percentage of fraudulent postings: {fraud_percent:.2f}%")
    st.write(f"Total number of postings: {len(df)}")
    st.write(f"Number of fraudulent postings: {df['fraudulent'].sum()}")
    st.write(f"Number of legitimate postings: {len(df) - df['fraudulent'].sum()}")

def handle_missing_values(df):
    """Handle missing values in all columns"""
    st.subheader("Missing Value Handling")

    # Store initial missing value counts
    initial_missing = df.isnull().sum()

    # Text columns
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    df[text_columns] = df[text_columns].fillna(' ')

    # Categorical columns
    categorical_columns = ['location', 'department', 'salary_range', 'employment_type',
                         'required_experience', 'required_education', 'industry', 'function']
    for col in categorical_columns:
        if col in df.columns:
            df[col].fillna('Not Specified', inplace=True)

    # Display missing value handling results
    final_missing = df.isnull().sum()

    # Create comparison dataframe
    missing_comparison = pd.DataFrame({
        'Initial Missing': initial_missing,
        'After Handling': final_missing,
        'Difference': initial_missing - final_missing
    })

    st.write("Missing Values Before and After Handling:")
    st.dataframe(missing_comparison[missing_comparison['Initial Missing'] > 0])

def preprocess_data(df):
    """Main preprocessing function"""
    st.write("Starting data preprocessing and analysis...")

    # Handle missing values
    handle_missing_values(df)

    # Plot class distribution
    plot_class_distribution(df)

    # Analyze word frequencies
    df['text'] = df[['title', 'company_profile', 'description', 'requirements', 'benefits']].apply(
        lambda x: ' '.join(x.dropna()), axis=1)
    analyze_word_frequencies(df)

    # Clean text
    for col in ['title', 'company_profile', 'description', 'requirements', 'benefits']:
        df[col] = df[col].apply(clean_text)

    # Filter for US jobs only
    df = df[df['location'].str.contains('US', na=False)]
    st.write("\nDataset Shape after filtering US jobs:", df.shape)

    # Create numerical features and handle outliers
    numerical_features = create_numerical_features(df)

    # Analyze correlations
    analyze_correlations(df, numerical_features)

    # Analyze outliers
    analyze_outliers(numerical_features)

    # Generate summary statistics
    generate_summary_stats(df, numerical_features)

    return df
def prepare_data(df):
    # Text preprocessing
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    df['text'] = df[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['fraudulent']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    return X_train_balanced, X_test, y_train_balanced, y_test, vectorizer

def display_data_analysis(df):
    st.header("Data Analysis Dashboard")

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview & Distribution",
        "Missing Data",
        "Word Analysis",
        "Correlations",
        "Model Results"
    ])

    with tab1:
        # Distribution of fraudulent vs non-fraudulent
        fig = px.histogram(df,
                          x='fraudulent',
                          title='Distribution of Fraudulent vs Non-Fraudulent Job Postings',
                          labels={'fraudulent': 'Fraudulent'},
                          color='fraudulent',
                          color_discrete_sequence=['#1f77b4', '#ff7f0e'])

        fig.update_layout(
            xaxis_title='Fraudulent',
            yaxis_title='Count',
            title_x=0.5
        )
        st.plotly_chart(fig)

        # Dataset info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Total Samples:", len(df))
            st.write("Legitimate Jobs:", len(df[df['fraudulent'] == 0]))
        with col2:
            st.write("Fraudulent Jobs:", len(df[df['fraudulent'] == 1]))
            st.write("Fraud Percentage:", f"{(df['fraudulent'].mean() * 100):.2f}%")

    with tab2:
        st.subheader("Missing Values Analysis")
        # Missing values heatmap
        missing_data = df.isnull().sum()
        fig = px.bar(x=missing_data.index,
                    y=missing_data.values,
                    title="Missing Values by Column")
        st.plotly_chart(fig)

        # Missing values table
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Values': missing_data.values,
            'Percentage': (missing_data.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df)

    with tab3:
        st.subheader("Word Analysis")

        # Separate fraudulent and non-fraudulent job postings
        fraudulent_jobs = df[df['fraudulent'] == 1]['text']
        non_fraudulent_jobs = df[df['fraudulent'] == 0]['text']

        def plot_top_words(text_series, title):
            word_freq = pd.Series(' '.join(text_series).split()).value_counts().head(20)
            fig = px.bar(word_freq,
                        x=word_freq.index,
                        y=word_freq.values,
                        title=title,
                        labels={'index': 'Words', 'y': 'Frequency'})
            st.plotly_chart(fig)

        plot_top_words(fraudulent_jobs, 'Top Words in Fraudulent Job Postings')
        plot_top_words(non_fraudulent_jobs, 'Top Words in Non-Fraudulent Job Postings')

    with tab4:
        st.subheader("Correlation Analysis")

        # Create numerical features for correlation
        numerical_features = pd.DataFrame()
        text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']

        for col in text_columns:
            numerical_features[f'{col}_length'] = df[col].str.len()
            numerical_features[f'{col}_word_count'] = df[col].str.split().str.len()

        numerical_features['fraudulent'] = df['fraudulent']

        # Plot correlation matrix
        corr = numerical_features.corr()
        fig = px.imshow(corr,
                       title='Feature Correlation Matrix',
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig)

    with tab5:
        st.subheader("Model Results")
        # This tab will be populated after model training

def display_model_results(accuracies, reports, conf_matrices):
    """Display model results in the Model Results tab"""
    st.subheader("Model Performance Comparison")

    # Create accuracy comparison
    accuracy_df = pd.DataFrame({
        'Model': list(accuracies.keys()),
        'Accuracy': list(accuracies.values())
    })

    fig = px.bar(accuracy_df,
                 x='Model',
                 y='Accuracy',
                 title='Model Accuracy Comparison')
    st.plotly_chart(fig)

    # Display classification reports
    for model_name, report in reports.items():
        st.write(f"\n{model_name} Classification Report:")
        st.text(report)

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100,
                                    max_depth=10,
                                    min_samples_split=5,
                                    class_weight='balanced',
                                    random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train):
    xgb_model = XGBClassifier(n_estimators=100,
                             max_depth=5,
                             learning_rate=0.1,
                             scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                             random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_logistic_regression(X_train, y_train):
    lr_model = LogisticRegression(class_weight='balanced',
                                 max_iter=1000,
                                 random_state=42)
    lr_model.fit(X_train, y_train)
    return lr_model

def train_svm(X_train, y_train):
    svm_model = LinearSVC(class_weight='balanced',
                         max_iter=2000,
                         random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    st.write(f"\n{model_name} Results:")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write("\nClassification Report:")
    st.text(report)

    return accuracy, report, conf_matrix

def plot_confusion_matrices(conf_matrices, model_names):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()

    for idx, (conf_matrix, model_name) in enumerate(zip(conf_matrices, model_names)):
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axes[idx])
        axes[idx].set_title(f'{model_name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.tight_layout()
    return fig

def compare_models(df):
    # Prepare data
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df)

    # First, show data analysis
    display_data_analysis(df)

    # Initialize lists to store results
    models = []
    model_names = ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM']
    accuracies = {}
    reports = {}
    conf_matrices = []

    # Train and evaluate Random Forest
    st.write("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    models.append(rf_model)
    accuracy, report, conf_matrix = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    accuracies["Random Forest"] = accuracy
    reports["Random Forest"] = report
    conf_matrices.append(conf_matrix)

    # Train and evaluate XGBoost
    st.write("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    models.append(xgb_model)
    accuracy, report, conf_matrix = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    accuracies["XGBoost"] = accuracy
    reports["XGBoost"] = report
    conf_matrices.append(conf_matrix)

    # Train and evaluate Logistic Regression
    st.write("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    models.append(lr_model)
    accuracy, report, conf_matrix = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    accuracies["Logistic Regression"] = accuracy
    reports["Logistic Regression"] = report
    conf_matrices.append(conf_matrix)

    # Train and evaluate SVM
    st.write("Training SVM...")
    svm_model = train_svm(X_train, y_train)
    models.append(svm_model)
    accuracy, report, conf_matrix = evaluate_model(svm_model, X_test, y_test, "SVM")
    accuracies["SVM"] = accuracy
    reports["SVM"] = report
    conf_matrices.append(conf_matrix)

    # Plot confusion matrices
    conf_matrix_fig = plot_confusion_matrices(conf_matrices, model_names)

    # Display model results
    display_model_results(accuracies, reports, conf_matrices)

    # Find best model
    best_model_name = max(accuracies.items(), key=lambda x: x[1])[0]
    best_model = models[model_names.index(best_model_name)]

    st.write(f"\nBest Model: {best_model_name}")
    st.write(f"Best Accuracy: {accuracies[best_model_name]:.4f}")

    return best_model, vectorizer, conf_matrix_fig
```

##### Ethics Analysis




---

**Data Collection:**
- Input Variables:
job_id, title, location, department, salary_range, company_profile, description, requirements, benefits, telecommuting, has_company_logo, has_questions, employment_type, required_experience, required_education, industry, function.

- Output Variable:
fraudulent (0 for real, 1 for fake job postings).

- Are the data sources properly licensed and legally available?

Ensure that the dataset has been sourced ethically, with proper licensing or permissions from the source organization.

- Has any sensitive information been anonymized?

- No personal data appears in the dataset. It primarily consists of job and company attributes, which do not seem to pose a risk to individual privacy.

- Have you obtained consent for data collected from private or proprietary sources?

This depends on the dataset's origin. If sourced from public job boards, there might not be a need for explicit consent, but confirmation of this is critical.

**Fairness & Justice:**
- How will you ensure the model’s predictions are fair and do not disproportionately affect specific regions or communities?

Analyze the location field to check for adequate representation across regions. Ensure fairness metrics (like demographic parity or equal opportunity) are incorporated in training and validation.

- What biases might exist in the historical data, and how will you address these to ensure the model does not unfairly target or neglect specific areas?

Historical bias may arise if certain industries, locations, or company types are disproportionately represented as fraudulent or legitimate. Balance the dataset or use reweighting techniques to mitigate this.

- How will you balance fairness in handling both false positives and false negatives?

A balance should be struck between identifying real fraudulent postings (reducing false negatives) and minimizing the flagging of legitimate postings as fraudulent (reducing false positives). Adjust the model threshold based on use-case priorities.

- Have you tested the model to ensure consistent performance across various conditions?

Conduct stratified testing based on attributes like location, industry, or company size to ensure fairness and consistency.

**Transparency:**
- How will you ensure transparency about the data sources, algorithms, and decision-making process of the model?

Provide clear documentation of the data sources, model architecture, and decision-making logic, including feature importance and model interpretability tools (e.g., SHAP or LIME).

- What information will you make available to government agencies, the public?

Share anonymized data summaries and algorithmic logic to build trust while protecting proprietary details.

- How will you communicate the model’s predictions and limitations to decision-makers so that they understand the risks involved?

Use straightforward visualizations and plain language to explain predictions, emphasizing both accuracy and limitations, especially for edge cases.

- How will you explain false positives and false negatives to the affected communities or stakeholders?

Provide examples of such cases and outline steps taken to minimize errors, emphasizing ongoing model improvement and accountability.

**Privacy:**
- How will you ensure the privacy of individuals whose data might be inadvertently captured?

The dataset appears anonymized, but any personal information should be redacted. Use robust security measures to protect data during processing and storage.

- What steps will you take to prevent the misuse of this data?

Limit access to the dataset to authorized personnel and audit its use. Incorporate ethical use clauses into agreements with stakeholders.

- If external data sources are integrated into the model, how will you balance the need for accurate predictions with protecting individual privacy?

Anonymize and aggregate external data sources before integration. Regularly review the model for unintended information leakage.

**Accountability:**
- Who will be held accountable if the model incorrectly predicts a job?

Define clear accountability, assigning responsibility to the organization deploying the model, with support from the data science team for technical issues.

- What system will you establish to monitor and adjust the model over time?

Develop monitoring pipelines to track model performance and fairness metrics, with a process for periodic retraining on updated data.

- How will you communicate accountability measures to the public?

Publish periodic reports on model performance, highlight efforts to mitigate errors, and establish a channel for feedback or grievances.

**Inclusivity:**
- How will you ensure the model includes diverse data, especially those that may be underrepresented in historical data collection?
Examine the distribution of attributes like location, industry, and employment_type for potential underrepresentation and balance the dataset or apply appropriate techniques to correct for this.

- How will you ensure the model accounts for the needs of different communities?

Validate the model across subsets of the dataset, ensuring consistent predictions across demographic and regional lines.

- If certain regions or communities lack sufficient data (e.g., underreporting, lack of resources), how will you address this to avoid biased predictions?

Augment the dataset by sourcing additional representative data or use transfer learning to better generalize for underrepresented communities.

**Sustainability:**
- How will the model’s predictions affect long-term strategies over time?

Regular monitoring will ensure the model continues to adapt to evolving trends in job postings. Fraud patterns may shift, requiring dynamic retraining.

- How will you ensure the model remains sustainable, considering its effects?

Implement a process for continuous improvement, leveraging new data to enhance performance and mitigate outdated predictions.

- What are the broader social and environmental implications if this model becomes widely adopted?

The model could deter fraud, saving time and resources for job seekers and companies. However, transparency and fairness are critical to prevent misclassification from eroding trust in job platforms.