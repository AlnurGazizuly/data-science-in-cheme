# Section 3: Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is the process of understanding and summarizing data before building models or making predictions. This section covers key steps in EDA, including data processing, handling outliers, dealing with missing values, and creating insightful visualizations.

#### 3.1. Why Use EDA

Exploratory Data Analysis (EDA) is a crucial step in any data science or machine learning project. It helps us understand the structure, quality, and key patterns in the data before applying models or algorithms. Here's why EDA is essential:

*Understand the Data*

---
EDA provides a clear view of the dataset, including its dimensions, types of variables, and potential issues like missing or inconsistent values.

*Detect Outliers and Errors*

---
By visualizing data and summarizing statistics, we can spot anomalies or errors that could negatively impact model performance.

*Uncover Patterns and Relationships*

---
EDA helps identify trends, correlations, or groupings in the data, which can inform feature selection or engineering.

*Choose the Right Approach*

---
Insights gained during EDA can guide model selection, preprocessing steps, and hyperparameter tuning.

*Communicate Findings*

---
EDA results are often visualized and shared to help stakeholders understand the data and proposed solutions.

### Exercise 3.1: Understanding a dataset

#### 3.2. Dataset Structure

Understanding the structure of a dataset is an essential part of Exploratory Data Analysis (EDA). It helps us grasp how the data is organized and what kind of information is available for analysis. Key components of a dataset's structure include **rows**, **columns**, and an overview of the data's content using methods like `df.head()`, `df.info()`, and `df.describe()`.

*Rows and Columns*


---
- **Rows** represent individual observations or data points in the dataset. For example, in a dataset about cars, each row might represent a single car.
- **Columns** represent variables or features. These can be numerical (e.g., engine size, mileage) or categorical (e.g., car brand, fuel type).



*Key Methods to Examine Dataset Structure*


---
1. **`df.head(n)`**  
   - Displays the first `n` rows of the dataset (default is 5).
   - Useful for a quick look at the data to understand the column names, variable types, and values.
   
   Example:
   ```{code-cell} ipython3
print(data.head(5))
```

2. **`df.info()`**  
   - Provides a summary of the dataset, including:
     - Number of rows and columns.
     - Column names and their data types (e.g., integer, float, object).
     - Non-null value counts for each column (helps identify missing data).
   - Essential for understanding the dataset's overall structure.

   Example:
   ```{code-cell} ipython3
print(data.info())
```

3. **`df.describe()`**  
   - Generates summary statistics for numerical columns, such as mean, standard deviation, minimum, maximum, and quartiles.
   - Useful for spotting trends or outliers in numerical data.

   Example:
   ```{code-cell} ipython3
print(data.describe())
```

*Output Analysis*


---


- **`df.head()`** shows the actual data and helps verify the column names and the format of values. For instance, we might see columns like `sepal_length`, `sepal_width`, and `species`.
- **`df.info()`** reveals critical details, like whether all values in `species` are non-null or if `sepal_length` is stored as a `float`.
- **`df.describe()`** highlights statistics such as the mean sepal length or the range of petal widths.

Understanding dataset structure allows us to identify potential issues (like missing values or incorrect data types) and helps us prepare the data for deeper analysis or modeling.

#### 3.3. Data Types

Data types define the kind of values stored in a dataset and determine what kind of operations can be performed on them. Understanding data types is crucial for data analysis and machine learning because models and preprocessing techniques often require specific data types.  

*Common Data Types*

---


1. **Numerical**
   - **Integer (`int`)**: Whole numbers (e.g., 1, 42, -3).
   - **Float (`float`)**: Numbers with decimals (e.g., 3.14, -0.001).
   - These are often used for features like counts, measurements, or continuous values.

2. **Categorical**
   - **Object (`str`)**: Text or string data (e.g., "cat", "apple").
   - Useful for labels or qualitative features.

3. **Boolean (`bool`)**
   - Represents binary values: `True` or `False`.
   - Often used for flags or binary classification.

4. **Datetime**
   - Represents dates or timestamps (e.g., `2025-01-01`, `2025-01-01 14:30`).
   - Useful for time-series data or analyzing temporal trends.

5. **Lists**
   - **Definition**: A list is an ordered, mutable collection of elements. Lists can store mixed data types (e.g., numbers, strings, other lists).
   - **Usage in ML**: Lists are often used for storing collections of features, labels, or intermediate results.
   - **Example**:
     ```{code-cell} ipython3
my_list = [1, 2, 3, 'a', 'b']
     print(my_list[0])  # Access first element
     my_list.append(4)  # Add an element
     print(my_list)
```

6. **Tuples**
   - **Definition**: A tuple is an ordered, immutable collection of elements. Tuples are like lists but cannot be modified after creation.
   - **Usage in ML**: Tuples are used when you need to group data that should not change, such as storing dataset dimensions or coordinates.
   - **Example**:
     ```{code-cell} ipython3
my_tuple = (1, 2, 3)
     print(my_tuple[1])  # Access second element
     # my_tuple[1] = 4  # This will raise an error because tuples are immutable
```

7. **Dictionaries**
   - **Definition**: A dictionary is an unordered collection of key-value pairs. Keys must be unique, and values can be of any data type.
   - **Usage in ML**: Dictionaries are useful for storing metadata, mapping column names to descriptions, or configuring model parameters.
   - **Example**:
     ```{code-cell} ipython3
my_dict = {'name': 'Alice', 'age': 25, 'city': 'New York'}
     print(my_dict['name'])  # Access value by key
     my_dict['age'] = 26  # Update value
     print(my_dict)
```

8. **Sets**
   - **Definition**: A set is an unordered collection of unique elements. Duplicate values are automatically removed.
   - **Usage in ML**: Sets are often used for finding unique labels or removing duplicates.
   - **Example**:
     ```{code-cell} ipython3
my_set = {1, 2, 3, 3}
     print(my_set)  # Output: {1, 2, 3}
```

     
*How to Check Data Types*

---


1. **Check the data type of an entire column**  
   Use `df.dtypes` to see the data types of all columns or `df['column_name'].dtype` for a specific column.

   Example:
   ```{code-cell} ipython3
print(data.dtypes)  # Check all column data types
   print(data['sepal_length'].dtype)  # Check data type of 'sepal_length'
```

2. **Check the data type of a specific data point**  
   Use Python’s built-in `type()` function to check the type of a single value.

   Example:
   ```{code-cell} ipython3
# Check the type of the first value in 'sepal_length'
   print(type(data['sepal_length'][0]))
```

3. **Convert data types**  
   Sometimes, data might be incorrectly typed (e.g., numbers stored as strings). You can convert types using `pd.to_numeric()`, `pd.to_datetime()`, or `.astype()`.

   Example:
   ```{code-cell} ipython3
# Convert a column to float
   data['sepal_length'] = data['sepal_length'].astype(float)
```

#### 3.4. Correcting the Dataset Formatting

#### Task:
1. Import the dataset titled 'Section3.xlsx'
2. Print the first five rows of the spreadsheet
3. Print the type of data in each column
4. If any columns contain the incorrect type of data, explain what type the column should be and write code to correct the error (Hint: one column should be fixed)

```{code-cell} ipython3
# Your code goes here
```

### Exercise 3.2: Missing values and duplicates

#### 3.5. Locating Missing Values

Missing values are a common issue in datasets and need to be identified before deciding how to handle them. In Python, the pandas library provides tools to locate missing values.


How to Locate Missing Values:

---


Using `isna()` or `isnull()`

These methods return a DataFrame of the same shape, indicating where values are missing (True for missing, False for not missing).


Summing Missing Values:

---


Count missing values in each column using `isna().sum()`

Visualizing Missing Values:

---


Libraries like matplotlib and seaborn can visualize missing values for better understanding.
```
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(data.isna(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
```

#### 3.6. Removing Missing Values

Sometimes it's best to remove rows or columns with missing values, especially if the missing data is sparse or cannot be reasonably filled.

Remove Rows with Missing Values:

---

Use .dropna() to remove rows containing any missing values.
Example:
```
# Drop rows with missing values
data_cleaned = data.dropna()
print(data_cleaned)
```

Remove Columns with Missing Values

---

Use .dropna(axis=1) to remove columns with missing values.
Example:
```
# Drop columns with missing values
data_cleaned = data.dropna(axis=1)
print(data_cleaned)
```

Removing Only Fully Missing Rows or Columns

---

Specify how='all' to remove rows or columns only if all values are missing.
Example:
```
# Drop rows where all values are missing
data_cleaned = data.dropna(how='all')
print(data_cleaned)
```

#### 3.7. Replacing Missing Values

Instead of removing missing values, you can replace them with meaningful values based on the dataset.


Replace with a Fixed Value

---

Use .fillna() to replace missing values with a specific value.
Example:
```
# Replace missing values with 0
data_filled = data.fillna(0)
print(data_filled)
```

Replace with Statistical Measures

---

Fill numerical data with the column mean, median, or mode.
Example:
```
# Replace missing values with the mean of 'Feature1'
data['Feature1'] = data['Feature1'].fillna(data['Feature1'].mean())
print(data)
```

Forward or Backward Filling

---

Use .fillna(method='ffill') (forward fill) or .fillna(method='bfill') (backward fill) to propagate non-missing values.
Example:
```
# Forward fill missing values
data_filled = data.fillna(method='ffill')
print(data_filled)
```

#### 3.8. Locating Duplicates

3.8. Locating Duplicates
Duplicate rows in a dataset can skew analysis and must be identified.


Using duplicated()

---

This method returns a Boolean Series indicating whether a row is a duplicate (i.e., has appeared earlier in the dataset).
Example:
```
# Create a dataset with duplicates
data = pd.DataFrame({'A': [1, 2, 2, 3], 'B': ['x', 'y', 'y', 'z']})

# Locate duplicates
print(data.duplicated())
```

Use .duplicated().sum() to count the total number of duplicate rows.
Example:
```
print(f"Number of duplicate rows: {data.duplicated().sum()}")
```

#### 3.9. Removing Duplicates

Once duplicates are identified, they can be removed to ensure the dataset contains unique rows.

Removing Duplicate Rows

---

Use .drop_duplicates() to remove duplicate rows.
Example:
```
# Remove duplicates
data_cleaned = data.drop_duplicates()
print(data_cleaned)
```

Keeping Specific Duplicates

---

Use the keep parameter to specify whether to keep the first (keep='first'), last (keep='last'), or no duplicate rows (keep=False).
Example:
```
# Remove duplicates but keep the last occurrence
data_cleaned = data.drop_duplicates(keep='last')
print(data_cleaned)
```

Removing Duplicates from Specific Columns

---

Use the subset parameter to check for duplicates only in specific columns.
Example:
```
# Remove duplicates based on column 'A' only
data_cleaned = data.drop_duplicates(subset='A')
print(data_cleaned)
```

#### Task:
1. Check for missing data and remove any rows that contain missing values
2. Check for duplicate rows and remove them

```{code-cell} ipython3
# Your code goes here
```

### Exercise 3.3: Outliers

#### 3.10. Locating Outliers

Outliers are data points that significantly differ from other observations. They can result from measurement errors or be legitimate extreme values. Detecting outliers is crucial as they can heavily influence statistical models and analyses.


Using Descriptive Statistics:

---

Outliers can be identified using the Interquartile Range (IQR) method:
* Calculate the 25th percentile (Q1) and 75th percentile (Q3).
* Compute the IQR as Q3 - Q1.
* Any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is considered an outlier.
Example:

```
import numpy as np
import pandas as pd

# Create a sample dataset
data = pd.DataFrame({'Values': [10, 12, 15, 18, 19, 120]})

# Calculate IQR
Q1 = data['Values'].quantile(0.25)
Q3 = data['Values'].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
outliers = data[(data['Values'] < Q1 - 1.5 * IQR) | (data['Values'] > Q3 + 1.5 * IQR)]
print(outliers)
```

Using Z-Scores:

---

The Z-score measures how many standard deviations a data point is from the mean. A common threshold for identifying outliers is a Z-score greater than 3 or less than -3.
Example:
```
from scipy.stats import zscore

# Calculate Z-scores
data['Z-Score'] = zscore(data['Values'])

# Identify outliers based on Z-score
outliers = data[(data['Z-Score'] > 3) | (data['Z-Score'] < -3)]
print(outliers)
```

#### 3.11. Removing Outliers
Once outliers are identified, they can be removed if they are likely to skew the analysis. You can either remove outliers using the IQR method.

Example:
```
# Remove outliers based on IQR
data_cleaned = data[(data['Values'] >= Q1 - 1.5 * IQR) & (data['Values'] <= Q3 + 1.5 * IQR)]
print(data_cleaned)
```

#### 3.12. Replacing Outliers


In some cases, removing outliers might not be ideal, especially if they are legitimate extreme values. Instead, they can be replaced with more reasonable values. Usually, outliers can be replaced with values like the mean or median, depending on the dataset presented and. your interpreation of the descriptive statistics.


Example:
```
# Replace outliers with the median
median = data['Values'].median()
data['Values'] = np.where(
    (data['Values'] < Q1 - 1.5 * IQR) | (data['Values'] > Q3 + 1.5 * IQR),
    median,
    data['Values']
)
print(data)
```

#### Task:
1. Use the IQR method to locate outliers
2. Use a box and whisker plot to visualize any outliers
3. Remove outliers
4. COnfirm that outliers have been successfully removed by checking for outliers after they have been removed. Print a message showing their are no outliers present

```{code-cell} ipython3
# Your code goes here
```

### Exercise 3.4: Heat Maps

#### 3.13. Correlations

A correlation measures the relationship between two variables. It shows whether an increase in one variable is associated with an increase or decrease in another.

Correlation coefficients range from -1 to 1:



*   1: Perfect positive correlation (as one variable increases, the other also increases).
*   0: No correlation (no relationship between the variables).
*  -1: Perfect negative correlation (as one variable increases, the other decreases).   

Correlation is a fundamental concept in machine learning to understand how features interact and contribute to the target variable.

#### 3.14. Why You Visualize Correlations
Visualizing correlations helps you:

- Quickly identify relationships between variables.
- Spot patterns that may not be evident in numerical data
- Detect multicollinearity, which occurs when variables are too closely related (a potential issue for some machine learning models).
- Understand which features may be important for predicting your target variable.
- Common visualization tools include scatter plots and heatmaps, which provide an intuitive way to interpret correlations.

#### 3.15. Heat Maps

A heatmap is a visualization tool that uses colors to represent the magnitude of values in a matrix. In the context of correlations, heatmaps can show relationships between features in a dataset.

Heatmaps simplify interpreting large correlation matrices.
Colors make it easier to identify strong positive or negative correlations. Patterns become apparent, helping to decide which features to include in a model.

Creating a Heatmap for Correlations
Example:
```
import seaborn as sns
import matplotlib.pyplot as plt

# Create a sample dataset
data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4],
    'Feature2': [2, 4, 6, 8],
    'Feature3': [10, 20, 30, 40]
})

# Compute the correlation matrix
correlation_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
```

This heatmap visually represents correlations, where:

- Strong positive correlations are shown in deep red.
- Strong negative correlations are shown in deep blue.
- Near-zero correlations are light-colored.

By using heatmaps, you can easily identify which features are highly correlated, aiding in feature selection for machine learning models.

#### Task:
1. Generate a heatmap for the dataset
2. What are the strongest positive correlations in the dataset?
3. What are the strongest negative correlations in the dataset?
4. Are there any factors (independant variables) that don't correlate with the target (dependant variable)?

```{code-cell} ipython3
# Your code goes here
```

### Excercise 3.5: Data Scaling

#### 3.15. What is Data Scaling?

Data scaling is the process of transforming the features in your dataset so that they are represented on a similar scale. Scaling ensures that the range of values for each feature does not vary widely. For example, one feature might represent income in dollars (ranging from thousands to millions), while another might represent age (ranging from 0 to 100). Without scaling, the feature with the larger range can dominate the results of machine learning algorithms that rely on distance or gradient descent.

#### 3.16. Why Scale Data?

Data scaling is crucial for many machine learning algorithms, especially those that calculate distances between data points or use gradient-based optimization. Here are some reasons why scaling is important:

1. **Improved Model Performance:**
Algorithms like Support Vector Machines (SVM) and K-Nearest Neighbors (KNN) use distance metrics that are sensitive to scale. Without scaling, features with larger ranges can disproportionately affect model decisions.

2. **Faster Convergence:**
Gradient-based algorithms, such as those used in neural networks or logistic regression, perform better when all features are on similar scales, as this ensures smoother gradients.

3. **Avoiding Bias:**
Features with larger scales can bias models, especially when interpreting coefficients in regression models.

4. **Compatibility with Regularization:**
Regularization techniques like L1 and L2 penalize weights. Without scaling, these penalties can become inconsistent due to varying magnitudes of feature values.

#### 3.17. Normalization, Standardization, and Regularization


---


Normalization transforms data to fit within a specific range, typically [0, 1]. This is done by rescaling each feature so that the minimum value becomes 0 and the maximum becomes 1.

Formula:
$$
X' = \frac{X-X_{min}}{X_{max}-X_{min}}
$$

When to Use:

When you know the data has a specific range or when dealing with algorithms that rely on bounded distances, like KNN or Neural Networks.

---

Standardization centers the data around 0 and scales it to have a standard deviation of 1.

Formula:
$$
X' = \frac{X-𝜇}{σ}
$$

Where:
* μ is the mean and
* σ is the standard deviation.

When to Use:

When you need data with a Gaussian-like distribution or when the algorithm (e.g., PCA, SVM) assumes data to be standardized.

---


Regularization refers to techniques used to reduce overfitting by penalizing large model coefficients during training. While it's not directly related to scaling, scaling ensures that regularization penalties (like L1 or L2) are applied consistently across all features.

#### Task:
1. Normalize the dataset
2. Plot a histogram of each variable to ensure it falls between 0 and 1
3. Why do you think it was important to normalize this dataset?

```{code-cell} ipython3
# Your code goes here
```

### Exercise 3.6: Multi-Collinearity

#### 3.18. What is Multi-Collinearity?

Multi-collinearity occurs when two or more independent variables in a dataset are highly correlated with each other. This means that one variable can be linearly predicted from another with a high degree of accuracy. While correlations describe relationships between two variables, multi-collinearity refers specifically to strong correlations between predictors (independent variables) in a model.

For example, in a dataset with height in inches and height in centimeters, these two variables would be perfectly collinear because one is a direct conversion of the other.

Multi-collinearity can be detected using statistical methods like correlation matrices to find highly correlated variables.

Types of Multi-Collinearity


---


1. Perfect Multi-Collinearity: When one variable is a perfect linear function of another (e.g., total price = price per item × number of items).
2. Imperfect Multi-Collinearity: When variables are strongly correlated but not perfectly.
Detecting Multi-Collinearity

#### 3.19. Why Deal with Multi-Collinearity?

Multi-collinearity can negatively affect machine learning models and statistical analyses, especially regression models. Here's why it's important to address it:

Unstable Coefficients:


---


When predictors are highly correlated, it becomes difficult for the model to determine which variable is contributing to the outcome. This can lead to unreliable or fluctuating coefficient estimates.

Inflated Standard Errors:


---


Multi-collinearity increases the standard errors of the estimated coefficients, making it harder to assess their statistical significance.

Reduced Interpretability:


---


In cases of multi-collinearity, the relationship between predictors and the target variable can become less clear, complicating interpretation.

Overfitting Risks:


---


Strong correlations between features can lead to redundancy in the model, which increases the risk of overfitting, particularly in machine learning.

Consequences of Ignoring Multi-Collinearity


---


* Models may produce counterintuitive or nonsensical results, such as a positive coefficient for one predictor and a negative coefficient for another predictor that measures a similar concept.
* Predictions may remain accurate, but the ability to understand or explain the model is compromised.
By identifying and addressing multi-collinearity (e.g., through feature selection, dimensionality reduction, or combining correlated features), you can ensure a more stable and interpretable model.

#### 3.20. Pricipal Component Analysis (PCA)

Principal Component Analysis (PCA) is a dimensionality reduction technique used to simplify a dataset while retaining as much of the original information as possible. PCA works by transforming the original features into a new set of features, called principal components, which are uncorrelated and capture the maximum variance in the data.

Why Use PCA?
* Reduce Dimensionality: Simplify datasets with many features, making them easier to visualize and analyze.
* Remove Redundancy: Address multi-collinearity by combining correlated variables into fewer components.
* Improve Model Performance: Reduce noise in the dataset, which can help improve the performance and generalization of machine learning models.
* Visualization: PCA enables visualization of high-dimensional data in 2D or 3D.

How Does PCA Work?
* Standardize the Data: PCA works best when data is standardized to have a mean of 0 and a standard deviation of 1.
* Compute the Covariance Matrix: PCA evaluates the relationships between variables.
* Calculate Eigenvectors and Eigenvalues: Eigenvectors define the directions of the new components, while eigenvalues represent the magnitude of variance captured by each component.
* Select Principal Components: Choose the top components that explain the majority of the variance.
* Transform the Data: Project the original data onto the selected principal components.

Code Example: PCA with Python

```
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Sample dataset
data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [2, 4, 6, 8, 10],
    'Feature3': [5, 7, 8, 10, 12],
    'Feature4': [10, 20, 30, 40, 50]
})

# Step 1: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 2: Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components
principal_components = pca.fit_transform(scaled_data)

# Step 3: Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
print("Principal Components:")
print(pca_df)

# Step 4: Explained Variance Ratio
explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance Ratio:", explained_variance)

# Step 5: Plot the principal components
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c='blue', edgecolor='k', s=50)
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()
```

Explanation of the Output
* Principal Components: The transformed data in terms of the two selected components, PC1 and PC2.
Explained Variance Ratio: The proportion of variance captured by each principal component. Use this to decide how many components to retain.
* Scatter Plot: Visualizes the data in terms of the top two principal components, which often reveals clusters or patterns.
When to Use PCA?
* When you have a large number of features, and many of them are correlated.
* When you need to visualize high-dimensional data.
* When you want to reduce the complexity of your dataset without losing much information.

#### Task:
1. Identify variables that have a high correlation which could skew the model.
2. Use PCA to rewrite the variables

```{code-cell} ipython3
# Your code goes here
```

#### Answer Key:

#### Excercise 3.1:

```
import pandas as pd

# Step 1: Import the dataset
df = pd.read_excel('Section3.xlsx')

# Step 2: Print the first five rows of the spreadsheet
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Print the type of data in each column
print("\nData types of each column:")
print(df.dtypes)

# Step 4: Check for incorrect data types and fix them
# For each issue with a column, e.g., 'ColumnName' being interpreted as a string instead of a numeric type
# Let's say 'ColumnName' should be of type float, but it's being read as object (string).

if df['ColumnName'].dtype == 'object':
    df['ColumnName'] = pd.to_numeric(df['ColumnName'], errors='coerce')

# After correcting, check the data types again
print("\nData types after correction:")
print(df.dtypes)

# Optional: Print the corrected first 5 rows
print("\nCorrected first 5 rows of the dataset:")
print(df.head())
```

#### Excercise 3.2:

```
import pandas as pd

# Step 1: Import the dataset
df = pd.read_excel('Section3.xlsx')

# Step 2: Check for missing data and remove any rows that contain missing values
print("Missing data before removal:")
print(df.isnull().sum())  # Shows the count of missing values per column

# Remove rows with missing data
df_cleaned = df.dropna()

# Verify that missing data has been removed
print("\nMissing data after removal:")
print(df_cleaned.isnull().sum())

# Step 3: Check for duplicate rows and remove them
print("\nDuplicate rows before removal:")
print(df_cleaned.duplicated().sum())  # Shows the count of duplicate rows

# Remove duplicate rows
df_cleaned = df_cleaned.drop_duplicates()

# Verify that duplicates have been removed
print("\nDuplicate rows after removal:")
print(df_cleaned.duplicated().sum())

# Optional: Show the cleaned dataset (first 5 rows)
print("\nFirst 5 rows of the cleaned dataset:")
print(df_cleaned.head())
```

#### Excercise 3.3:

```
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Import the dataset
df = pd.read_excel('Section3.xlsx')

# Step 2: Use the IQR method to locate outliers
Q1 = df.quantile(0.25)  # 25th percentile
Q3 = df.quantile(0.75)  # 75th percentile
IQR = Q3 - Q1  # Interquartile range

# Define outlier condition
outliers_condition = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

# Step 3: Use a box and whisker plot to visualize any outliers
plt.figure(figsize=(10, 6))
df.boxplot()
plt.title("Box and Whisker Plot to Visualize Outliers")
plt.show()

# Step 4: Remove outliers
df_no_outliers = df[~outliers_condition.any(axis=1)]  # Keep rows without any outliers

# Step 5: Confirm that outliers have been successfully removed
# Check for outliers in the cleaned dataset
Q1_cleaned = df_no_outliers.quantile(0.25)
Q3_cleaned = df_no_outliers.quantile(0.75)
IQR_cleaned = Q3_cleaned - Q1_cleaned

outliers_condition_cleaned = (df_no_outliers < (Q1_cleaned - 1.5 * IQR_cleaned)) | (df_no_outliers > (Q3_cleaned + 1.5 * IQR_cleaned))

# Check if any outliers remain
if outliers_condition_cleaned.any().any():
    print("Outliers still exist in the cleaned dataset.")
else:
    print("Outliers have been successfully removed from the dataset.")

# Optional: Show the first 5 rows of the dataset without outliers
print("\nFirst 5 rows of the dataset without outliers:")
print(df_no_outliers.head())
```

#### Excercise 3.4:

```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Import the dataset
df = pd.read_excel('Section3.xlsx')

# Step 2: Generate a correlation matrix for the dataset
correlation_matrix = df.corr()

# Step 3: Create a heatmap to visualize the correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title("Correlation Heatmap")
plt.show()

# Step 4: Find the strongest positive correlations
positive_corr = correlation_matrix[correlation_matrix > 0.7]  # Adjust threshold as needed
strongest_positive_corr = positive_corr.stack().sort_values(ascending=False)

print("\nStrongest Positive Correlations:")
print(strongest_positive_corr)

# Step 5: Find the strongest negative correlations
negative_corr = correlation_matrix[correlation_matrix < -0.7]  # Adjust threshold as needed
strongest_negative_corr = negative_corr.stack().sort_values(ascending=True)

print("\nStrongest Negative Correlations:")
print(strongest_negative_corr)

# Step 6: Check if any independent variables don't correlate with the target (assumed to be the last column)
target_column = df.columns[-1]  # Assuming target is the last column
independent_vars = df.columns[:-1]

no_corr_with_target = [var for var in independent_vars if correlation_matrix.loc[var, target_column] == 0]

print("\nIndependent Variables with No Correlation to the Target Variable:")
print(no_corr_with_target)
```

#### Excercise 3.5:

#### Excercise 3.6:
```
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Step 1: Import the dataset
df = pd.read_excel('Section3.xlsx')

# Step 2: Normalize the dataset using MinMaxScaler
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Step 3: Plot histograms of each variable to ensure it falls between 0 and 1
plt.figure(figsize=(12, 8))
df_normalized.hist(bins=20, figsize=(12, 8), layout=(4, 3), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Normalized Variables")
plt.show()

# Optional: Print the first 5 rows of the normalized dataset to verify
print("\nFirst 5 rows of the normalized dataset:")
print(df_normalized.head())
```

#### Excercise 3.7:

```
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Import the dataset
df = pd.read_excel('Section3.xlsx')

# Step 2: Compute the correlation matrix to identify highly correlated variables
correlation_matrix = df.corr()

# Step 3: Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title("Correlation Heatmap")
plt.show()

# Step 4: Identify pairs of variables that have a high correlation (e.g., above 0.9)
high_corr_pairs = []
threshold = 0.9  # You can adjust this threshold as needed

for col in correlation_matrix.columns:
    for row in correlation_matrix.index:
        if abs(correlation_matrix.loc[row, col]) > threshold and row != col:
            high_corr_pairs.append((row, col))

# Print the high correlation pairs
print("\nHigh Correlation Pairs:")
for pair in high_corr_pairs:
    print(pair)

# Step 5: Apply PCA to reduce the dimensionality and rewrite the variables
# Standardize the data before PCA (important for PCA)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Apply PCA
pca = PCA(n_components=len(df.columns))  # You can specify the number of components you want
pca_result = pca.fit_transform(df_standardized)

# Create a DataFrame of the PCA components
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(len(df.columns))])

# Step 6: Explain the explained variance ratio for each principal component
print("\nExplained Variance Ratio of each Principal Component:")
print(pca.explained_variance_ratio_)

# Optional: Show the first few rows of the PCA-rewritten dataset
print("\nFirst 5 rows of the PCA-transformed dataset:")
print(pca_df.head())
```