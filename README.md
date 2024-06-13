# Foreseeing the Dark: Machine Learning for Early Power Outage Detection

## Introduction

Welcome to the Power Outages Project! This website provides an overview of our project, aiming to make the findings accessible to a broad audience, including classmates, friends, family, recruiters, and random internet strangers. Here, we present the key elements of our analysis, focusing on understanding the leading indicators of power outages and exploring the potential to develop an early warning system.

## Understanding the Data

The dataset used for this project is the Power Outages dataset, which contains detailed information about various power outage events. Below are the key columns that are relevant to our analysis:

## Project Question

**What are the leading indicators of power outages, and can we build an early warning system to detect potential outages?**

## Importance of the Question

Developing an early warning system for power outages is crucial for several reasons:

- **Preparedness and Response**: Helps utility companies and emergency services prepare for and respond more effectively to impending outages.
- **Real-time Data Processing**: Demonstrates the ability to handle and analyze data in real-time, which is crucial for practical applications.
- **Anomaly Detection**: Identifying rare and unusual events shows proficiency in anomaly detection, a valuable skill in machine learning.
- **Practical Impact**: An early warning system has immediate practical applications, providing significant value to stakeholders by potentially preventing outages or mitigating their impact.

By investigating this question and developing an early warning system, we aim to create a practical solution that can help mitigate the effects of power outages and improve the resilience of power infrastructure.

## Dataset Info

- **OUTAGE.START**: The start time of the power outage.
- **OUTAGE.END**: The end time of the power outage.
- **CAUSE.CATEGORY**: The category of the cause of the power outage (e.g., weather, equipment failure).
- **CAUSE.CATEGORY.DETAIL**: Detailed description of the cause.
- **OUTAGE.DURATION**: Duration of the outage in minutes.
- **CUSTOMERS.AFFECTED**: Number of customers affected by the outage.
- **ZIP.CODE**: The ZIP code where the outage occurred.
- **DATE.EVENT**: The date of the power outage event.
- **CLIMATE.CATEGORY**: The climate category (e.g., humid, arid) of the area affected.
- **TEMPERATURE**: Temperature at the time of the outage.
- **PRECIPITATION**: Precipitation level at the time of the outage.

The dataset contains a total of 1540 rows and 57 columns, providing a comprehensive view of the factors involved in power outages.

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

In our data cleaning process, we performed the following steps:

1. **Removed Metadata**: The first four rows were metadata, so they were removed.
2. **Set Column Names**: The fifth row was used as the header for column names.
3. **Removed Empty Columns and Rows**: Columns and rows with all NaN values were dropped.
4. **Converted Data Types**: Appropriate columns were converted to numeric types.
5. **Handled Missing Values**: Replaced placeholders with NaN and dropped rows with significant missing values.
6. **Date and Time Processing**: Combined date and time columns to create datetime columns for OUTAGE.START and OUTAGE.END.
7. **Scaled Numerical Data**: Scaled numerical columns like demand loss and prices for analysis.

These steps ensured the dataset was clean and ready for analysis. Below is the head of the cleaned DataFrame:

---

### Uni/Bi-Variate Analysis

## Univariate Analysis

In our univariate analysis, we explored the distributions of several key columns from the Power Outages dataset. These distributions provide insights into the variability and characteristics of the data.

### Distribution of Outage Duration

We analyzed the distribution of outage durations to understand the range and frequency of different outage lengths.

![Distribution of Outage Duration](outage_duration.png)

**Explanation**: The distribution of outage durations shows high variability and significant outliers. While most outages are relatively short, there are some extreme cases with very long durations.

### Distribution of Customers Affected

We examined how many customers were affected by the outages to identify the scale and impact of different events.

![Distribution of Customers Affected](customers_affected.png)

**Explanation**: The distribution indicates that while many outages affect a smaller number of customers, there are instances where a large number of customers are impacted, highlighting the importance of understanding and mitigating large-scale outages.

### Distribution of Demand Loss

Understanding the distribution of demand loss helps to see how power demand is affected during outages.

![Distribution of Demand Loss](demand_loss.png)

**Explanation**: The demand loss data is mostly centered around zero, with a wide range of both positive and negative values, indicating that some outages result in substantial demand loss while others do not.

## Bivariate Analysis

In our bivariate analysis, we explored the relationships between pairs of columns to identify possible associations and trends.

### Outage Duration vs. Customers Affected

We plotted outage duration against the number of customers affected to see if there is a relationship between the two.

![Outage Duration vs. Customers Affected](outage_duration_vs_customers_affected.png)

**Explanation**: The scatter plot reveals a weak positive correlation (0.26) between outage duration and the number of customers affected, indicating that longer outages tend to affect more customers, but other factors also play significant roles.

### Outage Duration by Climate Category

We analyzed how different climate conditions impact the duration of power outages.

![Outage Duration by Climate Category](outage_duration_by_climate_category.png)

**Explanation**: The box plot shows that warm climates have the highest mean outage duration, but cold climates have the most variability and longest maximum outage durations. This indicates that climate plays a significant role in the duration of outages.

### Customers Affected vs. Demand Loss

We explored the relationship between the number of customers affected and the demand loss to highlight potential indicators of large-scale outages.

![Customers Affected vs. Demand Loss](customers_affected_vs_demand_loss.png)

**Explanation**: The scatter plot shows a moderate positive relationship (0.52) between customers affected and demand loss, highlighting demand loss as a significant indicator of large-scale outages.

### Outage Duration vs. Year

To analyze trends over time, we looked at how outage durations have changed over the years.

![Outage Duration vs. Year](outage_duration_vs_year.png)

**Explanation**: The scatter plot suggests a slight negative trend in outage duration over the years, indicating improvements in outage management and infrastructure over time.

These analyses provide a comprehensive understanding of the data and lay the groundwork for developing an early warning system for power outages.

---

# Interesting Aggregates

## Grouped and Pivot Tables

### Aggregate Statistics by Climate Category

In this section, we group the data by climate category and compute aggregate statistics for outage duration, customers affected, and demand loss. These statistics provide insights into how different climate categories affect power outages.

![Aggregate Statistics by Climate Category](aggregate_climate_category.png)

### Aggregate Statistics by Year

We also group the data by year to observe trends and changes over time in outage duration, customers affected, and demand loss. This helps us understand how power outage characteristics have evolved.

![Aggregate Statistics by Year](aggregate_year.png)

### Aggregate Statistics by State

Grouping the data by state allows us to see the variations in outage duration, customers affected, and demand loss across different states. This analysis is crucial for understanding regional differences.

![Aggregate Statistics by State](aggregate_state.png)

### Pivot Table: Mean Outage Duration by Climate Category and Year

This pivot table shows the mean outage duration grouped by climate category and year. It helps us understand how climate conditions and time periods affect the duration of power outages.

![Pivot Table: Mean Outage Duration by Climate Category and Year](pivot_climate_year.png)

### Pivot Table: Mean Customers Affected by State and Year

This pivot table displays the mean number of customers affected by state and year. It provides insights into the impact of power outages on different states over time.

![Pivot Table: Mean Customers Affected by State and Year](pivot_state_year.png)

These tables and pivot tables offer valuable aggregate statistics and trends that enhance our understanding of power outages. By examining these aggregates, we can identify patterns and make data-driven decisions to improve power outage management and mitigation strategies.


---

## Framing a Prediction Problem

### Step 1: Problem Identification

**Primary Question:** What are the leading indicators of power outages, and can we build an early warning system to detect potential outages?

**Importance of the Question:**
- **Preparedness and Response:** Helps utility companies and emergency services prepare for and respond more effectively to impending outages.
- **Practical Impact:** An early warning system has immediate practical applications, providing significant value to stakeholders by potentially preventing outages or mitigating their impact.

### Machine Learning Techniques and Approach

#### Model 1: Identifying Leading Indicators of Power Outages

**Objective:** Determine which features in the dataset are strong predictors of power outages.

**Technique:** Use supervised learning (e.g., logistic regression, decision trees) to identify significant features.

**Data:** Features such as weather conditions, time of year, population density, and previous outage history.

#### Model 2: Building an Early Warning System

**Objective:** Predict the likelihood of a power outage occurring within a given timeframe.

**Technique:** Use supervised learning (e.g., random forest, gradient boosting) to build a prediction model based on the identified indicators.

**Data:** Use the features identified in Model 1 to train the model.

### Prediction Problem Type and Details

**Prediction Problem Type:** Classification

**Type of Classification:** Binary classification (predicting whether a power outage will occur or not).

**Response Variable:** The variable we are predicting is `OUTAGE_OCCURRENCE`, a binary variable indicating whether a power outage occurs within a given timeframe.

**Chosen Metric:** F1-Score

**Justification for Metric Choice:** 
- **F1-Score** is chosen over accuracy because it provides a better balance between precision and recall, which is crucial in the context of predicting outages where both false positives (unnecessary alerts) and false negatives (missed outages) can have significant consequences.
  
### Information Known at the Time of Prediction

When predicting power outages, we ensure that the model only uses features that would be known before the outage occurs. This includes:
- Historical weather conditions
- Time of year
- Population density
- Previous outage history

These features are chosen based on their relevance and availability prior to the occurrence of an outage.

### Implementation

#### Model 1: Identifying Leading Indicators

#### Model 2: Building an Early Warning System

---

## Step 6: Baseline Model

### Model 1: Predicting the Likelihood of Power Outages

**Objective:** Predict whether a power outage will occur within a given timeframe.

**Technique:** Supervised learning using logistic regression.

**Data Preparation:**
- Features: YEAR, CLIMATE.CATEGORY, POPDEN_URBAN, RES.PRICE
- Target Variable: Binary (OUTAGE_OCCURRED)
- Steps:
  1. Extract the year from the OUTAGE.START column.
  2. Drop rows with missing values in the selected features and target (OUTAGE.DURATION).
  3. Create a binary target variable (OUTAGE_OCCURRED).

**Data Balancing:**
- Upsample the minority class to balance the dataset.

**Preprocessing:**
- Standardize numerical features (YEAR, POPDEN_URBAN, RES.PRICE).
- One-hot encode categorical features (CLIMATE.CATEGORY).

**Pipeline Creation:**
- Combine preprocessing steps and logistic regression into a single pipeline.

**Model Training and Evaluation:**
- Split the data into training and testing sets.
- Train the baseline model.
- Evaluate the model using accuracy, ROC-AUC score, and a classification report.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.utils import resample

# Load your dataset
# Assuming df is your DataFrame
df['YEAR'] = pd.to_datetime(df['OUTAGE.START']).dt.year

# Select relevant features and target
features = ['YEAR', 'CLIMATE.CATEGORY', 'POPDEN_URBAN', 'RES.PRICE']
df = df.dropna(subset=features + ['OUTAGE.DURATION'])
df['OUTAGE_OCCURRED'] = df['OUTAGE.DURATION'].apply(lambda x: 1 if x > 0 else 0)  # Binary target variable

# Balancing the dataset
df_majority = df[df.OUTAGE_OCCURRED == 1]
df_minority = df[df.OUTAGE_OCCURRED == 0]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=42) # reproducible results

df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced[features]
y = df_balanced['OUTAGE_OCCURRED']

# Define preprocessing for numerical and categorical features
numerical_features = ['YEAR', 'POPDEN_URBAN', 'RES.PRICE']
categorical_features = ['CLIMATE.CATEGORY']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the baseline model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
report = classification_report(y_test, y_pred, zero_division=0)

# Print evaluation results
print("Baseline Model Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"ROC-AUC Score: {roc_auc}")
print("Classification Report:")
print(report)
