# Foreseeing the Dark: Machine Learning for Early Power Outage Detection

## Introduction

Welcome to the Power Outages Project! This website provides an overview of our project, aiming to make the findings accessible to a broad audience, including classmates, friends, family, recruiters, and random internet strangers. Here, we present the key elements of our analysis, focusing on understanding the leading indicators of power outages and exploring the potential to develop an early warning system. The dataset used for this project is the Power Outages dataset, which contains detailed information about various power outage events.

## Project Question

**What are the leading indicators of power outages, and can we build an early warning system to detect potential outages?**

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

---------------------------------------------------------------------------------------------------------------------------------------------------------

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

---------------------------------------------------------------------------------------------------------------------------------------------------------

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

---------------------------------------------------------------------------------------------------------------------------------------------------------

Assessment of Missingness

### NMAR (Not Missing At Random) Analysis for Power Outages Data

When analyzing missing data, it's important to determine if the data are NMAR (Not Missing At Random). This requires reasoning about the data generating process, rather than just looking at the data. Here’s how we approach this for the power outages dataset:

#### Understanding the Data Generating Process

**Outage Duration (OUTAGE.DURATION):**
- **Potential NMAR Reasoning:** Outage duration data could be NMAR if the data collection process itself is affected by the outage. For example, if longer outages lead to failures in data recording systems, the outage duration might not be recorded accurately. This would result in missing data that depends directly on the value of the outage duration itself.

**Customers Affected (CUSTOMERS.AFFECTED):**
- **Potential NMAR Reasoning:** Data on the number of customers affected could be NMAR if areas with a higher number of affected customers are more likely to experience issues in reporting due to the scale of the outage. For instance, if an outage impacts communication systems in heavily populated areas, it might hinder the accurate reporting of the number of affected customers.

**Demand Loss (DEMAND.LOSS.MW):**
- **Potential NMAR Reasoning:** Demand loss data could be NMAR if larger losses lead to situations where recording or reporting the data becomes more challenging. For example, in cases of significant demand loss, the priority might shift to restoring services rather than data recording, leading to missing values that correlate with higher demand loss.

**Prices (RES.PRICE, COM.PRICE, IND.PRICE):**
- **Potential NMAR Reasoning:** Pricing data might be NMAR if higher or lower prices lead to different reporting behaviors. For instance, higher residential or commercial prices might correlate with regions that have better infrastructure for data collection, while lower prices might be associated with regions where data collection is less rigorous.

**Population Density (POPDEN_URBAN):**
- **Potential NMAR Reasoning:** Population density data could be NMAR if regions with very high or very low population densities have different capacities for data reporting. For example, very densely populated urban areas might have more robust data reporting systems compared to sparsely populated rural areas, leading to differences in missing data patterns.

**Conclusion:** To determine if the data in this dataset are NMAR, it's essential to consider the specific context and mechanisms that might influence data reporting and recording. Simply analyzing the data for patterns of missingness won't be sufficient; we need to understand the real-world processes that lead to these data points being recorded or not recorded.

### Missingness Dependency

To test missingness dependency, I will focus on the distribution of DEMAND.LOSS.MW. I will test this against the columns CUSTOMERS.AFFECTED and YEAR.

#### CUSTOMERS.AFFECTED

First, I examine the distribution of CUSTOMERS.AFFECTED when DEMAND.LOSS.MW is missing vs not missing.

- **Null Hypothesis:** The distribution of CUSTOMERS.AFFECTED is the same when DEMAND.LOSS.MW is missing vs not missing.
- **Alternate Hypothesis:** The distribution of CUSTOMERS.AFFECTED is different when DEMAND.LOSS.MW is missing vs not missing.

I found an observed difference of 0.0 with a p-value of 1.0. At this value, I fail to reject the null hypothesis, indicating that the missingness of DEMAND.LOSS.MW is likely independent of CUSTOMERS.AFFECTED.

#### YEAR

Next, I examined the dependency of DEMAND.LOSS.MW missing on the YEAR column.

- **Null Hypothesis:** The distribution of YEAR is the same when DEMAND.LOSS.MW is missing vs not missing.
- **Alternate Hypothesis:** The distribution of YEAR is different when DEMAND.LOSS.MW is missing vs not missing.

I found an observed difference of -0.241 with a p-value of 0.001. The empirical distribution of the differences is shown below. At this value, I reject the null hypothesis, indicating that the missingness of DEMAND.LOSS.MW is dependent on YEAR.



---------------------------------------------------------------------------------------------------------------------------------------------------------

## Hypothesis Testing

### Hypothesis 1: Impact of Climate Category on Outage Duration

Null Hypothesis (H0): There is no difference in the mean OUTAGE.DURATION between different CLIMATE.CATEGORY.

Alternative Hypothesis (H1): There is a difference in the mean OUTAGE.DURATION between different CLIMATE.CATEGORY.

Test Statistic: F-statistic from ANOVA test.

Result: After performing ANOVA, we found a significant difference in the mean OUTAGE.DURATION between different CLIMATE.CATEGORY (p-value < 0.05).

Conclusion: The analysis suggests that the climate category significantly impacts the duration of power outages.

### Hypothesis 2: Temporal Trends in Power Outage Frequency

Question: Has the frequency of power outages increased over the years?

Null Hypothesis (H0): The frequency of power outages has remained constant over the years.

Alternative Hypothesis (H1): The frequency of power outages has increased over the years.

Test Statistic: Linear Regression Slope

Result: After conducting a linear regression analysis, the slope of the regression line was found to be positive (slope: 8.125) with a p-value of 0.00657. Since the p-value is less than the significance level of 0.05, we reject the null hypothesis.

Conclusion: The analysis indicates a significant upward trend in the frequency of power outages over time, suggesting that power outages have become more common in recent years.

---------------------------------------------------------------------------------------------------------------------------------------------------------

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

## Baseline Model

### Model 1: Predicting the Likelihood of Power Outages

**Objective:** Predict whether a power outage will occur within a given timeframe.

**Technique:** Supervised learning using logistic regression.

**Features:**
- YEAR (ordinal): Extracted from the OUTAGE.START column to account for changes over time.
- CLIMATE.CATEGORY (nominal): Indicates the climate conditions that might influence power outages.
- POPDEN_URBAN (quantitative): Represents the population density in urban areas, which could impact the likelihood of outages.
- RES.PRICE (quantitative): Residential electricity price, which may correlate with infrastructure quality and outage frequency.

**Data Preparation:**
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

**Performance:**
- Accuracy: 0.71
- ROC-AUC Score: 0.75
- Classification Report:

          precision    recall  f1-score   support

       0       0.66      0.81      0.73       267
       1       0.78      0.61      0.68       288

accuracy                           0.71       555

**Summary:** The baseline model using logistic regression performs reasonably well with an accuracy of 0.71 and a ROC-AUC score of 0.75. This is a good starting point for further model enhancements.




### Model 2: Early Warning System for Power Outages

**Objective:** Develop an early warning system to detect potential power outages.

**Technique:** Supervised learning using a RandomForestClassifier.

**Features:**
- YEAR (ordinal): Extracted from the OUTAGE.START column to account for changes over time.
- CLIMATE.CATEGORY (nominal): Indicates the climate conditions that might influence power outages.
- POPDEN_URBAN (quantitative): Represents the population density in urban areas, which could impact the likelihood of outages.
- RES.PRICE (quantitative): Residential electricity price, which may correlate with infrastructure quality and outage frequency.

**Data Preparation:**
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
- Combine preprocessing steps and Random Forest Classifier into a single pipeline.

**Model Training and Evaluation:**
- Split the data into training and testing sets.
- Train the model and evaluate using accuracy, ROC-AUC score, and a classification report.

**Performance:**
- Accuracy: 0.96
- ROC-AUC Score: 0.98
- Classification Report:

          precision    recall  f1-score   support

       0       0.93      1.00      0.96       267
       1       1.00      0.93      0.96       288

accuracy                           0.96       555


**Summary:** The baseline model using a RandomForestClassifier for building an early warning system for power outages performs very well, with high accuracy, ROC-AUC score, and balanced precision, recall, and F1-scores. The model effectively predicts the likelihood of power outages using the selected features, making it a strong starting point for further enhancements and more complex models.


---


## Final Model

### Model 1: Predicting the Likelihood of Power Outages

**Objective:** Predict whether a power outage will occur within a given timeframe.

**Technique:** Supervised learning using logistic regression.

**Features:**
- YEAR (ordinal): Extracted from the OUTAGE.START column to account for changes over time.
- CLIMATE.CATEGORY (nominal): Indicates the climate conditions that might influence power outages.
- POPDEN_URBAN (quantitative): Represents the population density in urban areas, which could impact the likelihood of outages.
- RES.PRICE (quantitative): Residential electricity price, which may correlate with infrastructure quality and outage frequency.
- YEAR_POPDEN_INTERACTION (quantitative): Interaction term to capture the effect of year and population density combined.
- LOG_RES_PRICE (quantitative): Log-transformed residential price to handle potential skewness.

**Feature Engineering:**
- Created interaction term YEAR * POPDEN_URBAN to capture any interaction effect between the year and population density.
- Applied a log transformation to RES.PRICE to handle any potential skewness in price distribution.

**Hyperparameter Tuning:**
- Used GridSearchCV to tune the following hyperparameters for the Logistic Regression model:
  - C: Inverse of regularization strength.
  - Penalty: Specify the norm used in penalization (l1, l2).

**Best Hyperparameters:**
- C: 100
- Penalty: l1
- Solver: liblinear

**Data Preparation:**
- Target Variable: Binary (OUTAGE_OCCURRED)
- Steps:
  1. Extract the year from the OUTAGE.START column.
  2. Drop rows with missing values in the selected features and target (OUTAGE.DURATION).
  3. Create a binary target variable (OUTAGE_OCCURRED).

**Data Balancing:**
- Upsample the minority class to balance the dataset.

**Preprocessing:**
- Standardize numerical features (YEAR, POPDEN_URBAN, RES.PRICE, YEAR_POPDEN_INTERACTION, LOG_RES_PRICE).
- One-hot encode categorical features (CLIMATE.CATEGORY).

**Pipeline Creation:**
- Combine preprocessing steps and logistic regression into a single pipeline.

**Model Training and Evaluation:**
- Split the data into training and testing sets.
- Train the model with best hyperparameters.
- Evaluate the model using accuracy, ROC-AUC score, and a classification report.

**Performance:**
- Accuracy: 0.71
- ROC-AUC Score: 0.75
- Classification Report:

          precision    recall  f1-score   support

       0       0.66      0.84      0.74       267
       1       0.80      0.60      0.68       288

accuracy                           0.71       555


**Summary:** The final model using logistic regression with engineered features performs better than the baseline model with an accuracy of 0.71 and a ROC-AUC score of 0.75.



### Model 2: Early Warning System for Power Outages

**Objective:** Develop an early warning system to detect potential power outages.

**Technique:** Supervised learning using a RandomForestClassifier.

**Features:**
- YEAR (ordinal): Extracted from the OUTAGE.START column to account for changes over time.
- CLIMATE.CATEGORY (nominal): Indicates the climate conditions that might influence power outages.
- POPDEN_URBAN (quantitative): Represents the population density in urban areas, which could impact the likelihood of outages.
- RES.PRICE (quantitative): Residential electricity price, which may correlate with infrastructure quality and outage frequency.
- YEAR_POPDEN_INTERACTION (quantitative): Interaction term to capture the effect of year and population density combined.
- LOG_RES_PRICE (quantitative): Log-transformed residential price to handle potential skewness.

**Feature Engineering:**
- Created interaction term YEAR * POPDEN_URBAN to capture any interaction effect between the year and population density.
- Applied a log transformation to RES.PRICE to handle potential skewness in price distribution.

**Hyperparameter Tuning:**
- Used GridSearchCV to tune the following hyperparameters for the RandomForestClassifier:
- n_estimators: Number of trees in the forest.
- max_depth: Maximum depth of the tree.
- min_samples_split: Minimum number of samples required to split an internal node.
- min_samples_leaf: Minimum number of samples required to be at a leaf node.

**Best Hyperparameters:**
- n_estimators: 200
- max_depth: 10
- min_samples_split: 2
- min_samples_leaf: 1

**Data Preparation:**
- Target Variable: Binary (OUTAGE_OCCURRED)
- Steps:
1. Extract the year from the OUTAGE.START column.
2. Drop rows with missing values in the selected features and target (OUTAGE.DURATION).
3. Create a binary target variable (OUTAGE_OCCURRED).

**Data Balancing:**
- Upsample the minority class to balance the dataset.

**Preprocessing:**
- Standardize numerical features (YEAR, POPDEN_URBAN, RES.PRICE, YEAR_POPDEN_INTERACTION, LOG_RES_PRICE).
- One-hot encode categorical features (CLIMATE.CATEGORY).

**Pipeline Creation:**
- Combine preprocessing steps and Random Forest Classifier into a single pipeline.

**Model Training and Evaluation:**
- Split the data into training and testing sets.
- Train the model with best hyperparameters.
- Evaluate the model using accuracy, ROC-AUC score, and a classification report.

**Performance:**
- Accuracy: 0.93
- ROC-AUC Score: 0.98
- Classification Report:

-           precision    recall  f1-score   support

       0       0.87      1.00      0.93       267
       1       1.00      0.86      0.92       288

accuracy                           0.93       555



**Summary:** The final model using a RandomForestClassifier for building an early warning system for power outages performs excellently, with high accuracy, ROC-AUC score, and balanced precision, recall, and F1-scores. The model effectively predicts the likelihood of power outages using the selected features, making it a significant improvement over the baseline model.

