#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install imbalanced-learn


# In[2]:


pip install category_encoders


# In[3]:


pip install xgboost


# In[4]:


pip install scikit-plot


# In[5]:


import category_encoders as ce
import imblearn.combine as hib
import imblearn.over_sampling as os
import imblearn.under_sampling as us
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
import seaborn as sns
import shap
import time
import warnings
warnings.filterwarnings('ignore')

from collections import Counter
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import TomekLinks


from sklearn import feature_selection
from sklearn import model_selection
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import set_config

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# In[6]:


# Loading the dataset from the given path into a DataFrame
df = pd.read_csv('/Users/hung1/Desktop/Patient_Mortality_Prediction.csv')


# In[7]:


# Displaying the first 10 rows of the dataframe for a quick glance
df.head(10)


# In[8]:


# Displaying the last 10 rows of the dataframe for a quick glance
df.tail(10)


# Data overview

# In[9]:


# Displaying the general information about the DataFrame, including:
# the number of non-null values in each column, data type of each column, memory usage, etc.
df.info()


# In[10]:


# Displaying the summary statistics of the DataFrame's columns.
# This includes count, mean, std deviation, min value, 25th percentile, median, 75th percentile, and max value.
df.describe()


# In[11]:


# Calculating and displaying the number of missing values in each column of the DataFrame.
df.isnull().sum()


# Data Preprocessing

# 1. Elimination of redundant attributes

# In[12]:


# Define the target variable
df_output = 'hospital_death'

# Dropping unnecessary columns from the DataFrame
# 'Unnamed: 83' seems to be a redundant column, often generated during CSV export
# 'encounter_id' and 'hospital_id' are identifiers which might not be needed for analysis
df.drop(['Unnamed: 83', 'encounter_id', 'hospital_id'], axis=1, inplace=True)

# Displaying the general information about the DataFrame:
# This provides details such as data type, non-null counts, and memory usage for each column
df.info()


# In[13]:


# Displaying the summary statistics of the DataFrame's columns
# This will give insights like count, mean, std deviation, min value, quartiles, and max value for each column
df.describe()


# In[14]:


# Fetching and displaying rows with any missing values
# This helps in understanding which rows have missing data and may require imputation or deletion
missing_rows_df = df[df.isnull().any(axis=1)]
print(missing_rows_df)


# Data visualization

# In[15]:


# Getting the counts of each class in the target variable
counts = df[df_output].value_counts()

# Plotting the bar chart to visualize class distribution
plt.bar(counts.index, counts.values)

# Setting the x-ticks labels for better understanding
# Assuming 0 represents 'Surviving' and 1 represents 'Death' in the target variable
plt.xticks([0, 1], ['Surviving', 'Death'])

# Setting the label for the x-axis
plt.xlabel('Class')

# Setting the label for the y-axis
plt.ylabel('Number of instances')

# Setting the title for the plot to indicate we're checking imbalance
plt.title('Imbalance in the dataset')

# Displaying the plot
plt.show()


# In[16]:


# Specifying columns for which we want to visualize the correlations
category = [
    'age', 'bmi', 'elective_surgery', 'ethnicity', 'gender', 'height', 'weight', 'apache_2_diagnosis', 
    'apache_3j_diagnosis', 'apache_post_operative', 'arf_apache', 'gcs_eyes_apache', 
    'gcs_motor_apache', 'gcs_unable_apache', 'gcs_verbal_apache', 'heart_rate_apache', 
    'intubated_apache', 'map_apache', 'resprate_apache', 'temp_apache', 'ventilated_apache', 
    'd1_diasbp_max', 'd1_diasbp_min', 'd1_diasbp_noninvasive_max', 'd1_diasbp_noninvasive_min', 
    'd1_heartrate_max', 'd1_heartrate_min', 'd1_mbp_max', 'd1_mbp_min', 'd1_mbp_noninvasive_max', 
    'd1_mbp_noninvasive_min', 'd1_resprate_max', 'd1_resprate_min', 'd1_spo2_max', 'd1_spo2_min', 
    'd1_sysbp_max', 'd1_sysbp_min', 'd1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min', 
    'd1_temp_max', 'd1_temp_min', 'd1_glucose_max', 'd1_glucose_min', 
    'd1_potassium_max', 'd1_potassium_min', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 
    'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 
    'leukemia', 'lymphoma', 'solid_tumor_with_metastasis', 'apache_3j_bodysystem', 'apache_2_bodysystem'
]

# Compute the correlation matrix for the specified columns
corr = df[category].corr()

# Set up the figure and axis
plt.figure(figsize=(20,15))

# Display the heatmap of the correlation matrix
# Annotated with actual correlation values, using a blue color map
sns.heatmap(corr, annot=True, cmap='PuBu', linewidths=0.01, linecolor="white")

# Title for the heatmap
plt.title('Correlation Heatmap')

# Rotating x-axis labels for better visibility, keeping y-axis labels horizontal
plt.xticks(rotation=90)  
plt.yticks(rotation=0)

# Display the plotted heatmap
plt.show()


# In[17]:


# Check and print the number of duplicate rows in the dataframe
duplicate_rows = df[df.duplicated()]
print("Number of duplicate rows:", duplicate_rows.shape[0])


# In[18]:


# Display the count of different values in the 'hospital_death' column
print(Counter(df['hospital_death']))

# Group by 'patient_id' and keep the first occurrence. This helps in removing duplicates based on 'patient_id'
df = df.groupby('patient_id').first().reset_index()

# Drop the 'patient_id' column as it's no longer needed
df.drop(['patient_id'], axis=1, inplace=True)


# 2. Solution Description
# 
# Coding of categorical attributes and data imputation / 
# Elimination of correlated variables / 
# Data normalization / 
# Detection and management of Outliers / 
# Under Sampling /
# Noise reduction / 
# Parameter optimization for selected models

# 3. Data split

# In[19]:


# Initialize an empty list to store subsamples from each group
subsamples = []

# Group the dataframe by the output column (df_output) 
# and create a subsample containing 60% of each group
for group, data_group in df.groupby(df_output):
    subsample = data_group.sample(frac=0.6)
    subsamples.append(subsample)

# Concatenate all the subsamples to form the reduced dataframe
df_reduced = pd.concat(subsamples)


# In[20]:


# Display the count of different values in the 'df_output' column for the reduced dataframe
print(Counter(df_reduced[df_output]))


# In[21]:


# Required import for the train-test split
from sklearn.model_selection import train_test_split

# Splitting the reduced dataframe into features (X) and target/output (y)
X = df_reduced.drop(df_output, axis=1)
y = df_reduced[df_output]

# Splitting the data into training and testing sets
# The stratify parameter ensures that both training and testing sets have a similar ratio of target values
# A random seed (random_state) is set to 33 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=33)

# Displaying the features of the dataset
print(X)


# Imputation of missing values

# In[22]:


# Identifying numeric columns based on data types int64 and float64
numeric_variable_names = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()


# In[23]:


# Identifying categorical columns based on the object data type
categorical_variable_names = X_train.select_dtypes(include=['object']).columns.tolist()

# Extracting columns where the data type is object (another way of getting categorical variables)
cat = X_train.select_dtypes(include=['object'])

# Identifying categorical columns with unique values less than or equal to 2
categorical_variable_names_lessthan6 = cat.nunique()[cat.nunique() <= 2].index.tolist()

# Identifying categorical columns with more than 2 unique values
categorical_variable_names_morethan6_values = cat.nunique()[cat.nunique() > 2].index.tolist()


# Separated the categoricals into 2 types, those with less than 2 variables and those with more than 2

# In[24]:


# Define Pipelines for pre-processing techniques on categorical variables

# Pipeline for categorical variables with 6 or fewer unique values
# This pipeline first encodes the variables using an ordinal encoder and then imputes missing values using a KNN imputer.
categorical_less_than_6_imputer_transformer = Pipeline([
    ('encoder', ce.OrdinalEncoder(handle_missing='return_nan')),
    ('transformer', KNNImputer(missing_values=np.nan, n_neighbors=2))
])

# Pipeline for categorical variables with more than 6 unique values
# This pipeline first encodes the variables using a target encoder and then imputes missing values using a KNN imputer.
categorical_masDe6_imputer_transformer = Pipeline([
    ('encoder', ce.TargetEncoder(smoothing=0.0000001, handle_missing='return_nan')),
    ('transformer', KNNImputer(missing_values=np.nan, n_neighbors=2))
])

# ColumnTransformer for pre-processing the entire dataset 
# It applies different transformations to different subsets of columns based on their data type or characteristics.
preprocessing = ColumnTransformer([
    ('knn', KNNImputer(missing_values=np.nan), numeric_variable_names),  # KNN Imputation for numeric variables
    ('ordinal', categorical_less_than_6_imputer_transformer, categorical_variable_names_lessthan6),  # Ordinal encoding and imputation for categorical vars with <= 6 unique values
    ('output', categorical_masDe6_imputer_transformer, categorical_variable_names_morethan6_values)  # Target encoding and imputation for categorical vars with > 6 unique values
])


# Structure of data preprocessing

# In[25]:


# Define a Pipeline for the full data preprocessing
# First, the preprocessing step (imputation and encoding) is applied,
# followed by scaling of the data using StandardScaler.

data_preprocessing = Pipeline([
    ('preprocessing', preprocessing),
    ('scaler', StandardScaler())
])


# In[26]:


# Set the global option to display pipeline in a diagrammatic format
set_config(display="diagram")

# Display the configured data preprocessing pipeline
data_preprocessing


# In[27]:


# Making a copy of the original datasets to ensure the raw data remains untouched
X_train_full = X_train.copy()
X_test_full = X_test.copy()

# Capturing the start time for measuring preprocessing duration
start_time = time.time()

# Applying the preprocessing on the training dataset (fit and transform)
X_train_full = preprocessing.fit_transform(X_train_full, y_train)

# Applying the preprocessing on the test dataset (only transform)
X_test_full = preprocessing.transform(X_test_full)

# Capturing the end time
end_time = time.time()

# Printing the time taken for preprocessing in minutes
print("Runtime:", (end_time - start_time)/60, " minutes")


# In[28]:


# Converting the processed test dataset back to a DataFrame for better usability
X_test_full = pd.DataFrame(X_test_full, columns = X_train.columns)

# Displaying the first few rows of the processed test dataset
X_test_full.head()


# Eliminate variables based on correlations

# In[29]:


# Define the custom transformer class to select features based on correlation
class corr_selection(BaseEstimator, TransformerMixin):
    
    # Class constructor
    def __init__(self, threshold=0.9, verbose=False):
        """
        Initialize the transformer with the desired threshold for correlation 
        and verbosity option.
        
        Args:
        - threshold (float): Correlation threshold for dropping a variable.
        - verbose (bool): Flag to print details during the fit process.
        """
        self.threshold = threshold
        self.verbose = verbose

    # Fit method
    def fit(self, X, y=None):
        """
        Fit the transformer on the dataset. This method identifies the columns
        which are highly correlated based on the threshold.
        
        Args:
        - X (DataFrame or ndarray): Input data.
        - y (array-like, optional): Target variable.
        
        Returns:
        - self: The instance of the transformer.
        """
        X = pd.DataFrame(X)
        correlations = X.corr().abs()
        upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype('bool'))
        self.indices_variables_to_eliminate = [i for i, column in enumerate(upper.columns) if any(upper[column] > self.threshold)]

        if self.verbose:
            print('{} variables have been removed, which are: '.format(len(self.indices_variables_to_eliminate)))
            print(list(X.columns[self.indices_variables_to_eliminate]))

        return self

    # Transform method
    def transform(self, X):
        """
        Transform the dataset by removing the highly correlated columns.
        
        Args:
        - X (DataFrame or ndarray): Input data.
        
        Returns:
        - X_uncorr (DataFrame): The transformed dataset without correlated columns.
        """
        X = pd.DataFrame(X)
        X_uncorr = X.drop(columns=X.columns[self.indices_variables_to_eliminate])
        return X_uncorr

    # Override the set_params method to allow grid search
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # Override the get_params method to allow grid search
    def get_params(self, deep=True):
        return {"threshold": self.threshold}


# In[30]:


# Instantiate the custom transformer 'corr_selection' for feature selection based on correlation values
corr = corr_selection(threshold=0.9, verbose=True)

# Fit the transformer on the training data to identify highly correlated columns
corr.fit(X_train_full)

# Transform both training and test datasets to remove highly correlated features
X_train_uncorr = corr.transform(X_train_full)
X_test_uncorr = corr.transform(X_test_full)

# Store the names of the columns post-transformation
column_names = X_test_uncorr.columns


# Validation set to run tests

# In[31]:


# Keeping a complete version of the training dataset (prior to further splitting) for potential future use
X_train_comp, y_train_comp = X_train_uncorr, y_train

# Splitting the training dataset into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_uncorr,  # features
    y_train,         # target variable
    stratify=y_train,    # ensures distribution of target variable remains similar in both datasets
    test_size=0.2,       # 20% of the data is reserved for validation
    random_state=33      # setting a random state for reproducibility
)


# Detection and treatment of Outliers

# In[32]:


class OutlierDetection_treatment_MeanStd(TransformerMixin):
    """
    Outlier Detection and Treatment based on Mean and Standard Deviation.
    Outliers are identified as values that are 'k' standard deviations away from the mean.
    Detected outliers are then replaced with the median of the respective column.
    """

    def __init__(self, k=2, columns=None):
        """
        Initialize the transformer.
        
        :param k: Multiplier for the standard deviation to define outliers.
        :param columns: Columns to apply the outlier detection on. If None, all columns are considered.
        """
        self.k = k
        self.columns = columns

    def fit(self, X, y=None):
        """
        Compute statistics (mean, std, median) needed to detect outliers.
        
        :param X: Input data.
        :param y: Ignored, exists for compatibility.
        :return: self
        """
        X = pd.DataFrame(X)
        if self.columns is None:
            self.columns = X.columns
        self.stats = X[self.columns].describe()
        return self

    def transform(self, X):
        """
        Detect and treat outliers based on previously computed statistics.
        
        :param X: Input data with potential outliers.
        :return: Data with outliers replaced by median.
        """
        X = pd.DataFrame(X)
        Xaux = X.copy()

        for column in self.columns:
            lower_bound = self.stats.loc['mean', column] - self.k * self.stats.loc['std', column]
            upper_bound = self.stats.loc['mean', column] + self.k * self.stats.loc['std', column]
            
            outliers = (X[column] < lower_bound) | (X[column] > upper_bound)
            if outliers.any():
                Xaux.loc[outliers, column] = self.stats.loc['50%', column]
        return Xaux

    def set_params(self, **parameters):
        """
        Set hyper-parameters for the transformer.
        
        :param parameters: Dictionary of parameter names and their values.
        :return: self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """
        Get the hyper-parameters of the transformer.
        
        :param deep: Ignored, exists for compatibility.
        :return: Dictionary of hyper-parameters.
        """
        return {"k": self.k}


# In[33]:


# Instantiate the Outlier Detection and Treatment transformer with a k-value of 5. 
# This means outliers are defined as values that are 5 standard deviations away from the mean.
out_media_std = OutlierDetection_treatment_MeanStd(k=5)

# Fit the transformer on the training data. This computes necessary statistics on X_train.
out_media_std.fit(X_train)

# Transform the training data: Detect and treat outliers based on previously computed statistics.
X_train_std = out_media_std.transform(X_train)

# Similarly, treat outliers in the validation set using the statistics computed from the training set.
X_val_std = out_media_std.transform(X_val)


# In[34]:


# Initialize the Logistic Regression classifier
clf = LogisticRegression()

# Train the classifier on the training data and measure the time taken
start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()

# Predict classes and probabilities for the validation set
pred = clf.predict(X_val)
pred_proba = clf.predict_proba(X_val)[:, 1]

# Calculate various performance metrics
accuracy = accuracy_score(y_val, pred)
precision = precision_score(y_val, pred)
recall = recall_score(y_val, pred)
f1 = f1_score(y_val, pred)
fpr, tpr, _ = roc_curve(y_val, pred_proba)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_val, pred_proba)

# Display performance metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print("Runtime:", (end_time - start_time)/60, " minutes")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curve
plt.figure()
plt.plot(rec, prec, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Display confusion matrix using heatmap for better visualization
cm = confusion_matrix(y_val, pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Upon evaluating the models both with and without the presence of outliers, we have elected to retain the outliers in our dataset. This decision stems from our interest in identifying patients who exhibit atypical values, as these deviations may serve as potential indicators of medical anomalies. The consequent impact of this decision was evident in the results, particularly manifesting as a diminished recall rate.

# 4. Undersampling
# 
# In addressing the imbalance within the dataset, we employ undersampling rather than oversampling, given the abundance of examples present. To ascertain the most optimal approach, the geometric mean will be utilized as it offers enhanced robustness when confronted with class distribution disparities.

# In[35]:


# Define hyperparameters for Logistic Regression
hyperParams = { "logisticregression__C": [0.01, 0.1, 1, 10]}

# Define various undersampling techniques
rus = us.RandomUnderSampler(random_state=42, replacement=False)
tl = TomekLinks()
oss = OneSidedSelection(random_state=42)
cnn = CondensedNearestNeighbour(random_state=42)

# Define the scoring metric
geomean = make_scorer(geometric_mean_score)

# Lists to hold method names and their respective techniques
methodList = ['RUS', 'TL', 'OSS']
techniqueList = [rus, tl, oss]

# Initialize variables to store best results
bestResult = -1
bestMethod = None
bestConfig = None

# List to store validation results
GM_val_list = []

# Record start time for execution time calculation
start_time = time.time()

# Iterate over each undersampling technique
for name, technique in zip(methodList, techniqueList):
    # Create a pipeline with the current undersampling technique followed by Logistic Regression
    pipe = make_pipeline(technique, LogisticRegression(max_iter=300))
    
    # Perform grid search to find the best hyperparameters using cross-validation
    gsPipe = GridSearchCV(pipe, hyperParams, scoring=geomean, cv=5, n_jobs=-1)
    gsPipe.fit(X_train_comp, y_train_comp)
    
    # Append the best score to GM_val_list
    GM_val_list.append(gsPipe.best_score_)
    
    # Display the name of the technique and its performance
    print(f"{name}: {gsPipe.best_score_}")
    
    # Update best result variables if current result is better
    if gsPipe.best_score_ >= bestResult:
        bestResult = gsPipe.best_score_
        bestMethod = gsPipe.best_estimator_
        bestTec = technique
        name = name
        bestConfig = gsPipe.best_params_

# Record end time for execution time calculation
end_time = time.time()

# Display the results
print("\nBest sampling method:", bestMethod)
print("Best hyper-parameters:", bestConfig)
print("Best validation result:", bestResult)
print("Execution time:", (end_time - start_time)/60, " minutes")


# In[36]:


# Plotting the geometric mean validation scores for each method

# Define the X and Y axis data
plt.plot(methodList, GM_val_list)

# Set the label for the X axis
plt.xlabel('Method')

# Set the label for the Y axis
plt.ylabel('Geometric Mean Validation Score')

# Set the title of the plot
plt.title('Comparison of Different Methods')

# Display the plot
plt.show()


# In[37]:


# Obtain the number of examples for each class in the original dataset
examplesPerClass = Counter(y_train_comp)
print(f'Original dataset, examples of class 0: {examplesPerClass[0]}, examples of class 1: {examplesPerClass[1]}')

# Resample the training data using the best sampling technique
X_train_sampled, y_train_sampled = bestTec.fit_resample(X_train_comp, y_train_comp)

# Obtain the number of examples for each class after sampling
sampledExamplesPerClass = Counter(y_train_sampled)
print(f'Dataset sampled with {name}, examples of class 0: {sampledExamplesPerClass[0]}, examples of class 1: {sampledExamplesPerClass[1]}')

# Determine the indices of examples that were removed during the sampling process
indicesOfRemovedExamples = np.setdiff1d(np.arange(X_train_uncorr.shape[0]), bestTec.sample_indices_)
print(f'The number of removed examples is {len(indicesOfRemovedExamples)}')


# In[38]:


# Display the shape of the sampled dataset
print(X_train_sampled.shape, y_train_sampled.shape)


# Basic Model evaluation

# In[39]:


# Initializing a logistic regression classifier
clf_with_noise = LogisticRegression()

# Tracking the start time for model training
start_time = time.time()

# Fitting the model on the sampled data
clf_with_noise.fit(X_train_sampled, y_train_sampled)

# Making predictions on the validation set
pred_with_noise = clf_with_noise.predict(X_val)
pred_proba_with_noise = clf_with_noise.predict_proba(X_val)[:, 1]

# Calculating the time taken to train and predict
end_time = time.time()

# Computing various evaluation metrics
accuracy_with_noise = accuracy_score(y_val, pred_with_noise)
precision_with_noise = precision_score(y_val, pred_with_noise)
recall_with_noise = recall_score(y_val, pred_with_noise)
f1_with_noise = f1_score(y_val, pred_with_noise)
fpr_with_noise, tpr_with_noise, _ = roc_curve(y_val, pred_proba_with_noise)
roc_auc_with_noise = auc(fpr_with_noise, tpr_with_noise)
prec_with_noise, rec_with_noise, _ = precision_recall_curve(y_val, pred_proba_with_noise)

# Displaying the results
print(f'Accuracy: {accuracy_with_noise}')
print(f'Precision: {precision_with_noise}')
print(f'Recall: {recall_with_noise}')
print(f'F1 Score: {f1_with_noise}')
print(f"Execution time: {(end_time - start_time)/60} minutes")

# Plotting the ROC curve
plt.figure()
plt.plot(fpr_with_noise, tpr_with_noise, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_with_noise:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plotting the Precision-Recall curve
plt.figure()
plt.plot(rec_with_noise, prec_with_noise, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Plotting the Cumulative Gain chart
skplt.metrics.plot_cumulative_gain(y_val, clf_with_noise.predict_proba(X_val))
plt.show()

# Plotting the confusion matrix
cm_with_noise = confusion_matrix(y_val, pred_with_noise)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_with_noise, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Removing noise

# In[40]:


# Define noise removal techniques
tl = TomekLinks(sampling_strategy='all')
all_knn = AllKNN(sampling_strategy='all', kind_sel='mode')
enn = EditedNearestNeighbours(sampling_strategy='all', kind_sel='mode')

techniqueList = [tl, enn, all_knn]
methodList = ["TL", "ENN", "ALLKNN"]

bestResultNoise = -1
bestMethodNoise = None
bestConfigNoise = None

# Store validation results for each technique
val_list = []

# Define stratified k-fold cross validation
skf = StratifiedKFold(n_splits=5)

# Define hyperparameters for Grid Search
hyperPar = { "logisticregression__C": [0.01, 0.1, 1, 10]}

# Record the starting time for execution
start_time = time.time()

# Iterate through each noise removal technique
for name, technique in zip(methodList, techniqueList):
    # Create a pipeline with noise removal and logistic regression
    pipe = make_pipeline(technique, LogisticRegression(max_iter=300))
    gsPipe = GridSearchCV(pipe, hyperPar, scoring='recall', cv=skf, n_jobs=-1)
    gsPipe.fit(X_train_sampled, y_train_sampled)
    val_list.append(gsPipe.best_score_)
    
    # Display the validation result for the current technique
    print(f"{name}: {gsPipe.best_score_}")

    # Check if current technique performs better than previous ones and update the best if necessary
    if gsPipe.best_score_ >= bestResultNoise:
        bestResultNoise = gsPipe.best_score_
        bestMethodNoise = gsPipe.best_estimator_
        bestNoiseTec = technique
        nameNoise = name
        bestConfig = gsPipe.best_params_

# Record the ending time for execution
end_time = time.time()

# Calculate and display the time taken for execution
execution_time = (end_time - start_time) / 60
print(f"Best result with {nameNoise} with a recall of {bestResultNoise:.2f}")
print(f"Execution time: {execution_time:.2f} minutes")


# In the analysis, both ALLKNN and ENN demonstrated superior initial results. However, upon further examination, it was observed that these methods substantially reduced the number of instances for class 1, eliminating between 4,000 to 4,900 out of approximately 6,000 examples. Consequently, when subjected to testing, the performance, particularly in terms of recall, was suboptimal. Given these considerations, the decision was made to utilize Tomek Links. This approach focuses on the removal of border elements, potentially enhancing separability and reducing noise, without the extensive reduction of instances observed in the aforementioned methods.

# In[41]:


# Count the number of examples for each class in the sampled dataset
examplesPerClass = Counter(y_train_sampled)

# Display the count of each class in the original sampled dataset
print(f'Original dataset, examples of class 0: {examplesPerClass[0]}, examples of class 1: {examplesPerClass[1]}')

# Resample the dataset to remove noise using the best noise removal technique
X_train_noNoise, y_train_noNoise = bestNoiseTec.fit_resample(X_train_sampled, y_train_sampled)

# Count the number of examples for each class in the no-noise dataset
noNoiseExamplesPerClass = Counter(y_train_noNoise)

# Display the count of each class in the no-noise dataset
print(f'No noise dataset with {nameNoise}, examples of class 0: {noNoiseExamplesPerClass[0]}, examples of class 1: {noNoiseExamplesPerClass[1]}')

# Calculate the indices of examples that were removed during the noise removal process
indicesExamplesEliminated = np.setdiff1d(np.arange(X_train_sampled.shape[0]), bestNoiseTec.sample_indices_)

# Display the number of examples that were removed
print(f'The number of examples eliminated is {len(indicesExamplesEliminated)}')


# Prediction Model Evaluation

# 0. Logistic Regression

# In[42]:


# Initialize the Logistic Regression classifier
clf_without_noise = LogisticRegression()

# Record the start time for training the classifier
start_time = time.time()

# Train the classifier using the no-noise training dataset
clf_without_noise.fit(X_train_noNoise, y_train_noNoise)

# Predict the class labels and probabilities for the validation set
pred_without_noise = clf_without_noise.predict(X_val)
pred_proba_without_noise = clf_without_noise.predict_proba(X_val)[:, 1]

# Record the end time for training the classifier
end_time = time.time()

# Calculate evaluation metrics for the classifier's performance
accuracy_without_noise = accuracy_score(y_val, pred_without_noise)
precision_without_noise = precision_score(y_val, pred_without_noise)
recall_without_noise = recall_score(y_val, pred_without_noise)
f1_without_noise = f1_score(y_val, pred_without_noise)

# Display the evaluation metrics and execution time
print(f'Accuracy: {accuracy_without_noise}')
print(f'Precision: {precision_without_noise}')
print(f'Recall: {recall_without_noise}')
print(f'F1 Score: {f1_without_noise}')
print(f"Execution time: {(end_time - start_time) / 60:.2f} minutes")

# Plot the Receiver Operating Characteristic (ROC) curve
fpr_without_noise, tpr_without_noise, _ = roc_curve(y_val, pred_proba_without_noise)
roc_auc_without_noise = auc(fpr_without_noise, tpr_without_noise)
plt.figure()
plt.plot(fpr_without_noise, tpr_without_noise, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_without_noise:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot the Precision-Recall curve
precision_curve_without_noise, recall_curve_without_noise, _ = precision_recall_curve(y_val, pred_proba_without_noise)
plt.figure()
plt.plot(recall_curve_without_noise, precision_curve_without_noise, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Plotting the confusion matrix
cm_without_noise = confusion_matrix(y_val, pred_without_noise)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_without_noise, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot the Cumulative Gain using scikit-plot
skplt.metrics.plot_cumulative_gain(y_val, clf_without_noise.predict_proba(X_val))
plt.show()


# 1. Random Forest

# In[43]:


# Defining a grid of hyperparameters for the RandomForestClassifier
param_grid = {
    "criterion": ["gini", "entropy"],
    "min_samples_split": [0.02, 0.04, 0.06, 0.08, 0.1],
    "min_samples_leaf": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
    "n_estimators": [100, 200, 500],
    "max_features": ['auto', 'sqrt', 'log2']
}

# Initialize GridSearchCV with the RandomForestClassifier and the specified hyperparameters
gridSearch_pipe = model_selection.GridSearchCV(
    RandomForestClassifier(random_state=123), param_grid, scoring='recall', cv=5, n_jobs=-1
)

# Capture the starting time before fitting the grid search
start_time = time.time()

# Fitting the GridSearchCV on the no-noise training data
gridSearch_pipe.fit(X_train_noNoise, y_train_noNoise)

# Iterating through the results to display the mean and standard deviation of the test scores
results_dict = gridSearch_pipe.cv_results_
for i in range(len(results_dict["params"])):
    print(
        "{:.2f}% +/- {:.2f}%".format(
            results_dict["mean_test_score"][i] * 100.0,
            results_dict["std_test_score"][i] * 100.0,
        ),
        end=" ",
    )
    print(results_dict["params"][i])

# Fetch the best estimator and its parameters from the grid search
best_pipe_rf = gridSearch_pipe.best_estimator_
best_params_rf = gridSearch_pipe.best_params_

# Predict class labels and probabilities using the best model
pred_rf = best_pipe_rf.predict(X_test_uncorr)
pred_proba_rf = best_pipe_rf.predict_proba(X_test_uncorr)[:, 1]

# Calculate various metrics for the model's performance
accuracy_rf = accuracy_score(y_test, pred_rf)
precision_rf = precision_score(y_test, pred_rf)
recall_rf = recall_score(y_test, pred_rf)
f1_rf = f1_score(y_test, pred_rf)

# Display the performance metrics and execution time
print("Best params: ", best_params_rf)
print('Accuracy:', accuracy_rf)
print('Precision:', precision_rf)
print('Recall:', recall_rf)
print('F1 Score:', f1_rf)
end_time = time.time()
print(f"Execution time: {(end_time - start_time) / 60:.2f} minutes")

# Plot the Receiver Operating Characteristic (ROC) curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkorange', label=f'ROC curve (area = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Plot the Precision-Recall curve
precision_curve_rf, recall_curve_rf, _ = precision_recall_curve(y_test, pred_proba_rf)
plt.figure()
plt.plot(recall_curve_rf, precision_curve_rf, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Plotting the confusion matrix
cm_rf = confusion_matrix(y_test, pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot the Cumulative Gain using scikit-plot
skplt.metrics.plot_cumulative_gain(y_test, best_pipe_rf.predict_proba(X_test_uncorr))
plt.show()


# 2. GBM

# In[44]:


# Defining a hyperparameter grid for Gradient Boosting Classifier
param_grid = {
    "learning_rate": [0.1, 0.01, 0.001],
    "n_estimators": [100, 200, 300],
    "subsample": [0.5, 0.8, 1.0],
    "max_depth": [3, 5, 7],
}

# Initializing GridSearchCV with GradientBoostingClassifier and the specified hyperparameters
gridSearch_pipe = model_selection.GridSearchCV(
    GradientBoostingClassifier(random_state=123), param_grid, scoring='recall', cv=5, n_jobs=-1
)

# Capture the starting time before fitting the grid search
start_time = time.time()

# Fit the GridSearchCV on the no-noise training data
gridSearch_pipe.fit(X_train_noNoise, y_train_noNoise)

# Display the results in a readable format
results_dict = gridSearch_pipe.cv_results_
for i in range(len(results_dict["params"])):
    print(
        "{:.2f}% +/- {:.2f}%".format(
            results_dict["mean_test_score"][i] * 100.0,
            results_dict["std_test_score"][i] * 100.0,
        ),
        end=" ",
    )
    print(results_dict["params"][i])

# Fetch the best estimator and its parameters from the grid search
best_pipe_gbm = gridSearch_pipe.best_estimator_
best_params_gbm = gridSearch_pipe.best_params_

# Predict class labels and probabilities using the best model
pred_gbm = best_pipe_gbm.predict(X_test_uncorr)
pred_proba_gbm = best_pipe_gbm.predict_proba(X_test_uncorr)[:, 1]

# Calculate various performance metrics for the model
accuracy_gbm = accuracy_score(y_test, pred_gbm)
precision_gbm = precision_score(y_test, pred_gbm)
recall_gbm = recall_score(y_test, pred_gbm)
f1_gbm = f1_score(y_test, pred_gbm)

# Display performance metrics and execution time
print("Best params: ", best_params_gbm)
print('Accuracy:', accuracy_gbm)
print('Precision:', precision_gbm)
print('Recall:', recall_gbm)
print('F1 Score:', f1_gbm)
end_time = time.time()
print(f"Execution time: {(end_time - start_time) / 60:.2f} minutes")

# Plot the Receiver Operating Characteristic (ROC) curve
fpr_gbm, tpr_gbm, _ = roc_curve(y_test, pred_proba_gbm)
roc_auc_gbm = auc(fpr_gbm, tpr_gbm)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc_gbm:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for GBM')
plt.legend(loc="lower right")
plt.show()

# Plot the Precision-Recall curve
precision_curve_gbm, recall_curve_gbm, _ = precision_recall_curve(y_test, pred_proba_gbm)
plt.figure()
plt.plot(recall_curve_gbm, precision_curve_gbm, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for GBM')
plt.show()

# Plotting the confusion matrix
cm_gbm = confusion_matrix(y_test, pred_gbm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gbm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot the Cumulative Gain for GBM using scikit-plot
skplt.metrics.plot_cumulative_gain(y_test, best_pipe_gbm.predict_proba(X_test_uncorr))
plt.show()


# 3. XGB model

# In[45]:


original_columns = X_train_noNoise.columns.tolist()

new_columns = range(len(original_columns))

X_train_noNoise.columns = new_columns

print(X_train_noNoise.info())


# In[46]:


original_columns = X_test_uncorr.columns.tolist()

new_columns = range(len(original_columns))

X_test_uncorr.columns = new_columns

print(X_test_uncorr.info())


# In[47]:


# Create the hyperparameter grid
param_grid = {
    "learning_rate": [0.1, 0.01, 0.001],
    "n_estimators": [100, 200, 300],
    "subsample": [0.5, 0.8, 1.0],
    "max_depth": [3, 5, 7],
}

# Initialize GridSearchCV with XGBoost classifier
gridSearch_pipe = model_selection.GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=123), 
    param_grid, 
    scoring='recall', 
    cv=5, 
    n_jobs=-1
)

# Record starting time for performance measurement
start_time = time.time()

# Standardize column names to prevent errors
X_train_noNoise.columns = [str(i) for i in range(X_train_noNoise.shape[1])]
X_test_uncorr.columns = [str(i) for i in range(X_test_uncorr.shape[1])]

# Fit the GridSearchCV to the training data
gridSearch_pipe.fit(X_train_noNoise, y_train_noNoise)

# Store the DataFrame with the results
diccionarioResultados = gridSearch_pipe.cv_results_

# Display GridSearch results
for i in range(len(diccionarioResultados["params"])):
    print("{:.2f}% +/- {:.2f}%".format(diccionarioResultados["mean_test_score"][i] * 100.0, diccionarioResultados["std_test_score"][i] * 100.0), end=" ")
    print(diccionarioResultados["params"][i])
    
# Extract best estimator and its parameters
best_pipe_xgb = gridSearch_pipe.best_estimator_
best_params_xgb = gridSearch_pipe.best_params_

# Make predictions using best estimator
pred_xgb = best_pipe_xgb.predict(X_test_uncorr)
pred_proba_xgb = best_pipe_xgb.predict_proba(X_test_uncorr)[:, 1]

# Calculate various performance metrics for the model
accuracy_xgb = accuracy_score(y_test, pred_xgb)
precision_xgb = precision_score(y_test, pred_xgb)
recall_xgb = recall_score(y_test, pred_xgb)
f1_xgb = f1_score(y_test, pred_xgb)

# Display performance metrics and execution time
print("Best params: ", best_params_xgb)
print('Accuracy:', accuracy_score(y_test, pred_xgb))
print('Precision:', precision_score(y_test, pred_xgb))
print('Recall:', recall_score(y_test, pred_xgb))
print('F1 Score:', f1_score(y_test, pred_xgb))
end_time = time.time()
print("Execution time:", (time.time() - start_time) / 60, " minutes")

# Plot ROC Curve
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, pred_proba_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.figure()
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', label='ROC curve (area = %0.2f)' % auc(fpr_xgb, tpr_xgb))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XGB')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall Curve
precision_curve_xgb, recall_curve_xgb, _ = precision_recall_curve(y_test, pred_proba_xgb)
plt.figure()
plt.plot(recall_curve_xgb, precision_curve_xgb, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for XGB')
plt.show()

# Plotting the confusion matrix
cm_xgb = confusion_matrix(y_test, pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot Cumulative Gain Chart
skplt.metrics.plot_cumulative_gain(y_test, best_pipe_xgb.predict_proba(X_test_uncorr))
plt.title("Cumulative Gains Chart for XGB")
plt.show()


# Models with Feature Selection Evaluation

# 4. LR - APACHE model

# In[48]:


# List of features of interest
features_apache = [
    'age', 'temp_apache', 'map_apache', 'heart_rate_apache', 'resprate_apache', 
    'd1_spo2_max', 'd1_spo2_min', 'd1_potassium_max', 'd1_potassium_min', 
    'arf_apache', 'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_verbal_apache', 
    'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 
    'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis'
]

# Extract features and target from the DataFrame
X = df[features_apache]
y = df['hospital_death']

# Split the dataset into training and validation sets
X_train_noNoise, X_val, y_train_noNoise, y_val = model_selection.train_test_split(X, y, test_size=0.3, random_state=123)

# Create an imputer and a logistic regression classifier
imputer = SimpleImputer(strategy="median")
clf = LogisticRegression(max_iter=10000)

# Define the pipeline consisting of imputation step followed by the classifier
pipeline = Pipeline(steps=[('i', imputer), ('m', clf)])

# Record the start time for model training
start_time = time.time()

# Train the pipeline on the training data
pipeline.fit(X_train_noNoise, y_train_noNoise)

# Make predictions on the validation set
pred_apache = pipeline.predict(X_val)
pred_proba_apache = pipeline.predict_proba(X_val)[:, 1]

# Record the end time for model training
end_time = time.time()

# Compute performance metrics
accuracy_apache = accuracy_score(y_val, pred_apache)
precision_apache = precision_score(y_val, pred_apache)
recall_apache = recall_score(y_val, pred_apache)
f1_apache = f1_score(y_val, pred_apache)

# Display the performance metrics
print('Accuracy for Apache:', accuracy_apache)
print('Precision for Apache:', precision_apache)
print('Recall for Apache:', recall_apache)
print('F1 Score for Apache:', f1_apache)
print("Execution time:", (end_time - start_time) / 60, " minutes")

# Plot ROC curve for the Apache model
fpr_apache, tpr_apache, _ = roc_curve(y_val, pred_proba_apache)
roc_auc_apache = auc(fpr_apache, tpr_apache)

plt.figure()
plt.plot(fpr_apache, tpr_apache, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc_apache)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Apache')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curve for the Apache model
precision_curve_apache, recall_curve_apache, _ = precision_recall_curve(y_val, pred_proba_apache)
plt.figure()
plt.plot(recall_curve_apache, precision_curve_apache, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Apache')
plt.show()

# Plotting the confusion matrix
cm_apache = confusion_matrix(y_val, pred_apache)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_apache, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot Cumulative Gains chart for the Apache model
skplt.metrics.plot_cumulative_gain(y_val, pipeline.predict_proba(X_val))
plt.title("Cumulative Gains Chart for Apache")
plt.show()

print("Execution time for Apache:", (end_time - start_time) / 60, " minutes")


# In the subsequent analysis, we will utilize the tree explainer to ascertain the variables that possess significant relevance in the classification of a sample.

# In[49]:


# Print the number of columns in 'X_train_noNoise' and the length of 'column_names'
print(f"Number of columns in X_train_noNoise: {len(X_train_noNoise.columns)}")
print(f"Length of column_names: {len(column_names)}")


# In[50]:


# Preserve the original column names from 'X_train_noNoise' for later restoration
original_columns = X_train_noNoise.columns.tolist()

# Validate if the provided 'column_names' matches the length of columns in 'X_train_noNoise'
if len(column_names) == len(original_columns):
    X_train_noNoise.columns = column_names
else:
    print(f"Warning: Length mismatch between 'column_names' ({len(column_names)}) and 'X_train_noNoise' columns ({len(original_columns)}). Retaining original columns for SHAP analysis.")

# Compute SHAP values using the TreeExplainer with specific configurations
try:
    explainer = shap.TreeExplainer(best_pipe_rf, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_train_noNoise, check_additivity=False)
except Exception as e:
    print("Error during SHAP value computation:", str(e))
    shap_values = None

# Visualize the SHAP values using a summary plot, if they were successfully computed
if shap_values:
    shap.summary_plot(shap_values[1], X_train_noNoise)

# Revert the columns of 'X_train_noNoise' back to their original names
X_train_noNoise.columns = original_columns


# 5. XGB - Modified model

# In[51]:


# Define the features for the model
features_nm = [
    'd1_resprate_max', 'd1_glucose_max', 'heart_rate_apache', 
    'd1_diasbp_noninvasive_max', 'ventilated_apache', 'apache_2_diagnosis', 
    'd1_mbp_noninvasive_max', 'weight', 'gcs_verbal_apache', 'map_apache', 'd1_sysbp_max', 'arf_apache'
]

# Extract the features and target variable from the dataframe
X = df[features_nm]
y = df['hospital_death']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=123)

# Define hyperparameter grid for Grid Search
param_grid = {
    "learning_rate": [0.1, 0.01, 0.001],
    "n_estimators": [100, 200, 300],
    "subsample": [0.5, 0.8, 1.0],
    "max_depth": [3, 5, 7],
}

# Setup the GridSearch for XGBClassifier
gridSearch_pipe = model_selection.GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=123), 
    param_grid, 
    scoring='recall', 
    cv=5, 
    n_jobs=-1
)

# Train the models using Grid Search
start_time = time.time()
gridSearch_pipe.fit(X_train, y_train)

# Store the DataFrame with the results
diccionarioResultados = gridSearch_pipe.cv_results_

for i in range(len(diccionarioResultados["params"])):
    print(
        "{:.2f}% +/- {:.2f}%".format(
            diccionarioResultados["mean_test_score"][i] * 100.0,
           diccionarioResultados["std_test_score"][i] * 100.0,
        ),
        end=" ",
    )
    print(diccionarioResultados["params"][i])

# Extract the best model and its parameters
best_pipe_xgb = gridSearch_pipe.best_estimator_
best_params_xgb = gridSearch_pipe.best_params_
pred_xgb = best_pipe_xgb.predict(X_test)

# Calculate performance metrics
accuracy_xgb_nm = accuracy_score(y_test, pred_xgb)
precision_xgb_nm = precision_score(y_test, pred_xgb)
recall_xgb_nm = recall_score(y_test, pred_xgb)
f1_xgb_nm = f1_score(y_test, pred_xgb)

print("Best parameters: ", best_params_xgb)
print('Accuracy:', accuracy_xgb_nm)
print('Precision:', precision_xgb_nm)
print('Recall:', recall_xgb_nm)
print('F1 Score:', f1_xgb_nm)
print("Execution time:", (time.time() - start_time) / 60, " minutes")

# Plot the ROC Curve
probs_xgb_nm = best_pipe_xgb.predict_proba(X_test)[:,1]
fpr_xgb_nm, tpr_xgb_nm, _ = roc_curve(y_test, probs_xgb_nm)
roc_auc_xgb_nm = auc(fpr_xgb_nm, tpr_xgb_nm)

plt.figure()
plt.plot(fpr_xgb_nm, tpr_xgb_nm, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_xgb_nm)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Plotting the confusion matrix
cm_xgb_nm = confusion_matrix(y_test, pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb_nm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot the Precision-Recall Curve
precision_xgb_nm, recall_xgb_nm, _ = precision_recall_curve(y_test, probs_xgb_nm)
average_precision_xgb_nm = average_precision_score(y_test, probs_xgb_nm)
plt.step(recall_xgb_nm, precision_xgb_nm, where='post')
plt.fill_between(recall_xgb_nm, precision_xgb_nm, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve: AP={0:0.2f}'.format(average_precision_xgb_nm))
plt.show()


# Performance Comparison - Basic Line Models

# In[60]:


# Metrics values for LR model
accuracy_lr = 0.7681405208964264
precision_lr = 0.23000898472596587
recall_lr = 0.7191011235955056
f1_lr = 0.3485364193328795


# Metrics values for RF model
accuracy_rf = 0.7913062440939158
precision_rf = 0.2640134529147982
recall_rf = 0.793597304128054
f1_rf = 0.3962145110410095

# Metrics values for GBM model
accuracy_gbm = 0.7954495892999928
precision_gbm = 0.2724475524475524
recall_gbm = 0.8205560235888796
f1_gbm = 0.4090718185636287


# Metrics values for XGB model
accuracy_xgb = 0.8015555717089482
precision_xgb = 0.27811331607707795
recall_xgb = 0.814658803706824
f1_xgb = 0.4146655231560892


metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values_lr = [accuracy_lr, precision_lr, recall_lr, f1_lr]
values_rf = [accuracy_rf, precision_rf, recall_rf, f1_rf]
values_gbm = [accuracy_gbm, precision_gbm, recall_gbm, f1_gbm]
values_xgb = [accuracy_xgb, precision_xgb, recall_xgb, f1_xgb]

x = np.arange(len(metrics))

plt.figure(figsize=(12, 6))
barWidth = 0.2  # Adjusted bar width for 4 models
r1 = np.arange(len(values_rf))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Plotting bars and adding the metric score labels
for i in range(len(r1)):
    plt.bar(r1[i], values_rf[i], color='blue', width=barWidth, edgecolor='white', label='RF' if i == 0 else "")
    plt.text(r1[i], values_rf[i] + 0.01, f'{values_rf[i]:.2f}', ha = 'center', va = 'center')
    
    plt.bar(r2[i], values_gbm[i], color='green', width=barWidth, edgecolor='white', label='GBM' if i == 0 else "")
    plt.text(r2[i], values_gbm[i] + 0.01, f'{values_gbm[i]:.2f}', ha = 'center', va = 'center')
    
    plt.bar(r3[i], values_xgb[i], color='red', width=barWidth, edgecolor='white', label='XGB' if i == 0 else "")
    plt.text(r3[i], values_xgb[i] + 0.01, f'{values_xgb[i]:.2f}', ha = 'center', va = 'center')

    plt.bar(r4[i], values_lr[i], color='cyan', width=barWidth, edgecolor='white', label='LR' if i == 0 else "")
    plt.text(r4[i], values_lr[i] + 0.01, f'{values_lr[i]:.2f}', ha = 'center', va = 'center')

plt.xlabel('Metrics', fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.xticks([r + 1.5*barWidth for r in range(len(values_rf))], metrics)  # Adjusted x-ticks for 4 models
plt.legend()
plt.show()


# In[53]:


plt.figure(figsize=(10, 7))

# RF
plt.plot(fpr, tpr, color='blue', label=f'RF ROC curve (area = {roc_auc:.2f})')

# GBM
plt.plot(fpr_gbm, tpr_gbm, color='green', label=f'GBM ROC curve (area = {roc_auc_gbm:.2f})')

# XGB
plt.plot(fpr_xgb, tpr_xgb, color='red', label=f'XGB ROC curve (area = {roc_auc_xgb:.2f})')

# LR (without_noise)
plt.plot(fpr_without_noise, tpr_without_noise, color='cyan', lw=2, label=f'LR ROC curve (area = {roc_auc_without_noise:.2f})')

# Diagonal line
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.show()


# In[54]:


plt.figure(figsize=(10, 7))

# RF
plt.plot(recall_curve_rf, precision_curve_rf, color='blue', label='RF')

# GBM
plt.plot(recall_curve_gbm, precision_curve_gbm, color='green', label='GBM')

# XGB
plt.plot(recall_curve_xgb, precision_curve_xgb, color='red', label='XGB')

# LR (without_noise)
plt.plot(recall_curve_without_noise, precision_curve_without_noise, color='cyan', label='LR')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves Comparison')
plt.legend(loc="upper right")
plt.show()


# In[55]:


# Adjust the subplots for 4 matrices
fig, ax = plt.subplots(1, 4, figsize=(24, 5))

# RF
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('True')
ax[0].set_title('RF Confusion Matrix')

# GBM
sns.heatmap(cm_gbm, annot=True, fmt='d', cmap='Blues', ax=ax[1])
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('True')
ax[1].set_title('GBM Confusion Matrix')

# XGB
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=ax[2])
ax[2].set_xlabel('Predicted')
ax[2].set_ylabel('True')
ax[2].set_title('XGB Confusion Matrix')

# LR (without_noise)
sns.heatmap(cm_without_noise, annot=True, fmt='d', cmap='Blues', ax=ax[3])
ax[3].set_xlabel('Predicted')
ax[3].set_ylabel('True')
ax[3].set_title('LR Confusion Matrix')

plt.tight_layout()
plt.show()


# Performance Comparison - Models with Feature Selection

# In[56]:


# Combined ROC Curve
plt.figure()
plt.plot(fpr_apache, tpr_apache, color='darkblue', label='APACHE (area = %0.2f)' % roc_auc_apache)
plt.plot(fpr_xgb_nm, tpr_xgb_nm, color='darkorange', label='XGB Modified (area = %0.2f)' % roc_auc_xgb_nm)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.show()


# In[57]:


# Combined Precision-Recall Curve
plt.figure()
plt.step(recall_curve_apache, precision_curve_apache, where='post', color='darkblue', label='APACHE (AP = %0.2f)' % average_precision_score(y_val, pred_proba_apache))
plt.fill_between(recall_curve_apache, precision_curve_apache, step='post', alpha=0.2, color='darkblue')
plt.step(recall_xgb_nm, precision_xgb_nm, where='post', color='darkorange', label='XGB Modified (AP = %0.2f)' % average_precision_xgb_nm)
plt.fill_between(recall_xgb_nm, precision_xgb_nm, step='post', alpha=0.2, color='darkorange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves Comparison')
plt.legend(loc='upper right')
plt.show()


# In[58]:


# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(cm_apache, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix: APACHE')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

sns.heatmap(cm_xgb_nm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Confusion Matrix: XGB Modified')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()
plt.show()


# In[59]:


def plot_metrics_comparison(models_metrics, metrics_names, model_names):
    n_metrics = len(metrics_names)
    n_models = len(models_metrics)
    bar_width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create an array of positions for each bar
    index = np.arange(n_metrics)
    
    for i in range(n_models):
        values = [models_metrics[i][metric] for metric in metrics_names]
        bars = ax.barh(index + i * bar_width, values, bar_width, label=model_names[i], align='center')
        
        # Add labels to the bars
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.3f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3,0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center')
    
    ax.set_xlabel('Scores')
    ax.set_title('Metrics Comparison by Model')
    ax.set_yticks(index + bar_width / 2)
    ax.set_yticklabels(metrics_names)
    ax.legend()
    plt.tight_layout()
    
    plt.show()

# Metrics for LR - APACHE model
accuracy_apache = 0.9164061932107291
precision_apache = 0.598939929328622
recall_apache = 0.14054726368159204
f1_apache = 0.22766957689724646

# Metrics for XGB - Modified model
accuracy_xgb_nm = 0.9177509631460348
precision_xgb_nm = 0.5881656804733728
recall_xgb_nm = 0.2060530679933665
f1_xgb_nm = 0.30518882407123116

models_metrics = [
    {
        'Accuracy': accuracy_apache, 
        'Precision': precision_apache, 
        'Recall': recall_apache, 
        'F1': f1_apache
    },
    {
        'Accuracy': accuracy_xgb_nm, 
        'Precision': precision_xgb_nm, 
        'Recall': recall_xgb_nm, 
        'F1': f1_xgb_nm
    }
]

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
model_names = ["LR - APACHE model", "XGB - Modified model"]

# Call the function
plot_metrics_comparison(models_metrics, metrics_names, model_names)

