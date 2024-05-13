#!/usr/bin/env python
# coding: utf-8

# In[1]:


# task 1.1: show uncleaned dataset  
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
pd.set_option('display.max_rows', None)
ds = pd.read_csv('data.csv')


# In[2]:


# Task 1.2: cleaning and visualize the cleaned dataset
ds = pd.read_csv('data.csv')

ds = ds[:140]
ds = ds.drop([0,1,109])

for i in ds.index:
    if ds.reign[i]>1:
        ds = ds.drop(i)
        
ds


# In[4]:


#task 1.3 adn 1.4 EDA
# Select only numeric columns
numeric_ds = ds.select_dtypes(include=['number'])

# heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_ds.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Distribution of Days as Champion
plt.figure(figsize=(10, 6))
plt.hist(ds['days'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Days as Champion')
plt.xlabel('Days as Champion')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Cause of Death
plt.figure(figsize=(10, 6))
sns.countplot(y='cause_of_death', data=ds, order=ds['cause_of_death'].value_counts().index)
plt.title('Cause of Death')
plt.xlabel('Count')
plt.ylabel('Cause of Death')
plt.show()


# In[5]:


#task 2
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, errors='raise')
    except Exception as e:
        print(f"Error converting date: {date_str}. Error: {str(e)}")
        return pd.NaT  # Return NaT for invalid dates
# Convert 'date' and 'date_of_birth' columns to datetime using the custom function
try:
    ds['date'] = ds['date'].apply(parse_date)
    ds['date_of_birth'] = ds['date_of_birth'].apply(parse_date)
except ValueError:
    print("Error: Unable to convert some dates. These rows will be skipped.")

# doing feature engineer by adding new features
if 'date' in ds.columns:
    # Drop rows with NaT values
    ds.dropna(subset=['date'], inplace=True)

    # 1. Championship Density
    total_days = (ds['date'].max() - ds['date'].min()).days
    ds['championship_density'] = ds['days'] / total_days

    # 2. Age at the Time of Winning the Title
    ds['age_at_winning'] = (ds['date'] - ds['date_of_birth']).dt.days / 365

    # 3. Title Reigns per Year
    ds['title_reigns_per_year'] = ds['reign'] / ((ds['date'].max() - ds['date'].min()).days / 365)

    # 4. Location Frequency
    location_frequency = ds['location'].value_counts(normalize=True)
    ds['location_frequency'] = ds['location'].map(location_frequency)

    # 5. Reign Duration
    ds['reign_duration'] = ds['days']

    # 6. Experience Level
    ds['experience_level'] = (ds['date'] - ds['date'].min()).dt.days
    #7 Winner
    ds['winner'] = (ds['days'] > 365).astype(int)
ds


# In[6]:


# to detect the best model to use with this dataset
target_variable = 'winner'

if ds[target_variable].nunique() <= 2:
    print("Classification Task")
else:
    print("Regression Task or Clustering Task")
    
#Task 3 classification model
ds['successful_reign'] = (ds['days'] > 365).astype(int)

X = ds[['reign', 'days']]  # For simplicity, using only 'reign' and 'days' as features
y = ds['successful_reign']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Print results
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")


# In[7]:


#Task 4
# by using classification report it will produce the f1 and recall
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    print("\n")


# In[8]:


# Task 4 continue
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")
# Dictionary to store accuracies
accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy

# Create a bar plot for accuracies
plt.figure(figsize=(10, 6))
plt.barh(list(accuracies.keys()), list(accuracies.values()), color='skyblue')
plt.xlabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.xlim(0, 1)
plt.gca().invert_yaxis() 
plt.show()


# In[12]:


#Deployment Task
# install streamlit


# In[13]:


# add streamlit lib
import streamlit as st

st.title('WWE Championship Winner Prediction')

# Once user clicks on the "Predict" button
if st.button('Predict Winner'):
    # Prepare input features as a DataFrame using the last row of the dataset as an example
    features = ds[['reign', 'days']].iloc[-1:].values
    
    # Make prediction
    prediction_idx = model.predict(features)[0]  
    predicted_winner_name = ds.iloc[prediction_idx]['name']
    
    # Display predicted winner's name
    st.write(f'Predicted Winner: {predicted_winner_name}')
# In[ ]:




