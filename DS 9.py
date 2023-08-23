#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score , classification_report, ConfusionMatrixDisplay,precision_score,recall_score, f1_score,roc_auc_score,roc_curve


# In[2]:


df=pd.read_csv('bmi.csv')
df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Index'] = label_encoder.fit_transform(df['Index'])


# In[7]:


#Classification model 
#Train and Test the model


# In[8]:


from sklearn.model_selection import train_test_split
X=df.drop('Index',axis=1)
y=df['Index']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[9]:


X_train.shape


# In[10]:


X_test.shape


# In[11]:


y_train.shape


# In[12]:


y_test.shape


# In[13]:


from sklearn.preprocessing import StandardScaler
#Logistic Regression
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Initialize and train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
# Make predictions
predictions = model.predict(X_test_scaled)
# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


# In[ ]:





# In[14]:


#Kneighbours
# Initialize and train the k-NN model
k = 3  # You can adjust the number of neighbors (k) as needed
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# Evaluate the model
accuracy1 = accuracy_score(y_test, predictions)
print("Accuracy1:", accuracy1)


# In[ ]:





# In[16]:


#Decision Tree Classifier
# Initialize and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy2 = accuracy_score(y_test, predictions)
print("Accuracy2:", accuracy2)


# In[17]:


#Random Forest
# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy3 = accuracy_score(y_test, predictions)
print("Accuracy3:", accuracy3)


# In[18]:


#SVM
from sklearn.svm import SVC
model = SVC(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# Evaluate the model
accuracy4 = accuracy_score(y_test, predictions)
print("Accuracy4:", accuracy4)


# In[19]:


#Naive Bayes
# Initialize and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy5 = accuracy_score(y_test, predictions)
print("Accuracy5:", accuracy5)


# In[20]:


mse = metrics.mean_squared_error(y_test, predictions)
print('Mean Squared Error : '+ str(mse))


# In[21]:


#confusion matrix
matrix = pd.DataFrame(metrics.confusion_matrix(y_test, predictions))
matrix


# In[26]:


# User input for prediction
user_height = float(input("Enter your height (cm): "))
user_weight = float(input("Enter your weight (kg): "))
user_gender = input("Enter your gender (Male/Female): ")

# Ensure user gender is capitalized consistently
user_gender = user_gender.capitalize()

# Add the user gender to the label encoder classes and transform it
if user_gender not in label_encoder.classes_:
    label_encoder.classes_ = np.append(label_encoder.classes_, user_gender)

user_gender_encoded = label_encoder.transform([user_gender])[0]

# Create a feature array for prediction
user_features = np.array([[user_height, user_weight, user_gender_encoded]])

# Scale the user features
user_features_scaled = scaler.transform(user_features)

# Predict the user's BMI using the trained model
predicted_index = model.predict(user_features_scaled)

# Convert the predicted index back to BMI value
predicted_bmi = label_encoder.inverse_transform([predicted_index])[0]

print(f"Predicted BMI: {predicted_bmi}")


# In[ ]:





# In[ ]:




