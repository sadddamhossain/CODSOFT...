#!/usr/bin/env python
# coding: utf-8

# #  Convert movies.dat to movies.csv file
# 

# In[1]:


import pandas as pd

# Replace 'input_file_path' with the path to your input file
input_file_path = "movies.dat"

# Replace 'output_file_path' with the desired path for the output .csv file
output_file_path = "movies.csv"

# Read the .dat file into a DataFrame using a specific encoding
df = pd.read_csv(input_file_path, delimiter='\t', encoding='latin1')

# Write the DataFrame to a .csv file
df.to_csv(output_file_path, index=False)


# In[2]:


df


# # Convert  ratings.dat to ratings.csv File

# In[3]:


# Replace 'input_file_path' with the path to your input file
input_file_path = "ratings.dat"

# Replace 'output_file_path' with the desired path for the output .csv file
output_file_path = "ratings.csv"

# Read the .dat file into a DataFrame using a specific encoding
df = pd.read_csv(input_file_path, delimiter='\t', encoding='latin1')

# Write the DataFrame to a .csv file
df.to_csv(output_file_path, index=False)


# In[4]:


df


# # Convert users.dat to users.csv File

# In[ ]:


# Replace 'input_file_path' with the path to your input file
input_file_path = "users.dat"

# Replace 'output_file_path' with the desired path for the output .csv file
output_file_path = "users.csv"

# Read the .dat file into a DataFrame using a specific encoding
df = pd.read_csv(input_file_path, delimiter='\t', encoding='latin1')

# Write the DataFrame to a .csv file
df.to_csv(output_file_path, index=False)


# In[6]:


df


# # Data acquisition of the movies dataset

# In[7]:


df_movie=pd.read_csv('movies.csv', sep = '::', engine='python')
df_movie.columns =['MovieIDs','MovieName','Category']


# In[8]:


df_movie.head()


# # Analysing this Dataset

# In[9]:


df_movie.isnull()


# In[10]:


df_movie.isnull().sum()


# In[11]:


df_movie.info()


# # Data acquisition of the rating dataset

# In[12]:


df_rating = pd.read_csv("ratings.csv",sep='::', engine='python')
df_rating.columns =['ID','MovieID','Ratings','TimeStamp']


# In[13]:


df_rating


# # Analysing the Dataset

# In[14]:


df_rating.isnull()


# In[15]:


df_rating.isnull().sum()


# In[16]:


df_rating.info()


# # Data acquisition of the users dataset

# In[17]:


df_user = pd.read_csv("users.csv",sep='::',engine='python')
df_user.columns =['UserID','Gender','Age','Occupation','Zip-code']


# In[18]:


df_user


# # Analysing the Dataset

# In[19]:


df_user.isnull()


# In[20]:


df_user.isnull().sum()


# In[21]:


df_user.info()


# # Merge all the datasets and create One Dataset

# In[22]:


df = pd.concat([df_movie, df_rating,df_user], axis=1)


# In[24]:


df


# # Analysing the Dataset

# In[25]:


df.isnull()


# In[26]:


df.isnull().sum()


# In[27]:


df.head()


# In[30]:


print(df.dtypes)


# In[31]:


print(df.columns)


# In[32]:


print(df.describe())


# In[33]:


df.count()


# In[34]:


# Cleaning the dataset for build my model
df = df.dropna()


# In[35]:


df


# # Data Visualization

# In[50]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[49]:


#Visualize user age distribution
df['Age'].value_counts().plot(kind='barh',alpha=0.7,figsize=(8,5))
plt.show()


# In[42]:


df.hist(bins=10, figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[43]:


df.Age.plot.hist(bins=25)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('Age')


# In[47]:


#Visualize overall rating by users
df['Ratings'].value_counts().plot(kind='bar',alpha=0.7,figsize=(5,4))
plt.show()


# In[52]:


#Find and visualize the top 25 movies by viewership rating
top_25 = df[25:]
top_25['Ratings'].value_counts().plot(kind='barh',alpha=0.6,figsize=(7,4))
plt.show()


# In[53]:


#Create a histogram for movie
df.Age.plot.hist(bins=25)
plt.title("Movie & Rating")
plt.ylabel('MovieID')
plt.xlabel('Ratings')


# In[54]:


# For example, to create a scatter plot of 'MovieName' vs 'Category'
plt.scatter(df['MovieName'], df['Category'])
plt.xlabel('MovieName')
plt.ylabel('Category')
plt.title('MovieName vs Category')
plt.show()


# # Logestic Regression

# In[57]:


# Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# In[59]:


# Assuming you have a DataFrame called 'df' containing the dataset with the mentioned columns
# Make sure to preprocess the data and handle any missing values before proceeding

# Select the relevant columns for logistic regression
selected_columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code', 'Ratings']

# Create a new DataFrame with only the selected columns
df_selected = df[selected_columns].copy()

# Remove rows with non-numeric 'Zip-code' values
df_selected = df_selected[pd.to_numeric(df_selected['Zip-code'], errors='coerce').notna()]

# Perform label encoding for categorical variables
label_encoder = LabelEncoder()
df_selected['Gender'] = label_encoder.fit_transform(df_selected['Gender'])

# Split the dataset into training and testing sets
X = df_selected.drop('Ratings', axis=1)
y = df_selected['Ratings']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = logistic_model.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[61]:


df = logistic_model.predict(X_test)
print(df)


# In[62]:


df = logistic_model.predict(X_train)
print(df)


# In[63]:


print(X_train.shape, y_train.shape, X_test.shape)


# In[ ]:




