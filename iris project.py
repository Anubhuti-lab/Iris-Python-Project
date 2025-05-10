#!/usr/bin/env python
# coding: utf-8

# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df=pd.read_csv("iris data.csv")


# In[9]:


df.head()


# In[11]:


df.tail()


# In[13]:


# to display stats about dataset 
df.describe()


# In[15]:


df.info() # to display basic info about data


# In[17]:


# to display no of samples of each class
df["species"].value_counts()


# In[19]:


# to check for null values
df.isnull().sum()


# In[23]:


# for plotting histogram 
df["sepal_length"].hist()


# In[25]:


df[df["species"]=="setosa"]["sepal_length"].hist() # for a particular species 


# In[73]:


df["petal_length"].hist()





# In[91]:


df=sns.load_dataset("iris")


# In[93]:


# for scatter plot 
colors=["red","orange","blue"]
species=df['species'].unique()


# In[95]:


for i in range(3):
    x=df[df["species"] == species[i]]
    plt.scatter(x["sepal_length"],x["sepal_width"], color=colors[i], label=species[i])
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.legend()
# for colouring and other visualization effects
plt.title("sepal_length vs sepal_width by species")
plt.show()
     


# In[79]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# Load sample iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

species = ['setosa', 'versicolor', 'virginica']
colours = ['red', 'green', 'blue']

for i in range(3):
    x = df[df["species"] == species[i]]
    plt.scatter(x["sepal length (cm)"], x["sepal width (cm)"], color=colours[i], label=species[i])

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.title("Sepal Length vs Width by Species")
plt.show()


# In[97]:


for i in range(3):
    x=df[df["species"] == species[i]]
    plt.scatter(x["petal_length"],x["petal_width"], color=colors[i], label=species[i])
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.legend()
# for colouring and other visualization effects
plt.title("petal_length vs petal_width by species")
plt.show()
     


# In[127]:


corr = df.select_dtypes(include="number").corr()


# In[ ]:





# In[131]:


corr = df.corr


# In[137]:


corr=df.corr
fig,ax=plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True,cmap="coolwarm",ax=ax)
plt.show()


# In[143]:


iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)


# In[145]:


correlation_matrix = iris_df.corr()


# In[147]:


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Iris Dataset Features')















































plt.show()


# In[123]:


print(corr)


# In[149]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[151]:


df["species"]=le.fit_transform(df["species"])
df.head()


# In[153]:


from sklearn.model_selection import train_test_split
# train -70%,test-30% 
x=df.drop(columns=["species"])
y=df["species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[155]:


# logistic regression 
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[157]:


# model training
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)


# In[165]:


# to print matric to get perfermance
print("accuracy",model.score(x_train,y_train)) 


# In[175]:


# knn knearest neighbour 
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()


# In[177]:


model.fit(x_train,y_train)


# In[181]:


print("accuracy:",model.score(x_test,y_test)*100)


# In[185]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()


# In[187]:


model.fit(x_train,y_train)


# In[191]:


print("accuracy:",model.score(x_test,y_test)*100)


# In[ ]:




