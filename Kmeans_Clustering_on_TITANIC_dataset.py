
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn import cluster, preprocessing


# In[2]:

df = pd.read_excel("titanic.xls")
df.head()


# # A little insight on the dataset
# 
# ##TITANIC DATASET
# 
#     Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
#     survival Survival (0 = No; 1 = Yes)
#     name Name
#     sex Sex
#     age Age
#     sibsp Number of Siblings/Spouses Aboard
#     parch Number of Parents/Children Aboard
#     ticket Ticket Number
#     fare Passenger Fare (British pound)
#     cabin Cabin
#     embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
#     boat Lifeboat
#     body Body Identification Number
#     home.dest Home/Destination
# 

# In[3]:

#To find the probability of survival rate; Name and Body(Body Identification Number aren't required)

df.drop(['name','body'], 1, inplace=True)
df.head()


# In[5]:

#replacing the invalid values with a 0
df.fillna(0, inplace=True)
df.head()


# In[6]:

#now to convert the non numerical data into numerical for convinience
columns = df.columns.values
for i in columns:
    text_value_int = {}
    def text_to_val(val):
        return text_value_int[val]
    if df[i].dtype != np.int64 and df[i].dtype != np.float64:
        all_text = df[i].values.tolist()
        unique_elements = set(all_text)
        
        x = 0
        for unique in unique_elements:
            if unique not in text_value_int:
                text_value_int[unique] = x
                x+=1
        
        df[i] = list(map(text_to_val, df[i]))


# In[7]:

df.head()


# In[ ]:




# In[8]:

#so as now the dataframe has been converted into all numericals now we can use any algorithm to predict
#here as there are no labels we are gonna use the K Means Clustering

X = np.array(df.drop('survived', 1))
y = np.array(df['survived'])


# In[9]:

#Now initiating the classifier

clf = cluster.KMeans(n_clusters=2)
clf.fit(X)


# In[10]:

found = 0
for i in range(len(X)):
    new_prediction = np.array(X[i].astype(float))
    new_prediction = new_prediction.reshape(-1, len(new_prediction))
    prediction = clf.predict(new_prediction)
    if prediction[0] == y[i]:
        found += 1

accuracy = (found/len(X))*100
accuracy


# So we get around 50% accuracy
# 
# so let's see after preprocessing X
# 

# In[14]:

X = preprocessing.scale(X)
clf.fit(X)
found = 0
for i in range(len(X)):
    new_prediction = np.array(X[i].astype(float))
    new_prediction = new_prediction.reshape(-1, len(new_prediction))
    prediction = clf.predict(new_prediction)
    if prediction[0] == y[i]:
        found += 1

accuracy = (found/len(X))*100
print(accuracy)

#So we get around 70% accuracy now!!
# In[ ]:



