
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

import numpy as np


# In[3]:

import sklearn


# In[4]:

from sklearn.preprocessing import LabelEncoder


# In[5]:

#read file




data=pd.read_csv("/Users/Nandu/Documents/OneDrive/Documents/new era/PinNPark/ML/train_data.csv")




# In[6]:

data.head()


# In[7]:

#Seperate labels and features
y=LabelEncoder().fit_transform(data["Loan_Status"])
x=data.drop('Loan_Status',1)

y=LabelEncoder().fit_transform(data["Loan_Status"])


# In[8]:

#y


# In[9]:

x.head()


# In[10]:

y.shape


# In[11]:

def conv_to_num(data):
    #convert to categories
    x=data
    for columns in data.columns:
        x[columns]=data[columns].astype('category')
    
        
    #convert to numerical
    for columns in data.columns:
        x[columns]=data[columns].cat.codes #missing values??
    return x


    


# In[12]:

x.shape


# In[13]:

x=conv_to_num(x)
x.head()


# In[14]:

x.shape


# In[15]:

print(x.dtypes)


# In[16]:

x.head()


# In[17]:

from sklearn import tree



# In[18]:

clf=tree.DecisionTreeClassifier()


# In[19]:

clf=clf.fit(x,y)


# In[20]:

x.shape


# In[21]:

data_test=pd.read_csv("/Users/Nandu/Documents/OneDrive/Documents/new era/PinNPark/ML/test_data.csv")


# In[22]:

data_test.head()


# In[23]:

data_test.shape


# In[24]:

#data_num=data_test
#data_test


# In[25]:

temp=data_test.copy()
#data_test
x_test=conv_to_num(temp)#NOt a good move


# In[26]:

#temp


# In[27]:

x_test.shape


# In[28]:

x_test.head()


# In[29]:

y_test=clf.predict(x_test)


# In[30]:

y_test.dtype


# In[31]:

y_result=[]
i=0;

for i in range(len(y_test)):
    
    if y_test[i]==0:
        y_result.append("N")
       # print"test"
            
    elif y_test[i]==1:
        y_result.append("Y")
       # print(y_test[i])
        #print"test"


# In[32]:

y_result=pd.DataFrame(y_result)


# In[33]:

y_result.columns=["Loan_Status"]
y_result.head()


# In[34]:

y_result.shape


# In[35]:

data_test.head()


# In[36]:

data_result=pd.concat([data_test,y_result],1)


# In[37]:

data_result.head()


# In[38]:

data_result.to_csv("/Users/Nandu/Documents/OneDrive/Documents/new era/PinNPark/ML/train_result.csv")


# In[39]:

from sklearn.model_selection import train_test_split


# In[40]:

xv_train,xv_test,yv_train,yv_test=train_test_split(x,y,test_size=.5)


# In[41]:

xv_train.shape


# In[42]:

clf=clf.fit(xv_train,yv_train)


# In[43]:

from sklearn.metrics import accuracy_score


# In[44]:

print accuracy_score(clf.predict(xv_test),yv_test)


# In[45]:

from sklearn.naive_bayes import GaussianNB


# In[46]:

model_nb = GaussianNB()


# In[47]:


model_nb=model_nb.fit(x,y)


# In[48]:

print accuracy_score(model_nb.predict(xv_test),yv_test)


# In[49]:

from sklearn.linear_model import LogisticRegression


# In[50]:

model_lg=LogisticRegression()


# In[51]:

model_lg=model_lg.fit(x,y)
print accuracy_score(model_lg.predict(xv_test),yv_test)


# In[ ]:



