#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv(r"WineQuality.csv")
data


# In[2]:


import seaborn as sn
sn.pairplot(data)


# In[3]:


data.hist(bins=20, figsize=(10, 10))
plt.show()


# In[4]:


from sklearn.model_selection import train_test_split
X = data.drop(columns='quality')
Y=data['quality']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
print("Coeffiecents",lr.coef_)
print("Intercept=",lr.intercept_)
y_pred=lr.predict(x_test)
y_pred


# In[5]:


from sklearn.metrics import accuracy_score,confusion_matrix,r2_score, mean_absolute_error,mean_squared_error
print("R2 score:",r2_score(y_test,y_pred))
print('MAE:', mean_absolute_error(y_test,y_pred))
print('MSE:', mean_squared_error(y_test,y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test,y_pred)))
output=[7.4,0.700,0.00,1.9,0.076,11.0,34.0,0.99780,3.51,0.56,9.4]
print("Prediction:",lr.predict([output]))
pd.DataFrame(np.c_[y_test ,y_pred] , columns =['Actual' , 'Predicted(Linear Regression)'])



# In[ ]:




