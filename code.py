# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
import zipfile
with zipfile.ZipFile('learn_ml_2021_grand_ai_challenge-dataset.zip',mode='r') as zip_ref:
    zip_ref.extractall('C:\\Users\Rishabh\Desktop\Rishabh\Learn ML Dockship.io')
"""

# Importing the dataset
dataset = pd.read_csv('new_train.csv')

#Determining X and Y
Nmax = 5
labels= [4 + 7 * i for i in range(0, Nmax)]
predictors=[]
limb=[]
count=0
for i in range (1,37):
    if i in labels :
        continue       
    if(int((i-1)/7)==count):
        count+=1    
        predictors.append(limb)
        #predictors[count]=[]
        limb=[]
    limb.append(i)
        
count=0
X=[]
y=[]
for i in range(1,6):
    X.append(dataset.iloc[:, predictors[i]].values)
    y.append(dataset.iloc[:, labels[i-1]].values)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
sc2 = StandardScaler()
for i in range(5):
    X[i] = sc1.fit_transform(X[i])
    y[i]=y[i].reshape(-1, 1)
    y[i] = sc2.fit_transform(y[i])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train=[]
X_valid=[]
y_train=[]
y_valid=[]
for i in range(5):
    X_train.append(train_test_split(X[i], y[i], test_size = 0.2, random_state = 0)[0])
    X_valid.append(train_test_split(X[i], y[i], test_size = 0.2, random_state = 0)[1])
    y_train.append(train_test_split(X[i], y[i], test_size = 0.2, random_state = 0)[2])
    y_valid.append(train_test_split(X[i], y[i], test_size = 0.2, random_state = 0)[3])



def result(X_train,X_valid,y_train,y_valid,num=1):
    # Fitting classifier to the Training set\
    if num is 1:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif num is 2:
        from xgboost import XGBRFRegressor
        model = XGBRFRegressor()
    
    #computing Training and Validation scores
    train_score=model.score(X_train, y_train)
    valid_score=model.score(X_valid, y_valid)
    
    # Predicting the results
    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)
    
    #Computing Root Mean Square Errors
    from sklearn.metrics import mean_squared_error
    RMS_train= mean_squared_error(y_train, y_pred_train, squared=False)
    RMS_valid= mean_squared_error(y_valid,y_pred_valid, squared=False)
    
    
    print("For STOCK - "+ str(i)+ ":")
    print(" Training R2 = "+ str(train_score))
    print(" Validation R2 = "+ str(valid_score))
    print("Training RMS error = "+ str(RMS_train))
    print("Validation RMS error = "+ str(RMS_valid))
    


for i in range(5):
    X_train1=X_train[i]
    X_valid1= X_valid[i]
    y_train1=y_train[i]
    y_valid1=y_valid[i]
    result(X_train1,X_valid1,y_train1,y_valid1,1)
    result(X_train1,X_valid1,y_train1,y_valid1,2)