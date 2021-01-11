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



def result(X_train,X_valid,y_train,y_valid,X_test,i=0,num=1):
    # Fitting classifier to the Training set\
    if num == 1:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
    #elif num is 2:
        #from xgboost import XGBRFRegressor
        #model = XGBRFRegressor()
    
    #computing Training and Validation scores
    train_score=model.score(X_train, y_train)
    valid_score=model.score(X_valid, y_valid)
    
    # Predicting the results
    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)
    y_pred_test = model.predict(X_test)
    
    #Inverse transform the scalin gto calculate mean square errors
    y_pred_train=sc2.inverse_transform(y_pred_train)
    #Computing Root Mean Square Errors
    from sklearn.metrics import mean_squared_error
    RMS_train= mean_squared_error(y_train, y_pred_train, squared=False)
    RMS_valid= mean_squared_error(y_valid,y_pred_valid, squared=False)
    
    
    print("For STOCK - "+ str(i+1)+ ":")
    print("Training R2          = "+ str(train_score))
    print("Validation R2        = "+ str(valid_score))
    print("Training RMS error   = "+ str(RMS_train))
    print("Validation RMS error = "+ str(RMS_valid))
    
    return y_pred_test
 
# Importing the test dataset
test_dataset = pd.read_csv('new_test.csv')
X_test=[]
predictors_test=[]
limb=[]
count=0
for i in range (1,34):      
    if(int((i-1)/6)==count):
        count+=1    
        predictors_test.append(limb)
        limb=[]
    limb.append(i)

    
for i in range(1,6):
    X_test.append(test_dataset.iloc[:, predictors_test[i]].values)
for i in range(5): 
    X_test[i] = sc1.fit_transform(X_test[i])
    #y_pred_train[i] = model.predict(X_test[i])   

y_pred_test=[]
for i in range(5):
    bm=i
    X_train1=X_train[bm]
    X_valid1= X_valid[bm]
    y_train1=y_train[bm]
    y_valid1=y_valid[bm]
    X_test1=X_test[bm]
    num=3
    # Fitting classifier to the Training set\
    if num == 1:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif num == 2:
        from xgboost import XGBRFRegressor
        model = XGBRFRegressor()
    elif num == 3:
        from sklearn.preprocessing import PolynomialFeatures
        poly_reg = PolynomialFeatures(degree = 3)
        X_poly = poly_reg.fit_transform(X_train1)
        poly_reg.fit(X_poly, y_train1)
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(X_poly, y_train1)
    elif num == 4:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators = 200, random_state = 0)
    elif num == 5:
        from sklearn.svm import SVR
        kernel=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        model = SVR(kernel = 'rbf')
        
    #model.fit(X_train1, y_train1)
    #computing Training and Validation scores
    train_score=model.score(X_train1, y_train1)
    valid_score=model.score(X_valid1, y_valid1)
    
    # Predicting the results
    y_pred_train = model.predict(X_train1)
    y_pred_valid = model.predict(X_valid1)
    y_pred_test = model.predict(X_test1)
    
    #y_pred_train=sc2.inverse_transform(y_pred_train)
    #Computing Root Mean Square Errors
    from sklearn.metrics import mean_squared_error
    RMS_train= mean_squared_error(y_train1, y_pred_train, squared=False)
    RMS_valid= mean_squared_error(y_valid1,y_pred_valid, squared=False)
    
    
    #print("For STOCK - "+ str(i+1)+ ":")
    #print("Training R2          = "+ str(train_score))
    #print("Validation R2        = "+ str(valid_score))
    #print("Training RMS error   = "+ str(RMS_train))
    #print("Validation RMS error = "+ str(RMS_valid))
    
    print(str(train_score))
    print(str(valid_score))
    print(str(RMS_train))
    print(str(RMS_valid))
    #y_pred_test.insert(result(X_train1,X_valid1,y_train1,y_valid1,X_test1,i,1),i)
    #result(X_train1,X_valid1,y_train1,y_valid1,2)
    
y_pred_test=[]
for i in range(5):
    y_pred_test.append(model.predict(X_test[i]))
#y_pred_test=sc2.inverse_transform(y_pred_test)
import csv
row=[]
row.append(test_dataset.iloc[:,0:1])
row=row[0]
ror=row.to_numpy('str')
if(type(y_pred_list)=='list'):
    y_pred_test=np.array(y_pred_test,'float32')
y_pred_test=y_pred_test.transpose()
x=np.hstack((ror,y_pred_test))
with open('submission3.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
    csv_writer.writerow(['Date','Close-Stock-1','Close-Stock-2','Close-Stock-3','Close-Stock-4','Close-Stock-5'])
    for i in range(len(x)):
        csv_writer.writerow(x[i])