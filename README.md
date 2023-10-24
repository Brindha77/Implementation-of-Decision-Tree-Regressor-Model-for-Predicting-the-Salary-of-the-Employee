# EXP 7: Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

## STEP 1 :
Import the standard libraries. 2.Upload the dataset and check for any null values using .isnull() function.

## STEP 2 :
Import LabelEncoder and encode the dataset.

## STEP 3 :
Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

## STEP 4 :
Predict the values of arrays.

## STEP 5 :
Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset 7.Predict the values of array.

## STEP 6 :
Apply to new unknown values.
## Program:
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: R.BRINDHA

RegisterNumber:212222230023  
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
## data.head() :
![image](https://github.com/Brindha77/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118889143/5a123c8b-5367-4da5-9ebe-9750c11a2841)
## data.info() :
![image](https://github.com/Brindha77/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118889143/ce1b2f5d-7331-4989-b44b-496329485e86)
## isnull() & sum() function :
![image](https://github.com/Brindha77/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118889143/0ffad8ec-93c8-4751-9782-7fac260254a2)
## data.head() for Position :
![image](https://github.com/Brindha77/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118889143/1af9624c-82a3-42e3-a920-04cbd3f211b9)
## MSE value :
![image](https://github.com/Brindha77/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118889143/c451bf55-c8bc-4f9c-94bc-270668f8e75d)
## R2 value :
![image](https://github.com/Brindha77/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118889143/25311a40-c400-4534-a328-2f65c790c650)
## Prediction value :
![image](https://github.com/Brindha77/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118889143/416b4a43-a78f-41cd-af31-53f9f7fe9cfe)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
