# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
developer Name: R.K Pragalyaa shree
Reg No        : 212221040125
import pandas as pd
df=pd.read_csv('/content/Churn_Modelling(1).csv')
df.head()
df.isnull().sum()
df.drop(['RowNumber','Age','Gender','Geography','Surname'],inplace=True,axis=1)
print(df)
x=df.iloc[:,:-1].values
y=df.iloc[:,:-1].values
print(x)
print(y)
from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()
df1  = pd.DataFrame(Scaler.fit_transform(df))
print(df1)
from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
print(xtrain)
print(len(xtrain))
print(xtest)
print(len(xtest))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1 = sc.fit_transform(df)
print(df1)
```

## OUTPUT:
![nn 1](https://user-images.githubusercontent.com/128135934/230117245-47e45ed3-7f6e-4a06-beb7-5981dfa9d100.png)
   ![nn 2](https://user-images.githubusercontent.com/128135934/230117433-ef4de705-61dc-47a8-b262-c2203b20873c.png)

![nn 3](https://user-images.githubusercontent.com/128135934/230117854-8ef29969-69c4-44a8-a110-b5647e8e4882.png)
![nn 4](https://user-images.githubusercontent.com/128135934/230118131-c1d84e7e-99cf-4fbe-a18a-616cc7dcc013.png)

![nn 5](https://user-images.githubusercontent.com/128135934/230118317-c6060197-56da-446d-a62c-a66336fa3c01.png)
   ![nn 6](https://user-images.githubusercontent.com/128135934/230118490-db0d6c0e-4ddf-4a1b-a6b8-2ba7341a4986.png)

![nn 7](https://user-images.githubusercontent.com/128135934/230118787-ba2d3357-84a5-4eda-832e-2de17f4e5849.png)

## RESULT
Thus the above program for standardizing the given data was implemented successfully.
