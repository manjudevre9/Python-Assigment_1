#Que.1.What is the min-max scaling, & how is it used in data preprocessing?Provide an example to 
# illustrate it's application.
#Ans.Min-max scaling,also known as normalization,is a data preprocessing technique used to rescale numeric
# data to a common range,usually between 0 & 1.This helps to:
#1.Prevent feature dominance.
#2.Improve model interpretability.
#3.Enhance algorithm performance.

from sklearn.preprocessing
import MinMaxScaler
import pandas as pd

data={'Age':[25,30,35,40],'Income':[50000,60000,70000,80000]}
df=pd.DataFrame(data)
scaler=MinMaxScaler()
df_scaled=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
print(df_scaled)

#Que.2.What is the unit vector techniques in feture scaling, & how does it differ from min-max scaling?
# Provide an example to illustrate it's application.
#Ans.Unit vector techniques, also known as normalization or L2 normalization, is a feature scaling method that
# transforms each feature vector into a unit vector with a length of 1.This is achieved by dividing each feature
# value by its Euclidean norm.
from sklearn.preprocessing
import Normalizer
import pandas as pd
import numpy as np

data={'Age':[25,30,35,40],'Income':[5000,60000,70000,80000]}
df=pd.DataFrame(data)
Normalizer=Normalizer(norm='12')

df_scaled=pd.DataFrame(normalizer.fit_transform(df),columns=df.columns)
print(df_scaled)

#Que.3.What is PCA,& how is it used in dimensionality reduction?Provide an example to illustrate its application.
#Ans. Principle Component Analysis is a widely used dimensionality reduction technique in machine learning &
# data analysis.It's a statistical method that transform a set of correlated variables into a new set of 
# uncorrelated variables,called principle components.
# PCA for Dimensionality Reduction:PCA is used for dimensionality reduction by selecting a subset of the principle
# components that capture most of the variance in the data.
from sklearn.decomposition
import PCA 
import pandas as pd
import numpy as np

np.random.seed(0)
data=np.random.rand(10,5)
df=pd.DataFrame(data,columns=['A','B','C','D','E'])
pca=PCA(n_components=2)
df_pca=pca.fit_transform(df)

print(pca.explained_variance_ratio_)
print(df_pca)

#Que.4.What is the relationship between PCA & feature extraction, & how can PCA be used for feature extraction?
# Provide an example to illustrate this concept.
#Ans.The relationship between PCA & feature extraction is that PCA transforms the originl features into new features,
# called principle components,which are uncorrelated & capture the most variance in the data.
# PCA is Used for Feature Extraction:
# 1.Dimensionality Reduction.
# 2.Feature Transformation.
# 3.Noise Reduction.

from sklearn.decomposition
import PCA 
import pandas as pd
import numpy as np

np.random.seed(0)
data=np.random.rand(10,5)
df=pd.DataFrame(data,columns=['A','B','C','D','E'])
pca=PCA(n_components=2)
df_pca=pca.fit_transform(df)
print(pca.explained_variance_ratio_)
print(df_pca)

#Que.5.You are working on a project to build a recommendation system for a food delivery service.The dataset contains
# feature such as price,rating,& dilivery time.Explain how you would use min-max scaling to preprocess the data.
#Ans.
import pandas as pd
from sklearn.preprocessing
import MinMaxScaler

df=pd.read_csv("food_delivery_data.csv")
features_to_scale=["price","rating","delivery_time"]
scaler=MinMaxScaler()
df[features_to_scale]=scaler.fit_transform(df[features_to_scale])

print(df)

#1.Import Necessary Libraries:Import the necessary,including pandas for data manipulation & scikit-learn for min-max scaling.
#2.Load the Dataset:Load the dataset into a pandas DataFrame.
#3.Select the Features to Scale:Select the features that require scaling,such as price,rating, &  delivery time.
#4.Apply Min-Max Scaling:Apply min-max scaling to the selected features using the following formula:
# x_scaled=(x-x_min)/(x_max-x_min)
#5.Use Scikit-Learn's MinMaxScaler:Alternatively,use scikit-learn's MinMaxScaler to apply min-max scaling to the selected features.

#Que.6.You are working on a project to build a model to predict stock price.The dataset contains many features,
#such as company financial data & market trends.
#Ans.
import pandas as pd
from sklearn.decomposition
import PCA 
from sklearn.preprocessing
import StandardScaler
from sklearn.model_selection
import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df=pd.read_csv("stock_price_data.csv")
scaler.fit_transform(df)
pca=PCA(n_components=0.95)
df_pca=pca.fit_transform(df_scaled)
print(pca.explained_variance_ratio_)

x_train,x_test,y_train,y_test=train_test_split(df_pca,df["target"],test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

mse=mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error:{mse:.2f}")

#Que.7.For a dataset containing the following values:[1,5,10,15,20],perform min-max scaling to transform the value to a range of -1 to 1.
#Ans.
x_scaled=(x-x_min)/(x_max-x_min)*(max-min)+min
x_min=1
x_max=20

for x=1:
x_scaled=(1-1)/(20-1)*(1-(-1))+(-1)=-1

for x=5:
x_scaled=(5-1)/(20-1)*(1-(-1))+(-1)=0.6 

for x=10:
x_scaled(10-1)/(20-1)*(1-(-1))+(-1)=-0.2

for x=15:
x_scaled=(15-1)/(20-1)*(1-(-1))+(-1)=0.2

for x=20:
x_scaled=(20-1)/(20-1)*(1-(-1))+(-1)=1

The scaled dataset is [-1,-0.6,-0.2,0.2,1]

#Que.8.for the data set containing the follwoing fetature :[Height,weight,age,gender,blood Pressure],perform
# fetaure extraction using PCA.How many pricpile would you choose to retain and why?
#Ans.
import pandas as pd
from sklearn.decomposition
import PCA 
from sklearn.preprocessing
import StandardScaler
from sklearn.preprocessing
import OneHotEncoder
import matplotlib.pyplot as plt

data={'height':[165,170,180,160,175],'weight':[55,60,70,50,65],'age':[25,30,35,20,40],'gender':[0,1,0,1,0],'blood_pressure':[120,125,130,115,135]}
df=pd.DataFrame(data)

scaler=StandardScaler()
df[['height','weight','age','blood_pressure']]=scaler.fit_transform(df[['height','weight','age','blood_pressure']])
encoder=OneHotEncoder(sparse=False)
df_gender=encoder.fit_transform(df[['gender']])
df=pd.concat([df,pd.DataFrame(df_gender,columns=['gender_male','gender_female'])],axis=1)
df=df.drop('gender,'axis=1)

pca=PCA(n_components=None)
df_pca=pca.fit_transform(df)
plt.plot(pca.explained_variance_ratio)
plt.xlable('Principle Component')
plt.ylable('Explained Variance Ratio')
plt.show()

n_components=3
pca=PCA(n_components=n_components)
df_pca=pca.fit_transform(df)

print("Retained Principle Components:")
print(df_pca)

# The resulting dataset has three features(PC1,PC2,PC3) instead of the original five features.These new features
# are uncorrelated & capture the most variance in the data.