#Que.1. What is the difference between Ordinal encoding & lable encoding?Provide an example of when you might choose one over the other regression.
#Ans.Ordinal Encoding:
#1.Assigns numerical values to categories based on their inherent coder or ranking.
#2.Preserves the ordinal relationship between categories.
#3.Example:Transforming education level(High School, Bachelor's, Master's, Ph.D.)into numerical values(1,2,3,4)

# Label Encoding:1.Assigns numerical values to categories arbitrarily,without considering any inherent order.
#2.Does not preserve any ordinal relationship between categories.
#3.Example:Transforming colors (Red,Green,Blue) into numerical values(1,2,3).
import pandas as pd
from sklearn.preprocessing
import OrdinalEncoder,LabelEncoder

data={'Location':['Urban','Suburban','Rural','Urban','Suburban'],'Color':['Red','Green','Blue','Red','Green']}
df=pd.DataFrame

location_encoder=OrdinalEncoder()
df['Location']=location_encoder.fit_transform(df[['Location']])
location_encoder.fit_transform(df['Location'])

color_encoder=LabelEncoder()
df['Color']=color_encoder.fit_transform(df['Color'])

print(df)

#Que.2.Explain how Targer guieded ordinal encoding works & provide an example of when you might use it in a machine learning project.
#Ans.How Target Guided Ordinal Encoding Works:
#1.Calculate the mean target value:For each category in the categorical variable,calculate the mean target value.
#2.Rank categories:Rank the categories based on their corresponding mean target values.
#3.Assign numerical values:Assign numerical values to each category based on their ranking.Typically,the category with the lowest mean target value is assigned a value of 0.

import pandas as pd

data={'Location':['Urban','Urban','Suburban','Suburban','Rural','Rural'],'House Price':[500000,450000,350000,400000,200000,25000]}
df=pd.DataFrame(data)

mean_house_prices=df.groupby('Location')['House Price'].mean().reset_index()
mean_house_prices['Rank']=mean_house_prices['House Price'].rank(method='dense',ascending=False).astype(int)-1
df=pd.merge(df,mean_house_prices[['Location','Rank']],on='Location')
df=df.drop('Location',axis=1)
df=df.rename(columns={'Rank':'Location'})
print(df)

#Que.3.Define covariance & explain why it is important in statistical analysis. How is covariance calculated?
#Ans.Covariance is a statistical measure that calculates the extent to which two continuous random variables.X & Y, move together in a linear relationship.
# Importance of Covariance:
#1.Measure Linear Relationship:Covariance helps identify the direction & strength of the linear relationship between two variables.
#2.Predictive Modeling:Covariance is used in predictive modeling techniques,such as linear regression, to establish relationships between variables.
#3.Risk Analysis:In finance,covariance is used to calculate the risk of a portfolio by measuring the covariance between different assets.

#Que.4.For a dataset with the following categorical variables: Color(red,green,blue),Size(small,medium,large),& Material(wood,metal,plastic),perform label encoding using python's scikit-learn library. Show your code & example the output.
#Ans.
import pandas as pd
from sklearn.preprocessing
import LabelEncoder

data={'Color':['red','green','blue','red','green'],'Size':['small','medium','large','small','medium'],'Material':['wood','metal','plastic','wood','metal']}
df=pd.DataFrame(data)
print("Original Dataset:")
print(df)

le=LabelEncoder()

df['Color']=le.fit_transform(df['Color'])
df['Size']=le.fit_transform(df['Size'])
df['Material']=le.fit_transform(df['Material'])
print("\nDataset after Label Encoding:")
print(df)

#Que.5.You are working on a machine learning project with a dataset containing several categorical variables,Including 'gender(male/female),'education level'(high School/Bachelor's/Masters/PHD),& 'Employment Status'(Unemployes/ part-time/full time).Which encoding method would you use for each variable,& why?
#Ans.
import pandas as pd
from sklearn.preprocessing
import LabelEncoder,OrdinalEncoder

data={'Gender':['male','female','male','female'],'Education Level':['High school','Bachelor\'s','Master\'s','PhD'],'Employment Status':['Unemployed','part-time','full-time','Unemployed']}
df=pd.DataFrame(data)

gender_encoder=LabelEncoder()
df['Gender']=gender_encoder.fit_transform(df['Gender'])
education_encoder=OrdinalEncoder()
df['Education Level']=education_encoder.fit_transform(df[['Education Level']])
employment_encoder=pd.get_dummies(df['Employment Status'])
df=pd.contact([df,employment_encoder],axis=1)
print(df)

#Que.6.You are analyzing a dataset with two continuous variable."Temperature" & "Humidity",& two categorical variables,"Weather Condition"(Sunny/Cloudy/Rainy)&"Wind Direction"(North/South/East/West).Calculate the covariance between each pair of variables & interpret the results.
#Ans.
import pandas as pd
data={'Temperature':[25,30,20,28,32],'Humidity':[60,70,50,65,75],'Weather Condition':['Sunny','Cloudy','Rainy','Sunny','Cloudy'],'Wind Direction':['North','South','East','West','North']}
df=pd.DataFrame(data)
covariance=df['Temperature'].cov(df['Humidity'])
print("Covariance between Tempreature & Humidity:",covariance)