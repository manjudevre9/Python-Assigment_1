#Que.1.What is data encoding?How is it useful in data science?
#Ans.Data encoding is the process of converting data from one format to another to enable better processing,analysis,
# or modeling.In data science, encoding is essential for transforming raw data into a format that can be understood &
# processed by machine learning algorithms or statistical models.
import pandas as pd
from sklearn.preprocessing
import OneHotEncoder,LabelEncoder

data={'Species':['Lion','Tiger','Bear'],'Habitat':['']}

from sklearn.preprocessing
import OneHotEncoder

data={'color':['red','green','blue']}
df=pd.DataFrame(data)
encoder=OneHotEncoder(sparse=False)
encoded_data=encoder.fit_transform(df)
print(encoded_data)

#Que.2.What is nominal encoding?Provide an example of how you would use it in a real-world scenario.
#Ans.Nominal encoding,also known as lable encoding,is a technique used to convert categorical variables into numerical
# veriables.This is necessary because many machine learning algorithms require numerical inputs,but categorical variables
# are often represented as text or categories.
from sklearn.preprocessing
import LabelEncoder
import pandas as pd

data={'Favorite Color':['Red','Blue','Green','Red','Blue']}
df=pd.DataFrame(data)
encoder=LabelEncoder()
df['Favorite Color']=ebcoder.fit_transform(df['Favorite Color'])
print(df)

#Que.3.In what situation is nominal encoding preferred over one-hot encoding?Provide a practical example.
#Ans.#1.Number of categories is large:One hot encoding can create a large number of new features,leading to the curse of
# dimensionality,Nominal encoding is more efficient in such cases.
#2.Categories are ordinal:If the categories have a natural order or hierarchy, nominal encoding can preserve this order.
# One-hot encoding treats all categories as equally important.
#3.Memory constraints:Nominal encoding requires less memory than one-hot encoding,especially when dealing with large datasets.
from sklearn.preprocessing
import LabelEncoder
import pandas as pd

data={'Vehicle Type':['sedan','SUV','truck','motorcycle','sedan','SUV']}
df=pd.DataFrame(data)
encoder=LabelEncoder()
df['Vehicle Type']=encoder.fit_tranform(df['Vehicle Type'])
print(df)

#Que.4.Suppose you have a dataset containing categorical data with 5 unique values.Which encoding techniques would you
# use to transform this data into a format suitable for machine learning algorithms?Explain why you made this choice.
#Ans.
import pandas as pd
from sklearn.preprocessing
import OneHotEncoder

data={'Category':['A','B','C','D','E']}
df=pd.DataFrame(data)
encoder=OneHotEncoder(sparse=False)
encoded_data=encoder.fit_transform(df)
print(encoded_data)

#Que.5.In a machine learning project you have a dataset with 1000 rows & 5 columns.Two of the columns are categorical,&
# & the remaining three columns are numerical if you were to use nominal encoding to transform the categorical data,how 
# many new columns would be created?Show your calculations.
#Ans.Assumptions:We have 2 categorical columns,Each categorical column has a different number of unique categories.
# Calculation:Let's assume the first categorical column has k unique categories,& the second categorical column has m unique categories.
# When we apply nominal encoding to Column A,we create k-1 new columns.
# Similarly when we apply nominal encoding to Column B,we create m-1 new columns.
# Therefore, the total number of new columns created by nominal encoding is,
(k-1)+(m-1)=k+m-2
# So, the correct answer is:
# 2 new columns.

#Que.6.You are working with a dataset containing information about different types of animals,including their species,habitat,diet.
# Which encoding techniques would you use to transform the categorical data into a format suitable for machine learning algorithms?
#Ans.
import pandas as pd
from sklearn.preprocessing
import OneHotEncoder,LableEncoder

data={'Species':['Lion','Tiger','Bear'],'Habitat':['Desert','Forest','Ocean'],'Diet':['Carnivore','Herbivore','Omnivore']}
df=pd.DataFrame(data)

species_encoder=OneHotEncoder(sparse=False)
species_encoded=species_encoder.fit_transform(df[['Species']])
df_species=pd.DataFrame(species_encoded,columns=species_encoder.get_feature_names(['Species']))
habitat_encoder=LabelEncoder()
df['Habitat']=habitat_encoder.fit_transform(df['Habitat'])

diet_encoder=LabelEncoder()
df['Diet']=diet_encoder.fit_transform(df['Diet'])
print(df)
print(df_species)



