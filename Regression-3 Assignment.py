#Que.1.What is Ridge regression, & how does it differ from ordinary least squares regression?
#Ans.Ridge Regression:Ridge regression, also known as Tikhonov regularization or L2 regularization, is a type of linear regression that uses a regularization technique
# to reduce the magnitude of the coefficients. This is done by adding a penalty term to the loss function, which discourages large coefficient values.

# Difference from Qrdinary Least Squares Regression:
#1.Regularization term:Ridge regression adds a regularization term to the loss function,which penalizes large coefficient values. OLS regression does not have this term.
#2.Coefficient shrinkage:Ridge regression shrinks the coefficients towards zero,whereas OLS regression does not.
#3.Handling multicollinearity:Ridge regression can handle multicollinearity better than OLS regression by reducing the impact of correlated variables.

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

np.random.seed(0)
x=np.random.rand(100,10)
y=np.random.read(100)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size==0.2,random_state=42)
model=Ridge(alpha=1.0)
model.fit(x_train,y_train)
print("Coefficients:",model.coef_)

#Que.2.What are the assumption of ridge regression?
#Ans.1.Linearity:The relationship between the independent variables & the dependent variable is linear.
#2.Independence:Each observation is independent of the others.
#3.Homoscedasticity:The variance of the residuals is constant across all levels of the independent variables.
#4.Normality:The residuals are normally distributed.

#Que.3.How do you select the value of the tuning parameter in ridge regression?
#Ans.The tuning parameter in Ridge regression controls the strengh of the regularization.Selecting the optimal value if alpha is
#  crucial to achieve a good balance between fitting the data & preventing overfitting.

# Methods for Selecting Alpha:
#1.Cross-validation:Use corss-validation to evalute the models performance on unseen data for different values of alpha.
#2.Grid search:Perform a grid search over a range of alpha values to find the optimal value.
#3.Random search:Perform a random search over a range of alpha values to find the optimal value.

#Que.4. Can ridge regression be used for feature selection? If yes, how?
#Ans.Ridge regression can be used for feature selection indirectly by reducing the magnitude of coefficients. However, it does not 
# perform feature selection in the classical sense.

# How Ridge Regression Helps:
#1.Coefficient shrinkage:Ridge regression shrinks the coefficents of less important features towords zero, making them less influential in the model.
#2.Feature ranking:By examining the magnitude of coefficient, we can rank features in terms of their importance.

import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pylot as plt

np.random.seed(0)
x=np.random.rand(100,10)
y=np.random.rand(100)
model=Ridge(alpha=1.0)
model.fit(x,y)

plt.bar(range(10),model.coef_)
plt.xlable('Feature Index')
plt.ylable('Coefficient Value')
plt.show()

#Que.5.How does the ridge regression model perform in the presence of multicollinearity?
#Ans.Ridge Regression & Multicollinearity: Ridge regression is particularly useful in the presence of multicollinearity. 
# Multicollinearity occurs when two or more independent variables are highly correlated with each other.

# How Ridge Regression Handles Multicollinearity:
#1.Reducing coefficeint variance:Ridge regression reduces the variance of the coefficient by adding a penalty term to the loss function.
# This helps to stabilize the estimates & reduce the impact of multicollinearity.
#2.Shrinking coefficients:Ridge regression shrinks the coefficients of correlated variables towards each other, which can help to reduce the impact of multicollinearity.

#Que.6 Can ridge regression handle both categorical & continuous independent variables?
#Ans.Handling Categorical & Continous Variables:Ridge regression can handle both categorical & continuous independent variables.
# However, categorical variables need to be encoded properly before fitting the model.

# Encoding Categorical Variables:
#1.One-hot encoding:One-hot encoding is a common method for encoding categorical variables.It creates a new binary variable for each category.
#2.Dummy coding:Dummy coding is similar to one-hot encoding but uses a reference category to avoid multicollinearity.

#Que.7.How do you interpret the coefficient of ridge regression?
#Ans.Interpreting coefficient in Ridge regession is similar to interpreting coefficient in ordinary least squares regression, with some considerations.
#1.Direction of relationship:The sign of the coefficient indicates the direction of the relationship between the independent variable & the dependent variable.
#2.Magnitude of coefficient:The magnitude of the coefficient represents the change in the dependent variables for a one-unit change in the independent variable,while holding all other independent variables constant.

#Que.8.Can ridge regression be used for time series data analysis?If yes, how?
#Ans.Ridge regresiion for Time Series Data Analysis:Ridge regression can be used for time series data analysis,particularly when dealing with.
#1.Autocorrelated data:Ridge regression can help to reduce the impact of autocorrelation in time series data.
#2.Multiple features:Ridge regression can handle multiple features,indicators,& other time series-related features.

#Preparing Time Series Data:
#1.Feature engineering:Create relevant features from the time series data,such as lagged variables,moving averages, or seasonal indicators.
#2.Stationarity:Ensure that the time series data is stationary or difference the data to make it stationary.


