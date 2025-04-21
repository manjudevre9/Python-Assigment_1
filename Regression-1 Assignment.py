#Que.1.Explain the difference between simple linear regression & multiple linear regression . Provide an example of each.
#Ans.Simple Linear Regression:Simple linear regression is a statistical method that models the relationship between a single independent variable 
     # & a continuous dependent variable. The goal is to create a linear equation that best predicts the value of the dependent variable based on the independent variable.
     #Example of Simple Linear Regression:
     #Suppose we want to predict the price of a house based on its size.We collect data on house prices & sizes,& fit a simple linear regression model.

# Multiple Linear Regression:Multiple Linear Regression is a statistical method that models the relationship between multiple independent variables & a continuous dependent variable.
#The goal is to create a linear equation that best predicts the value of the dependent variable based on multiple independent variables.
# Example of Multiple Linear Regression:
# Suppose we want to predict the price of a house based on its size, number of bedrooms,& location. We collect data on house prices,size,number 
# of bedrooms, & locations,& fit a multiple linear regression model.
  
#Que.2.Discuss the assumption of linear regression . How can you check whether these assumptions hold in a given dataset?
#Ans.Assumpptions of linear Regression:
#1.Linearity:The relationship between the independent variables & the dependent variable is linear.
#2.Independence:Each observation is independent of the others.
#3.Homoscedasticity:The variance of the residuals is constant across all levels if the independent variables.
#4.Normality:The residuals are normally distributed.
#5.No multicollinearity:The independent variables are not highly correlated with each other.

# Checking Assumptions:
#1.Linearity:Plot the dependent variable againts each independent variable to check for linearity. Use residual plots to check for non-linearity.
#2.Independence:Check for independent by examining the data collection process & ensuring that observations are not paired or matched.
#3.Homoscedasticity:Plot the residuals againts the fitted values to check for constant variance.Use statistical tests such as Breusch-Pagan test.
#4.Normality:Plot a histogram or Q-Q plot of the residuals to check for normality.Use statistical tests such as Shapiro-Wilk test.
#5.No multicollinearity:Calculate the variance inflation factor for each independent variable to check for multicollinearity.

#Que.3.How do you interpret the slope & interpret in a linear regression model? Provide an example using a real world scenario.
#Ans.Slope:The change in the dependent variable for a one-unit change in the independent variable, while holding all other independent variables constant.
# Intercept:The value of the dependent variable when the independent variables is equal to zero.

import numpy as np
from sklearn.linear_model import LinearRegression
np.random.seed()
x=np.array([[1],[2],[3],[4],[5]])
y=np.array([250,300,350,400,450])

model=LinearRegression()
model.fit(x,y)

print("Intercept:",model.intercept_)
print("Slope:",model.coef_)

#Que.4. Explain the concept of gradient descent.How is it used in machine learning?
#Ans.Gradient descent is an optimization algorithm used to minimize the loss function in machine leanrning models.Its an iterative process that
#  adjusts the models parameters to reduce the difference between predicted & actual outputs.
# Gradient Descent in Machine Learning:
#1.Linear Regression:To optimize the coefficients of the linear regression model.
#2.Logistic Regression:To optimize the coefficients of the logistic regression model.
#3.Neural Networks:To optimize the weights & biases of the neural network.

# Types of Gradient Descent:
#1.Batch Gradient Descent:Uses the entire dataset to compute the gradient.
#2.Stochastic Gradient Descent:Uses a single data point to compute the gradient.
#3.Min-Batch Gradient Descent:Uses a small batch of data points to compute the gradient.

#Que.5.Describe the multiple linear regression model. How does it differ from simple linear regression?
#Ans.Multiple Linear Regression Model:
# Multiple linear regression is a statistical technique that models the relationship between multiple independent variables & a
# continuous dependent variables.The goal is to create a linear quuation that best predicts the value of the dependent
# variable based on multiple independent variables.

# Difference from Simple Linear Regression:
#1.Number of Independent Variables:Multiple linear regression involves more than one independent variable,whereas simple linear regression involves only one.
#2.Model Complexity:Multiple linear regression models are more complex & can capture more nuanced relationships between variables.
#3.Coefficient Interpretation:In multiple linear regression each coefficient represents the change in the dependent variable for a one-unit change in the corresponding 
# independent variable,while holding all other independent variables constant.

#Que.6.Explain the concept of multicollinearity in multiple linear regression. How can you detect & address this issue?
#Ans.Multicollinearity occurs when two or more independent variables in a multiple linear regression model are highly correlated with each other.
# This can lead to unstable estimates of the regression coeffficients & inflated variance.

# Detecting Multicollinearity:
#1.Correlation matrix:Examine the correlation matrix of the independent variables to identify highly correlated variables.
#2.Variance Inflation Factor:Calculate the VIF for each independent variable.A hogh VIF value indicates multicollinearity.

#Que.7.Describe the polynomial regression model. How is it different from linear regression?
#Ans.Polynomial regression is a type of regression analysis where the relationship between the independent variable & the dependent variable is 
# modeled using a polynomial equation.The polynomial equation can be of any degree, such as quadratic,cubic, or higher.

# Difference from Linear Regression:
#1.Non-linear relationship:Polynomial regression models non-linear relationships between the independent variable & the dependent variable,
   # whereas linear regression assumes a linear relationship.
#2.Higher degree terms:Polynomial regression includes higher degree terms to capture non-liner relationship.
#3.More complex models:Polynomial regression models can be more complex & flexible than linear regression models.

#Que.8.What are the advanteges & disadvantages of polynomial regression compared to linear regression?In what situation would you prefer to use polynomial regression?
#Ans.Advanteges:1.Polynomial regression can model non-linear relationships between variables, making it more flexible than linear regression.
#2.Polynomial regression can capture complex relationship between variables,such as curves & bends.

# Disadvanteges:
#1.Polynomial regression models can suffer from overfitting,especially when the degree of the polynomial is high.
#2.Polynomial regression models can be more complex & difficult to interpret than linear regression models.
#3.Polynomial regression models can behave erratically when extrapolating beyond the range of the training data.

# Example Situations:
#1.Modeling population growth:Polynomial regression can be used to model population growth,which often exhibits non-linear patterns.
#2.Predicting stock prices:Polynomial regression can be used to predict stock prices,which can exibit complex & non-linear patterns.

