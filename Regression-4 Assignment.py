#Que.1.What is Lasso regression, & how does it differ from other regression techniques?
#Ans.Lasso Regression:Lasso regression,also known as least absolute shrinkage & selection operator is a type of linear
# regression that uses L1 regularization.This regularization technique adds a penalty term to the loss function that is
# proportional to the absolute value of the coefficients.

# How Lasso Regression Differs:
#1.L1 regularization:Lasso regression uses L1 regularization, which can set coefficient to zero, effectively performing feature selection.
#2.Sparse solutions:Lasso regression can produce sparse solutions, where some coefficients are exactly zero,making it useful for feature selection & model interpretation.

#Que.2.What is the main advantage of using lasso regression in feature selection?
#Ans.Lasso regression can set coefficients to exactly zero,effectively eliminating irrelevant features from the model.This allows for:
#1.Automatic feature selection:Lasso regression automatically selects the most relevant feature by setting the coefficient of irrelevant features to zero.
#2.Reduced model complexity:By eliminating irrelevant feature,Lasso regression can reduce model complexity & improve interpretability.

#Que.3.How do you interpret the coefficient of a lasso regression model?
#Ans.Interpreting Coefficient in Lasso Regression:
#Ans.Interpreting coefficient in Lasso regression is similar to interpreting coefficients in ordinary least squares regression, with some considerations:
#1.Direction of relationship:The sign of the coefficient indicates the direction of the relationship between the independent variable & the dependent variable.
#2.Magnitude of coefficient:The magnitude of the coefficient represents the change in the dependent variable for a one-unit change in the independent variable,
# while holding all other independent variables constant.

#Que.4.What are the tuning parameter that can be adjusted in lasso regression, & how do they affect the model performance?
# Ans.Tuning Parameters in Lasso Regression:
# Alpha:Alpha controls the strength of the L1 regularization. A higher value of alpha increases the penalty on the coefficients, leading to.
# 1.More coefficients set to zero:Higher alpha values result in more coefficients being set to zero, effectively eliminating more features from the model.
# 2.Sparser models:Higher alpha values lead to sparser models, which can improve model interpretability but may compromise predictive performance.

# Effect on Model Performance:
# 1.Overfitting vs. underfitting:A low alpha value may result in overfitting, while a high alpha may result in underfitting.
# 2.Feature selection:Alpha controls the number of features selected by the model. A higher alpha value selects fewer features.

#Que.5.Can lasso regression be used for non-linear regression problem?If yes, how?
# Ans.Lasso Regression for Non-Linear Regression Problems:
# Lasso regression is typically used for linear regression problems.However,it can be adapted for non-linear regression problems by:
# Using Basis Expansions:
# 1.Polynomial basis:Use polynomial basis functions to transform the features & fit a Lasso model.
# 2.Spline basis:Use spline basis functions to transform the features & fit a Lasso model.

# Using Kernel Methods:
# 1.Kernal trick:Use the kernel trick to transfrom the features into a higher-dimensional space & fit a Lasso model.
   
#Que.6.What is the difference between ridge regression & lasso regression?
#Ans.Ridge Regression vs Lasso Regression:
# The main difference between Ridge regression & Lasso regression lies in the type of regularization used:
# Ridge Regression:
#1.L2 regularization:Ridge regression uses L2 regularization, which adds a penalty term proportional to the square of the magnitude of the coefficients.
#2.Shrinks coefficients:Ridge regression shrinks the coefficients towards zero but does not set them to zero.

# Lasso Regression:
#1.L1 regularization:Lasso regression uses L1 regularization, which adds a penalty term propertional to the absolute value of the coefficients.
#2.Set coefficients to zero:Lasso regression can set coefficients to exactly zero, effectively performing feature selection.

#Que.7.Can lasso regression handle multicollinearity in the input feature?If yes, how?
#Ans.Lasso Regression & Multicollinearity:Lasso regression can handle multicollinearity in the input features to some extent:
# How Lasso Regression Handles Multicollinearity:
#1.Coefficient selection:Lasso regression can select one feature from a group of highly correlated features & set the coefficients of the other featues to zero.
#2.Features selsection:By setting coefficients to zero,lasso regression can effectively eliminate some features,which can help to reduce multicollinearity.

#Que.8.How do you choose the optimal value of the regression parameter in lasso regression?
#Ans.The optimal value of the regularization parameter in lasso regression can be chosen using.
# Cross-validation:
#1.k-fold cross-validation:Split the data into training & validation sets,& use k-fold cross-validation to evaluate the models performance for different values of alpha.
#2.Grid search:Perform a grid search over a range of alpha values to find the optimal value.