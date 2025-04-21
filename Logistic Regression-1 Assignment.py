#Que.1.Explain the difference between linear regression & logistics regression models. Provide an example of a scenario where logistics regression would be more appropriate.
#Ans.Linear Regression:
# Purpose:Predict a continuous outcome variable based on one or more predictor vaeiables.
# Assumptions:Linear relationship between predictors & outcome,normality of residuals,homoscedasticity, & independence of observations.
# Output:Continuous value.

# Logistic Regression:
# Purpose:Predict a binary outcome variable based on one or more preductor variables.
# Assumptions:Independence of observations,linearity between predictors & log-odds, & no multicollinearity.
# Output:Probability of the outcome variable being in one of the two categories.

# Scenario: Predicting Customer Churm:
# Logistic Regression:Logistic regression would be more appropriate in this scenario because:
# Binary Outcome:Churn is a binary outcome variable(0=no churn,1=churn).
# Predicting Probability:Logistic regression predicts the probability of churn, which can be useful for prioritizing customers at high risk of churning.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df=pd.read_csv('customer_churn_data.csv')
x_train,x_test,y_train,y_train,y_test=train_test_split(df.drop('churn',axis=1),df['churn'],test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('Accuracy:',accuracy_score(y_test,y_pred))
print('Classification Report:')
print(classification_report(y_test,y_pred))

#Que.2. What is the cost function used in logistics regression,& how is it optimised?
#Ans.Cost Function in  Logistic Regression:The cost function used in logistic regression is the Log Loss or Cross-Entropy Loss.It meatures the different between the predicted 
# probabilities & the actual labels.

# Optimizing the Cost Function:
# To optimize the cost function, we use Maximum Likelihood Estimation(MLE). The goal is to fine the model parameters that maximize the likehood of observing the data.

# Optimization Algorithm:
# 1.Gradient Descent:An iterative algorithm that updates the model parameters based on the gradient of the cost function.
# 2.Newton's Method:A second-order optimization algorithm that uses the Hessian matrix to update the model parameters.
# 3.Quasi-Newton Methods:A family of optimization algorithms that approximate the Hessian matrix.

import numpy as np
from sklearn.linear_model import LogisticRegression
x=np.array([[1,2],[3,4],[5,6]])
y=np.array([0,1,1])
model=LogisticRegression()
model.fit(x,y)
print('Optimized model parameters:',model.coef_)

#Que.3.Explain the concept of regularisation in logistics regression & how it helps prevent over fitting.
#Ans.Regularization in Logistic Regression:Regularization is a technique used to prevent overfitting in logistic regression models by adding a penalty term to the cost function.

# How Regularization Helps Prevent Overfitting 
# 1.Reducing model complexity:By adding a penalty term, regularization encourages the model to reduce the magnitude of the coefficients, which helps to prevent overfitting.
# 2.Preventing feature dominance:Regularization helps to prevent any single feature from dominating the model,which can lead to overfitting.

import numpy as np 
from sklearn.linear_model import LogisticRegression
x=np.array([[1,2],[3,4],[5,6]])
y=np.array([0,1,1])
model_l1=LogisticRegression(penalty='l1',solver='liblinear')
model_l1.fit(x,y)
model_l2=LogisticRegression(penalty='l2')
model_l2.fit(x,y)
print('L1 Regularization Coefficients:', model_l1.coef_)
print('L2 Regularization Coefficients:',model_l2.coef_)

#Que.4.What is the ROC curve & how is it used to evaluate the performance of the logistics regression model?
#Ans.The Receiver Operating Characteristics(ROC) curve is a graphical representation of a model performance, plotting the True Positive Rate against the False Positive Rate
# at different thresholds.

# Evaluating Logistic Regression Model Performance:
# To evaluate the performance of a logistic regression model using the ROC curve:
# 1.Predict probabilities:Use the model to predict probabilities for the test dataset.
# 2.Calculate TPR & FPR:Calculate the TPR & FPR at different thresholds.
# 3.Plot the ROC curve:Plot the TPR against the FPR.

# Interpreting the ROC Curve:
# 1.Visualizing trade-offs:The ROC curve shows the trade-off between TPR & FPR,allowing us to identify the optimal threshold for classification.
# 2.Comparing models:ROC curves can be used to compare the performance of different models.

#Que.5.What are some common techniques for feature selection in logistics regression?How do these techniques help improve the model performance?
#Ans.Feature Selection Techniques:
#1.Correlation Analysis:Identify features that are highly correlated with the target variable.
#2.Recursive Feature Elimination(RFE):Recursively eliminate features based on their importance.
#3.L1 Regularization(Lasso):Use L1 regularization to select features by setting coefficients to zero.
#4.Mutual Information:Select features based on their mutual information with the target variable.
#5.Forward Selection:Select features one by one based on their performance.

# How Feature Selection Improves Model Performance:
#1.Reducing Overfitting:By selecting only relevant features,we reduce the risk of overfitting.
#2.Improving model Interpretability:Feature selection helps identify the most important features,making the model more interpretable.
#3.Reducing Computational Cost:Feature selection reduces the number of features,making the model training process faster.

#Que.6.How can you handle imbalance dataset in logistics regression?What are some strategies for dealing with class imbalance?
#Ans.To handle imbalanced dataset in logistic regression,we can use various stategies to improve the models pefrormance.
# Strategies for Dealing with Class Imbalance:
#1.Oversampling the minority class:Create additional instances of the minority class to balance the dataset.
#2.Undersampling the majority class:Reduce the number of instances in the majority class to balance the dataset.
#3.SMOTE(Synthetic Minority Over-sampling Technique):Generate synthetic instances of the minority class.
#4.Class weights:Assigh different weights to the classes during training.
#5.Thresholding:Adjust the decision threshold to optimize performance.

#Que.7.Can you discuss some common issue & challenge that may arise when implementing logistics regression, & how they can be addressed?For Example
# what can be done if there is multicollinearity among the independent variable?
#Ans.Common Issues & Challenges:
#1.Multicollinearity:High correlation between independent variables can lead to unstable estimates & inflated variance.
#2.Overfitting:Models that are too complex may fit the training data too well but perform poorly on new data.
#3.Non-linearity:Logistic regression assumes a linear relationship between the independent variables & the log-odds of the outcome.
#4.Outliers & influential observations:Outliers & influential observations can affect the models estimates & predictions.

# Addressingb Multicollinearity:
#1.Variance Inflation Factor:Calculate the VIF for each independent variable to detect multicollinearity.
#2.Remove or combine variables:Remove or combine variables that are highly correlated.
#3.Regularization:Use regularization techniques,such as L1 or L2 regularization, to reduce the impact of multicollinearity.

import numpy as np
import pandas as pd
from statistics.stats.outliers_influence import variance_inflation_factor

np.random.seed(0)

x=np.random.rand(100,3)
x=pd.DataFrame(x,columns=['x1','x2','x3'])x['x2']=x['x1']+np.random.rand(100)
vif=pd.DataFrame()
vif["VIF Factor"]=[variance_inflation_factor(x.values,i)for i in range(x.shape[1])]vif["features"]=x.columns
print(vif)