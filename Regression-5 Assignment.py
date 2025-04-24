#Quie.1.What is elastic net regression & how does it differ from other regression techniques?
#Ans.Elastic Net regression is a type of linear regression that combines L1 & L2 regularization.

# Comparison with Other Techniques:
#1.Lasso regression:Elastic Net regression can handle highly correlated features better than Lasso regression by Using L2 regularization.
#2.Ridge regression:Elastic Net regression can perform feature selection like Lasso regression, whereas Ridge regression does not.

import numpy as np
from sklearn.linear_model import ElasticNet

np.random.seed(0)
x=np.random.rand(100,10)
y=np.random.rand(100)

model=ElasticNet(alpha=0.1,l1_ratio=0.5)
model.fit(x,y)
print("Coefficient:",model.coef_)

#Que.2.How do you choose the optimal value of the regularization parameter for elastic net regression?
#Ans.The optimal value of the regularization parameter & the L1 ratio in Elastic Net regression can be chosen using.

# Cross-validation:
#1.K-fold cross-validation:Split the data into training & validation sets, & use K-fold cross-validation to evaluate the models 
# performance for different values of alpha & L1 ratio.
#2.Grid search:Perform a grid search over a range of alpha & L1 ratio values to find the optimal combination.

import numpy as np
from sklearn.linear_model import ElasticNetCV

np.random.seed(0)
x=np.random.rand(100,10)
y=np.random.rand(100)

model=ElasticNetCV(alphas=[0.01,0.1,1,10],l1_ratio=[0.2,0.5,0.8],cv=5)
model.fit(x,y)

print("Optimal alpha:",model.alpha_)
print("Optimal L1 ratio:"model.l1_ratio_)

#Que.3.What are the advantages & disadvantages of elastic net regression?
#Ans.Advantages of Elastic Net Regression:
# 1.Elastic Net regression provides a flexible regularization framework that combines L1 & L2 regularization, allowing for both feature selection & coefficient shrinkage.
#2.Handling multicollinearity:Elastic Net regression can handle multicollinearity by shrinking the coefficient of highly correlated features towards each other.
#3.Elastic Net regression can handle group of correlated features by shrinking thair coefficient towards each other.
#4.Elastic Net regression can improve model performance by reducing  overfitting & selecting relevant features.

# Disadvantages of Elastic Net Regression:
#1.Elastic Net regression requires tuning two hyperparameters which can be computationally expensive.
#2.Elastic Net regression can be more complex to implement & interpret than other regression techniques.
#3.Elastic Net regression can be computationally expensive,especially for large datasets.
#4.Carefully selecting the type of regularization & the values of hyperparameters is crucial to achieve the desired level of sparsity & coefficient shrinkage.

#Que.4.What are some common use cases for elastic net regression?
#Ans.Common Use Cases for Elastic Net Regression:Elastic Net regression is commonly used in
# High Dimentional Data Analysis:
#1.Elastic Net regression is used in genomica to identify rlevant genes & predict outcomes based on high-dimentional gene expression data.
#2.Elastic Net regression is used for feature selection in high-dimentional datasets,where the number of feture is large.
# Predictive Modeling:
#1.Elastic Net regression is used in finance to predict stock prices, credict risk, & other finantional outcomes.
#2.Elastic Net regression is used in marketing to predict customer behavior,churn, & response to marketing campaigns.
# Data with Multicollinearity:
#1.Elastic Net regression is used in economics to model economic systems with multicollinearity.
#2.Elastic Net regression used in social sciences to model complex relationships between variables with multicollinearity.
# Other Use Cases:
#1.Elastic Net regression is used in image processing to denoise & compress images.
#2.Elastic Net regression is used in signal processing to denoise & reconstruct signals.

#Que.5.How do you interpret the coefficient in elastic net regression?
#Ans.Intrepreting Coefficient in Elastic Net Regression:Interpreting coefficients in Elastic Net regression is similar to intrepreting coefficient in linear regression.
# Coefficients Interpretation:
#1.The sign of the coefficient indicates the direction of the relationship between the independent variables & the dependent bariable.
#2.The magnitude of the coefficient represents the change in the dependent variables for a one-unit change in the independent variables,while holding all other independent variables constant.

#6.How do you handle missing values when using elastic net regression?
#Ans.Methods for Handling Missing Values:
#1.Remove rows with missing values from the dataset.
#2.Replace missing values with the mean or median of the respective feature.
#3.Use a regression model to predict missing values based on other features.
#4.Create multiple versions of the daataset with different imputed values & analyze each version separately.

#Que.7.How do you use elastic net regression for feature selection?
#Ans.1.The L1 regularization term in Elastic Net regression can set coefficients to zero,effectively selecting features.
#2.Features with coefficients close to zero can be considered less important & potentialy removed.
#3.Fit an Elastic Net regression model to the data.
#4.Get the coefficient of the features from the model.
#5.Select features with non-zero coefficients.

#Que.8.How do you pickle & unpickle a trained elastic net regression model in python?
#Ans.Pickling the Model:
import pickle
from sklearn.linear_model import ElasticNet

model=ElasticNet(alpha=0.1,l1_ratio=0.5)
model_fit(x,y)
open('elastic_net_model.pkl','wb') as f:
pickle.dump(model,f)

# Unpickling the Model:
import pickle
open('elastic_net_model.pkl','rb') as f:
loaded_model=pickle.load(f)

y_pred=loaded_model.predict(x_new)

#Que.9.What is the purpose of pickle a model in machine learning?
#Ans.Save & Load Models:
#1.Pickling allows you to save trained models to a file, so you can load & use them later.
#2.Pickling enables you to load saved models & use them for predictions,evaluation, or further training.
# Benefits:
#1.Pickling saves time by avoiding the need to retrain models.
#2.Pickling makes it easy to share models between different scripts,environments, or users.
#3.Pickling enables deployment of trained models in production environments.