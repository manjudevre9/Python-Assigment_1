#Que.1.What is the filter method in feature selection, & how does it work?
#Ans.The filter method is a popular approach to feature selection that evaluates the relevance of each feature based on statistical or
# or information-theoretic criteria,regardless of the machine learning algorithm used. Here's how it works:
# Steps involved in th Filter Method:
# 1.Data Preparation:Collect & preprocess the dataset, including handling missing values & normalization.
# 2.Feature Evaluation:Evaluate each feature individually using a statistical or information-theoretic measure.
# 3.Feature Ranking:Rank the features based on their evaluation scores.
# 4.Feature Selection:Select the top-ranked features based on a predetermined threshold or number of features to select.
# 5. Model Training:Train a machine learning model using the selected features.

#Que.2.How does the wrapper method differ from the filter method in feature selection?
#Ans.Filter Method:
#1.Evaluates features independently:The filter method evaluates each feature individually based on statistical or information-theoretic criteria,such as
# correlation, matual information, or chi-squared tests.
#2.Does not consider feature interactions:The filter method does not consider interactions between features.
#3.Computational efficiency:The filter method is generally faster & more scalable, as it dose not require training a machine
# learning model.
#4.Threshold selection:The filter method requires selecting a threshold to determine which features to select.

# Wrapper Method:
#1.Evaluates features in conjunction with a model:The wrapper method evaluates features in conjunction with a machine
# learning model,using the model's performance as the evaluation criterion.
#2.Considers feature interactions:The Wrapper method considers interactions between features,as the models performance is 
# affected by the combination of features.
#3. Computational intensity:The wrapper method is generally more computationally intensive,as it requires training 
# a machine learning model multiple times.
#4.Model-dependent:The wrapper method is dependent on the choice of machine learning model.

#Que.3.What are some common techniques used in embedded feature selection methods?
#Ans.1.Regularization Techniques:Regularization techniques,such as L1 & L2 regularization, can be used to select features by adding a penalty term to the loss function.
#2.Tree-Based Methods:Tree-based methods,such as decision trees & random forests,can be used for features selection by analyzing the importance of each feature in the tree.
#3.Gradient Boosting Machines:GBMs select features by analyzing the importance of each features in the boosting process.
#4.Neural Networks with Dropout:Neural networks with dropout can be used for feature selection by randomly dropping out features during training.
#5.Recursive Feature Elimination:RFE is a technique that recursively eliminates the least important features until a specified number of features is reached.

#Que.4.What are some drawbacks of using the filter method for feature selection?
#Ans.1.Ignores Feature Interactions:The filter method evaluates features individually,ignoring potential interactions between features.
#2.Assumes Feature Independence:The filter method assumes that features are independent,which is often not the case.
#3.Does Not Consider Model-Specific Feature Importance:The filter method evaluates feature importance based on general statistical or
# information-theoretic criteria.
#4.Can Be Affected by Noise & Outliers:The filter method can be sensitive to noise & outliers in the data,which can affect the evaluation of feature importance.
#5.Requires Choosing a Threshold:The filter method requires choosing a threshold to determine which features to select.
#6.Can Be Computationally Expensive for Large Datasets:While the filter method is generally faster than wrapper methods,it can still be computationally expensive for very large datasets.

#Que.5.In which situation would you prefer using the filter method over the wrapper method for feature selection?
#Ans.1.High-Dimensional Data:When dealing with high-dimensional data,the filter method is generally faster & more scalable than the wrapper method.
#2.Large Datasets:When working with large datasets,the filter method can be more efficient,as it doesn't require training a machine learning model.
#3.Exploratory Data Analysis:During exploratory data analysis,the filter method can be useful for quickly identifying the most relevent features.
#4.Feature Ranking:whwn you need to rank features based on their importance, the filter method provides a clear ranking.
#5.Intertability:The filter method provides a clear understanding of the feature selection process, as each feature is evaluated individually.
#6.Avoiding Overfitting:When you want to avoid overfitting the filter method can be a better choice,as it doesn't rely on a specific machine learning model.

#Que.6.In a telecom company,you are working on a project to develop a predictive model for customer churn,You are unsure of which features to include in the model because the dataset contains
# several different ones.Describe how you would choose the most pertinent attributes for the model using the filter method.
#Ans.
import pandas as pd
from sklearn.feature_selection
import mutual_info_classif
from sklearn.feature_selection
import SelectKBest

df=pd.read_csv("customer_churn_data.csv")
x=df.drop("churn",axis=1)
y=df["churn"]
k=10
selector=SelectKBest(mutual_info_classif,k=k)
x_selected=selector.fit_transform(x,y)
selected_features=x.columns[selector.get_support()]

print("Selected Features:")
print(selected_features)

#Que.7.You are working on a project to predict the outcome of a soccer match.You have a large dataset with many features,including player statistics & tearm rankings.Explain
# how you would use the embedded method to select the most relevant features for the model.
#Ans.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv("Soccer_match_data.csv")
x=df.drop("match_outcome",axis=1)
y=df["match_outcome"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)

importance=rf.feature_importances_feature_names=x.columns
feature_importance=pd.DataFrame({"feature":feature_names,"importance":importance})
feature_importance.sort_values("importance",ascending=False)

k=10
selected_features=feature_importance["feature"].head(k)
x_train_selected=x_train[selected_features]
x_test_selected=x_test[selected_features]
rf.fit(x_train_selected,y_train)
y_pred=rf.predict(x_test_selected)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy:.4f}")

#Que.8.You are working on a project to predict the price of a house based on its features,such as size,location,& ahe. You have a limited number of features, & you want to ensure that yo select the most 
# important ones for the model.Explain how you would use the wrapper method to select the set of features for the predictor.
#Ans.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df=pd.read_csv("house_prices.csv")
x=df.drop("price",axis=1)
y=df["price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

best_subset=[]
best_mse=float("inf")
for feature in x.columns:
    subset=best_subset+[feature]
x_train_subset=x_train[subset]
x_test_subset=x_test[subset]
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train_subset,y_train)

y_pred=rf.predict(x_test_subset)
mse=mean_squared_error(y_test,y_pred)

if mse<best_mse:
    best_subset=subset
    best_mse=mse

print("Best Subset of Features:")
print(best_subset)    