#Que.1.What are the key features of the wine quality dataset?Discuss the importance of each feature in predicting the quality of wine.
#Ans.The wine quality dataset typically includes several key features that help predict wine quality, These features can be broadly
# categorized into chemical properties & sensory attributes.
# Importance of Features:
# 1.Alcohol Content:Has a positive correlation with wine quality,with higher alcohol content often associated with better wines.
# 2.Volatile Acidity:Inversely connectd to pH,indicating that wines with more volatiliy tend to have lower pH.
# 3.Fixed Acidity:Correlates positively with density,suggesting denser wines have more fixed acidity.
# 4.Sulphates & Citric Acid:Contribute to the wines freshness,flavor,& overall quality.

#Que.2.How did you handle missing data in the wine quality dataset during the features engineering process?Discuss the advanteges & diadvantages of different imputation techniques.
#Ans.Imputation Techniques:
#1.Mean/Median Imputation:Replace missing values with the mean or median of the respective feature.
#2.Mode Imputation:Replace missing values with the most frequent value of the respective feature.
#3.Regression Imputation:Use a regression model to predict missing vaalues based on other features.
#4.K-Nearest Neighbors(KNN) Imputation:Replace missing values with the values from the k-nearest neighbors.
#5.Multiple Imputation:Create multiple versions of the dataset with different imputed values & analyze each version separately.

# Advantages & Disadvanteges:
# Mean/Median Imputation:
# Advantages:Simple to implement,fast, & easy to understand.
# Disadvantages:May not capture complex relationships between features,can lead to biased results if data is not normally distributed.

# Mode Imputation:
# Advanteges:Simple to implement,fast & easy to understand.
# Disadvanteges:May not capture complex relationship between featuires,can lead to biased results if data is not categorical.

# Regression Imputation:
# Advantages:Can capture complex relationship between features,can handle both continuous & categorical data.
# Disadvanteges:Requires a good understanding of the relationships between features,can be computationally expensive.

# KNN Imputation:
# Advanteges:Can capture complex relationships between features, can handle both continuous & categorical data.
# Disadvanteges:Can be computationally expensive,requres careful choice of k.

# Multiple Imputation:
# Advanteges:Can capture uncertainty in missing data,provides a range of possible values.
# Disadvanteges:Can be computationally expensive,requires careful choice of imputation models.

#Que.3.What are the key factors that affect students performance in exam?How would you go about analyzing these factors using statstical techniques?
#Ans.Key Factors Affecting Student Performance:
#1.Academic Background:Prior academic,achievement,educational background, & learning style.
#2.Demographic Factors:Age,gender,socioeconomic status,& ethnicity.
#3.Learning Environment:Classroom enviroment,teacher quality,& access to resources.
#4.Psychological Factors:Motivation,anxiety,& self-efficacy.
#5.Study Habits:Time management, study techniques,& homework completion.
#6.Socioeconomic Factors:Parental education,income,& access to resoures.

# Statistical Techniques for Analysis:
#1.Descriptive Statistics:Summarize student performance data using means,medians & standard deviations.
#2.Inferential Statistics:Use t-tests,ANOVA, & regression analysis to compare groups & identify relationships.
#3.Correlation Analysis:Examine relationships between factors & student performance.
#4.Regression Analysis:Model the relationship between factors & student performance.
#5.Factor Analysis:Identify underlying factors that affect student performance.
#6.Cluster Analysis:Group students with similar characteristics & performance.

#Que.4.Describe the process of features engineering in the context of the students performance dataset. How did you select & transform the variables for your model?
#Ans.Feature Engineering Process:
#1.Data Cleaning:Removed missing values & outliers from the dataset.
#2.Features Selection:Selected relevant features that impact student performance,such as academic background,demographic factors,& learning environment.
#3.Feature Transformation:Transformed categorical variables into numerical variables using techniques likle one-hot encoding & lebel encoding.
#4.Feature Scaling:Scaled numerical variables to a common range using standardization or normalization.
#5.Feature Extraction:Extracted new features from existing ones,such as calculating the average grade or creating a composite score.

# Variable Selection:
#1.Academic Background:Selected features like prior academic achievement, educational background,& learning style.
#2.Demographic Factors:Selected features like age,gender,socioeconomic status,& ethnicity.
#3.Learning Environment:Selected features like classroom environment, teacher quality & access to resources.

# Variable Transformation:
#1.One-Hot encoding:Transformed categorical variables like gender & ethnicity into numerical variables.
#2.Label Encoding:Transformed categorical variables like educational background into numerical variables.
#3.Standardization:Scaled numerical variables like age & socioeconomic status to a common range.

#Que.5.Load the winequality dataset & perform exploratory data analysis to identify the distribution of each feature.Which features exhibit non-normality & what transformation could be applied to these features to improve normality?
#Ans.Loading the Wine Quality Dataset:
import pandas as pd
url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df=pd.read_csv(url,sep=';')
print(df.head())

# Exploratory Data Analysis:
# Summary Statistics
print(df.describe())

# Identifying Non-Normality
# Features with Non-Normality:
#1.Residual Sugar:Exhibits a skewed distribution with a long tail.
#2.Chlorides:Shows a non-normal distribution with outliers.
#3.Sulfur Dioxide:Displays a bimodal distribution.

# Transformation Techniques:
# Log Transformation:
#1.Residual Sugar:Apply a long transformation to reduce skewness.
#2.Chlorides:Use a long transformation to stabilize variance.

# Standardization:
# Sulfur Dioxide:Standardize the feature to have zero mean & unit variance.

#Que.6.Using the wine quality dataset, perform principle components analysis to reduce the number of features what is the minimum number of principle componetnts required to explain 90% of the varianve in the data?
#Ans.
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df=pd.read_csv(url,sep=';')
scaler=StandardScaler()
df_scaled=scaler.fit_transform(df)
pca=PCA()
pca_components=pca.fit_transform(df_scaled)
explained_variance_ratio=pca.explained_variance_ratio_

import matplotlib.pyplot as plt
plt.plot(explained_variance_ratio.cumsum())
plt.xlabel('Number of Principle Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()

# Minimum Number of Principle Components:
min_components=np.argmax(explained_variance_ratio.cumsum()>=0.9)=1
print(f'Minimum number of principle components required to explain 90% of the variance:'{min_components})
      