#Que.1.A company conducted a survey of its employees & found that 70% of the employees use the companys health 
# insurance plan,while 40% of the employees who use the plan are smokers.What is the probability that an employees 
# is a smoker given that he/she uses the health insurance plan?
# Ans.Given Information:
# 1.P(uses insurance):70% of employees use the companys health insurance plan.
# 2.P(smoker | uses insurance):40% of employees who use the plan are smokers.

# Goal:Find P(smoker | uses insurance), the probability that an employee is a smoker given that they use the health insurance plan.
# Solution:
# P(smoker | uses insurance)=P(uses insueance | smoker)*P(smoker)/P(uses insurance)
# However, we are already given P(smoker | uses insurance) as 40%.

# Clarification:If we interpret the question as asking for the probability that an employee is a smoker & uses the
# insurance plan,then the answer is indeed 40%. The answer is 40%.

#Que.2 What is the difference between Bernoulli Naive Bayes & Multinomial naive bayes?
# Ans.Bernoulli Naive Bayes(BNB):
#1.Binary features:BNB is used when features are binary (0/1, yes/no, etc.).
#2.Bernoulli distribution:Each feature is assumed to follow a Bernoulli distribution,where the probability of a featurebeing 1 (or yes) is a parameter to be learned.
#3.Suitable for:BNB is suitable for datasets with binary features,such as spam vs. non-spam emails,where each word is either presend (1)or absent (0).

# Multinomial Naive Bayes(MNB):
#1.Categorical features:MNB is used when features are categorical & represent frequencies or counts.
#2.Multinomial distribution:Each feature is assumed to follow a multinomial distribution,where the probability of each category is a parameter to be learned.
# 3.Suitable for:MNB is suitable for datasets with categorical features,such as text classification,where each document is represented by a bag-of-words.

#Que.3.How does Bernoulli Naive Bayes handle missing values?
# Ans.Strategies to handle missing values:
# 1.Imputation:Replace missing values with mean,median,or imputed values using other algorithms.
# 2.Listwise deletion:Remove samples with missing values.
# 3.Pairwise deletion:Remove features with missing values.

# Best practices:
#1.Explore & understand:Investigate the nature of missing values & their impact on the data.
#2.Choose an appropriate strategy:Select a suitable method to handle missing values based on the data & problem characteristics.

#que.4.Can Gaussian naive bayes be used fo multi class classification?
#Ans.1.Calculate mean & variance:For each class,calculate the mean & variance of each feature.
#2.Model the probability distribution:Use the mean & variance to model the probability distribution of each feature for each class.

#Advantages:
#1.Simple & efficient.
#2.Robust to noise.