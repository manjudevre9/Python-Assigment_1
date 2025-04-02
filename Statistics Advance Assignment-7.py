#Que.1.Write a python function that takes in two arryas of data & calculate the F-value for a variance 
# ratio test.The function shoulde return the F-value & the corresponding p-value for the test.
#Ans.
import numpy as np
from scipy.stats import f

def calculate_f_value(data1,data2):
    Parameters:
    data1 (array-like):
    data2 (array-like):

    Returns:
    f_value (float):
    p_value (float):

    var1=np.var(data1,ddof=1)
    var2=np.var(data2,ddof=1)

    f_value=var1/var2

    df1=len(data1)-1
    df2=len(data2)-1

    p_value=2*(1-f.cdf(f_value,df1,df2))

    return f_value, p_value

data1=np.array([23,11,19,24,17])
data2=np.array([16,18,20,22,19])

f_value,p_value=
calculate_f_value(data1,data2)
print(f"F-value: {f_value:.4f}")
print(f"p-value: {p_value:.4f}")

#Que.2.Given a significances level of 0.05 & the degree of freedom for the numerator & denominator of an
# F-distribution,write a python function that return the critical F-value for a two tailed test.
#Ans.
import scipy.stast as stats

def critical_f_value(alpha,df1,df2):
    Parameters:
    alpha(float):
    df1(int):
    df2(int):

    Returns:
    critical_f (float):
    critical_f_right=stats.f.ppf(1-data/2,df1,df2)
    critical_f_left=1/stats.f.ppf(1-alpha/2,df2,df1)

    return max(critical_f_right,critical_f_left)

alpha=0.05
df1=3
df2=10

critical_f=critical_f_value(alpha,df1,df2)
print(f"Critical F-value:{critical_f:.4f}")

#Que.3.Write a python program that generates random sample from  two normal distribution with known variance
# & uses an F-test to determine if the variance are queal.The program should output the F-value,degree of freedom,&
# p-value for the test.
#Ans.
import numpy as np
from scipy.stats import f

np.random.seed(0)
mu1,sigma1=0,1
mu2,sigma2=0,1.5

sample1=np.random.normal(mu1,sigma1,100)
sample2=np.random.normal(mu2,sigma2,100)

var1=np.var(sample1,ddof=1)
var2=np.var(sample2,ddof=1)

f_value=var1/var2
df1=len(sample1)-1
df2=len(sample2)-1

p_value=2*(1-f.cdf(f_value,df1,df2))

print(f"F-value:{f_value:.4f}")
print(f"Degree of freedom:({df1},{df2})")
print(f"p-value:{p_value:.4f}")

alpha=0.05
if p_value<alpha:
    print("Reject the null hypothesis: The variances are not equal.")
else:
    print("Fail to reject the null hypothesis:The variances are equal.")

#Que.4.The variance of two population are known to be 10 & 15.A sample of 12 abservation is taken from each 
# population.conduct an F-test at the 5% significance level to determine if the variance are significantly
# different.
# Ans.
import numpy as np
from scipy.stats import f

s1_squared=10
s2_squared=15

n1=12
n2=12

df1=n1-1
df2=n1-1

f_statistic=s1_squared/s2_squared

p_value=2*(1-f.cdf(f_statistic,df1,df2))
print(f"F-statistic:{f_statistic:.4f}")
print(f"p-value:{p_value:.4f}")

#Que.5.A manufacture claim that the variance of the dimeter of a certain product is 0.005.A sample of 25
# product is taken & the sample variance is found to be 0.006.Conduct an F-test at the 1% significance level
# to determine if the claim is justified.
#Ans.
import numpy as np
from scipy.stats import f

s_squared=0.006

sigma_squared=0.005
n=25

df1=n-1
df2=np.inf

f_statistic=s_squared/sigma_squared

p_value=2*(1-f.cdf(f_statistic,df1,df2))
print(f"F-statistic:{f_statistic:.4f}")
print(f"p-value:{p_value:.4f}")

#Que.6.Write a python function that takes in the degree of freedom for the numerator & denominator of an
# F-distribution & calculate the mean & variance of the distribution.The function should return the mean & variance as a tuple.
#Ans.
import numpy as np
def calculate_f_distribution_mean_variance(df1,df2):
    Parameters:
    df1(int):
    df2(int):
    Returns:
    tuple:

    if df2<=2:
        if df2<=4:
            variance=np.nan
        else:
            variance=(2*df2**2*(df1*df2-2))/(df1*(df2-2)**2(df2-4))
            mean=df2/(df2-2)

            returnmean,variance

df1=10
df2=20
mean,variance=calculate_f_distribution_mean_variance(df1,df2)
print(f"Mean:{mean:.4f}")
print(f"Variance:{variance:.4f}")

#Que.7.A random sample of 10 measurement is taken from a normal population with unknown variance.The sample
# variance is found to be 25.Another random sample of 15 measurement is taken from another normal population with
# unknown variance & The sample variance is found to be 20. conduct an F-test at the 10% significance level to 
# determine if the variance are significantly different.
#Ans.
import numpy as np
from scipy.stats import f

s1_squared=25
s2_squared=20

n1=10
n2=15

df1=n1-1
df2=n2-1

f_statistic=s1_squared/s2_squared
p_value=2*(1-f.cdf(f_statistic,df1,df2))

print(f"F-statistic:{f_statistic:.4f}")
print(f"p-value:{p_value:.4f}")

#Que.8.The following data represents the waiting time in minutes at two different restaurant an a Saturday night:
# Restaurant A:24,25,28,23,22,20,27; Restaurant B:31,33,35,30,32,36. Conduct an F-test at the 5% significance
# level to determine if the variance are significantly different.
#Ans.
import numpy as np
from scipy.stats import f

A=np.array([24,25,28,23,22,20,27])
B=np.array([31,33,35,30,32,36])

s_A_squared=np.var(A,ddof=1)
s_B_squared=np.var(B,ddof=1)

print(f"Sample variance for Restaurant A:{s_A_squared:.4f}")
print(f"Sample variance for Restaurant B:{s_B_squared:.4f}")

f_statistic=s_A_squared/s_B_squared

print(f"F-statistic:{f_statistic:.4f}")
p_value=2*(1-f.cdf(f_statistic,6,5))
print(f"p-value:{p_value:.4f}")

#Que.9.The following data represents the test score of two groups of students:group A:80,85,90,92,87,83;
#group B:75,78,82,79,81,84. Conduct an F-test at the 1% significance level to determine if the variance are significantly different.
#Ans.
import numpy as np
from scipy.stats import f

A=np.array([80,85,90,92,87,83])
B=np.array([75,78,82,79,81,84])

s_A_squared=np.var(A,ddof=1)
s_B_squared=np.var(B,ddof=1)

print(f"Sample variance for Group A:{s_A_squared:.4f}")
print(f"Sample variance for Group B:{s_B_squared:.4f}")

f_statistic=s_A_squared/s_B_squared

print(f"F-statistic:{f_statistic:.4f}")
p_value=2*(1-f.cdf(f_statistic,5,5))
print(f"p-value:{p_value:.4f}")


