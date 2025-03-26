#Que.1.Write a python code to implement the KNN classifier algorithm on load_iris dataset in sklearn.datasets.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris=load_iris()

x=iris.data
y=iris.target

x_train, x_test, y_train, y_test =train_test_split(x,y, test_size = 0.2, random_state = 42) 

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

k=5
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

y_pred=knn.predict(x_test)

accuracy=accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Que.2.Write a python code to implement the KNN respressor algorithm on load_boston dataset in sklearn.datasets.
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

boston=load_boston()

x=boston.data
y=boston.target

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

k=5
knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R-Squared:",r2)

#Que.3 Write a python code snippet to find the optimal value of k for the KNN classifier algorithm using cross-validation on load_iris datasets in sklearn.datasets.
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np

iris=load_iris()

x=iris.data
y=iris.target

k_values=np.arange(1,31)
accuracies=[]

for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k)
    cross-validationscores=cross_val_score(knn,x,y,cv=5)

    accuracy=np.mean(scores)
    accuracies.append(accuracy)
    optimal_k=k_values[np.argmax(accuracies)]

    print("Optimal k value:", optimal_k)
    print("Highest average accuracy:",np.max(accuracies))

#Que.4.Implement the KNN espressor algorithm with feature scaling on load_boston dataset in sklearn.datasets.
#Ans.
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

boston=load_boston()

x=boston.data
y=boston.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(x_train)
x_train_scaled=scaler.transform(x_test)

knn=KNeighborsRegressor(n_neighbors=5)

knn.fit(x_train_scaled,y_train)

y_pred=knn.predict(x_test_scaled)

mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)

r2=r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R-Squared:",r2)

#Que.5. Write a python code snippt to implement the KNN classifier algorithm with weighted voting on load_iris dataset in sklearn.dataset.
#Ans.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

iris=load_iris()

x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
knn=KNeighborsClassifier(n_neighbors=5,weights='distance')

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)
accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print("Classification Report:")
print(Classification_report(y_test,y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))

#Que.6.Implement a function to standardise the features before applying KNN classifier.
#Ans.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def standardize_features(x_train,x_test):
    Args:
    x_train
    x_test

    Returns:
    x_train_scaled
    x_train_scaled
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

return x_train_scaled,x_test_scaled

def knn_classifier (x_train,x_test,y_train,y_test):
    Args:
    x_train
    x_test
    y_train
    y_test

    Returns:
    accuracy(float):
    report(str):
    matrix(array_like):

    x_test_scaled,x_test_scaled=standardize_features(x_train,x_test)
    knn= KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train_scaled,y_train)
    y_pred=knn.predict(x_test_scaled)
    accuracy=accuracy_score(y_test,y_pred)

    report=classification_report(y_test,y_pred)
    matrix=confusion_matrix(y_test,y_pred)

    iris=load_iris()
    x=iris.data
    y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)
accuracy,report,matrix=knn_classifier(x_train,x_test,y_train,y_test)
print("Accuracy:",accuracy)
print("Clacification Report:")
print(report)
print("Confusion Matrix:")
print(matrix)
 
#Que.7.Write a python function to calculate the euclidean distance between two points.
#Ans.
import numpy as np

def euclidean_distance(point1,point2):
    return
np.linalg.norm(np.array(point1)-np.array(point2))

point1=(2,3,4)
point2=(5,6,7)
distance=euclidean_distance(point1,point2)
print("Euclidean distance:"distance)

#Que.8. Write a python function to calculate the Manhattan distance between two points.
#Ans.
import numpy as np

def manhattan_distance(point1,point2):
    return
np.sun(np.abs(np.array(point1)-np.array(point2)))

point1=(1,2,3)
point2=(4,5,6)
distance=manhattan_distance(point1,point2)
print("Manhattan distance:",distance)



