#   The  goal  of  this  exercise  is  to  take  out-of-the-box  models  (logistic  regression,  SVM,  kernelSVM, neural network) and apply it to the Iris flower classification datasets.
#   Built intuition for model-to-problem fit and explain the strength and weakness of the used models.
#   The importanceof feature normalization should also be highlighted (what are the benefits of normalization, and what would happen if we don’t normalize the input features).
#   The performance of each model should be calculated on the test data and discussion should be done on why a given model is performing well compared to the other.

#   DISABLE UNNECESSARY WARNING
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
#----------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier



iris=load_iris()
X = iris.data               #   Independent
Y = iris.target             #   Dependent


#   Basic visualisation of the iris data. Helps with understanding.
plt.figure(figsize=(10, 6))
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color='b', label='0')
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='r', label='1')
#plt.show()


#   Normalize the data into the xNorm variable
min_max_scaler = preprocessing.MinMaxScaler()
xNorm = min_max_scaler.fit_transform(X)

#   Toggle normalization on/off
X = xNorm

#   Get the test and train data results (toggle normalization above)
xTrain, xTest, yTrain, yTest = train_test_split(X,Y,test_size=0.4)

#
##  Create and display the models
#

#   Logistic model
print("\n\n!!>                        Logistic Model              ")
logReg = LogisticRegression()
logReg.fit(xTrain, yTrain)
logRegPred = logReg.predict(xTest)
print( classification_report(yTest, logRegPred))
print(accuracy_score(yTest, logRegPred))

#   SVM model
print("\n\n!!>                          SVM Model               ")
svmMod = SVC(kernel='linear')
svmMod.fit(xTrain, yTrain)
svmModPred = svmMod.predict(xTest)
print(classification_report(yTest, svmModPred))
print(accuracy_score(yTest, svmModPred))

#   kernelSVM model
print("\n\n!!>                        kernelSVM Model            ")
kerMod = SVC(kernel='rbf')
kerMod.fit(xTrain, yTrain)
kerModPred = kerMod.predict(xTest)
print(classification_report(yTest, kerModPred))
print(accuracy_score(yTest, kerModPred))

#   Neural Network
print("\n\n!!>                         Neural Network               ")
nncMod = MLPClassifier(solver='lbfgs',max_iter=1000)
nncMod.fit(xTrain, yTrain)
nncModPred = nncMod.predict(xTest)
print(classification_report(yTest, nncModPred))
print(accuracy_score(yTest, nncModPred))
print("")



#       UNDER HERE IS CODE IM WORKING ON, ITS IRRELEVANT.



#dsf = pd.DataFrame(classification_report(yTest, nncModPred, digits=2,output_dict=True)).T
#dsf['support'] = dsf.support.apply(int)
#dsf.style.background_gradient(cmap='viridis',subset=pd.IndexSlice['0':'9', :'f1-score'])


#Use [:,0] here because xTest is 2Dimensional, and we need to squish it to 1D
#plt.scatter(xTest[:,0], yTest, c='r', cmap='rainbow')
#plt.show()

#------------------------------------------------------------
