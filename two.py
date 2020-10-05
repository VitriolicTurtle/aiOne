#The importance of cost/loss function is inevitable in any machine learning problem.
#We haveseen that the aim of any machine learning algorithm is to tune the parameters of the models in
#such a way that the cost function yields the minimum value.
#For the linear regression model, we know the parameters are β0 and β1.
#The aim of this exercise is to plot the loss function over a 2-dimensional grid against the model parameters β0 and β1.
#You may choose any regressionproblem (exp,  house price prediction,  stock market,  etc.),  and can use any publicly availabledataset or define your own data.
#Note:  While training the model, at each iteration, we will getdifferent values of loss function. You may plot the values for a given number of iterations (100,500, 1000 etc)


#   Needed for reading csv
import pandas as pd
#   Needed for np data array
import numpy as np
#   Needed for plotting lines on the graph
import matplotlib.pyplot as plt
#   Needed for Linear regression Model object.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

#   Read CSV file
path = 'ONE\gold_price_annual0.csv'
df = pd.read_csv(path)
X = df[['Date']].values.reshape(-1,1)
Y = df['Price'].values.reshape(-1,1)

#   Create training and test set.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)


#   Creates model object
lrModel = LinearRegression()
lrModel.fit(x_train, y_train)
yPred = lrModel.predict(x_test)

#   The mean squared error using premade sklearn function
print('Mean squared error: %.2f'% mean_squared_error(y_test, yPred))

#   Plotting graph for better udnerstanding.
#plt.scatter(x_test, y_test, color = "m", marker = "o", s = 15)
#plt.plot(x_test, yPred, color = "g")
#plt.show()


#
#       THIS IS THE ACTUAL ANSWER TO THE TASK.
#
#       Calcuilate the cost over time of cost function.
#       Cost function = Mean squared error
iterations = 100
def linear_regression(x, y, epochs=iterations, lRate=0.00000025):
    mCurr = 0
    bCurr = 0
    costList = []
    N = float(len(y))
    for i in range(epochs):
        yCurr = (mCurr * x) + bCurr
        cost = (1/N) * sum([data**2 for data in (y-yCurr)])
        costList.append(cost)
        mGradient = -(2/N) * sum(x * (y - yCurr))
        bGradient = -(2/N) * sum(y - yCurr)
        mCurr = mCurr - (lRate * mGradient)
        bCurr = bCurr - (lRate * bGradient)
        print("m {}, b {}, cost {} iteration {}".format(mCurr,bCurr,cost, i))
    return mCurr, bCurr, cost, costList

m_current, b_current, cost, costList = linear_regression(X, Y)

plt.plot(list(range(iterations)), costList, '-r')
plt.show()
















#------------------------------------------------------------------------------
