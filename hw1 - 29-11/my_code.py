import numpy as np
from test_linreg_univariate_ import plotData1D, plotRegLine1D
from linreg import LinearRegression


'''
    # ----------------------------------------
    # 2.1 Visualizing the Data
    # ----------------------------------------
'''

filePath = "/home/minh/Desktop/CIS419/Assignment1/hw1_skeleton/data/univariateData.dat"
file = open(filePath, 'r')
allData = np.loadtxt(file, delimiter = ',')

X = np.matrix(allData[:,: -1])
y = np.matrix((allData[:, -1 ])).T
n , d = X.shape
plotData1D (X, y)


'''
    # ----------------------------------------
    # 2.2 Implementation
    # ----------------------------------------
'''

X = np.c_[np.ones((n, 1)), X] 
lr_model = LinearRegression (alpha = 0.01, n_iter = 1500 )
lr_model.fit(X, y)
plotRegLine1D (lr_model, X, y) 



'''
    # ----------------------------------------
    # DONE!!!!!!
    # ----------------------------------------
'''





