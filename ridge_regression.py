import numpy as np
import pandas as pd

def ridge_regression(x_train, y_train, lam):
	'''
	http://hyperanalytic.net/ridge-regression
	'''
    
    X = np.array(x_train)
    ones = np.ones(len(X))
    X = np.column_stack((ones, X))
    y = np.array(y_train)
    
    Xt = np.transpose(X)
    lambda_identity = lam * np.identity(len(Xt))
    theInverse = np.linalg.inv(np.dot(Xt, X)) * lambda_identity
    w = np.dot(np.dot(theInverse, Xt), y)
    
    return w, lambda x: dot(w, x)
    
 
if __name__ == '__main__':
	x = np.random.random_sample((10,))
	y = x * np.random.random_sample((10, ))

	print(ridge_regression(x, y, 0.5))