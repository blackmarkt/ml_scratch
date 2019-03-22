import numpy as np 
import pandas as pd 

def linear_regression(x, y):

	x = np.array(x)
	ones = np.ones(len(x))
	x = np.column_stack((ones, x))
	y = np.array(y)


	xT = np.transpose(x)
	xTx_inv = np.linalg.inv(np.dot(xT, x))
	beta = np.dot(np.dot(xTx_inv, xT), y)

	return beta

if __name__ == '__main__':

	x_train = np.random.random_sample((10, 2))
	y_train = np.random.random_sample((10, ))

	print(linear_regression(x_train, y_train))
