import numpy as np
import pandas as pd

import warnings


def sigmoid(x):
		return 1 / (1 + np.exp(-x))

class logistic_regression:
	
	tol=1e-8 # convergence tolderance

	def __init__(self, X, y, lam, max_iter):
		self.X = X
		self.y = y
		self.lam = lam
		self.max_iter = max_iter

	def catch_singularity(f):
		'''
		Used to catch LinAlg Errors & throws a warning instead
		like dividing by zero
		'''

		def silencer(*args, **kwargs):
			try:
				return f(*args, **kwargs)
			except np.LinAlg.LinAlgError:
				warnings.warn('Algorithm terminated - singular Hessian!')
				return args[0]
			return silencer

	@catch_singularity
	def newton_step(self, curr):
		# create probability matrix, minimum 2 dimensions, transpose (flip it)
		p = np.array(sigmoid(self.X.dot(curr[:, 0])), ndmin=2).T
		# create weight matrix
		W = np.diag((p*(1-p))[:,0])
		# derive the hessian (2nd order)
		hessian = self.X.T.dot(W).dot(self.X)
		# derive the gradient (1st order)
		grad = self.X.T.dot(y-p)

		# regularization step (avoiding overfitting)
		if lamb:
			step, *_ = np.linalg.lstsq(hessian + lamb * np.eye(curr.shape[0]), grad)
		else:
			step, *_ = np.linalg.lstsq(hessian, grad)

		## update our
		beta = curr + step

		return beta

	@catch_singularity
	def alt_newton_step(self, curr):
		'''
		One naive step of Newton's Method
		'''
		p = np.array(sigmoid(self.X.dot(curr[:, 0])), ndmin=2).T
		W = np.diag((p*(1-p))[:,0])
		hessian = self.X.T.dot(W).dot(self.X)
		grad = self.X.T.dot(y-p)

		## regularization
		if lam:
			# Compute the inverse of a matrix
			step = np.dot(np.linalg.inv(hessian + self.lam*np.eye(curr.shape[0])), grad)
		else:
			step = np.dot(np.linalg.inv(hessian), grad)

		beta = curr + step

		return beta

	def check_coefs_convergence(self, beta_old, beta_new, iters):
		'''
		Checks whether the coefficients have converged in the l-infinity norm.
		Returns true if they have converged, False otherwise.
		'''
		# calculate the change in coefficients
		coef_change = np.abs(beta_old - beta_new)

		# if change hasn't reached the threshold & we have more iterations to go, keep training
		return not (np.any(coef_change > self.tol) & (self.iters < self.max_iter))

	def _logistic_regression(self):

		# initial coefficients (weight values), 2 copies, we'll update one
		beta_old = np.ones((len(self.X.columns), 1))
		beta = np.zeros((len(self.X.columns), 1))

		# num iterations we've done so far
		iter_count = 0
		# have we reached convergence?
		coefs_converged = False

		# if we haven't reached convergence... (training step)
		while not coefs_converged:
			# set the old coefficients to our current
			beta_old = beta
			# perform a single step of newton's optimization on our data, set our updated beta values
			beta = self.newton_step(beta)
			# increment the number of iterations
			iter_count += 1

			# check for convergence between our old and new beta values
			coefs_converged = self.check_coefs_convergence(sbeta_old, beta)

		return iter_count, beta

if __name__=='__main__':

	r = 0.95 # covariance between x and z
	n = 1000 # number of observations (size of dataset to generate) 
	sigma = 1 # variance of noise - how spread out is the data?


	beta_x, beta_z, beta_v = -4, .9, 1 # true beta coefficients
	var_x, var_z, var_v = 1, 1, 4 # variances of inputs

	x, z = np.random.multivariate_normal([0,0], [[var_x,r],[r,var_z]], n).T
	v = np.random.normal(0,var_v,n)**3

	A = pd.DataFrame({'x' : x, 'z' : z, 'v' : v})
	A['log_odds'] = sigmoid(A[['x','z','v']].dot([beta_x,beta_z,beta_v]) + sigma*np.random.normal(0,1,n))
	A['y'] = [np.random.binomial(1,p) for p in A.log_odds]

	# print(A)
	logReg = logistic_regression(A.iloc[:, :-1], A.iloc[:, -1], 1, 20)

	logReg._logistic_regression()






