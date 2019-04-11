from __future__ import print_function, division
from builtins import range, input

import numpy as np 
import pandas as pd 

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt 
from scipy.stats import norm

B = 200
N = 20
X = np.random.randn(N)

print("sample mean of X:", X.mean())

individual_estimates = np.empty(B)
for b in range(B):
	sample = np.random.choice(X, size=N)
	individual_estimates[b] = sample.mean()

bmean = individual_estimates.mean()
bstd = individual_estimates.std()
lower = bmean + norm.ppf(0.025) * bstd
upper = bmean + norm.ppf(0.975) * bstd

lower2 = X.mean() + norm.ppf(0.025) * X.std() / np.sqrt(N)
upper2 = X.mean() + norm.ppf(0.975) * X.std() / np.sqrt(N)

print("bootstrap mean of X:", bmean)

plt.hist(individual_estimates, bins=20)
plt.axvline(x=lower, ls='--', color='green', label=f'lower bound for 95% lower CI (Bootstrap)')
plt.axvline(x=upper, ls='--', color='green', label=f'upper bound for 95% upper CI (Bootstrap)')
plt.axvline(x=lower2, ls='--', color='red', label=f'lower2 bound for 95% lower2 CI')
plt.axvline(x=upper2, ls='--', color='red', label=f'upper2 bound for 95% upper2 CI')
plt.legend()
plt.show()
