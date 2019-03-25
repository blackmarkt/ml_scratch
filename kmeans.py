from copy import deepcopy

import os
cwd = os.getcwd()
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def manhattan_distance(a, b, metric='cityblock'):
	return cdist(a, b, metric='cityblock')

def euclidean_distance(a, b, ax=1):
	return np.linalg.norm(a - b, axis=ax)

class kmeans:

	def __init__(self, data, k):
		self.data = data
		self.k = k

	def initialize_centroids(self):
		c_x = np.random.randint(0, np.max(self.data)-20, size=self.k)
		c_y = np.random.randint(0, np.max(self.data)-20, size=self.k)
		return np.array(list(zip(c_x, c_y)), dtype=np.float32)

	def kmeans(self, distance_metric='euclidean'):

		centroids = self.initialize_centroids()

		# store old centroid values post update
		centroids_old = np.zeros(centroids.shape)
		# store cluster labels
		clusters = np.zeros(len(self.data))

		# distance between updated & old centroid
		if distance_metric == 'euclidean':
			dist = euclidean_distance
		elif distance_metric == 'manhattan':
			dist = manhattan_distance
		else:
			print(f'Error: Only euclidean or manhattan distance is available at this time')

		# initiaize the distance between the centroids and the empty centroids array
		error_distance = dist(centroids, centroids_old, None)

		while error_distance != 0:
			for i in range(len(self.data)):
				distances = dist(self.data[i], centroids)
				cluster = np.argmin(distances)
				clusters[i] = cluster

			centroids_old = deepcopy(centroids)

			for i in range(self.k):
				points = [self.data[j] for j in range(len(self.data)) if clusters[j] == i]
				centroids[i] = np.mean(points, axis=0)
			error_distance = dist(centroids, centroids_old, None)

		return clusters, centroids

if __name__ == '__main__':

	import matplotlib.pyplot as plt

	path_xclara = os.path.join(cwd, "data/xclara.csv")
	data_xclara = pd.read_csv(path_xclara)
	X = np.array(list(zip(data_xclara.V1.values, data_xclara.V2.values)))
	kmeans_test = kmeans(X, 3)
	clusters, C = kmeans_test.kmeans('euclidean')
	# C = vkmeans_test.initialize_centroids()

	colors = ['r', 'g', 'b', 'y', 'c', 'm']
	fig, ax = plt.subplots()
	for i in range(3):
	        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
	        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
	ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
	plt.show()

