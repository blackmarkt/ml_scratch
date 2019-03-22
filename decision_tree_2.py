from __future__ import print_function

import numpy as np 
import pandas as pd 

training_data = [
    [1, 'Bull', 'Buy'],
    [1, 'Bull', 'Buy'],
    [1, 'Neutral', 'Buy'],
    [0, 'Neutral', 'Hold'],
    [0, 'Bear', 'Sell'],
    [2, 'Bear', 'Sell'],
]

header = ['lt_regime', 'st_regime', 'direction']

def unique_vals(rows, col):
	return set([row[col] for row in rows])

# ####################
# # Demo
# print(unique_vals(training_data, 0))
# print(unique_vals(training_data, 1))
# print(unique_vals(training_data, 2))
# ####################

def class_counts(rows):
	# Counts the number of each type of example in a dataset
	counts = {}
	for row in rows:
		# in our dataset format, the label is always the last column
		label = row[-1]
		if label not in counts:
			counts[label] = 0
		counts[label] += 1
	return counts


# ####################
# # Demo
# print(class_counts(training_data))
# ####################

def is_numeric(value):
	return isinstance(value, int) or isinstance(value, float)

####################
# Demo
# print(is_numeric(10))
# print(is_numeric('Buy'))
####################

class Question:
	'''
	A Question is used to partition a dataset.

	This class just records a 'column number' (i.e. 0 for lt_regime) & a 'column value'
	(i.e. 'Bull'). The 'match' method is used to compared the feature value in an example
	to the feature value stored in the question. See demo below:
	'''

	def __init__(self, column, value):
		self.column = column
		self.value = value

	def match(self, example):
		# Compare he feature value in an example to the feature value in this ?
		val = example[self.column]
		if is_numeric(val):
			return val >= self.value
		else:
			return val == self.value

	def __repr__(self):
		# This is a helper method to print the ? in a readable format
		condition = "=="
		if is_numeric(self.value):
			condition = ">="
		return "Is %s %s %s?" % (header[self.column], condition, str(self.value))


####################
# Demo
# q = Question(2, 'Bear')
# example = training_data[0]
# print(q.match(example))
####################

def partition(rows, question):
	"""
	Partitions a dataset

	For each row in the dataset, check if it matches the question. If so, add it to 'true rows'
	otherwise add it to 'false rows'
	"""
	true_rows, false_rows = [], []
	for row in rows:
		if question.match(row):
			true_rows.append(row)
		else:
			false_rows.append(row)

	return true_rows, false_rows

####################
# Demo
# true_rows, false_rows = partition(training_data, Question(2, 'Buy'))
# print(true_rows, false_rows)
####################

def gini(rows):
	'''
	Calculate the Gini Impurity for a list of rows

	There are a few different ways to do this, I thought this one was the most concise.
	'''

	counts = class_counts(rows)
	# print(counts)
	impurity = 1
	for lbl in counts:
		prob_of_lbl = counts[lbl] / float(len(rows))
		impurity -= prob_of_lbl**2
		# print(prob_of_lbl, impurity)
	return impurity

####################
# Demo
# lots_of_mixing = [['Apple'],
#                   ['Orange'],
#                   ['Grape'],
#                   ['Grapefruit'],
#                   ['Blueberry']]
# print(gini(lots_of_mixing))
####################

def info_gain(left, right, current_uncertainty):
	'''
	Information Gain

	The uncertainty of the starting node, minus the weighted impurity of 2 child nodes
	'''
	p = float(len(left)) / (len(left) + len(right))
	return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

# ####################
# # Demo
# print('Gini:')
# current_uncertainty = gini(training_data)
# print(f'current_uncertainty: {current_uncertainty}')
# ####################

# ####################
# # Demo
# print('IG (Sell):')
# true_rows, false_rows = partition(training_data, Question(2, 'Sell'))
# print(f'IG (Sell): {info_gain(true_rows, false_rows, current_uncertainty)}')

# ####################
# # Demo
# print('IG (Hold):')
# true_rows, false_rows = partition(training_data, Question(2, 'Hold'))
# print(f'IG (Hold): {info_gain(true_rows, false_rows, current_uncertainty)}')

# ####################
# # Demo
# print('IG (Buy):')
# true_rows, false_rows = partition(training_data, Question(2, 'Buy'))
# print(f'IG (Buy): {info_gain(true_rows, false_rows, current_uncertainty)}')

# print(true_rows)
# print(false_rows)

def find_best_split(rows):
	best_gain = 0 # keep track of the best IG
	best_question = None # keep track of the feature / value that produced it
	current_uncertainty = gini(rows)
	n_features = len(rows[0]) - 1 # number of columns

	for col in range(n_features): # for each value

		values = set([row[col] for row in rows]) # unique values in the column
		print(values)

		for val in values: # for each value

			question = Question(col, val)
			print(question)

			# # try splitting the dataset
			true_rows, false_rows = partition(rows, question)
			print(f'true rows: {true_rows}')
			print(f'false rows: {false_rows}')

			# Skip this split if it doesn't divide the dataset.
			if len(true_rows) == 0 or len(false_rows) == 0:
				continue

			# Calculate the IG from this split
			gain = info_gain(true_rows, false_rows, current_uncertainty)
			print(gain)

			# You actually can use '>' instead of '>=' heere
			# but I wanted the tre to look a certain way for our
			# toy dataset.
			if gain >= best_gain:
				best_gain, best_question = gain, question

	return best_gain, best_question

# ####################
# # Demo
# best_gain, best_question = find_best_split(training_data)
# print(f'BEST GAIN: {best_gain}')
# print(f'BEST QUESTION: {best_question}')

class Leaf:
	'''
	A Leaf node classifies data.

	This holds a dictionary class(i.e. "Apple") -> # of times
	it appears in the rows from the training data that reach this leaf.
	'''

	def __init__(self, rows):
		self.predictions = class_counts(rows)

class Decision_Node:
	'''
	A decision node asks a question.

	This holds a reference to the question & to the 2 child nodes.
	'''

	def __init__(self, question, true_branch, false_branch):
		self.question = question
		self.true_branch = true_branch
		self.false_branch = false_branch

def build_tree(rows):
	'''
	Builds the tree.

	Rules of Recursion:
	1) Believe that it works.
	2) Start by checking for the base case (no further info gain)
	3) Prepare giant stack traces.
	'''
	# try partitioning the dataset on each of the unique attribute,
	# calculate the IG
	# & return the ? that produces the highest gain
	gain, question = find_best_split(rows)
	print(f'GAIN: {gain}')
	print(f'QUESTION: {question}')

	# Base case: no further info gain
	# Since we ask no further questions,
	# we'll return a leaf
	if gain == 0:
		return Leaf(rows)

	# If we reach here, we have found a useful feature / value to partition on.
	true_rows, false_rows = partition(rows, question)
	print(true_rows, false_rows)

	# Recursively build the true branch
	true_branch = build_tree(true_rows)
	print(f'TRUE BRANCH: {true_branch}')

	# Recursively build the false branch
	false_branch = build_tree(false_rows)
	print(f'FALSE BRANCH: {false_branch}')

	return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):

	# Base case: we've reached a leaf
	if isinstance(node, Leaf):
		print(spacing + "Predict" , node.predictions)
		return

	print(spacing + str(node.question))

	print(spacing + '--> True:')
	print_tree(node.true_branch, spacing + "  ")
	print(spacing + '--> False:')
	print_tree(node.false_branch, spacing + "  ")

# ####################
# # Demo
my_tree = build_tree(training_data)
# print_tree(my_tree)

def classify(row, node):

	if isinstance(node, Leaf):
		return node.predictions

	if node.question.match(row):
		return classify(row, node.true_branch)
	else:
		return classify(row, node.false_branch)

# ####################
# # Demo
# print(classify(training_data[0], my_tree))

def print_leaf(counts):
	total = sum(counts.values()) * 1.0
	probs = {}
	for lbl in counts.keys():
		probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
	return probs

# ####################
# # Demo
# print(print_leaf(classify(training_data[4], my_tree)))

testing_data = [
    [0, 'Neutral', 'Sold'],
    [0, 'Neutral', 'Hold'],
    [2, 'Neutral', 'Hold'],
    [0, 'Neutral', 'Sold'],
    [2, 'Bear', 'Sell'],
    [0, 'Bear', 'Sell'],
]

for row in testing_data:
	print("Actual: %s. Predicted %s" % (row[-1], print_leaf(classify(row, my_tree))))

