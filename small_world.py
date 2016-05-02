import numpy as np
import math
import random

# Generates the underlying lattice structure of the graph
def getAdjNodes(i, j, lim, k):
	listOfNodes = []
	for x in range(i - k, i + k + 1):
		for y in range(j - k, j + k + 1):
			# check if x, y in range
			if x >= 0 and x < lim and y >= 0 and y < lim:
				# check if x, y within k of i, j
				if (abs(i - x) + abs(j - y)) <= k and (abs(i - x) + abs(j - y)) > 0:
					listOfNodes.append((x,y))
	return listOfNodes

# Calculates lattice distance where all nodes within k are connected
def dist (i, j, x, y, k):
	distance = float(abs(i - x) + abs(j - y))/k
	return math.ceil(distance)

# Generates a long range edge
def getLongEdge(i, j, lim, k, alpha):
	totalDist = 0
	for x in range(lim):
		for y in range(lim):
			if x != i or y != j:
				totalDist += math.pow(dist(i, j, x, y, k), alpha*(-1))
	probabilityDict = {}
	for x in range(lim):
		for y in range(lim):
				if x != i or y != j:
					probabilityDict[(x,y)] = math.pow(dist(i, j, x, y, k), alpha*(-1)) / totalDist
				else:
					probabilityDict[(x,y)] = 0
	xCoord = 0
	yCoord = 0
	p = random.random()
	while p >= 0.0:
		if xCoord == lim - 1:
			if yCoord == lim - 1:
				return (xCoord, yCoord)
			else:
				xCoord = 0
				yCoord += 1
		else:
			xCoord += 1
		p -= probabilityDict[(xCoord, yCoord)]
	return (xCoord, yCoord)


def genKleinberg(alpha, k, n):
	graph = {}
	for i in range(int(np.sqrt(n))):
		for j in range(int(np.sqrt(n))):
			# connect (i, j) to rest of graph
			graph[(i,j)] = getAdjNodes(i, j, int(np.sqrt(n)), k)
			# draw one long range edge
			graph[(i,j)].append(getLongEdge(i, j, int(np.sqrt(n)), k, alpha))
	new_graph = {}
	for i,j in graph:
		new_index = int(i + np.sqrt(n) * j)
		new_graph[new_index] = []
		for ni, nj in graph[(i,j)]:
			new_graph[new_index].append(int(ni + np.sqrt(n) * nj))
	return new_graph
