####

#TODO: STILL HAVE BUGS 
# the circle's init still has bugs.
# now the neighbours is already changed to 3 


# Author : Codegass
# This algo is based on the typical self_orgnized_map algo, but I add
# some optimization on it to solve the Traveling salesman problem.
# If you are interested in the TSP, you can check the description on
# wiki [here](https://en.wikipedia.org/wiki/Travelling_salesman_problem).

# Also if you are interested in the Self orgnized map algo, you can check
# my blog [article](http://chwei.xyz) about it.

# This exmaple is inspired by the repo
# [Solving the Traveling Salesman Problem using Self-Organizing Maps](https://github.com/DiegoVicen/som-tsp)

#### IMPORTANT ####
# This code is based on self_orgnized_maps.py in this repo, I import some
# class inside the self_orgnized_maps.py.

# The city data comes from http://www.math.uwaterloo.ca/tsp/world/countries.html

####

from self_orgnized_maps import ERROR, node, SOMs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import argparse
import random


nodenn = None


class node(node):
    '''Class for the nodes inside the soms.
    The node is just the vector contents the n-dim parameter.

    here we initialize all the points as a circle.
    '''
    def __init__(self, shape):
        try:
            if type(shape) == int or type(shape) == tuple:
                pass
            else:
                raise ERROR('The shape of the node definition number must be int or a tuple of int.')
        except ERROR as e:
            print('\x1b[6;30;42m' + e.error + '\x1b[0m')

        self.node = np.random.random(size=shape)


class SOMs(SOMs):
    def __init__(self, radius, data, nodeQuantity, rounds,
                 learningRate=0.1, *shape):
        self.map = []  # contents all the nodes, should be a list

        angle = 360/nodeQuantity

        for i in range(nodeQuantity):
            round_node = node(shape)
            round_node.node = np.array([[0.5 - math.cos(i*angle), math.sin(i*angle)]])
            self.map.append(round_node)  # initialize the nodes.

        self.neighbournodes = []
        self.data = data  # the training data set which is a np.array

        try:
            if type(data) == np.ndarray:
                self.data = self.data.tolist()
            elif type(data) == list:
                pass
            else:
                raise ERROR('The data set should be a list')

            if type(data[0]) == np.ndarray:
                pass
            else:
                raise ERROR('The nodes in the data set should be numpy array.')
        except ERROR as e:
            print('\x1b[6;30;42m' + e.error + '\x1b[0m')

        self.originalradius = radius  # the origninal radius
        self.radius = radius  # the rsdius for the BMU to caculate the neighbours.
        self.quantity = nodeQuantity
        self.originalLearningRate = learningRate
        self.learningRate = learningRate
        self.BMU = None
        self.round = rounds
        self.index = []
        self.resultmap = []

    def BMUCalculate(self, randomChosenData):
        global BMUindex
        BMUindex = 0  # the index number of the best matching unit in the map list
        BMUindexDist = 0
        for i in range(self.quantity):
            node = self.map[i]
            dist = np.linalg.norm(node.node - randomChosenData)
            if dist >= BMUindexDist:
                BMUindex = i
        self.BMU = self.map[BMUindex]

    def neighbours(self):
        if BMUindex == 0:
            self.neighbournodes = [self.map[nodenn-1], self.BMU, self.map[1]]
            self.index = [nodenn-1, 0, 1]
        elif BMUindex == nodenn-1:
            self.neighbournodes = [self.map[nodenn-2], self.BMU, self.map[0]]
            self.index = [ nodenn-2, nodenn-1, 0]
        else:
            self.neighbournodes = [self.map[BMUindex-1], self.BMU, self.map[BMUindex+1]]
            self.index = [BMUindex-1, BMUindex, BMUindex+1]

    def train(self):
        '''
        train the model with sitted parameter.
        4 STEPS:
        1. randomly choose a node for the data set
        2. calculate the best match unit
        3. calculate the neighbourhood of the best match unit
        4. adjust all the units inside the neighbourhood.
        5. Again from step one.
        '''
        for t in range(self.round):
            print('\nEpoch %d / %d' % (t+1, self.round))
            print('Choosing a node from the data set randomly.')
            target = random.sample(self.data, 1)
            self.BMUCalculate(target)
            print('Best match unit finded.')
            self.radiusDamping(t)
            self.neighbours()
            print('All neighbours finded. This BMU has %d neighbours include itself.' % len(self.neighbournodes))
            if len(self.neighbournodes) == 0:
                pass
            else:
                self.learningRateDamping(t)
                for nodeIndex in range(len(self.neighbournodes)):
                    node_next = self.Adjusting(self.neighbournodes[nodeIndex], target)
                    self.map[self.index[nodeIndex]] = node_next
                    print('.', end='')
        print('\nTraining ended.')



    def result(self):

        # print(self.map[0].node)
        for j in range(len(self.map)):
            self.resultmap.append(self.map[j].node[0])

        self.resultmap = np.array(self.resultmap)  # convert the arrays to np.array in order to matching the plt requirements
        return self.resultmap

def read_tsp(filename):
    """
    Read a file in .tsp format into a pandas DataFrame
    The .tsp files can be found in the TSPLIB project. Currently, the library
    only considers the possibility of a 2D map.
    """
    with open(filename) as f:
        node_coord_start = None
        lines = f.readlines()
        dimension = None

        # Obtain the information about the .tsp
        i = 0
        while not dimension or not node_coord_start:
            line = lines[i]
            if line.startswith('DIMENSION :'):
                dimension = int(line.split()[-1])
            if line.startswith('NODE_COORD_SECTION'):
                node_coord_start = i
            i = i+1

        print('Problem with {} cities read.'.format(dimension))

        global nodenn
        nodenn = dimension

        f.seek(0)

        # Read a data frame out of the file descriptor
        cities = pd.read_csv(
            f,
            skiprows=node_coord_start + 1,
            sep=' ',
            names=['city', 'y', 'x'],
            dtype={'city': str, 'x': np.float64, 'y': np.float64},
            header=None,
            nrows=dimension
        )

        return cities


def normalize(points):
    """
    points should be pd.df of x,y row.
    Return the normalized version of a given vector of points.
    For a given array of n-dimensions, normalize each dimension by removing the
    initial offset and normalizing the points in a proportional interval: [0,1]
    on y, maintining the original ratio on x.
    """
    ratio = (points.x.max() - points.x.min()) / (points.y.max() - points.y.min()), 1
    ratio = np.array(ratio) / max(ratio)
    norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    return norm.apply(lambda p: ratio * p, axis=1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--file', type=str, default='qa194.tsp', required=False, help='input video file name')

    args = parser.parse_args()

    filename = 'self_orgnized_maps/asset/qa194.tsp' + args.file
    cities = read_tsp(filename)
    cities_map = normalize(cities[['x', 'y']])

    i = 0
    j = 0
    # print(nodenn)
    # print(cities_map)

    X_ori = []
    Y_ori = []
    for i in range(nodenn):
        X_ori.append(cities_map['x'][i])
        Y_ori.append(cities_map['y'][i])

    # print('X_ori is', X_ori)

    data = cities_map.values

    som = SOMs(1.2, data, nodenn, 10000, 0.1, 1, 2)
    som.train()
    result = som.result()

    # print(result)

    X = []
    Y = []

    for j in range(len(result)):

        X.append(result[j][0])
        Y.append(result[j][1])

    fig = plt.figure(figsize=(5, 5))
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_aspect('equal', adjustable='datalim')
    plt.axis('off')

    axis.scatter(X_ori, Y_ori, color='red', s=1)
    axis.plot(X, Y, color='purple',linewidth=1)

    plt.show()
