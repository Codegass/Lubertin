#####
# Author : Codegass

# TODO: the map's generation is too random, till now is still a list
# we need to change it to a n by m dimention matrix.
# the matrix should be sorted, the loction of each node should have
# relationship with the node's parameters.
#####
import numpy as np
import matplotlib.pyplot as plt
import math
import random


class ERROR(Exception):
    '''
    customized error class to raise the input type error in the functions.
    '''
    def __init__(self, error):
        Exception.__init__(self)
        self.error = error


class node:
    '''Class for the nodes inside the soms.
    The node is just the vector contents the n-dim parameter.

    BE CAREFUL: the node is defualtly init as int
                the upper bounding is limited as 256
    '''
    def __init__(self, isFloat, shape):
        try:
            if type(shape) == int or type(shape) == tuple:
                pass
            else:
                raise ERROR('The shape of the node definition number must be int or a tuple of int.')
        except ERROR as e:
            print('\x1b[6;30;42m' + e.error + '\x1b[0m')

        self.highest = 256
        self.node = np.random.randint(self.highest, size=shape)
        if isFloat:
            self.node = self.node.astype(np.float64)

    def neighbourhood(self, radius, node):
        '''
        caculate the nodes inside the radius of the neighbourhood of the BMU
        if the node is the one inside, return it.
        else we return 0.
        '''
        dist = np.linalg.norm(self.node - node.node)  # self.node is for the node call this method , node.node is the node you want know if it is inside.
        if dist <= radius:
            return node
        else:
            # print('this node is not inside.')
            return 0


class SOMs(object):
    """
    Class for self orgnized maps.

    radius is for the best matching unit to calculate the nodes inside
    data is for the training data you want to cluster
    nodeQuantity is for how many nodes you want.
    round is for the trainning times
    shape is for shape of the node.

    nodes in the dataset shoulf have the same shape as som's nodes shape, and shold be numpy array.
    """
    def __init__(self, radius, data, nodeQuantity, rounds,
                 learningRate=0.1, isFloat=False, *shape):
        self.map = []  # contents all the nodes, should be a list

        for _ in range(nodeQuantity):
            self.map.append(node(isFloat, shape))  # initialize the nodes.

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
        BMUindex = 0  # the index number of the best matching unit in the map list
        BMUindexDist = 0
        for i in range(self.quantity):
            node = self.map[i]
            dist = np.linalg.norm(node.node - randomChosenData)
            if dist >= BMUindexDist:
                BMUindex = i
        self.BMU = self.map[BMUindex]
        # return self.map[BMUindex]

    def neighbours(self):
        for i in range(self.quantity):
            # print(i)
            node = self.BMU.neighbourhood(self.radius, self.map[i])
            if node == 0:
                pass
            else:
                self.neighbournodes.append(node)
                self.index.append(i)

    def radiusDamping(self, t):
        '''
        t means the current time-step(iteration of the loop)
        if you are using this on the picturn data, you can define the radius like
        radius = max( picWidth, picHeight ) / 2
        '''
        dampingParameter = self.round/math.log(self.originalradius)
        self.radius = self.originalradius * math.exp(-t/dampingParameter)

    def learningRateDamping(self, t):
        '''
        t means the current time-step(iteration of the loop)
        '''
        dampingParameter = self.round/math.log(self.originalLearningRate)
        self.learningRate = self.originalLearningRate * math.exp(-t/dampingParameter)

    def Adjusting(self, node, randomChosenData):
        '''
        node is the class level input. not the self.node level.

        Node(t+1)=Node(t) + Theta(t)*LearningRate(t)*(Inputing(t)-Node(t))
        '''
        dist = np.linalg.norm(self.BMU.node - node.node)
        Theta = math.exp(-(dist**2)/(2*(self.radius)**2))  # Theta is the amount of learning should fade over distance similar to the Gaussian decay.
        node.node = node.node + Theta * self.learningRate * (randomChosenData - node.node)
        return node

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
        for j in range(len(self.map)):
            self.resultmap.append(self.map[j].node)
        return self.resultmap


if __name__ == '__main__':
    # a = node(True, 2, 3)
    # b = node(True, 2, 3)
    # print(a.node)
    # print(b.node)
    # print(a.neighbourhood(400, b))

    # Training inputs for RGBcolors
    colors = np.array(
         [[0, 0, 0],
          [0, 0, 1],
          [0, 0, 139],
          [135, 206, 235],
          [122, 197, 205],
          [132, 112, 255],
          [0, 255, 0],
          [255, 0, 0],
          [0., 255, 255],
          [255, 0., 255],
          [255, 255, 0.],
          [255, 255, 255],
          [105, 105, 105],
          [190, 190, 190],
          [211, 211, 211]])
    color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
         'cadetblue', 'LightSlateBlue', 'green', 'red',
         'cyan', 'violet', 'yellow', 'white',
         'dimGrey', 'mediumgrey', 'lightgrey']

    som = SOMs(200, colors, 400, 200, 0.3, False, 1, 3)
    som.train()

    plt.imshow(som.map,interpolation='gaussian')
    plt.show()