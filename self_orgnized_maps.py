import numpy as np
import matplotlib.pyplot as plt
import math


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
    def __init__(self, isFloat=False, *shape):
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

    data 里的每一个点应该和node的shape一致, 并且一定是numpy array
    """
    def __init__(self, radius, data, nodeQuantity, round,
                 learningRate=0.1, isFloat=False, *shape):
        self.map = []  # contents all the nodes, should be a list

        for _ in range(nodeQuantity):
            self.map.append(node(isFloat, shape))  # initialize the nodes.

        self.neighbournodes = []
        self.data = data  # the training data set
        self.originalradius = radius  # the origninal radius
        self.radius = radius  # the rsdius for the BMU to caculate the neighbours.
        self.quantity = nodeQuantity
        self.originalLearningRate = learningRate
        self.learningRate = learningRate
        self.BMU = None

    def BMU(self, randomChosenData):
        BMUindex = 0  # the index number of the best matching unit in the map list
        BMUindexDist = 0
        for i in range(self.quantity):
            node = self.map[i]
            dist = np.linalg.norm(node.node - randomChosenData)
            if dist >= BMUindexDist:
                BMUindex = i
        self.BMU = self.map[BMUindex]
        return self.map[BMUindex]

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
        node is the class level input. not the self.node level
        Node(t+1)=Node(t) + Theta(t)*LearningRate(t)*(Inputing(t)-Node(t))
        '''
        dist = np.linalg.norm(self.BMU.node - node.node)
        Theta = math.exp(-(dist**2)/(2*(self.radius)**2))  # Theta is the amount of learning should fade over distance similar to the Gaussian decay.
        node.node = node.node + Theta * self.learningRate * (randomChosenData - node.node)

    def train(self):
        ''' 
        train the model with sitted parameter. 
        '''


if __name__ == '__main__':
    # a = node(True, 2, 3)
    # b = node(True, 2, 3)
    # print(a.node)
    # print(b.node)
    # print(a.neighbourhood(400, b))
    c = SOMs(radius=400, data, 10, 40)
