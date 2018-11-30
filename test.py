import matplotlib.pyplot as plt
import numpy as np

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

a = node(False,2)
print(a.node)