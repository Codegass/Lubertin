import tensorflow as tf
from tensorflow.contrib import rnn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# batch_size = 32
# input_size = 4
# state_size = 32 #hidden layer dimention
# time_steps = 10

# X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# Y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
# cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
# outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# 超参数
learning_rate = 0.03
batch_size = tf.placeholder(tf.float32, [])
input_size = 4
hidden_size = 
LSTM_layer_num = 
class_num = 2

X = tf.placeholder(tf.float32, [None, ]) 
Y = tf.placeholder(tf.float32, [None, class_num])

