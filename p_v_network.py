import numpy as np
import tensorflow as tf
import math
from tensorflow.python.training.moving_averages import assign_moving_average


class P_V_Network:
    def __init__(self, learning_rate=0.001, board_size=15):
        self.learning_rate = learning_rate
        self.board_size = board_size
        # total learning step
        self.learn_step_counter = 0
        # consist of [target_net, evaluate_net]
        self._build_net()
        p_v_network_params = tf.get_collection('p_v_network_params')

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        # ------------------ build p_v_network ------------------
        self.l2_reg = tf.contrib.layers.l2_regularizer(scale=0.001)
        self.x_plane = tf.placeholder(tf.float32, shape=[None, self.board_size, self.board_size, 3], name='x_plane')
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.board_size * self.board_size], name='y_')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.game_result = tf.placeholder(tf.float32, name="game_result")
        with tf.variable_scope('p_v_network'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['p_v_network_params', tf.GraphKeys.GLOBAL_VARIABLES]
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = self.conv_weight_variable([3, 3, 3, 32], name='w1', collections=c_names)
                b1 = self.bias_variable([32], name='b1', collections=c_names)
                l1 = tf.nn.relu(self.conv2d(self.x_plane, w1) + b1)
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l11'):
                w11 = self.conv_weight_variable([3, 3, 32, 64], name='w11', collections=c_names)
                b11 = self.bias_variable([64], name='b11', collections=c_names)
                l11 = self.batch_norm(self.conv2d(l1, w11) + b11, is_training=self.is_training)
                l11 = tf.nn.relu(l11)

            with tf.variable_scope('l12'):
                w12 = self.conv_weight_variable([3, 3, 64, 128], name='w12', collections=c_names)
                b12 = self.bias_variable([128], name='b12', collections=c_names)
                l12 = self.batch_norm(self.conv2d(l11, w12) + b12, is_training=self.is_training)
                l12 = tf.nn.relu(l12)

            with tf.variable_scope('l13'):
                w13 = self.conv_weight_variable([3, 3, 128, 256], name='w13', collections=c_names)
                b13 = self.bias_variable([256], name='b12', collections=c_names)
                l13 = self.batch_norm(self.conv2d(l12, w13) + b13, is_training=self.is_training)
                l13 = tf.nn.relu(l13)

            with tf.variable_scope('l14'):
                w14 = self.conv_weight_variable([3, 3, 256, 256], name='w14', collections=c_names)
                b14 = self.bias_variable([256], name='b14', collections=c_names)
                l14 = self.batch_norm(self.conv2d(l13, w14) + b14, is_training=self.is_training)
                l14 = tf.nn.relu(l14)

            with tf.variable_scope('l2'):
                w2 = self.conv_weight_variable([3, 3, 256, 64], name='w2', collections=c_names)
                b2 = self.bias_variable([64], name='b2', collections=c_names)
                l2 = tf.nn.relu(self.conv2d(l14, w2) + b2)
                l2_flat = tf.reshape(l2, [-1, self.board_size ** 2 * 64])

            with tf.variable_scope('l3_p'):
                w3_p = self.full_connected_weight_variable([self.board_size**2*64, 1024], name='w3_p', collections=c_names)
                b3_p = self.bias_variable([1024], name='b3_p', collections=c_names)
                l3_p = tf.nn.relu(tf.matmul(l2_flat, w3_p) + b3_p)

            with tf.variable_scope('l3_v'):
                w3_v = self.full_connected_weight_variable([self.board_size**2*64, 1024], name='w3_v', collections=c_names)
                b3_v = self.bias_variable([1024], name='b3_v', collections=c_names)
                l3_v = tf.nn.relu(tf.matmul(l2_flat, w3_v) + b3_v)

            with tf.variable_scope('l4_p'):
                w4_p = self.full_connected_weight_variable([1024, self.board_size**2], name='w4_p', collections=c_names)
                b4_p = self.bias_variable([self.board_size**2], name='b4_p', collections=c_names)
                self.y_p = tf.nn.relu(tf.matmul(l3_p, w4_p) + b4_p)

            with tf.variable_scope('l4_v'):
                w4_v = self.full_connected_weight_variable([1024, 256], name='w4_v', collections=c_names)
                b4_v = self.bias_variable([256], name='b4_v', collections=c_names)
                l4_v = tf.nn.relu(tf.matmul(l3_v, w4_v) + b4_v)

            with tf.variable_scope('l5_v'):
                w5_v = self.full_connected_weight_variable([256, 1], name='w5_v', collections=c_names)
                b5_v = self.bias_variable([1], name='b5_v', collections=c_names)
                self.y_v = tf.nn.relu(tf.matmul(l4_v, w5_v) + b5_v)

        with tf.variable_scope('loss'):
            self.reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.reg_term = tf.contrib.layers.apply_regularization(self.l2_reg, self.reg_variables)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_p) + (self.game_result - self.y_v) ** 2 + self.reg_term)

        with tf.variable_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        with tf.variable_scope('softmax'):
            self.prediction = tf.nn.softmax(self.y_p)

    def conv_weight_variable(self, shape, name, collections):
        w = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(0.001, 0.1), collections=collections) / math.sqrt(shape[0]*shape[1])
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)
        return w

    def full_connected_weight_variable(self, shape, name, collections):
        w = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(0.001, 0.1), collections=collections) / math.sqrt(shape[0])
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)
        return w

    def bias_variable(self, shape, name, collections):
        b = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.01), collections=collections)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, b)
        return b

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def batch_norm(self, x, is_training, eps=1e-05, decay=0.9, affine=True, name=None):
        with tf.variable_scope(name, default_name='BatchNorm2d'):
            params_shape = [x.shape[-1]]
            print(params_shape)
            print(tf.shape(x))
            moving_mean = tf.get_variable('mean', shape=params_shape, initializer=tf.zeros_initializer, trainable=False)
            moving_variance = tf.get_variable('variance', shape=params_shape, initializer=tf.ones_initializer, trainable=False)
            def mean_var_with_update():
                mean, variance = tf.nn.moments(x, list(range(len(x.shape)-1)), name='moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                              assign_moving_average(moving_variance, variance, decay)]):
                    return tf.identity(mean), tf.identity(variance)

            mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_variance))
            if affine:
                beta = tf.get_variable('beta', params_shape,
                                       initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape,
                                        initializer=tf.ones_initializer)
                x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            else:
                x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
            return x

    def get_action_probability(self, observation):
        action_probability = self.sess.run(self.prediction, feed_dict={self.x_plane: observation, self.is_training: False})
        return action_probability


if __name__ == '__main__':
    # import generate_self_play_data
    # import self_play_game_logic
    import time
    import os
    import numpy as np

    p_v_network = P_V_Network()

    saver = tf.train.Saver()
    path = "./p_v_network"
    if not os.path.exists(path):
        os.makedirs(path)
    plane1 = np.zeros((15, 15))
    plane2 = np.zeros((15, 15))
    legal_actions = np.ones((15, 15))
    arr = np.stack((plane1, plane2, legal_actions), axis=2)
    arr = arr[np.newaxis, :]
    print(p_v_network.get_action_probability(arr))
