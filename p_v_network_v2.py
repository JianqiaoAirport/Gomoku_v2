import numpy as np
import tensorflow as tf
import math
from tensorflow.python.training.moving_averages import assign_moving_average
import config


class P_V_Network:
    def __init__(self, name_scope="default", board_size=config.PLANE_SIZE):
        self.board_size = board_size
        # total learning step
        self.learn_step_counter = 0
        self.name_scope = name_scope
        # consist of [target_net, evaluate_net]
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_net()
            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            self.merged = tf.summary.merge_all()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 自适应
        self.sess = tf.Session(config=config, graph=self.graph)

        self.train_writer = tf.summary.FileWriter("logs/" + name_scope + '/train/', self.sess.graph)
        writer = tf.summary.FileWriter("logs/" + name_scope + "/", self.sess.graph)
        writer.add_graph(self.sess.graph)

        self.sess.run(init)

    def _build_net(self):
        # ------------------ build p_v_network ------------------
        self.learning_rate = tf.placeholder(tf.float32)
        self.x_plane = tf.placeholder(tf.float32, shape=[None, self.board_size, self.board_size, 3], name='x_plane')
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.board_size * self.board_size], name='y_')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.game_result = tf.placeholder(tf.float32, name="game_result")

        self.conv1 = tf.layers.conv2d(inputs=self.x_plane, name="conv1",
                                      filters=32, kernel_size=[3, 3],
                                      padding="same", activation=None)
        self.conv1_bn = self.batch_norm(self.conv1, is_training=self.is_training)
        self.conv1_act = tf.nn.relu(self.conv1_bn)
        conv1_w = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel')[0]
        conv1_b = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/bias')[0]
        tf.summary.histogram("conv1_w", conv1_w)
        tf.summary.histogram("conv1_b", conv1_b)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1_act, name="conv2", filters=64,
                                      kernel_size=[3, 3], padding="same",
                                      activation=None)
        self.conv2_bn = self.batch_norm(self.conv2, is_training=self.is_training)
        self.conv2_act = tf.nn.relu(self.conv2_bn)
        conv2_w = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv2/kernel')[0]
        conv2_b = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv2/bias')[0]
        tf.summary.histogram("conv2_w", conv2_w)
        tf.summary.histogram("conv2_b", conv2_b)
        self.conv3 = tf.layers.conv2d(inputs=self.conv2_act, name="conv3", filters=128,
                                      kernel_size=[3, 3], padding="same",
                                      activation=None)
        self.conv3_bn = self.batch_norm(self.conv3, is_training=self.is_training)
        self.conv3_act = tf.nn.relu(self.conv3_bn)
        conv3_w = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv3/kernel')[0]
        conv3_b = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv3/bias')[0]
        tf.summary.histogram("conv3_w", conv3_w)
        tf.summary.histogram("conv3_b", conv3_b)

        # 3-1 Action Networks
        self.action_conv = tf.layers.conv2d(inputs=self.conv3_act, filters=4,
                                            kernel_size=[1, 1], padding="same",
                                            activation=None, name="convp")
        self.action_conv_bn = self.batch_norm(self.action_conv, is_training=self.is_training)
        self.action_conv_act = tf.nn.relu(self.action_conv_bn)
        convp_w = tf.get_collection(tf.GraphKeys.VARIABLES, 'convp/kernel')[0]
        convp_b = tf.get_collection(tf.GraphKeys.VARIABLES, 'convp/bias')[0]
        tf.summary.histogram("convp_w", convp_w)
        tf.summary.histogram("convp_b", convp_b)
        # Flatten the tensor
        self.action_conv_flat = tf.reshape(
            self.action_conv_act, [-1, 4 * self.board_size * self.board_size])
        # 3-2 Full connected layer, the output is the log probability of moves
        # on each slot on the board
        self.y_p = tf.layers.dense(inputs=self.action_conv_flat,
                                   units=self.board_size * self.board_size,
                                   activation=tf.nn.log_softmax, name="dense_p")
        self.prediction = tf.exp(self.y_p)

        # 4 Evaluation Networks
        self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3_act, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                activation=None, name="convv")
        self.evaluation_conv_bn = self.batch_norm(self.evaluation_conv, is_training=self.is_training)
        self.evaluation_conv_act = tf.nn.relu(self.evaluation_conv_bn)
        convv_w = tf.get_collection(tf.GraphKeys.VARIABLES, 'convv/kernel')[0]
        convv_b = tf.get_collection(tf.GraphKeys.VARIABLES, 'convv/bias')[0]
        tf.summary.histogram("convv_w", convv_w)
        tf.summary.histogram("convv_b", convv_b)
        self.evaluation_conv_flat = tf.reshape(
            self.evaluation_conv_act, [-1, 2 * self.board_size * self.board_size])
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=64, activation=tf.nn.relu, name="dense1_v")
        # output the score of evaluation on current state
        self.y_v = tf.layers.dense(inputs=self.evaluation_fc1,
                                   units=1, activation=tf.nn.tanh, name="dense2_v")



        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        # 2. Predictions: the array containing the evaluation score of each state
        # which is self.evaluation_fc2
        # 3-1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.game_result,
                                                       self.y_v)
        tf.summary.scalar("value_loss", self.value_loss)
        # 3-2. Policy Loss function
        self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.y_, self.y_p), 1)))
        tf.summary.scalar("policy_loss", self.policy_loss)
        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = config.L2_REG
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + self.policy_loss + l2_penalty
        tf.summary.scalar("loss", self.loss)
        # Define the optimizer we use for training

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        tf.summary.scalar("learning_rate", self.learning_rate)

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.exp(self.y_p) * self.y_p, 1)))
        tf.summary.scalar("entropy", self.entropy)



    # def conv_weight_variable(self, shape, name, collections):
    #     w = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(0.001, 0.1), collections=collections) / math.sqrt(shape[0]*shape[1])
    #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)
    #     return w
    #
    # def full_connected_weight_variable(self, shape, name, collections):
    #     w = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(0.001, 0.1), collections=collections) / math.sqrt(shape[0])
    #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)
    #     return w
    #
    # def bias_variable(self, shape, name, collections):
    #     b = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.01), collections=collections)
    #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, b)
    #     return b
    #
    # def conv2d(self, x, W):
    #     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #
    # def max_pool_2x2(self, x):
    #     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

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
        action_probability = self.sess.run(self.y_p, feed_dict={self.x_plane: observation, self.is_training: False})
        return action_probability

    def save(self, u, path="network"):
        self.saver.save(self.sess, path + '/model-' + str(u) + '.ckpt')

    def restore(self, u, path="network"):
        self.saver.restore(self.sess, path + '/model-' + str(u) + '.ckpt')


if __name__ == '__main__':
    # import generate_self_play_data
    # import self_play_game_logic
    import time
    import os
    import numpy as np

    p_v_network = P_V_Network()

    plane1 = np.zeros((config.PLANE_SIZE, config.PLANE_SIZE))
    plane2 = np.zeros((config.PLANE_SIZE, config.PLANE_SIZE))
    legal_actions = np.ones((config.PLANE_SIZE, config.PLANE_SIZE))
    arr = np.stack((plane1, plane2, legal_actions), axis=2)
    arr = arr[np.newaxis, :]
    print(p_v_network.get_action_probability(arr))
