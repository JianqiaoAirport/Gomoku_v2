import time
import numpy as np
import math
import copy
import random


class MCTSPlayer:
    """
    AI player, use Monte Carlo Tree Search with UCB
    """

    def __init__(self, root, p_v_network, max_simulation=5):
        '''

        :param root: 初始局面
        :param c_puct: 这个值越大，越倾向于 exploration，否则，exploitation
        :param max_simulation:
        '''
        self.root = root
        self.p_v_network = p_v_network
        self.max_simulation = max_simulation

    def select(self, node):
        '''
        :param node: 当前结点
        :return: 最大边以及对应的子结点。如果对应子结点为空，不出意外的话会进入expand程序
        '''
        max_edge_index = 0
        max_edge = node.child_edges[0]
        max_edge_index_list = [0]
        max_edge_list = [max_edge]

        for index, edge in enumerate(node.child_edges):
            if edge.P > 0:
                max_edge_index = index
                max_edge = node.child_edges[index]
                max_edge_index_list = [index]
                max_edge_list = [max_edge]

        for index, edge in enumerate(node.child_edges[1:]):
            if edge.P == 0:
                continue
            if edge.calculate_upper_confidence_bound() >= max_edge.calculate_upper_confidence_bound():
                if edge.calculate_upper_confidence_bound() > max_edge.calculate_upper_confidence_bound():
                    max_edge_index = index + 1
                    max_edge = edge
                    max_edge_index_list = [max_edge_index]
                    max_edge_list = [edge]
                else:
                    max_edge_index_list.append(index+1)
                    max_edge_list.append(edge)
        max_edge_index = random.choice(max_edge_index_list)
        next_node = node.child_edges[max_edge_index].child_node
        return max_edge_index, next_node

    def expand_and_evaluate(self, node, max_edge_index):
        gl = copy.deepcopy(node.state)
        x = int(max_edge_index / gl.plane_size)
        y = max_edge_index % gl.plane_size
        gl.play(x, y)
        result = gl.game_result_fast_version(x, y)
        if result != 2:
            new_node = MCTSNode(gl, node.child_edges[max_edge_index], self.p_v_network, is_terminal=True)
            if result != gl.current_player:  # 注意current_player在调用play方法后会切换
                new_node.value = 1
            else:
                new_node.value = -1
            pass
        else:
            new_node = MCTSNode(gl, node.child_edges[max_edge_index], self.p_v_network)
            # new_node.value = new_node.get_current_value_by_neural_network(self.sess, self.p_v_network)

        node.child_edges[max_edge_index].child_node = new_node
        return new_node

    def back_up(self, node, c_puct=4.90):
        value = node.value
        depth = 0  # 计算深度，如果只有根节点，则深度为0
        node_for_count = node

        while node_for_count.father_edge != None:
            depth += 1
            node_for_count = node_for_count.father_edge.father_node

        while depth > 0:
            '''
            注意：以下一段在 MCTSEdge 的构造方法中有出现，如若需更改，那边也得改。
            '''
            node.father_edge.N += 1
            #  如果深度为奇数，说明当前节点的值函数是对对手的局面的评价，对手局面越好，自己越差，累加W的时候应当加上负号
            if depth % 2 == 1:
                node.father_edge.W += -value
            else:
                node.father_edge.W += value
            node.father_edge.Q = node.father_edge.W / node.father_edge.N
            node = node.father_edge.father_node
            depth -= 1

    def simulate(self, root):
        for i in range(self.max_simulation):
            max_edge_index, next_node = self.select(root)
            node = root
            while next_node != None and next_node.is_terminal == False:
                node = next_node
                max_edge_index, next_node = self.select(node)
            if next_node == None:
                # 此时到达了空空如也的叶节点
                new_node = self.expand_and_evaluate(node, max_edge_index)
            elif next_node.is_terminal == True:
                # 说明到的是棋局结束的节点，则不继续往下展开了
                new_node = next_node
            self.back_up(new_node)
        return root

    def get_action_and_probability(self):
        root = self.simulate(self.root)
        max_index = 0
        max_edge = root.child_edges[0]
        for index, edge in enumerate(root.child_edges[1:]):
            if edge.N > max_edge.N:
                max_index = index + 1
                max_edge = edge
        x = int(max_index / root.state.plane_size)
        y = max_index % root.state.plane_size
        π = [edge.N/edge.get_sum_of_Nb() for edge in root.child_edges]
        next_node = root.child_edges[max_index].child_node
        root.child_edges[max_index] = None
        next_node.father_edge = None
        self.root = next_node
        return x, y, np.array([π])

    def get_opponents_action(self, x, y):
        '''
        :param action: 对方的动作的index
        :return: root
        '''
        action = int(x*self.root.state.plane_size + y)
        next_node = self.root.child_edges[action].child_node
        if next_node == None:
            next_node = self.expand_and_evaluate(self.root, action)
            self.back_up(next_node)
        else:
            self.root.child_edges[action] = None
            next_node.father_edge = None
        self.root = next_node

class MCTSNode:
    def __init__(self, state, father_edge, p_v_network, value=0, is_terminal=False):
        '''
        :param state: 当前局面, 是一个plane_logic
        :param value: 当前局面的值
        :param father_node: 父节点
        注意，初始化时需要将当前局面喂给神经网络，得到下一步动作的概率分布，用一个列向量表示，列向量的index代表动作。
        列向量除以棋盘宽度的整数部分代表第几列，余数代表第几行（因为棋盘用dataframe表示，df[x][y]中，x代表第几列，y代表第几行）。
        '''
        self.state = state
        self.value = value
        self.father_edge = father_edge
        self.is_terminal = is_terminal
        self.child_edges = []

        action_probability_distribution, value = self.get_current_action_probability_distribution_and_value_by_neural_network(p_v_network)
        self.value = value
        # action_probability_distribution = self.get_current_action_probability_distribution_by_neural_network(sess, p_v_network)
        for index, Psa in enumerate(action_probability_distribution):
            edge = MCTSEdge(self, index, P=Psa)
            self.child_edges.append(edge)


    def get_current_action_probability_distribution_by_neural_network(self, p_v_network):
        '''
        注意， self_play_game_logic 中也有类似的定义，这边如果要改，那边也得改
        '''
        input_data = self.generate_input_data_for_neural_network()
        y_prediction = p_v_network.sess.run(p_v_network.y_prediction, feed_dict={p_v_network.x_plane: input_data, p_v_network.is_training: False})
        y_prediction = np.array(y_prediction)
        plane = self.state.plane
        for i in range(self.state.plane_size):
            for j in range(self.state.plane_size):
                if plane[0][i][j] != 0:
                    y_prediction[0][i * plane.shape[0] + j] = 0
        total = y_prediction[0].sum()
        action_probability_distribution = y_prediction[0] / total
        return action_probability_distribution

    def get_current_value_by_neural_network(self, p_v_network):
        input_data = self.generate_input_data_for_neural_network()
        value = p_v_network.sess.run(p_v_network.y_value, feed_dict={p_v_network.x_plane: input_data, p_v_network.is_training: False})
        value = float(value)
        return value

    def get_current_action_probability_distribution_and_value_by_neural_network(self, p_v_network):
        input_data = self.generate_input_data_for_neural_network()
        result = p_v_network.sess.run([p_v_network.prediction, p_v_network.y_v], feed_dict={p_v_network.x_plane: input_data, p_v_network.is_training: False})
        y_prediction = np.array(result[0])

        for i in range(self.state.plane_size):
            for j in range(self.state.plane_size):
                if not self.state.play_is_legal(i, j):
                    y_prediction[0][i * self.state.plane_size + j] = 0

        total = y_prediction[0].sum() + 0.000001
        action_probability_distribution = y_prediction[0] / total
        # print("action_probability.shape: ", action_probability_distribution.shape)
        return action_probability_distribution, result[1]

    def generate_input_data_for_neural_network(self):
        '''
        注意， generate_self_play_data中也有类似的定义，这边如果要改，那边也得改
        :return:
        '''
        size = self.state.plane_size
        plane_record = self.state.plane
        arr1 = np.zeros((size, size), dtype=np.float32)
        arr2 = np.zeros((size, size), dtype=np.float32)
        if (self.state.current_turn-1) % 2 == 1:
            arr3 = np.zeros((size, size), dtype=np.float32)
            for i in range(size):
                for j in range(size):
                    if plane_record[1][i][j] <= (self.state.current_turn-1):
                        if plane_record[0][i][j] == 1:
                            arr1[i][j] = 1
                        elif plane_record[0][i][j] == -1:
                            arr2[i][j] = 1

            arr = np.concatenate((np.array([arr1]), np.array([arr2])))
            arr = np.concatenate((arr, np.array([arr3])))

        else:
            arr3 = np.ones((size, size), dtype=np.float32)
            for i in range(size):
                for j in range(size):
                    if plane_record[1][i][j] <= (self.state.current_turn-1):
                        if plane_record[0][i][j] == -1:
                            arr1[i][j] = 1
                        elif plane_record[0][i][j] == 1:
                            arr2[i][j] = 1

            arr = np.concatenate((np.array([arr1]), np.array([arr2])))
            arr = np.concatenate((arr, np.array([arr3])))
        arr = arr.swapaxes(0, 1)
        arr = arr.swapaxes(1, 2)
        return np.array([arr])

class MCTSEdge:
    def __init__(self, father_node, action, child_node=None, N=0, W=0, Q=0, P=0, c_puct=4.90):
        '''
        注意：以下一段在 back_up 方法中有出现，如若需更改，那边也得改。
        :param c_puct:
        '''

        self.father_node = father_node
        self.child_node = child_node
        self.action = action
        self.N = N
        self.W = W
        self.Q = Q
        self.P = P
        self.c_puct = c_puct

    def calculate_upper_confidence_bound(self):
        sum_of_Nb = 0.0
        for edge in self.father_node.child_edges:
            sum_of_Nb += edge.N
        U = self.c_puct * self.P * math.sqrt(sum_of_Nb) / (1 + self.N)
        upper_confidence_bound = self.Q + U
        return upper_confidence_bound

    def get_sum_of_Nb(self):
        sum_of_Nb = 0.0
        for edge in self.father_node.child_edges:
            sum_of_Nb += edge.N
        return sum_of_Nb

if __name__ == '__main__':
    import generate_self_play_data
    import self_play_game_logic
    import p_v_network
    import os
    import numpy as np
    import tensorflow as tf
    import game_logic

    p_v_network = p_v_network.P_V_NeuralNetwork()

    saver = tf.train.Saver()
    path = "./p_v_network"
    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # self_player = self_play_game_logic.GameLogic(plane_size=15)
        # data_generator = generate_self_play_data.GenerateSelfPlayData(self_play_game_logic=self_player)
        # plane_records, game_result_, y_ = data_generator.generate_self_play_data(10, sess=sess, neural_network=p_v_network)
        # data = [plane_records, game_result_, y_]

        pl = game_logic.PlaneLogic()
        root = MCTSNode(pl, None, p_v_network)
        mcts = MCTSPlayer(root, p_v_network=p_v_network)
        root = mcts.simulate(root, max_simulation=5)
        pass
