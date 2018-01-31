import game_logic as pl
import play
import time
import random
import numpy as np
import pandas as pd
import p_v_mcts_player

class GenerateSelfPlayData:
    def __init__(self, self_play_game_logic):
        self.self_play_game_logic = self_play_game_logic
        self.play_record = []

    def generate_self_play_data(self, number_of_games, player1, player2, method="MCTS"):
        '''
        :param number: 局数
        :param sess:
        :param p_v_neural_network:
        :return: 很多棋谱，对局结果（列向量），下一步棋的走法
        '''
        if method != "MCTS":
            # winner, plane_record, action_list, turn = self.self_play_game_logic.play(p_v_network, self.self_play_game_logic.get_current_move_by_neural_network)
            pass
        elif method == "MCTS":
            start_time = time.time()
            winner, plane_record, action_list, turn = self.self_play_game_logic.play(player1, player2)
            end_time = time.time()
            print(end_time - start_time)
        #winner不能直接喂给神经网络，如果当时是白棋走，得把winner反一下得到z再作为神经网络的z
        z, arr, y_ = self.select_and_generate_data(winner, plane_record, action_list, turn)
        plane_records = arr
        game_result = np.array([np.zeros(number_of_games, dtype=np.float32)])
        game_result[0][0] = z
        # 注意，上面改了for循环下面的也得改
        for i in range(1, number_of_games):
            if method != "MCTS":
                winner, plane_record, action_list, turn = self.self_play_game_logic.self_play(sess, p_v_neural_network, self.self_play_game_logic.get_current_move_by_neural_network)
            elif method == "MCTS":
                start_time = time.time()
                winner, plane_record, action_list, turn = self.self_play_game_logic.self_play_by_mcts(sess, p_v_neural_network, p_v_neural_network)
                end_time = time.time()
                print(end_time - start_time)
            z, arr, y__ = self.select_and_generate_data(winner, plane_record, action_list, turn)
            plane_records = np.concatenate((plane_records, arr))
            y_ = np.concatenate((y_, y__))
            game_result[0][i] = z

        return plane_records, game_result.T, y_


    def select_and_generate_data(self, winner, plane_record, action_list, turn):
        '''
        turn: 一共走了多少步，注意：不是步数加1，selfplay最后有turn+=1，但这个判断胜负的方法中，如果比赛结束，current_trun会减1
        注意: monte_carlo_tree_search 中也有类似的定义，这边如果要改，那边也得改
        '''
        size = plane_record.shape[0]
        situation = random.randint(0, turn-1)# 走了sitation步之后的局面
        if situation % 2 == 1: #走了奇数步，该白走
            z = -winner
        else:
            z = winner
        y_ = np.zeros(size**2, dtype=np.float32)
        arr1 = np.zeros((size, size), dtype=np.float32)
        arr2 = np.zeros((size, size), dtype=np.float32)
        if situation % 2 == 1: # 走了奇数步，该白棋走
            arr3 = np.zeros((size, size), dtype=np.float32)
            for i in range(size):
                for j in range(size):
                    if plane_record[i][j][3] <= situation:
                        if plane_record[i][j][2] == 1:
                            arr1[i][j] = 1
                        elif plane_record[i][j][2] == -1:
                            arr2[i][j] = 1
                    elif plane_record[i][j][3] == situation+1:
                        y_[i * size + j] = 1 #找出下一步棋在哪儿，返回一个类似 one_hot_key的向量
            arr = np.concatenate((np.array([arr1]), np.array([arr2])))
            arr = np.concatenate((arr, np.array([arr3])))

        else:
            arr3 = np.ones((size, size), dtype=np.float32)
            for i in range(size):
                for j in range(size):
                    if plane_record[i][j][3] <= situation:
                        if plane_record[i][j][2] == -1:
                            arr1[i][j] = 1
                        elif plane_record[i][j][2] == 1:
                            arr2[i][j] = 1
                    elif plane_record[i][j][3] == situation+1:
                        y_[i * size + j] = 1
            arr = np.concatenate((np.array([arr1]), np.array([arr2])))
            arr = np.concatenate((arr, np.array([arr3])))
        arr = arr.swapaxes(0, 1)
        arr = arr.swapaxes(1, 2)
        action_probability_distribution = action_list[situation]
        return z, np.array([arr]), action_probability_distribution

    def generate_mcts_self_play_data(self, number, sess, p_v_neural_network):
        '''
        大半复制于 generate_self_play_data
        '''
        winner, plane_record, turn = self.self_play_game_logic.self_play_by_mcts(sess, p_v_neural_network)
        # winner不能直接喂给神经网络，如果当时是白棋走，得把winner反一下得到z再作为神经网络的z
        z, arr, y_ = self.select_and_generate_data(winner, plane_record, turn)
        plane_records = arr
        game_result = np.array([np.zeros(number, dtype=np.float32)])
        game_result[0][0] = z
        # ！！！！！！！！！！！！！！注意，上面改了for循环下面的也得改！！！！！！！！！！！1
        for i in range(1, number):
            winner, plane_record, turn = self.self_play_game_logic.self_play_by_mcts(sess, p_v_neural_network)
            z, arr, y__ = self.select_and_generate_data(winner, plane_record, turn)
            plane_records = np.concatenate((plane_records, arr))
            y_ = np.concatenate((y_, y__))
            game_result[0][i] = z
        return plane_records, game_result.T, y_


if __name__ == "__main__":

    self_play_game = self_play_game_logic.GameLogic(plane_size=15)
    data_generator = GenerateSelfPlayData(self_play_game)
    arr, result, y_ = data_generator.generate_self_play_data(1)
    print(arr.shape, result.shape, y_.shape)
