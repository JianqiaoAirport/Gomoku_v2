import game_logic as gl
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

    def generate_self_play_data(self, player1, player2, number_of_games=2, numbuer_of_samples_in_each_game=8):
        '''
        :param number: 局数
        :param sess:
        :param p_v_neural_network:
        :return: 很多棋谱，对局结果（列向量），下一步棋的走法
        '''
        start_time = time.time()
        winner, plane_record, action_list, turn = self.self_play_game_logic.play(player1, player2)
        end_time = time.time()
        print(end_time - start_time)
        game_result = np.array([np.zeros(number_of_games * numbuer_of_samples_in_each_game, dtype=np.float32)])
        # winner不能直接喂给神经网络，如果当时是白棋走，得把winner反一下得到z再作为神经网络的z
        z, arr, y_ = self.select_and_generate_data(winner, plane_record, action_list, turn)
        plane_records = arr
        game_result[0][0] = z
        for k in range(1, numbuer_of_samples_in_each_game):
            z, arr, y__ = self.select_and_generate_data(winner, plane_record, action_list, turn)
            plane_records = np.concatenate((plane_records, arr))
            y_ = np.concatenate((y_, y__))
            game_result[0][0+k] = z
        # 注意，上面改了for循环下面的也得改
        for i in range(1, number_of_games):
            start_time = time.time()
            player1.refresh()
            player2.refresh()
            winner, plane_record, action_list, turn = self.self_play_game_logic.play(player1, player2)
            end_time = time.time()
            print(end_time - start_time)
            for k in range(numbuer_of_samples_in_each_game):
                z, arr, y__ = self.select_and_generate_data(winner, plane_record, action_list, turn)
                plane_records = np.concatenate((plane_records, arr))
                y_ = np.concatenate((y_, y__))
                game_result[0][i+k] = z
        return plane_records, game_result.T, y_

    def select_and_generate_data(self, winner, plane_record, action_list, turn):
        '''
        turn: 一共走了多少步，注意：不是步数加1，selfplay最后有turn+=1，但这个判断胜负的方法中，如果比赛结束，current_trun会减1
        注意: monte_carlo_tree_search 中也有类似的定义，这边如果要改，那边也得改
        '''
        size = plane_record.shape[1]
        situation = random.randint(0, turn-1)   # 走了situation步之后的局面
        if situation % 2 == 1:  # 走了奇数步，该黑走
            z = winner
        else:
            z = -winner
        y_ = np.zeros(size**2, dtype=np.float32)
        arr1 = np.zeros((size, size), dtype=np.float32)
        arr2 = np.zeros((size, size), dtype=np.float32)
        if situation % 2 == 1:  # 走了奇数步，该黑棋走
            arr3 = np.ones((size, size), dtype=np.float32)
            for i in range(size):
                for j in range(size):
                    if plane_record[1][i][j] <= situation:
                        if plane_record[0][i][j] == 1:
                            arr1[i][j] = 1
                        elif plane_record[0][i][j] == -1:
                            arr2[i][j] = 1
                    elif plane_record[1][i][j] == situation+1:
                        y_[i * size + j] = 1  # 找出下一步棋在哪儿，返回一个类似 one_hot_key的向量，注意，后来不用这个东西了
            arr = np.concatenate((np.array([arr1]), np.array([arr2])))
            arr = np.concatenate((arr, np.array([arr3])))

        else:
            arr3 = np.zeros((size, size), dtype=np.float32)
            for i in range(size):
                for j in range(size):
                    if plane_record[1][i][j] <= situation:
                        if plane_record[0][i][j] == 1:
                            arr1[i][j] = 1
                        elif plane_record[0][i][j] == -1:
                            arr2[i][j] = 1
                    elif plane_record[1][i][j] == situation+1:
                        y_[i * size + j] = 1
            arr = np.concatenate((np.array([arr1]), np.array([arr2])))
            arr = np.concatenate((arr, np.array([arr3])))
        arr = arr.swapaxes(0, 1)
        arr = arr.swapaxes(1, 2)
        action_probability_distribution = action_list[situation]
        return z, np.array([arr]), action_probability_distribution



if __name__ == "__main__":
    import p_v_network
    import play

    self_play_game = play.PlayLogic(plane_size=15)
    data_generator = GenerateSelfPlayData(self_play_game)


    p_v_network = p_v_network.P_V_Network()
    root1 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=15), father_edge=None, p_v_network=p_v_network)
    root2 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=15), father_edge=None, p_v_network=p_v_network)
    player1 = p_v_mcts_player.MCTSPlayer(root=root1, p_v_network=p_v_network, max_simulation=5)
    player2 = p_v_mcts_player.MCTSPlayer(root=root2, p_v_network=p_v_network, max_simulation=5)

    arr, result, y_ = data_generator.generate_self_play_data(2, 8, player1, player2)
    print(arr.shape, result.shape, y_.shape)
