import game_logic as gl
import play
import time
import random
import numpy as np
import pandas as pd
import p_v_mcts_player
import logging

class GenerateSelfPlayData:
    def __init__(self, self_play_game_logic):
        self.self_play_game_logic = self_play_game_logic
        self.play_record = []

    def generate_self_play_data(self, player1, player2, number_of_games=2, numbuer_of_samples_in_each_game=1):
        '''
        :param number: 局数
        :param sess:
        :param p_v_neural_network:
        :return: 很多棋谱，对局结果（列向量），下一步棋的走法
        '''
        start_time = time.time()
        winner, plane_record, action_list, turn = self.self_play_game_logic.play(player1, player2)
        end_time = time.time()
        logging.info(end_time - start_time)
        print(end_time - start_time)
        # winner不能直接喂给神经网络，如果当时是白棋走，得把winner反一下得到z再作为神经网络的z
        z, arr, y_ = self.select_and_generate_data(winner, plane_record, action_list, turn)
        plane_records = arr
        game_result = z
        for k in range(1, numbuer_of_samples_in_each_game):
            z, arr, y__ = self.select_and_generate_data(winner, plane_record, action_list, turn)
            plane_records = np.concatenate((plane_records, arr))
            y_ = np.concatenate((y_, y__))
            game_result = np.concatenate((game_result, z))
        # 注意，上面改了for循环下面的也得改
        for i in range(1, number_of_games):
            start_time = time.time()
            player1.refresh()
            player2.refresh()
            winner, plane_record, action_list, turn = self.self_play_game_logic.play(player1, player2)
            end_time = time.time()
            logging.info(end_time - start_time)
            print(end_time - start_time)
            for k in range(numbuer_of_samples_in_each_game):
                z, arr, y__ = self.select_and_generate_data(winner, plane_record, action_list, turn)
                plane_records = np.concatenate((plane_records, arr))
                y_ = np.concatenate((y_, y__))
                game_result = np.concatenate((game_result, z))
        # q.put((plane_records, game_result.T, y_))
        return plane_records, game_result, y_

    def select_and_generate_data(self, winner, plane_record, action_list, turn):
        '''
        turn: 一共走了多少步，注意：不是步数加1，selfplay最后有turn+=1，但这个判断胜负的方法中，如果比赛结束，current_trun会减1
        注意: monte_carlo_tree_search 中也有类似的定义，这边如果要改，那边也得改
        '''
        size = plane_record.shape[1]
        situation = random.randint(0, turn-1)   # 走了situation步之后的局面
        if situation % 2 == 1:  # 最后一步的回合计数是奇数，说明黑棋刚走完，下一步该白棋走
            z = -winner
        else:
            z = winner
        result = np.array([[z]])
        y_ = np.zeros(size**2, dtype=np.float32)
        arr1 = np.zeros((size, size), dtype=np.float32)
        arr2 = np.zeros((size, size), dtype=np.float32)
        if situation % 2 == 1:  # 最后一步的回合计数是奇数，说明黑棋刚走完，下一步该白棋走
            arr3 = np.zeros((size, size), dtype=np.float32)
            for i in range(size):
                for j in range(size):
                    if plane_record[1][i][j] <= situation:
                        if plane_record[0][i][j] == -1:
                            arr1[i][j] = 1
                        elif plane_record[0][i][j] == 1:
                            arr2[i][j] = 1
                    elif plane_record[1][i][j] == situation+1:
                        y_[i * size + j] = 1  # 找出下一步棋在哪儿，返回一个类似 one_hot_key的向量，注意，后来不用这个东西了

        else:
            arr3 = np.ones((size, size), dtype=np.float32)
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

        board = arr.copy()
        board = board.swapaxes(0, 1)
        board = board.swapaxes(1, 2)
        board = np.array([board])

        action_probability_distribution = action_list[situation]
        action_matrix = action_probability_distribution.reshape((size, size))

        for i in range(3):  # 旋转90°，一共3次
            arr_data_augment_board1 = arr.copy()
            arr_data_augment_board2 = arr.copy()  # 旋转90°之前，先左右翻一下
            arr_data_augment_act1 = action_matrix.copy()
            arr_data_augment_act2 = action_matrix.copy()
            arr_data_augment_act1 = np.rot90(m=arr_data_augment_act1, k=i + 1)
            arr_data_augment_act2 = np.fliplr(arr_data_augment_act2)
            arr_data_augment_act2 = np.rot90(m=arr_data_augment_act2, k=i + 1)

            for j in range(2):  #分别对arr1，2进行操作
                arr_data_augment_board1[j] = np.rot90(m=arr_data_augment_board1[j], k=i+1)

                arr_data_augment_board2[j] = np.fliplr(arr_data_augment_board2[j])
                arr_data_augment_board2[j] = np.rot90(m=arr_data_augment_board2[j], k=i+1)

            arr_data_augment_board1 = arr_data_augment_board1.swapaxes(0, 1)
            arr_data_augment_board1 = arr_data_augment_board1.swapaxes(1, 2)
            arr_data_augment_board1 = np.array([arr_data_augment_board1])
            board = np.concatenate((board, arr_data_augment_board1))

            arr_data_augment_board2 = arr_data_augment_board2.swapaxes(0, 1)
            arr_data_augment_board2 = arr_data_augment_board2.swapaxes(1, 2)
            arr_data_augment_board2 = np.array([arr_data_augment_board2])
            board = np.concatenate((board, arr_data_augment_board2))

            action_probability_distribution = np.concatenate((action_probability_distribution, np.array([arr_data_augment_act1.reshape(size ** 2)])))
            action_probability_distribution = np.concatenate((action_probability_distribution, np.array([arr_data_augment_act2.reshape(size ** 2)])))

            result = np.concatenate((result, np.array([[z]])))
            result = np.concatenate((result, np.array([[z]])))

        arr_data_augment_board = arr.copy()
        arr_data_augment_act = action_matrix.copy()
        for j in range(2):
            arr_data_augment_board[j] = np.fliplr(arr_data_augment_board[j])
        arr_data_augment_board = arr_data_augment_board.swapaxes(0, 1)
        arr_data_augment_board = arr_data_augment_board.swapaxes(1, 2)
        arr_data_augment_act = np.fliplr(arr_data_augment_act)
        board = np.concatenate((board, np.array([arr_data_augment_board])))
        action_probability_distribution = np.concatenate((action_probability_distribution, np.array([arr_data_augment_act.reshape(size ** 2)])))
        result = np.concatenate((result, np.array([[z]])))

        return result, board, action_probability_distribution



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

    arr, result, y_ = data_generator.generate_self_play_data(player1, player2, number_of_games=2, numbuer_of_samples_in_each_game=8)
    print(arr.shape, result.shape, y_.shape)



