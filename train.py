import numpy as np
import p_v_network
import p_v_mcts_player
import time
import game_logic as gl
import random
import generate_self_play_data
import play
import time
import os
import tensorflow as tf
import copy
import logging

class TrainAndUpdate:
    def __init__(self):
        pass

    def train_and_update(self, plane_size=15, number_of_epoch=1, number_of_update_network=200, number_of_games=200, numbuer_of_samples_in_each_game=9, min_batch=100, max_simulation=3):
        '''
        :param number_of_epoch:
        :param number_of_update_network:
        :param number_of_games:
        :param numbuer_of_samples_in_each_game:
        :param min_batch: 需要是 number_of_games 乘以 numbuer_of_samples_in_each_game 的积的约数
        :return:
        '''
        p_v_network_new = p_v_network.P_V_Network()
        p_v_network_old = p_v_network.P_V_Network()

        path = "./network"
        if not os.path.exists(path):
            os.makedirs(path)

        for u in range(number_of_update_network):
            print("the %dth update" % (u))
            p_v_network_new.save(u)

            self_play_game = play.PlayLogic(plane_size=plane_size)
            data_generator = generate_self_play_data.GenerateSelfPlayData(self_play_game)

            root1 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=plane_size), father_edge=None, p_v_network=p_v_network_new)
            root2 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=plane_size), father_edge=None, p_v_network=p_v_network_new)
            player1 = p_v_mcts_player.MCTSPlayer(root=root1, p_v_network=p_v_network_new, max_simulation=max_simulation)
            player2 = p_v_mcts_player.MCTSPlayer(root=root2, p_v_network=p_v_network_new, max_simulation=max_simulation)

            plane_records, game_result_, y_ = data_generator.generate_self_play_data(player1, player2, number_of_games=number_of_games, numbuer_of_samples_in_each_game=numbuer_of_samples_in_each_game)

            for e in range(number_of_epoch):
                for i in range(int(number_of_games*numbuer_of_samples_in_each_game/min_batch)):
                    # min-batch 100， 由于只有1000个局面样本，所以只循环10次
                    batch = [plane_records[i * min_batch: (i + 1) * min_batch], game_result_[i * min_batch: (i + 1) * min_batch], y_[i * min_batch: (i + 1) * min_batch]]
                    if e % 10 == 0:
                        # loss = p_v_network_new.loss.eval(feed_dict={p_v_network_new.x_plane: batch[0], p_v_network_new.game_result: batch[1], p_v_network_new.y_: batch[2], p_v_network_new.is_training: False})
                        # p_v_network_new.sess.run([p_v_network_new.loss.eval], feed_dict={p_v_network_new.x_plane: batch[0], p_v_network_new.game_result: batch[1], p_v_network_new.y_: batch[2], p_v_network_new.is_training: False})
                        # print("step %d, loss %g" % (i, loss))
                        pass
                    p_v_network_new.sess.run([p_v_network_new.train_step], feed_dict={p_v_network_new.x_plane: batch[0], p_v_network_new.game_result: batch[1], p_v_network_new.y_: batch[2], p_v_network_new.is_training: True})

            if self.evaluate_new_neural_network(p_v_network_old, p_v_network_new, plane_size=plane_size, number_of_battles=5):
                print("old_network changed")
                p_v_network_old.restore(u)


    def evaluate_new_neural_network(self, p_v_network_old, p_v_network_new, number_of_battles=11, plane_size=15):

        root1 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=plane_size), father_edge=None,
                                         p_v_network=p_v_network_new)
        root2 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=plane_size), father_edge=None,
                                         p_v_network=p_v_network_old)
        player1 = p_v_mcts_player.MCTSPlayer(root=root1, p_v_network=p_v_network_new, max_simulation=50)
        player2 = p_v_mcts_player.MCTSPlayer(root=root2, p_v_network=p_v_network_old, max_simulation=50)

        new_pure_win = 0
        for i in range(number_of_battles):
            player1.refresh()
            player2.refresh()
            winner, plane_record, action_list, turn = play.PlayLogic().play(player1, player2)
            new_pure_win += winner
        if new_pure_win > 2:
            new_pure_win = 0
            for i in range(number_of_battles):
                player1.refresh()
                player2.refresh()
                winner, plane_record, action_list, turn = play.PlayLogic().play(player2, player1)
                new_pure_win += winner
            if new_pure_win > 2:
                return True
            else:
                return False
        else:
            return False
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    logging.basicConfig(filename='network/example.log', filemode="w", level=logging.DEBUG)
    train_and_update = TrainAndUpdate()
    train_and_update.train_and_update(number_of_epoch=1, number_of_update_network=200, number_of_games=16, numbuer_of_samples_in_each_game=9, min_batch=3, max_simulation=3)
