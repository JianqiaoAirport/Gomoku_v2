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
import shutil
import tensorflow as tf
import copy
import logging
import config

class TrainAndUpdate:
    def __init__(self):
        pass

    def train_and_update(self, plane_size=15, number_of_epoch=1, number_of_update_network=200, number_of_games=200, number_of_samples_in_each_game=9, min_batch=100):
        '''
        :param number_of_epoch:
        :param number_of_update_network:
        :param number_of_games:
        :param numbuer_of_samples_in_each_game:
        :param min_batch: 需要是 number_of_games 乘以 numbuer_of_samples_in_each_game 的积的约数
        :return:
        '''
        p_v_network_new = p_v_network.P_V_Network()
        p_v_network_new.save(0)
        p_v_network_old = p_v_network.P_V_Network()
        current_best_model = 0
        path = "./network"
        if not os.path.exists(path):
            os.makedirs(path)

        for u in range(1, number_of_update_network + 1):
            print("the %dth update" % u)

            # get data
            while True:
                data_loaded = False
                time.sleep(10)
                generator_files = os.listdir("data")
                for generator in generator_files:
                    data_batches = os.listdir("data/"+generator)

                    for item in data_batches:
                        if not item.startswith("self_play_data"):
                            data_batches.remove(item)

                    if len(data_batches) != 0:
                        for i in range(len(data_batches)):
                            data_list = os.listdir("data/"+generator+"/"+data_batches[i])
                            data_is_used = False
                            for item in data_list:
                                if item.endswith("data_is_used"):
                                    data_is_used = True
                                    break
                            if (not data_is_used) and len(data_list) >= 3:
                                plane_records = np.load("data/"+generator+"/"+data_batches[i]+"/plane_records.npy")
                                game_result_ = np.load("data/"+generator+"/"+data_batches[i]+"/game_result.npy")
                                y_ = np.load("data/"+generator+"/"+data_batches[i]+"/y_.npy")
                                # shutil.rmtree("data/"+generator+"/"+data_batches[i])
                                os.mkdir("data/"+generator+"/"+data_batches[i]+"/data_is_used")
                                data_loaded = True
                                break
                        if data_loaded:
                            break
                if data_loaded:
                    break
            # get data end

            # train
            for e in range(number_of_epoch):
                for i in range(int(number_of_games*number_of_samples_in_each_game*8/min_batch)):
                    # min-batch 100， 由于只有1000个局面样本，所以只循环10次
                    batch = [plane_records[i * min_batch: (i + 1) * min_batch], game_result_[i * min_batch: (i + 1) * min_batch], y_[i * min_batch: (i + 1) * min_batch]]
                    if e % 10 == 0:
                        # loss = p_v_network_new.loss.eval(feed_dict={p_v_network_new.x_plane: batch[0], p_v_network_new.game_result: batch[1], p_v_network_new.y_: batch[2], p_v_network_new.is_training: False})
                        # p_v_network_new.sess.run([p_v_network_new.loss.eval], feed_dict={p_v_network_new.x_plane: batch[0], p_v_network_new.game_result: batch[1], p_v_network_new.y_: batch[2], p_v_network_new.is_training: False})
                        # print("step %d, loss %g" % (i, loss))
                        pass
                    p_v_network_new.sess.run([p_v_network_new.train_step], feed_dict={p_v_network_new.x_plane: batch[0], p_v_network_new.game_result: batch[1], p_v_network_new.y_: batch[2], p_v_network_new.is_training: True})

            if self.evaluate_new_neural_network(p_v_network_old, p_v_network_new, plane_size=plane_size, number_of_battles=config.NUMBER_of_BATTLES_WHEN_EVALUATING):
                print("old_network changed")
                p_v_network_new.save(u)
                p_v_network_old.restore(u)
                current_best_model = u
                logging.info("Evaluation passed")
                logging.info("current_best: "+str(current_best_model))
            else:
                logging.info("Evaluation failed")
                p_v_network_new.restore(current_best_model)


    def evaluate_new_neural_network(self, p_v_network_old, p_v_network_new, number_of_battles=config.NUMBER_of_BATTLES_WHEN_EVALUATING, plane_size=config.PLANE_SIZE):

        root1 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=plane_size), father_edge=None, p_v_network=p_v_network_new)
        root2 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=plane_size), father_edge=None, p_v_network=p_v_network_old)
        player1 = p_v_mcts_player.MCTSPlayer(root=root1, p_v_network=p_v_network_new, max_simulation=config.MAX_SIMULATION_WHEN_EVALUATING)
        player2 = p_v_mcts_player.MCTSPlayer(root=root2, p_v_network=p_v_network_old, max_simulation=config.MAX_SIMULATION_WHEN_EVALUATING)

        new_pure_win = 0
        logging.info("新白旧黑")
        for i in range(number_of_battles):
            player1.refresh()
            player2.refresh()
            winner, plane_record, action_list, turn = play.PlayLogic().play(player2, player1)
            new_pure_win -= winner
        if new_pure_win >= 0:
            new_pure_win = 0
            logging.info("新黑旧白")
            for i in range(number_of_battles):
                player1.refresh()
                player2.refresh()
                winner, plane_record, action_list, turn = play.PlayLogic().play(player1, player2)
                new_pure_win += winner
            if new_pure_win >= 1:
                return True
            else:
                # return True  # 测试用
                return False
        else:
            # return True  # 测试用
            return False


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_WHEN_TRAINING

    logging.basicConfig(filename='network/training_record.log', filemode="w", level=logging.DEBUG)
    train_and_update = TrainAndUpdate()
    train_and_update.train_and_update(plane_size=config.PLANE_SIZE, number_of_epoch=1, number_of_update_network=config.NUMBER_of_UPDATE_NEURAL_NETWORK, number_of_games=config.NUMBER_of_GAMES_IN_EACH_BATCH, number_of_samples_in_each_game=config.NUMBER_of_SAMPLES_IN_EACH_GAME, min_batch=config.MIN_BATCH)
