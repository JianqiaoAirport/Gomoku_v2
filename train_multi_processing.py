import numpy as np
import p_v_network_v2 as p_v_network
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
import random_player


class TrainAndUpdate:
    def __init__(self, kl_target=config.KL_TARGET, learning_rate_multi=config.LEARNING_RATE_MULTI):
        self.KL_target = kl_target
        self.learning_rate_multi = learning_rate_multi

    def train_and_update(self, plane_size=15, number_of_epoch=1, number_of_update_network=200, number_of_games=200, number_of_samples_in_each_game=9, min_batch=100, start_iteration=1):
        '''
        :param number_of_epoch:
        :param number_of_update_network:
        :param number_of_games:
        :param numbuer_of_samples_in_each_game:
        :param min_batch: 需要是 number_of_games 乘以 numbuer_of_samples_in_each_game 的积的约数
        :return:
        '''

        path = "./network"
        if not os.path.exists(path):
            os.makedirs(path)

        p_v_network_new = p_v_network.P_V_Network(name_scope="train_new")
        p_v_network_old = p_v_network.P_V_Network(name_scope="train_old")

        model_list = os.listdir("network")
        current_best_model = 0
        for item in model_list:
            if item.startswith("model-"):
                model_number = int(item[6:].split('.')[0])
                if model_number > current_best_model:
                    current_best_model = model_number
        logging.info("model-" + str(current_best_model))
        if len(model_list) >= 4:
            p_v_network_new.restore(u=current_best_model)
            p_v_network_old.restore(u=current_best_model)
        else:
            p_v_network_new.save(0)
            p_v_network_old.restore(0)

        for u in range(start_iteration, number_of_update_network+1):
            print("the %dth update" % u)

            # get data
            while True:
                data_loaded = False

                generator_files = os.listdir("data")
                for file in generator_files:
                    if not file.startswith("brain"):
                        generator_files.remove(file)
                for generator in generator_files:
                    if not generator.startswith("brain"):
                        continue
                    data_batches = os.listdir("data/"+generator)

                    for item in data_batches:
                        if not item.startswith("self_play_data"):
                            data_batches.remove(item)

                    if len(data_batches) != 0:
                        for i in range(len(data_batches)):
                            if not os.path.isdir("data/"+generator+"/"+data_batches[i]):
                                continue
                            data_list = os.listdir("data/"+generator+"/"+data_batches[i])
                            data_is_used = False
                            for data_file in data_list:
                                if data_file.endswith("data_is_used"):
                                    data_is_used = True
                                    break
                            if (not data_is_used) and len(data_list) >= 3:
                                try:
                                    plane_records = np.load("data/"+generator+"/"+data_batches[i]+"/plane_records.npy")
                                    game_result_ = np.load("data/"+generator+"/"+data_batches[i]+"/game_result.npy")
                                    y_ = np.load("data/"+generator+"/"+data_batches[i]+"/y_.npy")
                                # shutil.rmtree("data/"+generator+"/"+data_batches[i])
                                    os.mkdir("data/"+generator+"/"+data_batches[i]+"/data_is_used")
                                    data_loaded = True
                                except Exception:
                                    time.sleep(2)
                                    plane_records = np.load(
                                        "data/" + generator + "/" + data_batches[i] + "/plane_records.npy")
                                    game_result_ = np.load(
                                        "data/" + generator + "/" + data_batches[i] + "/game_result.npy")
                                    y_ = np.load("data/" + generator + "/" + data_batches[i] + "/y_.npy")
                                    # shutil.rmtree("data/"+generator+"/"+data_batches[i])
                                    os.mkdir("data/" + generator + "/" + data_batches[i] + "/data_is_used")
                                    data_loaded = True
                                break
                        if data_loaded:
                            break
                if data_loaded:
                    break

                time.sleep(5)
                logging.info("sleep 5 s")
            # get data end

            # train
            for e in range(number_of_epoch):
                for i in range(int(number_of_games*number_of_samples_in_each_game*8/min_batch)):
                    # min-batch 100， 由于只有1000个局面样本，所以只循环10次
                    batch = [plane_records[i * min_batch: (i + 1) * min_batch], game_result_[i * min_batch: (i + 1) * min_batch], y_[i * min_batch: (i + 1) * min_batch]]
                    # if e % 10 == 0:
                        # loss = p_v_network_new.loss.eval(feed_dict={p_v_network_new.x_plane: batch[0], p_v_network_new.game_result: batch[1], p_v_network_new.y_: batch[2], p_v_network_new.is_training: False})
                        # p_v_network_new.sess.run([p_v_network_new.loss.eval], feed_dict={p_v_network_new.x_plane: batch[0], p_v_network_new.game_result: batch[1], p_v_network_new.y_: batch[2], p_v_network_new.is_training: False})
                        # print("step %d, loss %g" % (i, loss))
                        # pass

                    result_old = p_v_network_new.sess.run([p_v_network_new.prediction, p_v_network_new.y_v],
                                                          feed_dict={p_v_network_new.x_plane: batch[0],
                                                             p_v_network_new.is_training: False})
                    old_prediction = result_old[0]

                    summary, _ = p_v_network_new.sess.run([p_v_network_new.merged, p_v_network_new.train_step], feed_dict={p_v_network_new.x_plane: batch[0], p_v_network_new.game_result: batch[1], p_v_network_new.y_: batch[2], p_v_network_new.is_training: True, p_v_network_new.learning_rate: config.LEARNING_RATE*self.learning_rate_multi})
                    p_v_network_new.train_writer.add_summary(summary, u)

                    result_new = p_v_network_new.sess.run([p_v_network_new.prediction, p_v_network_new.y_v],
                                                          feed_dict={p_v_network_new.x_plane: batch[0],
                                                             p_v_network_new.is_training: False})
                    new_prediction = result_new[0]

                    kl_expansion = np.mean(np.sum(old_prediction * (np.log(old_prediction + 1e-10) - np.log(new_prediction + 1e-10)), axis=1))
                    with open('network/KL_Expasion.txt', 'a+') as f:
                        f.write(str(u) + "," + str(kl_expansion) + "\n")

                    if kl_expansion > self.KL_target * 4:  # early stop if KL diverges badly
                        logging.info("early stop!!!!!!!!!")
                        print("early stop!!!!!!!!!!!")
                        break
                    else:
                        #logging.info("not early stop!")
                        pass
            # adaptively adjust the learning rate
            if kl_expansion > self.KL_target * 2 and self.learning_rate_multi > 0.1:
                self.learning_rate_multi /= 1.5
            elif kl_expansion < self.KL_target / 2 and self.learning_rate_multi < 10:
                self.learning_rate_multi *= 1.5

            # evaluate

            if u % 2 == 0:
                self.evaluate_new_network_with_random_player(p_v_network_new, u=u, max_simulation=1, number_of_battles=15)
                self.evaluate_new_network_with_random_player(p_v_network_new, u=u, max_simulation=3, number_of_battles=10)

            if u % 10 != 0:
                # print("old_network changed")
                # p_v_network_new.save(u)
                # p_v_network_old.restore(u)
                # current_best_model = u
                # logging.info("current_best: " + str(current_best_model))
                continue

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
        # return True  # 测试用
        root1 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=plane_size), father_edge=None, p_v_network=p_v_network_new)
        root2 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=plane_size), father_edge=None, p_v_network=p_v_network_old)
        player1 = p_v_mcts_player.MCTSPlayer(root=root1, p_v_network=p_v_network_new, max_simulation=config.MAX_SIMULATION_WHEN_EVALUATING)
        player2 = p_v_mcts_player.MCTSPlayer(root=root2, p_v_network=p_v_network_old, max_simulation=config.MAX_SIMULATION_WHEN_EVALUATING)

        new_pure_win = 0
        logging.info("新白旧黑")
        for i in range(number_of_battles):
            player1.refresh()
            player2.refresh()
            winner, plane_record, action_list, turn = play.PlayLogic(plane_size=config.PLANE_SIZE).play(player2, player1)
            new_pure_win -= winner
        if new_pure_win >= 0:
            new_pure_win = 0
            logging.info("新黑旧白")
            for i in range(number_of_battles):
                player1.refresh()
                player2.refresh()
                winner, plane_record, action_list, turn = play.PlayLogic(plane_size=config.PLANE_SIZE).play(player1, player2)
                new_pure_win += winner
            if new_pure_win >= 0:
                return True
            else:
                # return True  # 测试用
                return False
        else:
            # return True  # 测试用
            return False

    def evaluate_new_network_with_random_player(self, p_v_network_new, number_of_battles=25, plane_size=config.PLANE_SIZE, u=1, max_simulation=1):
        root1 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=plane_size), father_edge=None, p_v_network=p_v_network_new)
        player1 = p_v_mcts_player.MCTSPlayer(root=root1, p_v_network=p_v_network_new, max_simulation=max_simulation)
        player2 = random_player.RandomPlayer(gl.GameLogic(plane_size=plane_size))
        new_pure_win = 0
        print("------神黑随白------")
        for i in range(number_of_battles):
            player1.refresh()
            player2.refresh()
            winner, plane_record, action_list, turn = play.PlayLogic().play(player1, player2)
            new_pure_win += winner
        print("------神白随黑------")
        for i in range(number_of_battles):
            player1.refresh()
            player2.refresh()
            winner, plane_record, action_list, turn = play.PlayLogic().play(player2, player1)
            new_pure_win -= winner
        win_rate = (new_pure_win+number_of_battles*2.0)/(2*2*number_of_battles)
        with open('network/win_rate_max_simulation'+str(max_simulation)+'.txt', 'a+') as f:
            f.write(str(u)+","+str(win_rate)+"\n")

        return new_pure_win


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_WHEN_TRAINING

    logging.basicConfig(filename='network/training_record.log', filemode="a", level=logging.DEBUG)
    train_and_update = TrainAndUpdate()
    train_and_update.train_and_update(plane_size=config.PLANE_SIZE, number_of_epoch=config.NUMBER_of_EPOCH, number_of_update_network=config.NUMBER_of_UPDATE_NEURAL_NETWORK, number_of_games=config.NUMBER_of_GAMES_IN_EACH_BATCH, number_of_samples_in_each_game=config.NUMBER_of_SAMPLES_IN_EACH_GAME, min_batch=config.MIN_BATCH, start_iteration=config.START_ITERATION)
