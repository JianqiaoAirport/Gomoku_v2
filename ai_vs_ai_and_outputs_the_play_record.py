import numpy as np

import p_v_network_v2 as p_v_network
import p_v_mcts_player
import game_logic as gl
import play
import config
import random_player

game_logic = gl.GameLogic(plane_size=config.PLANE_SIZE)

# p_v_network_1 = p_v_network.P_V_Network()
# p_v_network_1.restore(0)
# root1 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=plane_size), father_edge=None, p_v_network=p_v_network_new)
# player1 = p_v_mcts_player.MCTSPlayer(root=root1, p_v_network=p_v_network_new, max_simulation=2)
player1 = random_player.RandomPlayer(gl.GameLogic(plane_size=config.PLANE_SIZE))

p_v_network_2 = p_v_network.P_V_Network()
p_v_network_2.restore(0)
root2 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=config.PLANE_SIZE), father_edge=None, p_v_network=p_v_network_2)
player2 = p_v_mcts_player.MCTSPlayer(root=root2, p_v_network=p_v_network_2, max_simulation=2)


def evaluate_new_neural_network(player1, player2, number_of_battles=1):
    new_pure_win = 0
    print("------新黑旧白------")
    for i in range(number_of_battles):
        player1.refresh()
        player2.refresh()
        winner, plane_record_1, action_list, turn = play.PlayLogic().play(player1, player2)
        new_pure_win += winner
    print("------新白旧黑------")
    for i in range(number_of_battles):
        player1.refresh()
        player2.refresh()
        winner, plane_record_2, action_list, turn = play.PlayLogic().play(player2, player1)
        new_pure_win -= winner

    return plane_record_1, plane_record_2

plane_record_1, plane_record_2 = evaluate_new_neural_network(player1, player2)
np.save("plane_record/plane_record_1.npy", plane_record_1)
np.save("plane_record/plane_record_2.npy", plane_record_2)
