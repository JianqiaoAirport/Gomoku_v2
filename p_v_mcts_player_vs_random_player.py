import p_v_network_v2 as p_v_network
import p_v_mcts_player
import random_player
import game_logic as gl
import play
import config
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

game_logic = gl.GameLogic(plane_size=config.PLANE_SIZE)

p_v_network_1 = p_v_network.P_V_Network()
p_v_network_1.restore(5)


def evaluate_new_neural_network(p_v_network_new, number_of_battles=4, plane_size=config.PLANE_SIZE):
    root1 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=plane_size), father_edge=None,
                                     p_v_network=p_v_network_new)
    player1 = p_v_mcts_player.MCTSPlayer(root=root1, p_v_network=p_v_network_new, max_simulation=3)
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
    return new_pure_win


print(evaluate_new_neural_network(p_v_network_1))
