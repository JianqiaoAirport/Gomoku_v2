import p_v_network
import p_v_mcts_player
import game_logic as gl
import play

game_logic = gl.GameLogic(plane_size=15)

p_v_network_1 = p_v_network.P_V_Network()
p_v_network_1.restore(0)
p_v_network_2 = p_v_network.P_V_Network()
p_v_network_2.restore(5)

def evaluate_new_neural_network(p_v_network_old, p_v_network_new, number_of_battles=4, plane_size=15):
    root1 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=plane_size), father_edge=None,
                                     p_v_network=p_v_network_new)
    root2 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=plane_size), father_edge=None,
                                     p_v_network=p_v_network_old)
    player1 = p_v_mcts_player.MCTSPlayer(root=root1, p_v_network=p_v_network_new, max_simulation=80)
    player2 = p_v_mcts_player.MCTSPlayer(root=root2, p_v_network=p_v_network_old, max_simulation=80)

    new_pure_win = 0
    print("------新黑旧白------")
    for i in range(number_of_battles):
        player1.refresh()
        player2.refresh()
        winner, plane_record, action_list, turn = play.PlayLogic().play(player1, player2)
        new_pure_win += winner
    print("------新白旧黑------")
    for i in range(number_of_battles):
        player1.refresh()
        player2.refresh()
        winner, plane_record, action_list, turn = play.PlayLogic().play(player2, player1)
        new_pure_win -= winner

    return new_pure_win

print(evaluate_new_neural_network(p_v_network_1, p_v_network_2))
