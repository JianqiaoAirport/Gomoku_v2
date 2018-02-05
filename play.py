import time
import game_logic as gl

class PlayLogic:
    def __init__(self, plane_size=15):
        self.plane_size = plane_size
        self.game_logic = gl.GameLogic(plane_size=plane_size)
        self.play_record = []

    def play(self, player1, player2):
        self.game_logic = gl.GameLogic(plane_size=self.plane_size)
        action_probability_distribution_list = []
        x, y, action_probability_distribution = player1.get_action_and_probability()
        action_probability_distribution_list.append(action_probability_distribution)
        self.game_logic.play(x, y)
        player2.get_opponents_action(x, y)
        while self.game_logic.game_result_fast_version(x, y) == 2:
            if self.game_logic.current_player == 1:
                x, y, action_probability_distribution = player1.get_action_and_probability()
                action_probability_distribution_list.append(action_probability_distribution)
                self.game_logic.play(x, y)
                player2.get_opponents_action(x, y)
            else:
                x, y, action_probability_distribution = player2.get_action_and_probability()
                action_probability_distribution_list.append(action_probability_distribution)
                self.game_logic.play(x, y)
                player1.get_opponents_action(x, y)
        result = self.game_logic.game_result_fast_version(x, y)
        if result == 1:
            self.play_record.append(self.game_logic.plane.copy())
            print("黑胜")
            return 1, self.game_logic.plane, action_probability_distribution_list, self.game_logic.current_turn - 1
        elif result == -1:
            self.play_record.append(self.game_logic.plane.copy())
            print("白胜")
            return -1, self.game_logic.plane, action_probability_distribution_list, self.game_logic.current_turn - 1
        elif result == 0:
            self.play_record.append(self.game_logic.plane.copy())
            print("和棋")
            return 0, self.game_logic.plane, action_probability_distribution_list, self.game_logic.current_turn - 1
        else:
            print("程序出错了，3秒后退出...")
            time.sleep(3)
            exit()

if __name__ == "__main__":
    import p_v_mcts_player
    import p_v_network
    import game_logic as gl
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    pl = PlayLogic()
    p_v_network = p_v_network.P_V_Network()
    root1 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=15), father_edge=None, p_v_network=p_v_network)
    root2 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=15), father_edge=None, p_v_network=p_v_network)
    player1 = p_v_mcts_player.MCTSPlayer(root=root1, p_v_network=p_v_network, max_simulation=5)
    player2 = p_v_mcts_player.MCTSPlayer(root=root2, p_v_network=p_v_network, max_simulation=5)
    start_time = time.time()
    pl.play(player1, player2)
    end_time = time.time()

    print(end_time-start_time)


