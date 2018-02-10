import tkinter as tk
import time
import p_v_network
import p_v_mcts_player
import game_logic as gl

game_logic = gl.GameLogic(plane_size=15)

p_v_network_1 = p_v_network.P_V_Network()
p_v_network_1.restore(0)
p_v_network_2 = p_v_network.P_V_Network()
p_v_network_2.restore(2)
root1 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=15), father_edge=None, p_v_network=p_v_network_1)
root2 = p_v_mcts_player.MCTSNode(gl.GameLogic(plane_size=15), father_edge=None, p_v_network=p_v_network_2)
player1 = p_v_mcts_player.MCTSPlayer(root=root1, p_v_network=p_v_network_1, max_simulation=20)
player2 = p_v_mcts_player.MCTSPlayer(root=root2, p_v_network=p_v_network_2, max_simulation=20)

def click_callback(event):
    x = event.x
    y = event.y
    print(x, y)
    if x > game_logic.plane_size*30+15 or x < 15 or y > game_logic.plane_size*30+15 or y < 15:
        return

    if game_logic.current_player == 1:
        print("current player: "+str(game_logic.current_player))
        result_x, result_y, prob = player1.get_action_and_probability()
        player2.get_opponents_action(result_x, result_y)
    elif game_logic.current_player == -1:
        print("current player: " + str(game_logic.current_player))
        result_x, result_y, prob = player2.get_action_and_probability()
        player1.get_opponents_action(result_x, result_y)
    print(result_x, result_y)
    if game_logic.play(result_x, result_y):
        if game_logic.current_player == 1:
            canvas.create_oval(30 + result_x * 30 - 11, 30 + result_y * 30 - 11, 30 + result_x * 30 + 10,
                               30 + result_y * 30 + 10, fill='white')
        elif game_logic.current_player == -1:
            canvas.create_oval(30 + result_x * 30 - 11, 30 + result_y * 30 - 11, 30 + result_x * 30 + 10,
                               30 + result_y * 30 + 10, fill='black')
        else:
            print("程序出错了，3秒后退出...")
            time.sleep(3)
            exit()

        result = game_logic.game_result_fast_version(result_x, result_y)

        if result == 2:
            pass
        elif result == 0:
            print("和棋")
        elif result == 1:
            print("黑胜")
        elif result == -1:
            print("白胜")
        else:
            print("程序出错了，3秒后退出...")
            time.sleep(3)
            exit()
    else:
        pass

root = tk.Tk()
root.title("Gomoku")
root.resizable(0, 0)
root.wm_attributes("-topmost", 1)
canvas = tk.Canvas(root, width=game_logic.plane_size*30+30, height=game_logic.plane_size*30+30, bd=0, bg='khaki', highlightthickness=0)

for i in range(1, game_logic.plane_size+1):
    canvas.create_line(i*30, 30, i*30, game_logic.plane_size*30, width=2)
for i in range(1, game_logic.plane_size+1):
    canvas.create_line(30, i*30, game_logic.plane_size*30, i*30, width=2)
#之所以是123，因为create_line宽度是2个像素，如果124的话会不合适
if game_logic.plane_size == 15:
    canvas.create_oval(116, 116, 123, 123, fill='black')
    canvas.create_oval(116, 356, 123, 363, fill='black')
    canvas.create_oval(356, 116, 363, 123, fill='black')
    canvas.create_oval(356, 356, 363, 363, fill='black')
    canvas.create_oval(236, 236, 243, 243, fill='black')

canvas.bind('<Button-1>', click_callback)

canvas.pack()

button = tk.Button(root)
button.pack()
root.update()
root.mainloop()
