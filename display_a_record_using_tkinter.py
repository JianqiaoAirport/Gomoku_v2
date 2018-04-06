import tkinter as tk
import time
import numpy as np

import game_logic as gl
import config

game_logic = gl.GameLogic(plane_size=config.PLANE_SIZE)

plane_record = np.load("plane_record/plane_record25.npy")

root = tk.Tk()
root.title("Gomoku")
root.resizable(0, 0)
root.wm_attributes("-topmost", 1)
canvas = tk.Canvas(root, width=game_logic.plane_size*30+30, height=game_logic.plane_size*30+30, bd=0, bg='khaki', highlightthickness=0)

for i in range(1, game_logic.plane_size+1):
    canvas.create_line(i*30, 30, i*30, game_logic.plane_size*30, width=2)
for i in range(1, game_logic.plane_size+1):
    canvas.create_line(30, i*30, game_logic.plane_size*30, i*30, width=2)
#  之所以是123，因为create_line宽度是2个像素，如果124的话会不合适
if game_logic.plane_size == 15:
    canvas.create_oval(116, 116, 123, 123, fill='black')
    canvas.create_oval(116, 356, 123, 363, fill='black')
    canvas.create_oval(356, 116, 363, 123, fill='black')
    canvas.create_oval(356, 356, 363, 363, fill='black')
    canvas.create_oval(236, 236, 243, 243, fill='black')
canvas.pack()

for i in range(1, config.PLANE_SIZE**2+1):
    for x in range(config.PLANE_SIZE):
        for y in range(config.PLANE_SIZE):
            if plane_record[1][x][y] == i:
                result_x = x
                result_y = y
                break
    time.sleep(0.3)
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
            print(game_logic.plane)
            print("和棋")
            break
        elif result == 1:
            print(game_logic.plane)
            print("黑胜")
            break
        elif result == -1:
            print(game_logic.plane)
            print("白胜")
            break
        else:
            print("程序出错了，3秒后退出...")
            time.sleep(3)
            exit()
    else:
        pass

    root.update()



root.update()
root.mainloop()
