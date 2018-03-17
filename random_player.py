import random
import game_logic

import config

class RandomPlayer:
    def __init__(self, game_logic):
        self.game_logic = game_logic

    def get_action_and_probability(self):
        legal_points_num = 0
        legal_action = []
        for i in range(self.game_logic.plane_size):
            for j in range(self.game_logic.plane_size):
                if self.game_logic.plane[0][i][j] == 0:
                    legal_points_num += 1
                    legal_action.append(self.game_logic.plane_size*i + j)

        action = random.sample(legal_action, k=1)[0]
        x = int(action/self.game_logic.plane_size)
        y = action % self.game_logic.plane_size
        p = []
        for i in range(self.game_logic.plane_size**2):
            if i in legal_action:
                p.append(1.0/legal_points_num)
            else:
                p.append(0.0)
        self.game_logic.play(x, y)
        return x, y, p

    def get_opponents_action(self, x, y):
        self.game_logic.play(x, y)

    def refresh(self):
        self.game_logic = game_logic.GameLogic(plane_size=config.PLANE_SIZE)
