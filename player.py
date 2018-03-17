import numpy as np

class Player:
    def __init__(self):
        pass

    def get_action_and_probability(self):
        x = 0
        y = 0
        p = np.array(range(225))
        return x, y, p

    def get_opponents_action(self, x, y):
        pass

    def refresh(self):
        pass
