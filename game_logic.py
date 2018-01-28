import numpy as np
from pandas import DataFrame


class GameLogic:
    def __init__(self, plane_size=15):
        self.plane_size = plane_size
        self.current_turn = 1
        self.current_player = 1  # 1表示当前该黑棋走，-1 表示当前白棋下
        # 第一层指示 1黑 -1白 0空, 第二层指示下子的回合数
        self.plane = np.zeros((2, plane_size, plane_size))
        self.legal_actions = np.ones((plane_size, plane_size))

    def set_current_player(self, player):
        if player == 1:
            self.current_player = 1
            return True
        elif player == -1:
            self.current_player = -1
            return True
        else:
            return False

    def play_is_legal(self, x, y):

        '''
        :param x: 落子坐标x
        :param y: 落子坐标y
        :return: 是否合法
        '''

        if self.plane[0][x][y] == 0:
            return True
        else:
            return False

    def play(self, x, y):
        '''

        :param x:
        :param y:
        :param player: 1代表黑棋，-1代表白棋
        :return:
        '''
        if self.play_is_legal(x, y):
            self.plane[0][x][y] = self.current_player
            self.plane[1][x][y] = self.current_turn
            if self.current_player == 1:
                self.current_player = -1
            else:
                self.current_player = 1
            self.current_turn += 1
            return True
        else:
            return False

    def game_result(self):

        '''
        :return: 2:还没结束， 0:和棋 1:黑方胜 -1:白方胜
        '''

        is_full = True
        # 先检查竖的方向上有没有五子，再检查横的方向上有没有五子，然后检查两个斜线方向有没有五子
        i = 0
        j = 0
        connect_point = j
        while i < self.plane_size:
            while j < self.plane_size - 4:
                if self.plane[0][i][j] == 0:
                    j += 1
                    connect_point = j
                    is_full = False
                    continue
                if self.plane[0][i][j] != self.plane[0][i][j+1]:
                    j += 1
                    connect_point = j
                else:
                    connect_point += 1
                    while ((connect_point + 1 < self.plane_size)
                            and (self.plane[0][i][j] == self.plane[0][i][connect_point+1])):#这里利用了短路机制，否则后面的表达式可能溢出
                        connect_point += 1

                    if connect_point - j >= 4:
                        return self.plane[0][i][j]  #[i][j][2]表示花色
                    else:
                        j = connect_point + 1
                        connect_point = j
            i += 1
            j = 0
            connect_point = j

        # 先检查横的方向上有没有五子，再检查竖的方向上有没有五子，然后检查两个斜线方向有没有五子
        i = 0
        j = 0
        connect_point = j
        while j < self.plane_size:
            while i < self.plane_size - 4:
                if self.plane[0][i][j] == 0:
                    i += 1
                    connect_point = i
                    is_full = False
                    continue
                if self.plane[0][i][j] != self.plane[0][i+1][j]:
                    i += 1
                    connect_point = i
                else:
                    connect_point += 1
                    while ((connect_point + 1 < self.plane_size)
                           and (self.plane[0][i][j] == self.plane[0][connect_point+1][j])):  # 这里利用了短路机制，否则后面的表达式可能溢出
                        connect_point += 1

                    if connect_point - i >= 4:
                        return self.plane[0][i][j]  # [i][j][2]表示花色
                    else:
                        i = connect_point + 1
                        connect_point = i
            j += 1
            i = 0
            connect_point = i

        # 先检查竖的方向上有没有五子，再检查横的方向上有没有五子，然后检查两个斜线方向有没有五子
        #一下代码用于检测斜线方向的胜利情况，应该是对的吧，深感脑子不够用
        i = self.plane_size - 5
        j = 0
        connect_point_x = i
        connect_point_y = j
        while i >= 0:
            while j < self.plane_size - i - 4:
                if self.plane[0][i+j][j] == 0:
                    j += 1
                    connect_point_x = i + j
                    connect_point_y = j
                    is_full = False
                    continue
                if self.plane[0][i+j][j] != self.plane[0][i+j+1][j+1]:
                    j += 1
                    connect_point_x = i + j
                    connect_point_y = j
                else:
                    connect_point_x += 1
                    connect_point_y += 1
                    while ((connect_point_x + 1 < self.plane_size) and (connect_point_y + 1 < self.plane_size)
                           and (self.plane[0][i+j][j] == self.plane[0][connect_point_x+1][connect_point_y+1])):  # 这里利用了短路机制，否则后面的表达式可能溢出
                        connect_point_x += 1
                        connect_point_y += 1
                    if connect_point_y - j >= 4:
                        return self.plane[0][i+j][j]  # [i][j][2]表示花色
                    else:
                        j = connect_point_y + 1
                        connect_point_x = i + j
                        connect_point_y = j
            i -= 1
            j = 0
            connect_point_x = i
            connect_point_y = j

        i = 0
        j = 0
        while j < self.plane_size - 4:
            while i < self.plane_size - j - 4:
                if self.plane[0][i][j+i] == 0:
                    i += 1
                    connect_point_x = i
                    connect_point_y = j + i
                    is_full = False
                    continue
                if self.plane[0][i][j+i] != self.plane[0][i+1][j+i+1]:
                    i += 1
                    connect_point_x = i
                    connect_point_y = j + i
                else:
                    connect_point_x += 1
                    connect_point_y += 1
                    while ((connect_point_x + 1 < self.plane_size) and (connect_point_y + 1 < self.plane_size)
                           and (self.plane[0][i][j+i] == self.plane[0][connect_point_x+1][connect_point_y+1])):  # 这里利用了短路机制，否则后面的表达式可能溢出
                        connect_point_x += 1
                        connect_point_y += 1

                    if connect_point_x - i >= 4:
                        return self.plane[0][i][j+i]  # [i][j][2]表示花色
                    else:
                        i = connect_point_x + 1
                        connect_point_x = i
                        connect_point_y = j + i
            j += 1
            i = 0
            connect_point_x = i
            connect_point_y = j

        #一下代码用于检测斜率为+1的斜线方向的胜利情况，应该是对的吧，深感脑子不够用
        i = 4
        j = 0
        connect_point_x = i
        connect_point_y = j
        while i < self.plane_size:
            while j < i + 1 - 4:
                if self.plane[0][i-j][j] == 0:
                    j += 1
                    connect_point_x = i-j
                    connect_point_y = j
                    is_full = False
                    continue
                if self.plane[0][i-j][j] != self.plane[0][i-j-1][j+1]:
                    j += 1
                    connect_point_x = i-j
                    connect_point_y = j
                else:
                    connect_point_x -= 1
                    connect_point_y += 1
                    while ((connect_point_x - 1 >= 0) and (connect_point_y + 1 < self.plane_size)
                           and (self.plane[0][i-j][j] == self.plane[0][connect_point_x-1][connect_point_y+1])):  # 这里利用了短路机制，否则后面的表达式可能溢出
                        connect_point_x -= 1
                        connect_point_y += 1
                    if connect_point_y - j >= 4:
                        return self.plane[0][i-j][j]  # [i][j][2]表示花色
                    else:
                        j = connect_point_y + 1
                        connect_point_x = i - j
                        connect_point_y = j
            i += 1
            j = 0
            connect_point_x = i
            connect_point_y = j

        i = self.plane_size - 1
        j = 0
        while j < self.plane_size - 4:
            while i > j + 4:
                if self.plane[0][i][j+self.plane_size-1-i] == 0:
                    i -= 1
                    connect_point_x = i
                    connect_point_y = j + self.plane_size - 1 - i
                    is_full = False
                    continue
                if self.plane[0][i][j+self.plane_size-1-i] != self.plane[0][i-1][j+self.plane_size-1-i+1]:
                    i -= 1
                    connect_point_x = i
                    connect_point_y = j + self.plane_size - 1 - i
                else:
                    connect_point_x -= 1
                    connect_point_y += 1
                    while ((connect_point_x - 1 >= 0) and (connect_point_y + 1 < self.plane_size)
                           and (self.plane[0][i][j+self.plane_size-1-i] == self.plane[0][connect_point_x-1][connect_point_y+1])):  # 这里利用了短路机制，否则后面的表达式可能溢出
                        connect_point_x -= 1
                        connect_point_y += 1

                    if i - connect_point_x >= 4:
                        return self.plane[0][i][j+self.plane_size-1-i]  # [i][j][2]表示花色
                    else:
                        i = connect_point_x - 1
                        connect_point_x = i
                        connect_point_y = j + self.plane_size - 1 - i
            j += 1
            i = self.plane_size - 1
            connect_point_x = i
            connect_point_y = j

        for n in range(5):
            for m in range(5):
                if self.plane[0][self.plane_size-1-n][self.plane_size-1-m] == 0:
                    is_full = False
                    break
            if not is_full:
                break
        if is_full:
            return 0
        else:
            return 2

    def game_result_fast_version(self, x, y):

        '''
        获取上一步坐标，向8个方向探索
        :param x:
        :param y:
        :return:
        '''
        # print("current_play: ", x, y)
        point_x = x - 1
        point_y = y
        count = 1
        while point_x >= 0:
            if self.plane[0][x][y] == self.plane[0][point_x][point_y]:
                count += 1
                point_x -= 1
            else:
                break
        point_x = x + 1
        point_y = y
        while point_x < self.plane_size:
            if self.plane[0][x][y] == self.plane[0][point_x][point_y]:
                count += 1
                point_x += 1
            else:
                break
        if count >= 5:
            return self.plane[0][x][y]

        point_x = x
        point_y = y - 1
        count = 1
        while point_y >= 0:
            if self.plane[0][x][y] == self.plane[0][point_x][point_y]:
                count += 1
                point_y -= 1
            else:
                break
        point_x = x
        point_y = y + 1
        while point_y < self.plane_size:
            if self.plane[0][x][y] == self.plane[0][point_x][point_y]:
                count += 1
                point_y += 1
            else:
                break
        if count >= 5:
            return self.plane[0][x][y]

        point_x = x - 1
        point_y = y - 1
        count = 1
        while point_x >= 0 and point_y >= 0:
            if self.plane[0][x][y] == self.plane[0][point_x][point_y]:
                count += 1
                point_x -= 1
                point_y -= 1
            else:
                break
        point_x = x + 1
        point_y = y + 1
        while point_x < self.plane_size and point_y < self.plane_size:
            if self.plane[0][x][y] == self.plane[0][point_x][point_y]:
                count += 1
                point_x += 1
                point_y += 1
            else:
                break
        if count >= 5:
            return self.plane[0][x][y]

        point_x = x - 1
        point_y = y + 1
        count = 1
        while point_x >= 0 and point_y < self.plane_size:
            if self.plane[0][x][y] == self.plane[0][point_x][point_y]:
                count += 1
                point_x -= 1
                point_y += 1
            else:
                break
        point_x = x + 1
        point_y = y - 1
        while point_x < self.plane_size and point_y > 0:
            if self.plane[0][x][y] == self.plane[0][point_x][point_y]:
                count += 1
                point_x += 1
                point_y -= 1
            else:
                break
        if count >= 5:
            return self.plane[0][x][y]

        is_full = True
        for n in range(self.plane_size):
            for m in range(self.plane_size):
                if self.plane[0][n][m] == 0:
                    is_full = False
                    break
            if not is_full:
                break

        if is_full:
            return 0
        else:
            return 2

