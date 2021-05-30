
# This is a lightweight ML agent trained by self-play.
# After sharing this notebook,
# we will add Hungry Geese environment in our HandyRL library.
# https://github.com/DeNA/HandyRL
# We hope you enjoy reinforcement learning!


import base64
import bz2
import math
import pickle
import time
from collections import defaultdict, deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, translate
from kaggle_environments.helpers import histogram

# MCTS


class MCTS:
    def __init__(self, game, nn_agent, eps=1e-8, cpuct=1.0, pb_c_base=19652, pb_c_init=1.25):
        self.game = game
        self.nn_agent = nn_agent
        self.eps = eps
        self.cpuct = cpuct
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init

        self.Qsa = {}  # 状態 s でプレイヤー i が行動 a を行ったあとの状態の状態評価値(訪問回数で平均)
        self.Nsa = {}  # 状態 s でプレイヤー i が行動 a を行ったあとの状態への訪問回数
        self.Ns = {}  # 状態 s の訪問回数
        self.Ps = {}  # 状態 s でプレイヤー i の行動の評価値。policy networkの出力

        self.Vs = {}  # 状態 s でのプレイヤー i の有効手

        self.last_obs = None

    def getActionProb(self, obs, timelimit=1.0):
        start_time = time.time()
        while time.time() - start_time < timelimit:
            self.search(obs, self.last_obs)

        s = self.game.stringRepresentation(obs)
        i = obs.index
        counts = [self.Nsa[(s, i, a)] if (s, i, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        prob = counts / np.sum(counts)

        print(f"player: {i}, count: {counts} / {np.sum(counts)}")

        self.last_obs = obs
        return prob

    def search(self, obs, last_obs):
        """
        用語:
            葉ノード: まだシミュレーションが行われていないノード
        """
        s = self.game.stringRepresentation(obs)

        # 現在の局面が葉ノードならば
        if s not in self.Ns:
            values = [-10] * 4
            for i in range(4):
                if len(obs.geese[i]) == 0:
                    continue

                # ニューラルネットワークで局面を評価する
                self.Ps[(s, i)], values[i] = self.nn_agent.predict(obs, last_obs, i)

                valids = self.game.getValidMoves(obs, last_obs, i)
                self.Ps[(s, i)] = self.Ps[(s, i)] * valids  # masking invalid moves
                sum_Ps_s = np.sum(self.Ps[(s, i)])
                if sum_Ps_s > 0:
                    self.Ps[(s, i)] /= sum_Ps_s  # renormalize

                self.Vs[(s, i)] = valids
                self.Ns[s] = 0

            # 各プレイヤーの現在の局面の 状態の評価値 を返す
            return values

        best_acts = [None] * 4
        for i in range(4):
            if len(obs.geese[i]) == 0:
                continue

            valids = self.Vs[(s, i)]
            cur_best = -float("inf")
            best_act = self.game.actions[-1]

            # pick the action with the highest upper confidence bound
            # 現在の局面 s でプレイヤー i の最適な行動を PUCTアルゴリズム で決定する
            for a in range(self.game.getActionSize()):
                if valids[a]:
                    cs = math.log((1 + self.Ns[s] + self.pb_c_base) / self.pb_c_base) + self.pb_c_init
                    if (s, i, a) in self.Qsa:
                        # u = self.Qsa[(s, i, a)] + self.cpuct * self.Ps[(s, i)][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, i, a)])
                        u = self.Qsa[(s, i, a)] + cs * self.Ps[(s, i)][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, i, a)])
                    else:
                        # u = self.cpuct * self.Ps[(s, i)][a] * math.sqrt(self.Ns[s] + self.eps)
                        u = cs * self.Ps[(s, i)][a] * math.sqrt(self.Ns[s] + self.eps)

                    if u > cur_best:
                        cur_best = u
                        best_act = self.game.actions[a]

            best_acts[i] = best_act

        # 各プレイヤーがベストな行動を行ったあとの局面を生成
        next_obs = self.game.getNextState(obs, last_obs, best_acts)

        # 生成した次の局面を探索
        values = self.search(next_obs, obs)

        for i in range(4):
            if len(obs.geese[i]) == 0:
                continue

            a = self.game.actions.index(best_acts[i])
            v = values[i]

            if (s, i, a) in self.Qsa:
                self.Qsa[(s, i, a)] = (self.Nsa[(s, i, a)] * self.Qsa[(s, i, a)] + v) / (self.Nsa[(s, i, a)] + 1)
                self.Nsa[(s, i, a)] += 1

            else:
                self.Qsa[(s, i, a)] = v
                self.Nsa[(s, i, a)] = 1

        self.Ns[s] += 1
        return values


class HungryGeese(object):
    def __init__(
        self, rows=7, columns=11, actions=[Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST], hunger_rate=40
    ):
        self.rows = rows
        self.columns = columns
        self.actions = actions
        self.hunger_rate = hunger_rate

    def getActionSize(self):
        return len(self.actions)

    def getNextState(self, obs, last_obs, directions):
        next_obs = deepcopy(obs)
        next_obs.step += 1
        geese = next_obs.geese
        food = next_obs.food

        for i in range(4):
            goose = geese[i]

            if len(goose) == 0:
                continue

            head = translate(goose[0], directions[i], self.columns, self.rows)

            # Check action direction
            if last_obs is not None and head == last_obs.geese[i][0]:
                geese[i] = []
                continue

            # Consume food or drop a tail piece.
            if head in food:
                food.remove(head)
            else:
                goose.pop()

            # Add New Head to the Goose.
            goose.insert(0, head)

            # If hunger strikes remove from the tail.
            if next_obs.step % self.hunger_rate == 0:
                if len(goose) > 0:
                    goose.pop()

        goose_positions = histogram(position for goose in geese for position in goose)

        # Check for collisions.
        for i in range(4):
            if len(geese[i]) > 0:
                head = geese[i][0]
                if goose_positions[head] > 1:
                    geese[i] = []

        return next_obs

    def getValidMoves(self, obs, last_obs, index):
        geese = obs.geese
        pos = geese[index][0]
        obstacles = {position for goose in geese for position in goose[:-1]}
        if last_obs is not None:
            obstacles.add(last_obs.geese[index][0])

        valid_moves = [translate(pos, action, self.columns, self.rows) not in obstacles for action in self.actions]

        return valid_moves

    def stringRepresentation(self, obs):
        return str(obs.geese + obs.food)


# Neural Network for Hungry Geese


class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:, :, :, -self.edge_size[1] :], x, x[:, :, :, : self.edge_size[1]]], dim=3)
        h = torch.cat([h[:, :, -self.edge_size[0] :], h, h[:, :, : self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h


class GeeseNetAlpha(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 64
        self.conv0 = TorusConv2d(28, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])

        self.conv_p = TorusConv2d(filters, filters, (3, 3), True)
        self.conv_v = TorusConv2d(filters, filters, (3, 3), True)

        self.head_p1 = nn.Linear(filters * 5 + 77, filters * 3, bias=False)
        self.head_p2 = nn.Linear(filters * 3, 4, bias=False)
        self.head_v1 = nn.Linear(filters * 5 + 77, filters * 3, bias=False)
        self.head_v2 = nn.Linear(filters * 3, 1, bias=False)

    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))

        h_p = F.relu_(self.conv_p(h))
        h_head_p = (h_p * x[:, :1]).view(h_p.size(0), h_p.size(1), -1).sum(-1)
        h_head_p2 = (h_p * x[:, 1:2]).view(h_p.size(0), h_p.size(1), -1).sum(-1)
        h_head_p3 = (h_p * x[:, 2:3]).view(h_p.size(0), h_p.size(1), -1).sum(-1)
        h_head_p4 = (h_p * x[:, 3:4]).view(h_p.size(0), h_p.size(1), -1).sum(-1)
        h_avg_p1 = h_p.view(h_p.size(0), h_p.size(1), -1).mean(-1)
        h_avg_p2 = h_p.view(h_p.size(0), h_p.size(1), -1).mean(1)

        h_p = F.relu_(self.head_p1(torch.cat([h_head_p, h_head_p2, h_head_p3, h_head_p4, h_avg_p1, h_avg_p2], 1)))
        p = torch.softmax(self.head_p2(h_p), 1)

        h_v = F.relu_(self.conv_v(h))
        h_head_v = (h_v * x[:, :1]).view(h_v.size(0), h_v.size(1), -1).sum(-1)
        h_head_v2 = (h_v * x[:, 1:2]).view(h_v.size(0), h_v.size(1), -1).sum(-1)
        h_head_v3 = (h_v * x[:, 2:3]).view(h_v.size(0), h_v.size(1), -1).sum(-1)
        h_head_v4 = (h_v * x[:, 3:4]).view(h_v.size(0), h_v.size(1), -1).sum(-1)
        h_avg_v1 = h_v.view(h_v.size(0), h_v.size(1), -1).mean(-1)
        h_avg_v2 = h_v.view(h_v.size(0), h_v.size(1), -1).mean(1)

        h_v = F.relu_(self.head_v1(torch.cat([h_head_v, h_head_v2, h_head_v3, h_head_v4, h_avg_v1, h_avg_v2], 1)))
        v = torch.tanh(self.head_v2(h_v))

        return p, v  # {"policy": p, "value": v}


def identity(image):
    return image.copy(), [0, 1, 2, 3]


def horizontal_flip(image):
    image = image[:, :, ::-1]
    return image.copy(), [0, 1, 3, 2]


def vertical_flip(image):
    image = image[:, ::-1, :]
    return image.copy(), [1, 0, 2, 3]


def horizontal_vertical_flip(image):
    image = image[:, ::-1, ::-1]
    return image.copy(), [1, 0, 3, 2]


class NNAgent:

    next_position_map = {}
    for pos in range(77):
        position = []
        position.append((11 * (1 + pos // 11) + pos % 11) % 77)
        position.append((11 * (-1 + pos // 11) + pos % 11) % 77)
        position.append((11 * (pos // 11) + (pos + 1) % 11) % 77)
        position.append((11 * (pos // 11) + (pos - 1) % 11) % 77)
        next_position_map[pos] = set(position)

    def __init__(self, state_dict):
        self.model = GeeseNetAlpha()
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, obs, last_obs, index):
        x = self._make_input(obs, last_obs, index)

        p, v = self._predict(x, identity)
        # p_h, v_h = self._predict(x, horizontal_flip)
        # p_v, v_v = self._predict(x, vertical_flip)
        # p_hv, v_hv = self._predict(x, horizontal_vertical_flip)

        # p = (p + p_h + p_v + p_hv) / 4
        # v = (v + v_h + v_v + v_hv) / 4

        return p, v

    def _predict(self, x, transform):
        x, slices = transform(x)
        with torch.no_grad():
            xt = torch.from_numpy(x).unsqueeze(0)
            p, v = self.model(xt)

        p = p.squeeze(0).detach().numpy()
        p = p[slices]
        return p, v.item()

    # Input for Neural Network
    def _make_input(self, obs, last_obs, index):
        x_ = []
        x_.append(self._make_input_normal(obs, last_obs, index))
        x_.append(self._get_reverse_cube(obs, index))
        x_.append(self._get_next_disappear_cube(obs, index))
        x_.append(self._get_step_cube_v2(obs))
        x_.append(self._get_length_cube(obs))
        x = np.concatenate(x_)
        return x

    def _make_input_normal(self, obs, last_obs, index):
        b = np.zeros((17, 7 * 11), dtype=np.float32)

        for p, pos_list in enumerate(obs.geese):
            # head position
            for pos in pos_list[:1]:
                b[0 + (p - index) % 4, pos] = 1
            # tip position
            for pos in pos_list[-1:]:
                b[4 + (p - index) % 4, pos] = 1
            # whole position
            for pos in pos_list:
                b[8 + (p - index) % 4, pos] = 1

        # previous head position
        if last_obs is not None:
            for p, pos_list in enumerate(last_obs.geese):
                for pos in pos_list[:1]:
                    b[12 + (p - index) % 4, pos] = 1

        # food
        for pos in obs.food:
            b[16, pos] = 1

        return b.reshape(-1, 7, 11)

    def _get_reverse_cube(self, obs, index):
        """
        尻尾から順番に 1, 0.9, 0.8, ... という並び
        """
        b = np.zeros((4, 7 * 11), dtype=np.float32)

        for p, geese in enumerate(obs["geese"]):
            # whole position reverse
            for num_reverse, pos in enumerate(geese[::-1]):
                b[(p - index) % 4, pos] = 1 - num_reverse * 0.1

        return b.reshape(-1, 7, 11)

    def _get_next_disappear_cube(self, obs, index):
        """
        次になくなる場所: 1
        次になくなる可能性のある場所: 0.5
        """
        b = np.zeros((4, 7 * 11), dtype=np.float32)
        step = obs["step"]

        # foodを食べる可能性があるか。
        eat_food_possibility = defaultdict(int)
        for p, geese in enumerate(obs["geese"]):
            for pos in geese[:1]:
                if not self.next_position_map[pos].isdisjoint(obs["food"]):
                    eat_food_possibility[p] = 1

        if (step % 40) == 39:  # 1つ短くなる
            for p, geese in enumerate(obs["geese"]):
                if eat_food_possibility[p]:  # 尻尾が1、尻尾の１つ前0.5
                    for pos in geese[-1:]:
                        b[(p - index) % 4, pos] = 1
                    for pos in geese[-2:-1]:
                        b[(p - obs["index"]) % 4, pos] = 0.5
                else:  # 食べる可能性なし -> 尻尾が1, 尻尾の1つ前1
                    for pos in geese[-2:]:
                        b[(p - index) % 4, pos] = 1
        else:  # 1つ短くならない
            for p, geese in enumerate(obs["geese"]):
                if eat_food_possibility[p]:  # 食べる可能性があり -> 尻尾を0.5
                    for pos in geese[-1:]:
                        b[(p - index) % 4, pos] = 0.5
                else:  # 食べる可能性なし # 尻尾を1
                    for pos in geese[-1:]:
                        b[(p - index) % 4, pos] = 1

        return b.reshape(-1, 7, 11)

    def _get_step_cube_v2(self, obs):
        """
        step0: 0, step199: 1
        step0: 0, step39 + 40n: 1
        """
        b = np.zeros((1, 7, 11), dtype=np.float32)
        step = obs["step"]

        b[:, :, :5] = (step % 200) / 199
        b[:, :, 5:] = (step % 40) / 39

        return b

    def _get_length_cube(self, obs):
        b = np.zeros((2, 7, 11), dtype=np.float32)

        my_length = len(obs["geese"][obs["index"]])
        opposite1_length = len(obs["geese"][(obs["index"] + 1) % 4])
        opposite2_length = len(obs["geese"][(obs["index"] + 2) % 4])
        opposite3_length = len(obs["geese"][(obs["index"] + 3) % 4])

        b[0] = my_length / 10
        max_opposite_length = max(opposite1_length, opposite2_length, opposite3_length)
        b[1, :, 0:2] = (my_length - max_opposite_length) / 10
        b[1, :, 2:5] = (my_length - opposite1_length) / 10
        b[1, :, 5:8] = (my_length - opposite2_length) / 10
        b[1, :, 8:11] = (my_length - opposite3_length) / 10

        return b


# Load PyTorch Model


state_dict = pickle.loads(bz2.decompress(base64.b64decode(PARAM)))

game = HungryGeese()
agent = NNAgent(state_dict)
mcts = MCTS(game, agent, pb_c_base=10, pb_c_init=1.3)


def alphageese_agent(obs, config):
    action = game.actions[np.argmax(mcts.getActionProb(obs, timelimit=1.0))]  # timelimit=config.actTimeout
    return action.name