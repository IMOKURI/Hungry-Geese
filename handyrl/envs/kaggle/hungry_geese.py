# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# kaggle_environments licensed under Copyright 2020 Kaggle Inc. and the Apache License, Version 2.0
# (see https://github.com/Kaggle/kaggle-environments/blob/master/LICENSE for details)

# wrapper of Hungry Geese environment from kaggle

import importlib
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# You need to install kaggle_environments, requests
from kaggle_environments import make

from ...environment import BaseEnvironment
from ...model import ModelWrapper
from .geese.smart_goose import model as smart_model


class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn, groups=1):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, groups=groups)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:, :, :, -self.edge_size[1]:], x, x[:, :, :, :self.edge_size[1]]], dim=3)
        h = torch.cat([h[:, :, -self.edge_size[0]:], h, h[:, :, :self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h


class Conv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn=True, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class GeeseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])

        self.conv_p = TorusConv2d(filters, filters, (3, 3), True)
        self.conv_v = TorusConv2d(filters, filters, (3, 3), True)

        self.head_p = nn.Linear(filters, 4, bias=False)
        self.head_v1 = nn.Linear(filters * 2, filters, bias=False)
        self.head_v2 = nn.Linear(filters, 1, bias=False)

    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))

        h_p = F.relu_(self.conv_p(h))
        h_head_p = (h_p * x[:, :1]).view(h_p.size(0), h_p.size(1), -1).sum(-1)
        p = self.head_p(h_head_p)

        h_v = F.relu_(self.conv_v(h))
        h_head_v = (h_v * x[:, :1]).view(h_v.size(0), h_v.size(1), -1).sum(-1)
        h_avg_v = h_v.view(h_v.size(0), h_v.size(1), -1).mean(-1)

        h_v = F.relu_(self.head_v1(torch.cat([h_head_v, h_avg_v], 1)))
        v = torch.tanh(self.head_v2(h_v))

        return {"policy": p, "value": v, "h_head_p": h_head_p, "h_head_v": h_head_v, "h_avg_v": h_avg_v}


class GeeseNetAlpha(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
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
        p = self.head_p2(h_p)

        h_v = F.relu_(self.conv_v(h))
        h_head_v = (h_v * x[:, :1]).view(h_v.size(0), h_v.size(1), -1).sum(-1)
        h_head_v2 = (h_v * x[:, 1:2]).view(h_v.size(0), h_v.size(1), -1).sum(-1)
        h_head_v3 = (h_v * x[:, 2:3]).view(h_v.size(0), h_v.size(1), -1).sum(-1)
        h_head_v4 = (h_v * x[:, 3:4]).view(h_v.size(0), h_v.size(1), -1).sum(-1)
        h_avg_v1 = h_v.view(h_v.size(0), h_v.size(1), -1).mean(-1)
        h_avg_v2 = h_v.view(h_v.size(0), h_v.size(1), -1).mean(1)

        h_v = F.relu_(self.head_v1(torch.cat([h_head_v, h_head_v2, h_head_v3, h_head_v4, h_avg_v1, h_avg_v2], 1)))
        v = torch.tanh(self.head_v2(h_v))

        return {"policy": p, "value": v}


class RandomModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, _=None):
        xh = x[:, 0, :]

        h = torch.argmax(xh.sum(axis=2), axis=1)
        w = torch.argmax(xh.sum(axis=1), axis=1)

        whole = x[:, 8:12, :, :].sum(axis=1)
        prev = x[:, 12, :, :]
        whole_prev = whole + prev
        whole_prev = torch.clip(whole_prev, 0, 1)

        north = 1 - whole_prev[torch.arange(len(x)), (h - 1) % 7, w]
        south = 1 - whole_prev[torch.arange(len(x)), (h + 1) % 7, w]
        east = 1 - whole_prev[torch.arange(len(x)), h, (w + 1) % 11]
        west = 1 - whole_prev[torch.arange(len(x)), h, (w - 1) % 11]

        p = torch.rand(len(x), 4)
        p = p * torch.stack([north, south, west, east], axis=1)
        v = torch.rand(len(x), 1)
        return {'policy': p, 'value': v}


def get_random_model():
    model = RandomModel()
    return ModelWrapper(model)


def get_smart_model():
    return ModelWrapper(smart_model)


def get_alpha_model(path):
    model = GeeseNetAlpha()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return ModelWrapper(model)


random_model_agent = get_random_model()
smart_model_agent = get_smart_model()
pre_train_agent = get_alpha_model("weights/geese_net_fold0_best_64_be3500afc1c1b1778e2acea8678d0238c692f11b.pth")
alpha_64_1436_agent = get_alpha_model("weights/alpha_64_1436.pth")
alpha_64_3391_agent = get_alpha_model("weights/alpha_64_4093.pth")
alpha_64_4093_agent = get_alpha_model("weights/alpha_64_4093.pth")


class Environment(BaseEnvironment):
    ACTION = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    NUM_AGENTS = 4
    NUM_ROW = 7
    NUM_COL = 11
    CENTER_ROW = NUM_ROW // 2
    CENTER_COL = NUM_COL // 2

    next_position_map = {}
    for pos in range(77):
        position = []
        position.append((11 * (1 + pos // 11) + pos % 11) % 77)
        position.append((11 * (-1 + pos // 11) + pos % 11) % 77)
        position.append((11 * (pos // 11) + (pos + 1) % 11) % 77)
        position.append((11 * (pos // 11) + (pos - 1) % 11) % 77)
        next_position_map[pos] = set(position)

    def __init__(self, args={}):
        super().__init__()
        self.env = make("hungry_geese")
        self.reset()

    def reset(self, args={}):
        obs = self.env.reset(num_agents=self.NUM_AGENTS)
        self.update((obs, {}), True)

    def update(self, info, reset):
        obs, last_actions = info
        if reset:
            self.obs_list = []
        self.obs_list.append(obs)
        self.last_actions = last_actions

    def action2str(self, a, player=None):
        return self.ACTION[a]

    def str2action(self, s, player=None):
        return self.ACTION.index(s)

    def direction(self, pos_from, pos_to):
        if pos_to is None:
            return None
        x_from, y_from = pos_from // 11, pos_from % 11
        x_to, y_to = pos_to // 11, pos_to % 11
        if x_from == x_to:
            if (y_from + 1) % 11 == y_to:
                return 3
            if (y_from - 1) % 11 == y_to:
                return 2
        if y_from == y_to:
            if (x_from + 1) % 7 == x_to:
                return 1
            if (x_from - 1) % 7 == x_to:
                return 0

    def __str__(self):
        # output state
        obs = self.obs_list[-1][0]['observation']
        colors = ['\033[33m', '\033[34m', '\033[32m', '\033[31m']
        color_end = '\033[0m'

        def check_cell(pos):
            for i, geese in enumerate(obs['geese']):
                if pos in geese:
                    if pos == geese[0]:
                        return i, 'h'
                    if pos == geese[-1]:
                        return i, 't'
                    index = geese.index(pos)
                    pos_prev = geese[index - 1] if index > 0 else None
                    pos_next = geese[index + 1] if index < len(geese) - 1 else None
                    directions = [self.direction(pos, pos_prev), self.direction(pos, pos_next)]
                    return i, directions
            if pos in obs['food']:
                return 'f'
            return None

        def cell_string(cell):
            if cell is None:
                return '.'
            elif cell == 'f':
                return 'f'
            else:
                index, directions = cell
                if directions == 'h':
                    return colors[index] + '@' + color_end
                elif directions == 't':
                    return colors[index] + '*' + color_end
                elif max(directions) < 2:
                    return colors[index] + '|' + color_end
                elif min(directions) >= 2:
                    return colors[index] + '-' + color_end
                else:
                    return colors[index] + '+' + color_end

        cell_status = [check_cell(pos) for pos in range(7 * 11)]

        s = 'turn %d\n' % len(self.obs_list)
        for x in range(7):
            for y in range(11):
                pos = x * 11 + y
                s += cell_string(cell_status[pos])
            s += '\n'
        for i, geese in enumerate(obs['geese']):
            s += colors[i] + str(len(geese) or '-') + color_end + ' '
        return s

    def step(self, actions):
        # state transition
        obs = self.env.step([self.action2str(actions.get(p, None) or 0) for p in self.players()])
        self.update((obs, actions), False)

    def diff_info(self, _):
        return self.obs_list[-1], self.last_actions

    def turns(self):
        # players to move
        return [p for p in self.players() if self.obs_list[-1][p]['status'] == 'ACTIVE']

    def terminal(self):
        # check whether terminal state or not
        for obs in self.obs_list[-1]:
            if obs['status'] == 'ACTIVE':
                return False
        return True

    def outcome(self):
        # return terminal outcomes
        # 1st: 1.00 2nd: 0.33 3rd: -0.33 4th: -1.00
        rewards = {o['observation']['index']: o['reward'] for o in self.obs_list[-1]}
        outcomes = {p: 0.0 for p in self.players()}
        for p, r in rewards.items():
            for pp, rr in rewards.items():
                if p != pp:
                    if r > rr:
                        outcomes[p] += 1 / (self.NUM_AGENTS - 1)
                    elif r < rr:
                        outcomes[p] -= 1 / (self.NUM_AGENTS - 1)
        return outcomes

    def legal_actions(self, player):
        # return legal action list
        return list(range(len(self.ACTION)))

    def action_length(self):
        # maximum action label (it determines output size of policy function)
        return len(self.ACTION)

    def players(self):
        return list(range(self.NUM_AGENTS))

    def rule_based_action(self, player):
        from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, GreedyAgent
        action_map = {'N': Action.NORTH, 'S': Action.SOUTH, 'W': Action.WEST, 'E': Action.EAST}

        agent = GreedyAgent(Configuration({'rows': 7, 'columns': 11}))
        agent.last_action = action_map[self.ACTION[self.last_actions[player]]
                                       [0]] if player in self.last_actions else None
        obs = {**self.obs_list[-1][0]['observation'], **self.obs_list[-1][player]['observation']}
        action = agent(Observation(obs))
        return self.ACTION.index(action)

    def rule_based_action_smart_geese(self, player, goose=None):
        from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, GreedyAgent
        if goose is None:
            agent_path = 'handyrl.envs.kaggle.geese.smart_goose'
        else:
            agent_path = 'handyrl.envs.kaggle.geese.' + goose
        agent_module = importlib.import_module(agent_path)
        if agent_module is None:
            print("No environment %s" % agent_path)

        obs = {**self.obs_list[-1][0]['observation'], **self.obs_list[-1][player]['observation']}
        action = agent_module.agent(Observation(obs), None)
        return self.ACTION.index(action)

    def net(self):
        return GeeseNetAlpha

    def to_offset(self, x):
        row = self.CENTER_ROW - x // self.NUM_COL
        col = self.CENTER_COL - x % self.NUM_COL
        return row, col

    def to_row(self, offset, x):
        return (x // self.NUM_COL + offset) % self.NUM_ROW

    def to_col(self, offset, x):
        return (x + offset) % self.NUM_COL

    def around(self, x):
        return [
            ((x[0] - 1) % 7, x[1]),
            ((x[0] + 1) % 7, x[1]),
            (x[0], (x[1] - 1) % 11),
            (x[0], (x[1] + 1) % 11),
        ]

    def observation(self, player=None):
        obses = []
        obses.append(self.observation_normal(player))
        # obses.append(self.observation_centering_head(player))
        obses.append(self.observation_reverse_pos(player))
        obses.append(self.observation_disappear_next(player))
        obses.append(self.observation_num_step(player))
        obses.append(self.observation_length(player))
        x = np.concatenate(obses)
        return x

    def observation_normal(self, player=None):
        if player is None:
            player = 0

        b = np.zeros((self.NUM_AGENTS * 4 + 1, 7 * 11), dtype=np.float32)
        obs = self.obs_list[-1][0]['observation']

        for p, geese in enumerate(obs['geese']):
            pid = (p - player) % self.NUM_AGENTS

            # head position
            for pos in geese[:1]:
                b[0 + pid, pos] = 1
            # tip position
            for pos in geese[-1:]:
                b[4 + pid, pos] = 1
            # whole position
            for pos in geese:
                b[8 + pid, pos] = 1

        # previous head position
        if len(self.obs_list) > 1:
            obs_prev = self.obs_list[-2][0]['observation']
            for p, geese in enumerate(obs_prev['geese']):
                pid = (p - player) % self.NUM_AGENTS

                for pos in geese[:1]:
                    b[12 + pid, pos] = 1

        # food
        for pos in obs['food']:
            b[16, pos] = 1

        return b.reshape(-1, 7, 11)

    def observation_centering_head(self, player=None):
        if player is None:
            player = 0

        b = np.zeros((self.NUM_AGENTS * 4 + 1, self.NUM_ROW, self.NUM_COL), dtype=np.float32)
        obs = self.obs_list[-1][0]['observation']

        player_goose_head = obs['geese'][player][0]
        o_row, o_col = self.to_offset(player_goose_head)

        for p, geese in enumerate(obs['geese']):
            # head position
            for pos in geese[:1]:
                b[0 + (p - player) % self.NUM_AGENTS, self.to_row(o_row, pos), self.to_col(o_col, pos)] = 1
            # tip position
            for pos in geese[-1:]:
                b[4 + (p - player) % self.NUM_AGENTS, self.to_row(o_row, pos), self.to_col(o_col, pos)] = 1
            # whole position
            for pos in geese:
                b[8 + (p - player) % self.NUM_AGENTS, self.to_row(o_row, pos), self.to_col(o_col, pos)] = 1

        # previous head position
        if len(self.obs_list) > 1:
            obs_prev = self.obs_list[-2][0]['observation']
            for p, geese in enumerate(obs_prev['geese']):
                for pos in geese[:1]:
                    b[12 + (p - player) % self.NUM_AGENTS, self.to_row(o_row, pos), self.to_col(o_col, pos)] = 1

        # food
        for pos in obs['food']:
            b[16, self.to_row(o_row, pos), self.to_col(o_col, pos)] = 1

        return b

    def observation_reverse_pos(self, player=None):
        """
        尻尾から順番に 1, 0.9, 0.8, ... という並び
        """
        if player is None:
            player = 0

        b = np.zeros((self.NUM_AGENTS, 7 * 11), dtype=np.float32)
        obs = self.obs_list[-1][0]['observation']

        for p, geese in enumerate(obs['geese']):
            pid = (p - player) % self.NUM_AGENTS

            # whole position reverse
            for nr, pos in enumerate(geese[::-1]):
                b[pid, pos] = 1 - nr * 0.1

        return b.reshape(-1, 7, 11)

    def observation_disappear_next(self, player=None):
        """
        次になくなる場所: 1
        次になくなる可能性のある場所: 0.5
        """
        if player is None:
            player = 0

        b = np.zeros((self.NUM_AGENTS, 7 * 11), dtype=np.float32)
        obs = self.obs_list[-1][0]['observation']
        step = obs["step"]

        # foodを食べる可能性があるか。
        eat_food_possibility = defaultdict(int)
        for p, geese in enumerate(obs["geese"]):
            for pos in geese[:1]:
                if not self.next_position_map[pos].isdisjoint(obs["food"]):
                    eat_food_possibility[p] = 1

        if (step % 40) == 39:  # 1つ短くなる
            for p, geese in enumerate(obs['geese']):
                pid = (p - player) % self.NUM_AGENTS

                if eat_food_possibility[p]:  # 尻尾が1、尻尾の１つ前0.5
                    for pos in geese[-1:]:
                        b[pid, pos] = 1
                    for pos in geese[-2:-1]:
                        b[pid, pos] = 0.5
                else:  # 食べる可能性なし -> 尻尾が1, 尻尾の1つ前1
                    for pos in geese[-2:]:
                        b[pid, pos] = 1

        else:  # 1つ短くならない
            for p, geese in enumerate(obs["geese"]):
                pid = (p - player) % self.NUM_AGENTS

                if eat_food_possibility[p]:  # 食べる可能性があり -> 尻尾を0.5
                    for pos in geese[-1:]:
                        b[pid, pos] = 0.5
                else:  # 食べる可能性なし # 尻尾を1
                    for pos in geese[-1:]:
                        b[pid, pos] = 1

        return b.reshape(-1, 7, 11)

    def observation_num_step(self, player=None):
        """
        step0: 0, step199: 1
        step0: 0, step39 + 40n: 1
        """
        if player is None:
            player = 0

        b = np.zeros((1, 7, 11), dtype=np.float32)
        obs = self.obs_list[-1][0]['observation']
        step = obs["step"]

        b[:, :, :5] = (step % 200) / 199
        b[:, :, 5:] = (step % 40) / 39

        return b

    def observation_length(self, player=None):
        if player is None:
            player = 0

        b = np.zeros((2, 7, 11), dtype=np.float32)
        obs = self.obs_list[-1][0]['observation']

        my_length = len(obs['geese'][player])
        opposite1_length = len(obs['geese'][(player + 1) % self.NUM_AGENTS])
        opposite2_length = len(obs['geese'][(player + 2) % self.NUM_AGENTS])
        opposite3_length = len(obs['geese'][(player + 3) % self.NUM_AGENTS])

        b[0] = my_length / 10
        max_opposite_length = max(opposite1_length, opposite2_length, opposite3_length)
        b[1, :, 0:2] = (my_length - max_opposite_length) / 10
        b[1, :, 2:5] = (my_length - opposite1_length) / 10
        b[1, :, 5:8] = (my_length - opposite2_length) / 10
        b[1, :, 8:11] = (my_length - opposite3_length) / 10

        return b


if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            print(e)
            actions = {p: e.legal_actions(p) for p in e.turns()}
            print([[e.action2str(a, p) for a in alist] for p, alist in actions.items()])
            e.step({p: random.choice(alist) for p, alist in actions.items()})
        print(e)
        print(e.outcome())
