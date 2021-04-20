# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# kaggle_environments licensed under Copyright 2020 Kaggle Inc. and the Apache License, Version 2.0
# (see https://github.com/Kaggle/kaggle-environments/blob/master/LICENSE for details)

# wrapper of Hungry Geese environment from kaggle

import importlib
import io
import math
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# You need to install kaggle_environments, requests
from kaggle_environments import make

from ...environment import BaseEnvironment


class Dense(nn.Module):
    def __init__(self, units0, units1, bnunits=0, bias=True):
        super().__init__()
        if bnunits > 0:
            bias = False
        self.dense = nn.Linear(units0, units1, bias=bias)
        self.bnunits = bnunits
        self.bn = nn.BatchNorm1d(bnunits) if bnunits > 0 else None

    def forward(self, x):
        h = self.dense(x)
        if self.bn is not None:
            size = h.size()
            h = h.view(-1, self.bnunits)
            h = self.bn(h)
            h = h.view(*size)
        return h


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
    def __init__(self, input_dim, output_dim, kernel_size, bn, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class ChannelSELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# https://github.com/Kaixhin/Rainbow/blob/master/model.py
# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class GeeseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32

        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.cnn_blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.cse_blocks = nn.ModuleList([ChannelSELayer(filters, 4) for _ in range(layers)])

        self.conv_p = TorusConv2d(filters, filters, (3, 3), True)
        self.conv_v = TorusConv2d(filters, filters, (3, 3), True)

        self.head_p = NoisyLinear(filters, 4)
        self.head_v = NoisyLinear(77, 1)

    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))
        for cnn, cse in zip(self.cnn_blocks, self.cse_blocks):
            h = cnn(h)
            h = F.relu_(h + cse(h))

        p = F.relu_(self.conv_p(h))
        head = x[:, :1]
        p_head = (p * head).view(h.size(0), h.size(1), -1).sum(-1)
        p = self.head_p(p_head)

        v = F.relu_(self.conv_v(h))
        v_gap = v.view(h.size(0), h.size(1), -1).mean(1)
        v = torch.tanh(self.head_v(v_gap))

        return {'policy': p, 'value': v}

    def reset_noise(self):
        for name, module in self.named_children():
            if "head" in name:
                module.reset_noise()


class GeeseImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 6, 32

        self.conv0 = Conv2d(4, filters, (8, 8), True, 8)
        self.cnn_blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])

        self.conv_p = TorusConv2d(filters, filters, (3, 3), True)
        self.conv_v = TorusConv2d(filters, filters, (3, 3), True)

        self.head_p = NoisyLinear(77, 4)
        self.head_v = NoisyLinear(77, 1)

    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))

        for cnn in self.cnn_blocks:
            h = F.relu_(h + cnn(h))

        p = F.relu_(self.conv_p(h))
        v = F.relu_(self.conv_v(h))

        p_gmp, _ = torch.max(p.view(h.size(0), h.size(1), -1), dim=1)
        v_gap = v.view(h.size(0), h.size(1), -1).mean(1)

        p = self.head_p(p_gmp)
        v = torch.tanh(self.head_v(v_gap))

        return {'policy': p, 'value': v}

    def reset_noise(self):
        for name, module in self.named_children():
            if "head" in name:
                module.reset_noise()


class GeeseImage:
    GOOSE_ID_TO_COLOR = {0: "blue", 1: "orange", 2: "orange", 3: "orange"}
    FOOD_COLOR = "green"
    NUM_AGENTS = 4
    NUM_ROW, NUM_COL = 7, 11

    def __init__(self, **kwargs):
        width = 0.88
        height = 0.56
        padding = 0.0

        self.fig = plt.figure(dpi=100, figsize=(width, height), facecolor=(1, 1, 1))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.axes()
        plt.axis("off")

        self.plot_objects = self.init_func(ax, padding=padding, **kwargs)

    def flat_to_point(self, _x):
        assert 0 <= _x < self.NUM_COL * self.NUM_ROW
        return _x % self.NUM_COL, self.NUM_ROW - _x // self.NUM_COL - 1

    def food_positions(self, state):
        points = state[0]["observation"]["food"]
        points = [self.flat_to_point(x) for x in points]
        assert len(points) == 2
        x, y = [x[0] for x in points], [x[1] for x in points]
        if x[0] == x[1] and y[0] == y[1]:
            x[0] -= 0.25
            x[1] += 0.25
        return x, y

    def player_positions(self, state, goose_id: int):
        points = state[0]["observation"]["geese"][goose_id]
        points = [self.flat_to_point(x) for x in points]

        if len(points) == 0:
            return [], []

        if len(points) == 1:
            return [points[0][0]], [points[0][1]]

        last_x, last_y = points[0][0], points[0][1]
        xx, yy = [last_x], [last_y]
        for p in points[1:]:
            x, y = p[0], p[1]

            if x == 0 and last_x == self.NUM_COL - 1:
                xx += [self.NUM_COL + 2, self.NUM_COL + 2, -2, -2]
                yy += [y, -2, -2, y]

            if x == self.NUM_COL - 1 and last_x == 0:
                xx += [-2, -2, self.NUM_COL + 2, self.NUM_COL + 2]
                yy += [y, -2, -2, y]

            if y == 0 and last_y == self.NUM_ROW - 1:
                xx += [x, -2, -2, x]
                yy += [self.NUM_ROW + 2, self.NUM_ROW + 2, -2, -2]

            if y == self.NUM_ROW - 1 and last_y == 0:
                xx += [x, -2, -2, x]
                yy += [-2, -2, self.NUM_ROW + 2, self.NUM_ROW + 2]

            xx.append(x)
            yy.append(y)
            last_x, last_y = x, y

        return xx, yy

    def head_positions(self, state, goose_id: int):
        points = state[0]["observation"]["geese"][goose_id]
        points = [self.flat_to_point(x) for x in points]
        return [x[0] for x in points[:1]], [x[1] for x in points[:1]]

    def init_func(self, ax, padding=0.5):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_ylim(-0.5 - padding, self.NUM_ROW - 0.5 + padding)
        ax.set_xlim(-0.5 - padding, self.NUM_COL - 0.5 + padding)

        # FOOD
        food, *_ = ax.plot([], [], "D", color=self.FOOD_COLOR, ms=3)

        # GEESE
        geese = []
        heads = []
        for i in range(self.NUM_AGENTS):
            color = self.GOOSE_ID_TO_COLOR[i]
            g, *_ = ax.plot([], [], color=color)
            g.set_linewidth(1)
            h, *_ = ax.plot([], [], "o", color=color, ms=2)
            geese.append(g)
            heads.append(h)

        return food, geese, heads

    def to_image(self, state, player=0):
        food, geese, heads = self.plot_objects

        # FOOD
        food.set_data(self.food_positions(state))

        # GEESE
        for i in range(self.NUM_AGENTS):
            geese[(i - player) % self.NUM_AGENTS].set_data(self.player_positions(state, i))
            heads[(i - player) % self.NUM_AGENTS].set_data(self.head_positions(state, i))

        return (food, *geese, *heads)

    def to_numpy(self, state, player=0):
        self.to_image(state, player)

        buf = io.BytesIO()
        self.fig.savefig(buf, format="png")

        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()

        img = cv2.imdecode(img_arr, 1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img


class Environment(BaseEnvironment):
    ACTION = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    NUM_AGENTS = 4
    NUM_ROW = 7
    NUM_COL = 11
    CENTER_ROW = NUM_ROW // 2
    CENTER_COL = NUM_COL // 2

    def __init__(self, args={}):
        super().__init__()
        self.env = make("hungry_geese")
        self.reset()
        self.geese_image = GeeseImage()

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
        # 1st: 1.0 2nd: 0.33 3rd: -0.33 4th: -1.00
        rewards = {o['observation']['index']: o['reward'] for o in self.obs_list[-1]}
        outcomes = {p: 0 for p in self.players()}
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
        return GeeseImageNet

    def to_offset(self, x):
        row = self.CENTER_ROW - x // self.NUM_COL
        col = self.CENTER_COL - x % self.NUM_COL
        return row, col

    def to_row(self, offset, x):
        return (x // self.NUM_COL + offset) % self.NUM_ROW

    def to_col(self, offset, x):
        return (x + offset) % self.NUM_COL

    def observation(self, player=None):
        # x = self.observation_normal(player)
        # x = self.observation_centering_head(player)
        x = self.observation_image(player)
        return x

    def observation_normal(self, player=None):
        if player is None:
            player = 0

        b = np.zeros((self.NUM_AGENTS * 4 + 1, 7 * 11), dtype=np.float32)
        obs = self.obs_list[-1][0]['observation']

        for p, geese in enumerate(obs['geese']):
            # head position
            for pos in geese[:1]:
                b[0 + (p - player) % self.NUM_AGENTS, pos] = 1
            # tip position
            for pos in geese[-1:]:
                b[4 + (p - player) % self.NUM_AGENTS, pos] = 1
            # whole position
            for pos in geese:
                b[8 + (p - player) % self.NUM_AGENTS, pos] = 1

        # previous head position
        if len(self.obs_list) > 1:
            obs_prev = self.obs_list[-2][0]['observation']
            for p, geese in enumerate(obs_prev['geese']):
                for pos in geese[:1]:
                    b[12 + (p - player) % self.NUM_AGENTS, pos] = 1

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

    def observation_image(self, player=None):
        if player is None:
            player = 0

        b = np.full((4, 56, 88), 255, dtype=np.float32)

        for c, obs in enumerate(self.obs_list[:-5:-1]):
            b[c] = self.geese_image.to_numpy(obs, player)

        # normalize
        b = b / 255.0

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
