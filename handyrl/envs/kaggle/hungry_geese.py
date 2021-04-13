# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# kaggle_environments licensed under Copyright 2020 Kaggle Inc. and the Apache License, Version 2.0
# (see https://github.com/Kaggle/kaggle-environments/blob/master/LICENSE for details)

# wrapper of Hungry Geese environment from kaggle

import random
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# You need to install kaggle_environments, requests
from kaggle_environments import make

from ...environment import BaseEnvironment


class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
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


class GeeseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32

        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.cnn_blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.se_blocks = nn.ModuleList([SELayer(filters) for _ in range(layers)])

        self.conv_p = TorusConv2d(filters, filters, (3, 3), True)
        self.conv_v = TorusConv2d(filters, filters, (3, 3), True)

        self.head_p1 = nn.Linear(filters + 77, 48, bias=False)
        self.head_p2 = nn.Linear(48, 4, bias=False)
        self.head_v = nn.Linear(filters + 77, 1, bias=False)

    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))
        for cnn, se in zip(self.cnn_blocks, self.se_blocks):
            h = F.relu_(h + se(cnn(h)))

        p = F.relu_(self.conv_p(h))
        v = F.relu_(self.conv_v(h))

        p_head = (p * x[:, :1]).view(h.size(0), h.size(1), -1).sum(-1)
        v_head = (v * x[:, :1]).view(h.size(0), h.size(1), -1).sum(-1)

        # Global Average Pooling
        p_gap = p.view(h.size(0), h.size(1), -1).mean(1)
        v_gap = v.view(h.size(0), h.size(1), -1).mean(1)

        p = F.relu_(self.head_p1(torch.cat([p_head, p_gap], dim=1)))
        p = self.head_p2(p)
        v = torch.tanh(self.head_v(torch.cat([v_head, v_gap], dim=1)))

        return {'policy': p, 'value': v}


class Environment(BaseEnvironment):
    ACTION = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    NUM_AGENTS = 4

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
        agent.last_action = action_map[self.ACTION[self.last_actions[player]][0]] if player in self.last_actions else None
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
        return GeeseNet

    def observation(self, player=None):
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
