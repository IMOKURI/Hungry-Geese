# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# kaggle_environments licensed under Copyright 2020 Kaggle Inc. and the Apache License, Version 2.0
# (see https://github.com/Kaggle/kaggle-environments/blob/master/LICENSE for details)

# wrapper of Hungry Geese environment from kaggle

import math
import random
import itertools
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# You need to install kaggle_environments, requests
from kaggle_environments import make

from ...environment import BaseEnvironment
from ...model import BaseModel, Dense
from .models.gtrxl_torch import GTrXL


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


class Conv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class GeeseNet(BaseModel):
    def __init__(self, env, args={}):
        super().__init__(env, args)
        input_shape = env.observation().shape
        blocks, filters = 12, 32
        self.conv0 = TorusConv2d(input_shape[0], filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(blocks)])

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


class GeeseNetIMO(BaseModel):
    class GeeseEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(20, 3)

        def forward(self, x):
            x = self.embed(x).view(1, x.size()[0], -1)
            return x

    class GeeseBlock(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim, num_heads)

        def forward(self, x):
            h, _ = self.attention(x, x, x)
            return h

    class GeeseControll(nn.Module):
        def __init__(self, filters):
            super().__init__()
            self.filters = filters
            self.fc_control = Dense(filters * 2, filters, bnunits=filters)

        def forward(self, x, e):
            h = torch.cat([x, e], dim=2).view(x.size(1), -1)
            h = self.fc_control(h)
            return h

    class GeeseHead(nn.Module):
        def __init__(self, filters):
            super().__init__()
            f = filters // 2
            self.head_p_1 = nn.Linear(filters, f, bias=False)
            self.head_p_2 = nn.Linear(f, 4, bias=False)
            self.head_v_1 = nn.Linear(filters, f, bias=True)
            self.head_v_2 = nn.Linear(f, 1, bias=True)

        def forward(self, x):
            p = F.relu_(self.head_p_1(x))
            p = self.head_p_2(p)
            v = F.relu_(self.head_v_1(x))
            v = torch.tanh(self.head_v_2(v))
            return p, v

    def __init__(self, env, args={}):
        super().__init__(env, args)
        blocks = 6
        filters = 240

        self.encoder = self.GeeseEncoder()
        # self.geese_net = GeeseNet(env, args)

        self.blocks = nn.ModuleList([self.GeeseBlock(filters, 8) for _ in range(blocks)])
        self.control = self.GeeseControll(filters)
        self.head = self.GeeseHead(filters)

    def forward(self, x, _=None):
        e = self.encoder(x)
        # x_ = self.geese_net(x)
        # e = torch.cat([x_["h_head_p"], x_["h_avg_v"]], 1).view(1, x.size()[0], -1)

        h = e
        for block in self.blocks:
            h = h + block(h)
        h = self.control(h, e)
        p, v = self.head(h)
        return {"policy": p, "value": v}


class GeeseNetGTrXL(BaseModel):
    class GeeseEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = 8
            self.embed = nn.Embedding(18, self.d_model)

        def forward(self, x):
            x = self.embed(x) * math.sqrt(self.d_model)
            x = x.view(1, x.size()[0], -1)
            return x

    class GeeseHead(nn.Module):
        def __init__(self, filters):
            super().__init__()
            f = filters // 2
            self.head_p_1 = nn.Linear(filters, f, bias=False)
            self.head_p_2 = nn.Linear(f, 4, bias=False)
            self.head_v_1 = nn.Linear(filters, f, bias=True)
            self.head_v_2 = nn.Linear(f, 1, bias=True)

        def forward(self, x):
            p = F.relu_(self.head_p_1(x))
            p = self.head_p_2(p)
            v = F.relu_(self.head_v_1(x))
            v = torch.tanh(self.head_v_2(v))
            return p, v

    def __init__(self, env, args={}):
        super().__init__(env, args)
        d_model = 616
        n_heads = 8
        t_layers = 2

        self.encoder = self.GeeseEncoder()
        # self.geese_net = GeeseNet(env, args)

        self.gtrxl = GTrXL(d_model, n_heads, t_layers)
        self.head = self.GeeseHead(d_model)

    def forward(self, x, _=None):
        e = self.encoder(x)
        # x_ = self.geese_net(x)
        # e = torch.cat([x_["h_head_p"], x_["h_avg_v"]], 1).view(1, x.size()[0], -1)

        out = self.gtrxl(e).view(x.size()[0], -1)
        p, v = self.head(out)
        return {"policy": p, "value": v}


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
        # 1st: 1.0 2nd: 0.33 3rd: -0.33 4th: -1.0
        rewards = {o['observation']['index']: o['reward'] for o in self.obs_list[-1]}
        outcomes = {p: 0.0 for p in self.players()}
        for p, r in rewards.items():
            for pp, rr in rewards.items():
                if p != pp:
                    if r > rr:
                        outcomes[p] += 1 / (self.NUM_AGENTS - 1)
                    elif r < rr:
                        outcomes[p] -= 1 / (self.NUM_AGENTS - 1)
        # for p, o in outcomes.items():
        #     if o == 0.5:
        #         outcomes[p] = 1.0

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
        return GeeseNetGTrXL

    def to_offset(self, x):
        row = self.CENTER_ROW - x // self.NUM_COL
        col = self.CENTER_COL - x % self.NUM_COL
        return row, col

    def to_row(self, offset, x):
        return (x // self.NUM_COL + offset) % self.NUM_ROW

    def to_col(self, offset, x):
        return (x + offset) % self.NUM_COL

    def observation(self, player=None):
        if player is None:
            player = 0

        b = np.zeros((self.NUM_ROW, self.NUM_COL), dtype=np.long)
        obs = self.obs_list[-1][0]['observation']

        player_goose_head = obs['geese'][player][0]
        o_row, o_col = self.to_offset(player_goose_head)

        for p, geese in enumerate(obs['geese']):
            # whole position
            for pos in geese:
                b[self.to_row(o_row, pos), self.to_col(o_col, pos)] = 9 + (p - player) % self.NUM_AGENTS
            # tip position
            for pos in geese[-1:]:
                b[self.to_row(o_row, pos), self.to_col(o_col, pos)] = 5 + (p - player) % self.NUM_AGENTS
            # head position
            for pos in geese[:1]:
                b[self.to_row(o_row, pos), self.to_col(o_col, pos)] = 1 + (p - player) % self.NUM_AGENTS

        # previous head position
        if len(self.obs_list) > 1:
            obs_prev = self.obs_list[-2][0]['observation']
            for p, geese in enumerate(obs_prev['geese']):
                for pos in geese[:1]:
                    b[self.to_row(o_row, pos), self.to_col(o_col, pos)] = 13 + (p - player) % self.NUM_AGENTS

        # food
        for pos in obs['food']:
            b[self.to_row(o_row, pos), self.to_col(o_col, pos)] = 17

        # padding
        # b = np.pad(b.reshape(-1), (0, 3))

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
