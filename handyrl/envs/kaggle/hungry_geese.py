# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# kaggle_environments licensed under Copyright 2020 Kaggle Inc. and the Apache License, Version 2.0
# (see https://github.com/Kaggle/kaggle-environments/blob/master/LICENSE for details)

# wrapper of Hungry Geese environment from kaggle

import importlib
import random

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
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
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


class GeeseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32

        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.cnn_blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.cse_blocks = nn.ModuleList([ChannelSELayer(filters, 4) for _ in range(layers)])

        self.conv_p = TorusConv2d(filters, filters, (3, 3), True)
        self.conv_v = TorusConv2d(filters, filters, (3, 3), True)

        self.head_p1 = nn.Linear(filters + 77, 48, bias=False)
        self.head_p2 = nn.Linear(48, 4, bias=False)
        self.head_v = nn.Linear(filters + 77, 1, bias=False)

    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))
        for cnn, cse in zip(self.cnn_blocks, self.cse_blocks):
            h = cnn(h)
            h = F.relu_(h + cse(h))

        p = F.relu_(self.conv_p(h))
        v = F.relu_(self.conv_v(h))

        p_head = (p * x[:, :1]).view(h.size(0), h.size(1), -1).sum(-1)
        v_head = (v * x[:, :1]).view(h.size(0), h.size(1), -1).sum(-1)

        # Global Average Pooling
        p_gap = p.view(h.size(0), h.size(1), -1).mean(1)
        v_gap = v.view(h.size(0), h.size(1), -1).mean(1)

        hidden = {"p": p_gap, "v": v_gap}

        p = F.relu_(self.head_p1(torch.cat([p_head, p_gap], dim=1)))
        p = self.head_p2(p)
        v = torch.tanh(self.head_v(torch.cat([v_head, v_gap], dim=1)))

        return {'policy': p, 'value': v, 'hidden': hidden}


class GeeseNetA(nn.Module):
    class GeeseEncoder(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            # self.edge_size = (1, 2)
            self.d_model = d_model
            self.filters = d_model // 28

            self.conv0 = Conv2d(17, self.filters, (1, 1), True)

        def forward(self, x):
            x = self.conv0(x)

            h = x.view(x.size(0), self.filters, 7, 11)

            # make torus. size: (..., 9, 15)
            # h = torch.cat([h[:,:,:,-self.edge_size[1]:], h, h[:,:,:,:self.edge_size[1]]], dim=3)
            # h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)

            # drop edge 2 cols. size: (..., 7, 7)
            h = h[:, :, :, 2:-2]

            # Info: ['NORTH', 'SOUTH', 'WEST', 'EAST']
            # split into patch 7x4 and flatten
            n = torch.rot90(h[:, :, :4], 1, (2, 3)).reshape(h.size(0), -1)
            s = torch.rot90(h[:, :, 3:], 3, (2, 3)).reshape(h.size(0), -1)
            w = h[:, :, :, :4].reshape(h.size(0), -1)
            e = torch.rot90(h[:, :, :, 3:], 2, (2, 3)).reshape(h.size(0), -1)

            z = torch.stack([n, s, w, e])

            # z = z.permute(1, 0, 2)

            return z

    class GeeseBlock(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            self.ln = nn.LayerNorm(embed_dim)
            self.attention = nn.MultiheadAttention(embed_dim, num_heads)

        def forward(self, x):
            h = self.ln(x)
            h, _ = self.attention(h, h, h)
            h = h + x
            return h

    class GeeseControll(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.d_model = d_model

        def forward(self, x, e):
            # h = x.permute(1, 0, 2)
            h = x
            return h

    class GeeseHead(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.filters = d_model // 28
            self.head_v = nn.Linear(49, 1, bias=False)

        def forward(self, x):
            p = x.permute(1, 0, 2).mean(-1)

            v = x.view(x.size(0), x.size(1), self.filters, 7, 4)

            # Info: ['NORTH', 'SOUTH', 'WEST', 'EAST']
            n = torch.rot90(v[0], 3, (2, 3))
            s = torch.rot90(v[1], 1, (2, 3))
            w = v[2]
            e = torch.rot90(v[3], 2, (2, 3))

            n = F.pad(n, (0, 0, 0, 3, 0, 0, 0, 0), mode="constant", value=0)
            s = F.pad(s, (0, 0, 3, 0, 0, 0, 0, 0), mode="constant", value=0)
            w = F.pad(w, (0, 3, 0, 0, 0, 0, 0, 0), mode="constant", value=0)
            e = F.pad(e, (3, 0, 0, 0, 0, 0, 0, 0), mode="constant", value=0)

            v = n + s + w + e
            v = v.mean(1).view(x.size(1), -1)

            v = torch.tanh(self.head_v(v))
            return p, v

    def __init__(self):
        super().__init__()
        d_model = 224  # 28x8
        n_heads = 8
        blocks = 6

        # self.geese_net = GeeseNet()
        self.encoder = self.GeeseEncoder(d_model)

        self.blocks = nn.ModuleList([self.GeeseBlock(d_model, n_heads) for _ in range(blocks)])
        # self.gtrxl = GTrXL(d_model, n_heads, blocks)

        # self.control = self.GeeseControll(d_model)
        self.head = self.GeeseHead(d_model)

    def forward(self, x, _=None):
        # e = self.geese_net(x)["hidden"]
        # h = self.encoder(e["p"])
        h = self.encoder(x)

        for block in self.blocks:
            h = block(h)
        # h = self.gtrxl(h)

        # h = self.control(h, e)
        p, v = self.head(h)
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
        return GeeseNetA

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
        x = self.observation_centering_head(player)
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
