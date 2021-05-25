# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# kaggle_environments licensed under Copyright 2020 Kaggle Inc. and the Apache License, Version 2.0
# (see https://github.com/Kaggle/kaggle-environments/blob/master/LICENSE for details)

# wrapper of Hungry Geese environment from kaggle

import importlib
import math
import random
from collections import defaultdict, deque

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
    def __init__(self, input_dim, output_dim, kernel_size, bn=True, stride=1):
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


class MultiGeeseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.offensive = GeeseNet()
        self.defensive = GeeseNet()

        layers, filters = 12, 32
        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])

        self.conv_p = TorusConv2d(filters, filters, (3, 3), True)
        self.conv_v = TorusConv2d(filters, filters, (3, 3), True)

        self.head_p1 = nn.Linear(filters * 2, filters, bias=False)
        self.head_p2 = nn.Linear(filters, 1, bias=False)
        self.head_v1 = nn.Linear(filters * 2, filters, bias=False)
        self.head_v2 = nn.Linear(filters, 1, bias=False)

    def forward(self, x, _=None):
        o = self.offensive(x)
        d = self.defensive(x)

        # Training danger rate
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))

        h_p = F.relu_(self.conv_p(h))
        h_head_p = (h_p * x[:, :1]).view(h_p.size(0), h_p.size(1), -1).sum(-1)
        h_avg_p = h_p.view(h_p.size(0), h_p.size(1), -1).mean(-1)

        h_p = F.relu_(self.head_p1(torch.cat([h_head_p, h_avg_p], 1)))
        drp = torch.sigmoid(self.head_p2(h_p))

        h_v = F.relu_(self.conv_v(h))
        h_head_v = (h_v * x[:, :1]).view(h_v.size(0), h_v.size(1), -1).sum(-1)
        h_avg_v = h_v.view(h_v.size(0), h_v.size(1), -1).mean(-1)

        h_v = F.relu_(self.head_v1(torch.cat([h_head_v, h_avg_v], 1)))
        drv = torch.sigmoid(self.head_v2(h_v))

        p = drp * o["policy"] + (1 - drp) * d["policy"]
        v = drv * o["value"] + (1 - drv) * d["value"]

        return {"policy": p, "value": v}


class GeeseNetAlpha(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 64
        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
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

    def head_tail_bonus(self, danger_rate=1.0):
        bonus_rate = 50

        bonus = {i: 0 for i in range(4)}

        if danger_rate < 0.5:
            return bonus

        try:
            prev_obs = self.obs_list[-2]
        except IndexError:
            return bonus

        tails = [goose[-1] for goose in prev_obs[0]["observation"]["geese"] if len(goose) > 0]

        obs = self.obs_list[-1][0]["observation"]["geese"]
        for i, goose in enumerate(obs):
            if len(goose) > 0 and goose[0] in tails:
                bonus[i] = bonus_rate * danger_rate

        return bonus

    def death_bonus(self, danger_rate=1.0):
        death_rate = 200

        try:
            prev_obs = self.obs_list[-2]
        except IndexError:
            prev_obs = None
        obs = self.obs_list[-1]

        num_alive = len([o for o in obs if o["status"] == "ACTIVE"])
        prev_num_alive = len([o for o in prev_obs if o["status"] == "ACTIVE"]) if prev_obs is not None else num_alive

        return (prev_num_alive - num_alive) * death_rate * danger_rate

    def move_to(self, x, a):
        actions = {
            "NORTH": (-1, 0),
            "SOUTH": (1, 0),
            "WEST": (0, -1),
            "EAST": (0, 1),
        }
        y = ((x[0] + actions[a][0]) % 7, (x[1] + actions[a][1]) % 11)
        return y

    # def reward(self):
    #     x = self.reward_default()
    #     # x = self.reward_offensive()
    #     # x = self.reward_defensive()
    #     return x

    def reward_default(self):
        """
        もともと以下の値となっている
        reward = steps survived * (configuration.max_length + 1) + goose length
        """
        obs = self.obs_list[-1]
        rewards = {}
        for p, o in enumerate(obs):
            rewards[p] = o["reward"]

        return rewards

    def reward_offensive(self):
        """
        長さ * 100 + 行動可能なマス数
        """
        obs = self.obs_list[-1]
        geese = obs[0]["observation"]["geese"]

        rewards = {}
        for p, o in enumerate(obs):
            length_reward = o["reward"] % 100 * 100

            if o["status"] == "ACTIVE":
                head = (self.to_row(0, geese[p][0]), self.to_col(0, geese[p][0]))
                field_reward = self.bfs(self.field, self.move_to(head, o["action"])).sum()
                rewards[p] = length_reward + field_reward
            else:
                rewards[p] = length_reward

        return rewards

    def reward_defensive(self):
        """
        default reward + head tail 報酬(50)  # + death数 * 200
        """
        obs = self.obs_list[-1]
        ht_bonus = self.head_tail_bonus()
        d_bonus = self.death_bonus()

        rewards = {}
        for p, o in enumerate(obs):
            if o["status"] == "ACTIVE":
                rewards[p] = o["reward"] + ht_bonus[p]  # + d_bonus
            else:
                rewards[p] = o["reward"]

        return rewards

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

    def empty_around_head(self, field, x):
        return [e for e in self.around(x) if field[e[0], e[1]] == 0]

    def food_around_head(self, head, food):
        food_ = [
            (self.to_row(0, f), self.to_col(0, f))
            for f in food
        ]
        for a in self.around(head):
            if a in food_:
                return True
        return False

    def bfs(self, field, head):
        q = deque([head])
        movable = np.zeros([7, 11])
        searched = defaultdict(bool)
        while len(q) != 0:
            v = q.popleft()
            movable[v] = 1
            searched[v] = True
            edges = [a for a in self.empty_around_head(field, v) if not searched[a]]
            for edge in edges:
                q.append(edge)
        return movable

    def observation(self, player=None):
        obses = []
        obses.append(self.observation_normal(player))
        # obses.append(self.observation_centering_head(player))
        # obses.append(self.observation_tip_as_food(player))
        # obses.append(self.observation_num_step(player))
        x = np.concatenate(obses)
        return x

    def observation_normal(self, player=None):
        if player is None:
            player = 0

        b = np.zeros((self.NUM_AGENTS * 4 + 1, 7 * 11), dtype=np.float32)
        # head = defaultdict(tuple)
        obs = self.obs_list[-1][0]['observation']

        for p, geese in enumerate(obs['geese']):
            pid = (p - player) % self.NUM_AGENTS

            # head position
            for pos in geese[:1]:
                b[0 + pid, pos] = 1
                # head[pid] = (self.to_row(0, pos), self.to_col(0, pos))
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

        # movable position
        # self.field = b[8:13].sum(0).reshape(7, 11)
        # b[17] = self.bfs(b[8:13].sum(0).reshape(7, 11), head[0]).reshape(-1)

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

    def observation_tip_as_food(self, player=None):
        if player is None:
            player = 0

        b = np.zeros((self.NUM_AGENTS * 4 + 1, 7 * 11), dtype=np.float32)
        obs_all = self.obs_list[-1]
        obs = obs_all[0]['observation']

        # Danger Rate
        num_geese = len([g for g in obs_all if g["status"] == "ACTIVE"])
        num_filled_cell = len([pos for geese in obs["geese"] for pos in geese])
        if num_geese == 2:
            danger_rate = min(1.0, num_filled_cell ** 2 / 3500)
        elif num_geese == 3:
            danger_rate = min(1.0, num_filled_cell ** 2 / 2500)
        else:
            danger_rate = min(1.0, num_filled_cell ** 2 / 2000)

        # Geese
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
        if danger_rate < 0.5:
            for pos in obs['food']:
                b[16, pos] = 1

        return b.reshape(-1, 7, 11)

    def observation_num_step(self, player=None):
        if player is None:
            player = 0

        b = np.zeros((7, 11), dtype=np.float32)
        obs_all = self.obs_list[-1]
        obs = obs_all[0]['observation']

        num_step = obs["step"]  # 0-198
        b[0, 0] = num_step / 198

        return b.reshape(1, 7, 11)


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
