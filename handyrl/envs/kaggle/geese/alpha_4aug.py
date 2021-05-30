
# This is a lightweight ML agent trained by self-play.
# After sharing this notebook,
# we will add Hungry Geese environment in our HandyRL library.
# https://github.com/DeNA/HandyRL
# We hope you enjoy reinforcement learning!


import base64
import bz2
import math
import pickle
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.set_printoptions(formatter={'float': '{:.4f}'.format})


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

        return {"policy": p, "value": v}
        

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

        return {"policy": p, "value": v}


# Input for Neural Network


next_position_map = {}
for pos in range(77):
    position = []
    position.append((11 * (1 + pos // 11) + pos % 11) % 77)
    position.append((11 * (-1 + pos // 11) + pos % 11) % 77)
    position.append((11 * (pos // 11) + (pos + 1) % 11) % 77)
    position.append((11 * (pos // 11) + (pos - 1) % 11) % 77)
    next_position_map[pos] = set(position)


def make_input(obses):
    b = np.zeros((17, 7 * 11), dtype=np.float32)
    obs = obses[-1]

    for p, pos_list in enumerate(obs["geese"]):
        # head position
        for pos in pos_list[:1]:
            b[0 + (p - obs["index"]) % 4, pos] = 1
        # tip position
        for pos in pos_list[-1:]:
            b[4 + (p - obs["index"]) % 4, pos] = 1
        # whole position
        for pos in pos_list:
            b[8 + (p - obs["index"]) % 4, pos] = 1

    # previous head position
    if len(obses) > 1:
        obs_prev = obses[-2]
        for p, pos_list in enumerate(obs_prev["geese"]):
            for pos in pos_list[:1]:
                b[12 + (p - obs["index"]) % 4, pos] = 1

    # food
    for pos in obs["food"]:
        b[16, pos] = 1

    return b.reshape(-1, 7, 11)


def get_reverse_cube(obses):
    """
    尻尾から順番に 1, 0.9, 0.8, ... という並び
    """
    b = np.zeros((4, 7 * 11), dtype=np.float32)
    obs = obses[-1]

    for p, geese in enumerate(obs["geese"]):
        # whole position reverse
        for num_reverse, pos in enumerate(geese[::-1]):
            b[(p - obs["index"]) % 4, pos] = 1 - num_reverse * 0.1

    return b.reshape(-1, 7, 11)


def get_next_disappear_cube(obses):
    """
    次になくなる場所: 1
    次になくなる可能性のある場所: 0.5
    """
    b = np.zeros((4, 7 * 11), dtype=np.float32)
    obs = obses[-1]
    step = obs["step"]

    # foodを食べる可能性があるか。
    eat_food_possibility = defaultdict(int)
    for p, geese in enumerate(obs["geese"]):
        for pos in geese[:1]:
            if not next_position_map[pos].isdisjoint(obs["food"]):
                eat_food_possibility[p] = 1

    if (step % 40) == 39:  # 1つ短くなる
        for p, geese in enumerate(obs["geese"]):
            if eat_food_possibility[p]:  # 尻尾が1、尻尾の１つ前0.5
                for pos in geese[-1:]:
                    b[(p - obs["index"]) % 4, pos] = 1
                for pos in geese[-2:-1]:
                    b[(p - obs["index"]) % 4, pos] = 0.5
            else:  # 食べる可能性なし -> 尻尾が1, 尻尾の1つ前1
                for pos in geese[-2:]:
                    b[(p - obs["index"]) % 4, pos] = 1
    else:  # 1つ短くならない
        for p, geese in enumerate(obs["geese"]):
            if eat_food_possibility[p]:  # 食べる可能性があり -> 尻尾を0.5
                for pos in geese[-1:]:
                    b[(p - obs["index"]) % 4, pos] = 0.5
            else:  # 食べる可能性なし # 尻尾を1
                for pos in geese[-1:]:
                    b[(p - obs["index"]) % 4, pos] = 1

    return b.reshape(-1, 7, 11)


def get_step_cube_v2(obses):
    """
    step0: 0, step199: 1
    step0: 0, step39 + 40n: 1
    """
    b = np.zeros((1, 7, 11), dtype=np.float32)
    obs = obses[-1]
    step = obs["step"]

    b[:, :, :5] = (step % 200) / 199
    b[:, :, 5:] = (step % 40) / 39

    return b


def get_length_cube(obses):
    b = np.zeros((2, 7, 11), dtype=np.float32)
    obs = obses[-1]

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
# model = GeeseNet()
model = GeeseNetAlpha()
model.load_state_dict(state_dict)
model.eval()


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


def predict_action(cube_board, transform, cube_info=None):
    cube_board, slices = transform(cube_board)
    if cube_info is not None:
        cube = np.concatenate([cube_board, cube_info], axis=0)
    else:
        cube = cube_board

    with torch.no_grad():
        cube_tensor = torch.from_numpy(cube).unsqueeze(0)
        out = model(cube_tensor)
    p = out['policy'].squeeze(0).detach().numpy()
    p = p[slices]
    return p


# Main Function of Agent

obses = []


def agent(obs, _):
    obses.append(obs)

    obses_ = []
    obses_.append(make_input(obses))
    obses_.append(get_reverse_cube(obses))
    obses_.append(get_next_disappear_cube(obses))
    obses_.append(get_step_cube_v2(obses))
    obses_.append(get_length_cube(obses))
    x = np.concatenate(obses_)

    p_i = predict_action(x, identity)
    p_h = predict_action(x, horizontal_flip)
    p_v = predict_action(x, vertical_flip)
    p_hv = predict_action(x, horizontal_vertical_flip)

    actions = ["NORTH", "SOUTH", "WEST", "EAST"]
    pred = (p_i + p_h + p_v + p_hv) / 4
    print(f"player: {obs['index']}, pred: {pred}")
    return actions[np.argmax(pred)]