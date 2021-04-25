from collections import defaultdict

import numpy as np

NUM_ROW = 7
NUM_COL = 11
CENTER_ROW = NUM_ROW // 2
CENTER_COL = NUM_COL // 2


def to_offset(x):
    row = CENTER_ROW - x // NUM_COL
    col = CENTER_COL - x % NUM_COL
    return row, col


def to_row(offset, x):
    return (x // NUM_COL + offset) % NUM_ROW


def to_col(offset, x):
    return (x + offset) % NUM_COL


def make_input_centering_head_for_rule(obses):
    b = {}
    for i in range(4):
        b[i] = defaultdict(list)
    obs = obses[-1]

    player_goose_head = obs["geese"][obs["index"]][0]
    o_row, o_col = to_offset(player_goose_head)

    for p, geese in enumerate(obs["geese"]):
        # whole position
        for pos in geese:
            b[(p - obs["index"]) % 4]["body"].append((to_row(o_row, pos), to_col(o_col, pos)))

    # previous head position
    if len(obses) > 1:
        obs_prev = obses[-2]
        for p, geese in enumerate(obs_prev["geese"]):
            for pos in geese[:1]:
                if (p - obs["index"]) % 4 == 0:
                    b[0]["previous"].append((to_row(o_row, pos), to_col(o_col, pos)))

    # food
    for pos in obs["food"]:
        b[0]["food"].append((to_row(o_row, pos), to_col(o_col, pos)))

    return b


def distance(a, b):
    x = b[0] - a[0]
    y = b[1] - a[1]
    return (x, y), abs(x) + abs(y)


def around(a):
    return [
        (a[0] - 1, a[1]),
        (a[0] + 1, a[1]),
        (a[0], a[1] - 1),
        (a[0], a[1] + 1),
    ]


def apply_rule(b, prob):
    """
    player head = (3, 5)
    ["NORTH", "SOUTH", "WEST", "EAST"]
    """
    north = (2, 5)
    south = (4, 5)
    west = (3, 4)
    east = (3, 6)
    neighbor = [north, south, west, east]

    # 隣接している場所に行けないケース
    for i, n in enumerate(neighbor):
        # 自分の直前の場所
        if n in b[0]["previous"]:
            prob[i] = -np.inf

        for p in range(4):
            # ガチョウの体がある場所 (しっぽ除く)
            if n in b[p]["body"][:-1]:
                prob[i] = -np.inf

    north_2step = [(2, 4), (1, 5), (2, 6)]
    south_2step = [(4, 4), (5, 5), (4, 6)]
    west_2step = [(2, 4), (3, 3), (4, 4)]
    east_2step = [(2, 6), (3, 7), (4, 6)]
    two_step = [north_2step, south_2step, west_2step, east_2step]

    # 2step 先のマスがすべて2step後に埋まっている場合移動不可とする
    for i, ts in enumerate(two_step):
        death = 0
        for s in ts:
            for p in range(4):
                # 体がある場合
                if s in b[p]["body"][:-2]:
                    death += 1
                    break
                # しっぽがあるけど周りに頭があってすぐ埋まりそうな場合
                if (
                    s in b[p]["body"][-2:]
                    and any(
                        (b[pp]["body"] != [] and b[pp]["body"][0] in around(b[p]["body"][-1]))
                        for pp in range(1, 4)
                    )
                ):
                    death += 0.5
                    break

            else:
                break
        if death == len(ts):
            prob[i] -= 10_000
        elif death == len(ts) - 0.5:
            prob[i] -= 5_000
            
    # 3step 先のまずがすべて埋まっているケース
    

    # 次の移動で頭がぶつかる可能性のあるケース
    for p in range(1, 4):
        if b[p]["body"] != []:
            (x, y), d = distance(b[0]["body"][0], b[p]["body"][0])
            if d == 2:
                if x < 0:
                    prob[0] -= 100 if north in b[0]["food"] else 50
                elif x > 0:
                    prob[1] -= 100 if south in b[0]["food"] else 50
                if y < 0:
                    prob[2] -= 100 if west in b[0]["food"] else 50
                elif y > 0:
                    prob[3] -= 100 if east in b[0]["food"] else 50

    # しっぽが伸びる可能性のあるケース
    for i, n in enumerate(neighbor):
        for p in range(1, 4):
            if (
                n in b[p]["body"][-1:]
                and any(food in around(b[p]["body"][0]) for food in b[0]["food"])
            ):
                prob[i] -= 100

    return prob
