import numpy as np
import poc.rule as R


def test_to_offset():
    row, col = R.to_offset(38)
    assert row == 0
    assert col == 0

    row, col = R.to_offset(25)
    assert row == 1
    assert col == 2

def test_to_row_to_col():
    assert R.to_row(1, 17) == 2
    assert R.to_col(2, 17) == 8

    assert R.to_row(1, 75) == 0
    assert R.to_col(2, 75) == 0

def test_own_previous_position():
    prob = list(range(4))
    obses = [
        {
            'geese': [[1], [], [], []],
            'food': [61, 62],
            'index': 0
        },
        {
            'geese': [[2], [], [], []],
            'food': [61, 62],
            'index': 0
        },
    ]
    prob = R.apply_rule(R.make_input_centering_head_for_rule(obses), prob)
    assert prob == [0, 1, -np.inf, 3]

def test_body_neighbor():
    prob = list(range(4))
    obses = [
        {
            'geese': [[1], [], [], []],
            'food': [61, 62],
            'index': 0
        },
        {
            'geese': [[2], [13, 14], [3], []],
            'food': [61, 62],
            'index': 0
        },
    ]
    prob = R.apply_rule(R.make_input_centering_head_for_rule(obses), prob)
    assert prob == [0, -np.inf, -np.inf, 3]

def test_body_neighbor():
    prob = list(range(4))
    obses = [
        {
            'geese': [[37], [], [], []],
            'food': [1, 2],
            'index': 0
        },
        {
            'geese': [[38], [17, 28, 29, 40, 51, 50, 61, 62], [], []],
            'food': [1, 2],
            'index': 0
        },
    ]
    prob = R.apply_rule(R.make_input_centering_head_for_rule(obses), prob)
    assert prob == [0, 1, -np.inf, -9997]

    prob = list(range(4))
    obses = [
        {
            'geese': [[37], [], [], []],
            'food': [1, 2],
            'index': 0
        },
        {
            'geese': [[38], [17, 28, 29, 40, 51, 50, 61], [], []],
            'food': [1, 2],
            'index': 0
        },
    ]
    prob = R.apply_rule(R.make_input_centering_head_for_rule(obses), prob)
    assert prob == [0, 1, -np.inf, 3]

def test_body_neighbor_with_food():
    prob = list(range(4))
    obses = [
        {
            'geese': [[37], [], [], []],
            'food': [1, 2],
            'index': 0
        },
        {
            'geese': [[38], [17, 28, 29, 40, 51, 50, 61], [62], []],
            'food': [1, 2],
            'index': 0
        },
    ]
    prob = R.apply_rule(R.make_input_centering_head_for_rule(obses), prob)
    assert prob == [0, 1, -np.inf, -4997]

    prob = list(range(4))
    obses = [
        {
            'geese': [[37], [], [], []],
            'food': [1, 2],
            'index': 0
        },
        {
            'geese': [[38], [17, 28, 29, 40, 51, 50], [61], []],
            'food': [1, 2],
            'index': 0
        },
    ]
    prob = R.apply_rule(R.make_input_centering_head_for_rule(obses), prob)
    assert prob == [0, 1, -np.inf, -4997]

def test_2step_to_opponent_head():
    prob = list(range(4))
    obses = [
        {
            'geese': [[37], [], [], []],
            'food': [1, 2],
            'index': 0
        },
        {
            'geese': [[38], [50], [], []],
            'food': [1, 39],
            'index': 0
        },
    ]
    prob = R.apply_rule(R.make_input_centering_head_for_rule(obses), prob)
    assert prob == [0, -49, -np.inf, -97]

def test_grow_tip():
    prob = list(range(4))
    obses = [
        {
            'geese': [[37], [], [], []],
            'food': [1, 2],
            'index': 0
        },
        {
            'geese': [[38], [41, 40, 39], [], []],
            'food': [1, 42],
            'index': 0
        },
    ]
    prob = R.apply_rule(R.make_input_centering_head_for_rule(obses), prob)
    assert prob == [0, 1, -np.inf, -97]
