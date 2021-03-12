# Training output

Something like this.

```
epoch 35
win rate = 0.436 (80.2 / 184)
generation stats = -0.000 +- 0.729
loss = p:0.008 v:0.018 ent:0.640 total:0.025
updated model(794)
```

- `win rate`: the winning rate against the random agent. opponent agent should set `Evaluator` class and `eval_main` function in `evaluation.py`
- `generation stats`: average and standard deviation of outcome. training opponent is myself, so average is almost zero.
- `loss`: p is policy loss, v is value loss, ent is entropy of the policy, and total is total loss. Total loss is an weighted sum of each loss components. If the loss reached NaN or some big value, there should be something failed.
    - エントロピーはすべての手が均等に選ばれる場合に最大となり特定の1手の確率が高くなる場合低い値になる
    - すなわちエントロピーが低くなる = ある局面での最適解がわかってきている (それが局所解になっていないかは注意が必要)
