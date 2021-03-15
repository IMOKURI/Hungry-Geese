# Kaggle - Hungry Geese with HandyRL

## Training Configuration

### First Stage

<details>

<summary> `config.yaml` </summary>

```yaml
env_args:
    # env: 'TicTacToe'
    # source: 'handyrl.envs.tictactoe'
    # env: 'Geister'
    # source: 'handyrl.envs.geister'
    env: 'HungryGeese'
    source: 'handyrl.envs.kaggle.hungry_geese'


train_args:
    turn_based_training: False
    observation: False
    gamma: 0.8
    forward_steps: 32
    compress_steps: 4
    entropy_regularization: 2.0e-3
    entropy_regularization_decay: 0.3
    update_episodes: 500
    batch_size: 500  # GPU memory 12GB
    minimum_episodes: 10000
    maximum_episodes: 500000
    num_batchers: 8
    eval_rate: 0.1
    worker:
        num_parallel: 6
    lambda: 0.7
    policy_target: 'TD' # 'UPGO' 'VTRACE' 'TD' 'MC'
    value_target: 'TD' # 'VTRACE' 'TD' 'MC'
    seed: 2456
    restart_epoch: 0


worker_args:
    server_address: '127.0.0.1'
    num_parallel: 6

```

</details>

### Second Stage

<details>

<summary> `config.yaml` </summary>

```yaml
env_args:
    # env: 'TicTacToe'
    # source: 'handyrl.envs.tictactoe'
    # env: 'Geister'
    # source: 'handyrl.envs.geister'
    env: 'HungryGeese'
    source: 'handyrl.envs.kaggle.hungry_geese'


train_args:
    turn_based_training: False
    observation: False
    gamma: 0.9
    forward_steps: 32
    compress_steps: 4
    entropy_regularization: 2.0e-3
    entropy_regularization_decay: 0.3    update_episodes: 500
    batch_size: 500  # GPU memory 12GB
    minimum_episodes: 10000
    maximum_episodes: 500000
    num_batchers: 8
    eval_rate: 0.1
    worker:
        num_parallel: 6
    lambda: 0.8
    policy_target: 'UPGO' # 'UPGO' 'VTRACE' 'TD' 'MC'
    value_target: 'TD' # 'VTRACE' 'TD' 'MC'
    seed: 2789
    restart_epoch: 2237


worker_args:
    server_address: '127.0.0.1'
    num_parallel: 6

```

</details>






