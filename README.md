# Kaggle - Hungry Geese with HandyRL

## Training Configuration

### First Stage

<details>

<summary> config.yaml </summary>

[Link](https://github.com/IMOKURI/Hungry-Geese2/blob/007e3c7be02fafb2cbe0e379d9fe96957df505a7/config.yaml)

```yaml
env_args:
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

- [[HandyRL] Enjoy Distributed Reinforcement Learning (Rating ~1100, February 10)](https://www.kaggle.com/c/hungry-geese/discussion/218190)

### Second Stage

<details>

<summary> config.yaml </summary>

[Link](https://github.com/IMOKURI/Hungry-Geese2/blob/7a50dda9737653480f15f241d5aa9326aafbe1f9/config.yaml)

```diff
diff --git config.yaml config.yaml
index 0fce8a6..e58770d 100755
--- config.yaml
+++ config.yaml
@@ -11,7 +11,7 @@ env_args:
 train_args:
     turn_based_training: False
     observation: False
-    gamma: 0.8
+    gamma: 0.9
     forward_steps: 32
     compress_steps: 4
     entropy_regularization: 2.0e-3
@@ -24,11 +24,11 @@ train_args:
     eval_rate: 0.1
     worker:
         num_parallel: 6
-    lambda: 0.7
-    policy_target: 'TD' # 'UPGO' 'VTRACE' 'TD' 'MC'
+    lambda: 0.8
+    policy_target: 'UPGO' # 'UPGO' 'VTRACE' 'TD' 'MC'
     value_target: 'TD' # 'VTRACE' 'TD' 'MC'
-    seed: 2456
-    restart_epoch: 0
+    seed: 2789
+    restart_epoch: 2237


 worker_args:
```

</details>






