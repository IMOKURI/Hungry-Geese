# 🦆 Hungry Geese 🦆

https://www.kaggle.com/c/hungry-geese

## 使い方

- GPUサーバーで `python main.py -ts`
- CPUサーバーの `config.yaml` で
    - `num_parallel` を使用するCPU数にする
    - `server_address` をGPUサーバーのアドレスにする
- CPUサーバーで `python main.py -ts`

### その他

- GPUが遊んでいるようならGPUサーバーの `config.yaml` の `num_batchers` を増やす
- `config.yaml` の `batch_size` は GPUのメモリサイズに合わせる
    - ただし learning rate がこの `batch_size` に依存しているっぽい [このあたり](https://github.com/IMOKURI/Hungry-Geese/blob/825c94ead47638ed56479de87481838ee8a58bff/handyrl/train.py#L318-L322)
- `config.yaml` の `maximum_episodes` はGPUサーバーのメモリサイズに依存する (128GBで 100万くらい)

## ハイパーパラメータ

[parameters.md](./docs/parameters.md) のメモ (間違っているかも)

- `gamma`: 先の報酬をどのくらい減らすか - 例えば gamma が 0.8 なら 10ステップ先の評価は 0.8 ^ 10 = 0.1 倍される
- `forward_steps`: 何ステップ先までの行動を評価の対象とするか
- `entropy_regularization`: 最適な行動ではなくランダムな行動をする確率 (Noisy Layerを使うなら0でよさそう)
- `update_episodes`: 何エピソードごとにモデルを更新するか (大きいほど安定するが時間がかかる)
- `lambda`: n が 1 だと最終的な報酬を重視, n が 0 だと直近の報酬を重視
- `policy_target`, `value_target`: 損失関数(行動評価用と状態価値評価用) (基本はTD)

## 対戦相手

- 学習時の対戦相手は直前の自分のモデル
- 学習時の評価用の相手は [ここで定義](https://github.com/IMOKURI/Hungry-Geese/blob/09acf84a9ecec0cd67277a301f0959263c9c565f/handyrl/evaluation.py#L123)
- シュミレーション `python main.py -e` の対戦相手は [ここで定義](https://github.com/IMOKURI/Hungry-Geese/blob/09acf84a9ecec0cd67277a301f0959263c9c565f/handyrl/evaluation.py#L278-L284)
