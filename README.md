<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [🦆 Hungry Geese 🦆](#-hungry-geese-)
  - [🎉 Month 5 Winners - Goose luck! 🎉](#-month-5-winners---goose-luck-)
  - [はじめに](#%E3%81%AF%E3%81%98%E3%82%81%E3%81%AB)
  - [ソリューション](#%E3%82%BD%E3%83%AA%E3%83%A5%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3)
  - [教師あり学習](#%E6%95%99%E5%B8%AB%E3%81%82%E3%82%8A%E5%AD%A6%E7%BF%92)
  - [強化学習 (HandyRL)](#%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92-handyrl)
    - [使い方](#%E4%BD%BF%E3%81%84%E6%96%B9)
      - [その他](#%E3%81%9D%E3%81%AE%E4%BB%96)
    - [学習](#%E5%AD%A6%E7%BF%92)
      - [ハイパーパラメータ](#%E3%83%8F%E3%82%A4%E3%83%91%E3%83%BC%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF)
      - [対戦相手](#%E5%AF%BE%E6%88%A6%E7%9B%B8%E6%89%8B)
  - [MCTS](#mcts)
  - [評価](#%E8%A9%95%E4%BE%A1)
  - [Kaggle への Submit](#kaggle-%E3%81%B8%E3%81%AE-submit)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 🦆 Hungry Geese 🦆

https://www.kaggle.com/c/hungry-geese


## 🎉 Month 5 Winners - Goose luck! 🎉

[Congratulations to our Month 5 Winners - Goose luck!](https://www.kaggle.com/c/hungry-geese/discussion/248986)

https://twitter.com/imokurity/status/1408578446645547012


## はじめに

このレポジトリは [HandyRL](https://github.com/DeNA/HandyRL) の fork です


## 私のソリューション

- 教師あり学習 + 強化学習 + MCTS (rating 1200くらい)

> チームのベストソリューションは、また別です。。😂


## 教師あり学習

[rating 1200 以上の agent の対戦履歴](https://www.kaggle.com/imokuri/hungrygeeseepisode)をもとに、
勝者の行動を教師データとして学習する

[Notebook](./hungry_geese_train_cnn.ipynb)


## 強化学習 (HandyRL)

### 使い方

- GPUサーバーで `python main.py -ts`
- CPUサーバーの `config.yaml` で
    - `num_parallel` を使用するCPU数にする
    - `server_address` をGPUサーバーのアドレスにする
- CPUサーバーで `python main.py -w`

#### その他

- GPUが遊んでいるようならGPUサーバーの `config.yaml` の `num_batchers` を増やす
- `config.yaml` の `batch_size` は GPUのメモリサイズに合わせる
    - ただし learning rate がこの `batch_size` に依存しているっぽい [このあたり](https://github.com/IMOKURI/Hungry-Geese/blob/825c94ead47638ed56479de87481838ee8a58bff/handyrl/train.py#L318-L322)
- `config.yaml` の `maximum_episodes` はGPUサーバーのメモリサイズに依存する (モデルサイズによるが128GBで 100万くらい)

### 学習

#### ハイパーパラメータ

[parameters.md](./docs/parameters.md) のメモ (間違っているかも)

- `gamma`: 先の報酬をどのくらい減らすか - 例えば gamma が 0.8 なら 10ステップ先の評価は 0.8 ^ 10 = 0.1 倍される
- `forward_steps`: 何ステップ先までの行動を評価の対象とするか
- `entropy_regularization`: 最適な行動ではなくランダムな行動をする確率 (Noisy Layerを使うなら0でよさそう)
- `update_episodes`: 何エピソードごとにモデルを更新するか (大きいほど安定するが時間がかかる)
- `lambda`: n が 1 だと最終的な報酬を重視, n が 0 だと直近の報酬を重視
- `policy_target`, `value_target`: 損失関数(行動評価用と状態価値評価用) (基本はTD)

#### 対戦相手

- 学習時の対戦相手は [ここで定義](https://github.com/IMOKURI/Hungry-Geese/blob/b0cebefa45b77cd07a19ccca996a18e5913857ab/handyrl/worker.py#L51-L57)
- 学習時の評価用の相手は [ここで定義](https://github.com/IMOKURI/Hungry-Geese/blob/09acf84a9ecec0cd67277a301f0959263c9c565f/handyrl/evaluation.py#L123)
- シュミレーション `python main.py -e` の対戦相手は [ここで定義](https://github.com/IMOKURI/Hungry-Geese/blob/09acf84a9ecec0cd67277a301f0959263c9c565f/handyrl/evaluation.py#L278-L284)

## MCTS

[こちら](https://www.kaggle.com/shoheiazuma/alphageese-baseline) を ~~パクる~~ 参考にして、以下の更新をした。

- 探索のたびに、(探索済みの局面も)推論を行う
    - 推論のたびに、ランダムにモデルが選ばれる(アンサンブル効果)
    - 推論のたびに、局面をランダムにスライドさせる
    - 推論のたびに、敵 3体のチャネルをシャッフルする
- ベストな行動は、直近での推論結果のみで決める
    - 推論結果は、4択しかなく、選択肢の差がモデルによっては付きやすいため
- ある確率で、推論結果にルールを適用する
    - それによって、正面衝突を回避しやすくする


## 評価

モデル同士で対戦し、成績の良いモデルを submit する

- [Notebook](https://www.kaggle.com/imokuri/hungry-geese-eval-models)
- [Notebook](https://www.kaggle.com/imokuri/hungry-geese-vs)


## Kaggle への Submit

- 学習済みモデルの upload: `make model`
- ソースコードの upload: `make source`
- [エージェント Notebook](./ds/submit/alpha/alpha-geese.ipynb)の submit: `make submit`
