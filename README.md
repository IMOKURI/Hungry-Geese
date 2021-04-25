# 🦆 Hungry Geese 🦆

https://www.kaggle.com/c/hungry-geese

## 使い方

- GPUサーバーで `python main.py -ts`
- CPUサーバーの `config.yaml` で `num_parallel` を使用するCPU数にする
- CPUサーバーで `python main.py -ts`

### その他

- GPUが遊んでいるようならGPUサーバーの `config.yaml` の `num_batchers` を増やす
- `config.yaml` の `batch_size` は GPUのメモリサイズに合わせる
    - ただし learning rate がこの `batch_size` に依存しているっぽい
- `config.yaml` の `maximum_episodes` はGPUサーバーのメモリサイズに依存する (128GBで 100万くらい)
