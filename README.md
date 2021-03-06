<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [ð¦ Hungry Geese ð¦](#-hungry-geese-)
  - [ð Month 5 Winners - Goose luck! ð](#-month-5-winners---goose-luck-)
  - [ã¯ããã«](#%E3%81%AF%E3%81%98%E3%82%81%E3%81%AB)
  - [ã½ãªã¥ã¼ã·ã§ã³](#%E3%82%BD%E3%83%AA%E3%83%A5%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3)
  - [æå¸«ããå­¦ç¿](#%E6%95%99%E5%B8%AB%E3%81%82%E3%82%8A%E5%AD%A6%E7%BF%92)
  - [å¼·åå­¦ç¿ (HandyRL)](#%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92-handyrl)
    - [ä½¿ãæ¹](#%E4%BD%BF%E3%81%84%E6%96%B9)
      - [ãã®ä»](#%E3%81%9D%E3%81%AE%E4%BB%96)
    - [å­¦ç¿](#%E5%AD%A6%E7%BF%92)
      - [ãã¤ãã¼ãã©ã¡ã¼ã¿](#%E3%83%8F%E3%82%A4%E3%83%91%E3%83%BC%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF)
      - [å¯¾æ¦ç¸æ](#%E5%AF%BE%E6%88%A6%E7%9B%B8%E6%89%8B)
  - [MCTS](#mcts)
  - [è©ä¾¡](#%E8%A9%95%E4%BE%A1)
  - [Kaggle ã¸ã® Submit](#kaggle-%E3%81%B8%E3%81%AE-submit)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# ð¦ Hungry Geese ð¦

https://www.kaggle.com/c/hungry-geese


## ð Month 5 Winners - Goose luck! ð

[Congratulations to our Month 5 Winners - Goose luck!](https://www.kaggle.com/c/hungry-geese/discussion/248986)

https://twitter.com/imokurity/status/1408578446645547012


## ã¯ããã«

ãã®ã¬ãã¸ããªã¯ [HandyRL](https://github.com/DeNA/HandyRL) ã® fork ã§ã


## ç§ã®ã½ãªã¥ã¼ã·ã§ã³

- æå¸«ããå­¦ç¿ + å¼·åå­¦ç¿ + MCTS (rating 1200ããã)

> ãã¼ã ã®ãã¹ãã½ãªã¥ã¼ã·ã§ã³ã¯ãã¾ãå¥ã§ãããð


## æå¸«ããå­¦ç¿

[rating 1200 ä»¥ä¸ã® agent ã®å¯¾æ¦å±¥æ­´](https://www.kaggle.com/imokuri/hungrygeeseepisode)ããã¨ã«ã
åèã®è¡åãæå¸«ãã¼ã¿ã¨ãã¦å­¦ç¿ãã

[Notebook](./hungry_geese_train_cnn.ipynb)


## å¼·åå­¦ç¿ (HandyRL)

### ä½¿ãæ¹

- GPUãµã¼ãã¼ã§ `python main.py -ts`
- CPUãµã¼ãã¼ã® `config.yaml` ã§
    - `num_parallel` ãä½¿ç¨ããCPUæ°ã«ãã
    - `server_address` ãGPUãµã¼ãã¼ã®ã¢ãã¬ã¹ã«ãã
- CPUãµã¼ãã¼ã§ `python main.py -w`

#### ãã®ä»

- GPUãéãã§ãããããªãGPUãµã¼ãã¼ã® `config.yaml` ã® `num_batchers` ãå¢ãã
- `config.yaml` ã® `batch_size` ã¯ GPUã®ã¡ã¢ãªãµã¤ãºã«åããã
    - ãã ã learning rate ããã® `batch_size` ã«ä¾å­ãã¦ããã£ã½ã [ãã®ããã](https://github.com/IMOKURI/Hungry-Geese/blob/825c94ead47638ed56479de87481838ee8a58bff/handyrl/train.py#L318-L322)
- `config.yaml` ã® `maximum_episodes` ã¯GPUãµã¼ãã¼ã®ã¡ã¢ãªãµã¤ãºã«ä¾å­ãã (ã¢ãã«ãµã¤ãºã«ããã128GBã§ 100ä¸ããã)

### å­¦ç¿

#### ãã¤ãã¼ãã©ã¡ã¼ã¿

[parameters.md](./docs/parameters.md) ã®ã¡ã¢ (ééã£ã¦ãããã)

- `gamma`: åã®å ±é¬ãã©ã®ãããæ¸ããã - ä¾ãã° gamma ã 0.8 ãªã 10ã¹ãããåã®è©ä¾¡ã¯ 0.8 ^ 10 = 0.1 åããã
- `forward_steps`: ä½ã¹ãããåã¾ã§ã®è¡åãè©ä¾¡ã®å¯¾è±¡ã¨ããã
- `entropy_regularization`: æé©ãªè¡åã§ã¯ãªãã©ã³ãã ãªè¡åãããç¢ºç (Noisy Layerãä½¿ããªã0ã§ãããã)
- `update_episodes`: ä½ã¨ãã½ã¼ããã¨ã«ã¢ãã«ãæ´æ°ããã (å¤§ããã»ã©å®å®ãããæéãããã)
- `lambda`: n ã 1 ã ã¨æçµçãªå ±é¬ãéè¦, n ã 0 ã ã¨ç´è¿ã®å ±é¬ãéè¦
- `policy_target`, `value_target`: æå¤±é¢æ°(è¡åè©ä¾¡ç¨ã¨ç¶æä¾¡å¤è©ä¾¡ç¨) (åºæ¬ã¯TD)

#### å¯¾æ¦ç¸æ

- å­¦ç¿æã®å¯¾æ¦ç¸æã¯ [ããã§å®ç¾©](https://github.com/IMOKURI/Hungry-Geese/blob/b0cebefa45b77cd07a19ccca996a18e5913857ab/handyrl/worker.py#L51-L57)
- å­¦ç¿æã®è©ä¾¡ç¨ã®ç¸æã¯ [ããã§å®ç¾©](https://github.com/IMOKURI/Hungry-Geese/blob/09acf84a9ecec0cd67277a301f0959263c9c565f/handyrl/evaluation.py#L123)
- ã·ã¥ãã¬ã¼ã·ã§ã³ `python main.py -e` ã®å¯¾æ¦ç¸æã¯ [ããã§å®ç¾©](https://github.com/IMOKURI/Hungry-Geese/blob/09acf84a9ecec0cd67277a301f0959263c9c565f/handyrl/evaluation.py#L278-L284)

## MCTS

[ãã¡ã](https://www.kaggle.com/shoheiazuma/alphageese-baseline) ã ~~ãã¯ã~~ åèã«ãã¦ãä»¥ä¸ã®æ´æ°ãããã

- æ¢ç´¢ã®ãã³ã«ã(æ¢ç´¢æ¸ã¿ã®å±é¢ã)æ¨è«ãè¡ã
    - æ¨è«ã®ãã³ã«ãã©ã³ãã ã«ã¢ãã«ãé¸ã°ãã(ã¢ã³ãµã³ãã«å¹æ)
    - æ¨è«ã®ãã³ã«ãå±é¢ãã©ã³ãã ã«ã¹ã©ã¤ãããã
    - æ¨è«ã®ãã³ã«ãæµ 3ä½ã®ãã£ãã«ãã·ã£ããã«ãã
- ãã¹ããªè¡åã¯ãç´è¿ã§ã®æ¨è«çµæã®ã¿ã§æ±ºãã
    - æ¨è«çµæã¯ã4æãããªããé¸æè¢ã®å·®ãã¢ãã«ã«ãã£ã¦ã¯ä»ãããããã
- ããç¢ºçã§ãæ¨è«çµæã«ã«ã¼ã«ãé©ç¨ãã
    - ããã«ãã£ã¦ãæ­£é¢è¡çªãåé¿ãããããã


## è©ä¾¡

ã¢ãã«åå£«ã§å¯¾æ¦ããæç¸¾ã®è¯ãã¢ãã«ã submit ãã

- [Notebook](https://www.kaggle.com/imokuri/hungry-geese-eval-models)
- [Notebook](https://www.kaggle.com/imokuri/hungry-geese-vs)


## Kaggle ã¸ã® Submit

- å­¦ç¿æ¸ã¿ã¢ãã«ã® upload: `make model`
- ã½ã¼ã¹ã³ã¼ãã® upload: `make source`
- [ã¨ã¼ã¸ã§ã³ã Notebook](./ds/submit/alpha/alpha-geese.ipynb)ã® submit: `make submit`
