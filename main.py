# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import os
import sys
import yaml


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'

    with open('config.yaml') as f:
        args = yaml.safe_load(f)
    print(args)

    if len(sys.argv) < 2:
        print('Please set mode of HandyRL.')
        exit(1)

    mode = sys.argv[1]

    if mode == '--train' or mode == '-t':
        from handyrl.train import train_main as main
        main(args)
    elif mode == '--train-server' or mode == '-ts':
        import wandb

        config_defaults = {
            "gamma": args["train_args"]["gamma"],
            "forward_steps": args["train_args"]["forward_steps"],
            "entropy_regularization": args["train_args"]["entropy_regularization"],
            "entropy_regularization_decay": args["train_args"]["entropy_regularization_decay"],
            "update_episodes": args["train_args"]["update_episodes"],
            "batch_size": args["train_args"]["batch_size"],
            "minimum_episodes": args["train_args"]["minimum_episodes"],
            "maximum_episodes": args["train_args"]["maximum_episodes"],
            "lambda": args["train_args"]["lambda"],
            "policy_target": args["train_args"]["policy_target"],
            "value_target": args["train_args"]["value_target"],
            "seed": args["train_args"]["seed"],
        }

        if args["train_args"]["debug"]:
            wandb.init(project="hungry-geese-handyrl", config=config_defaults, mode="disabled")
        else:
            wandb.init(project="hungry-geese-handyrl", config=config_defaults)

        from handyrl.train import train_server_main as main
        main(args)
    elif mode == '--worker' or mode == '-w':
        from handyrl.worker import worker_main as main
        main(args)
    elif mode == '--eval' or mode == '-e':
        from handyrl.evaluation import eval_main as main
        main(args, sys.argv[2:])
    elif mode == '--eval-server' or mode == '-es':
        from handyrl.evaluation import eval_server_main as main
        main(args, sys.argv[2:])
    elif mode == '--eval-client' or mode == '-ec':
        from handyrl.evaluation import eval_client_main as main
        main(args, sys.argv[2:])
    else:
        print('Not found mode %s.' % mode)
