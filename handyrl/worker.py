# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# worker and gather

import functools
import multiprocessing as mp
import pickle
import random
import threading
import time
from collections import deque
from socket import gethostname

from .connection import (QueueCommunicator, accept_socket_connections,
                         connect_socket_connection,
                         open_multiprocessing_connections, send_recv)
from .environment import make_env, prepare_env
from .envs.kaggle.hungry_geese import (pre_train_model, random_model_model, smart_model_model)
from .envs.kaggle.hungry_geese_center_5 import (Agent005_9321_model,
                                                Agent005_13726_model,
                                                Agent005_17669_model,
                                                Agent005_18128_model,
                                                Agent005_21099_model)
from .envs.kaggle.hungry_geese_center_6 import (Agent006_13777_model,
                                                Agent006_17586_model,
                                                Agent006_22300_model,
                                                Agent006_24147_model)
from .evaluation import Evaluator
from .generation import Generator
from .model import ModelWrapper


class Worker:
    def __init__(self, args, conn, wid):
        print('opened worker %d' % wid)
        self.worker_id = wid
        self.args = args
        self.conn = conn
        self.latest_model = -1, None

        env = make_env({**args['env'], 'id': wid})
        self.generator = Generator(env, self.args)
        self.evaluator = Evaluator(env, self.args)

        random.seed(args['seed'] + wid)

    def __del__(self):
        print('closed worker %d' % self.worker_id)

    def _gather_models(self, model_ids, args):
        model_pool = {}
        for model_id in model_ids:
            if model_id not in model_pool:
                if model_id < 0:
                    model_pool[model_id] = None
                    if args['role'] == 'g':
                        models = {
                            random_model_model: 10,
                            smart_model_model: 100,
                            pre_train_model: 100,
                            Agent005_9321_model: 10,
                            Agent005_13726_model: 10,
                            Agent005_17669_model: 10,
                            Agent005_18128_model: 10,
                            Agent005_21099_model: 10,
                            Agent006_13777_model: 10,
                            Agent006_17586_model: 10,
                            Agent006_22300_model: 10,
                            Agent006_24147_model: 10,
                        }

                        def normalize(w):
                            s = sum(w)
                            return [p / s for p in w]

                        agent_ = random.choices(list(models.keys()), k=1, weights=normalize(list(models.values())))[0]
                        model_pool[model_id] = agent_

                elif model_id == self.latest_model[0]:
                    # use latest model
                    model_pool[model_id] = self.latest_model[1]
                else:
                    # get model from server
                    model_pool[model_id] = ModelWrapper(pickle.loads(send_recv(self.conn, ('model', model_id))))
                    # update latest model
                    if model_id > self.latest_model[0]:
                        self.latest_model = model_id, model_pool[model_id]
        return model_pool

    def run(self):
        while True:
            args = send_recv(self.conn, ('args', None))
            role = args['role']

            models = {}
            if 'model_id' in args:
                model_ids = list(args['model_id'].values())
                model_pool = self._gather_models(model_ids, args)

                # make dict of models
                for p, model_id in args['model_id'].items():
                    models[p] = model_pool[model_id]

            if role == 'g':
                episode = self.generator.execute(models, args)
                send_recv(self.conn, ('episode', episode))
            elif role == 'e':
                result = self.evaluator.execute(models, args)
                send_recv(self.conn, ('result', result))


def make_worker_args(args, n_ga, gaid, wid, conn):
    return args, conn, wid * n_ga + gaid


def open_worker(args, conn, wid):
    worker = Worker(args, conn, wid)
    worker.run()


class Gather(QueueCommunicator):
    def __init__(self, args, conn, gaid):
        print('started gather %d' % gaid)
        super().__init__()
        self.gather_id = gaid
        self.server_conn = conn
        self.args_queue = deque([])
        self.data_map = {'model': {}}
        self.result_send_map = {}
        self.result_send_cnt = 0

        n_pro, n_ga = args['worker']['num_parallel'], args['worker']['num_gathers']

        num_workers_per_gather = (n_pro // n_ga) + int(gaid < n_pro % n_ga)
        worker_conns = open_multiprocessing_connections(
            num_workers_per_gather,
            open_worker,
            functools.partial(make_worker_args, args, n_ga, gaid)
        )

        for conn in worker_conns:
            self.add_connection(conn)

        self.args_buf_len = 1 + len(worker_conns) // 4
        self.result_buf_len = 1 + len(worker_conns) // 4

    def __del__(self):
        print('finished gather %d' % self.gather_id)

    def run(self):
        while True:
            conn, (command, args) = self.recv()
            if command == 'args':
                # When requested arguments, return buffered outputs
                if len(self.args_queue) == 0:
                    # get multiple arguments from server and store them
                    self.server_conn.send((command, [None] * self.args_buf_len))
                    self.args_queue += self.server_conn.recv()

                next_args = self.args_queue.popleft()
                self.send(conn, next_args)

            elif command in self.data_map:
                # answer data request as soon as possible
                data_id = args
                if data_id not in self.data_map[command]:
                    self.server_conn.send((command, args))
                    self.data_map[command][data_id] = self.server_conn.recv()
                self.send(conn, self.data_map[command][data_id])

            else:
                # return flag first and store data
                self.send(conn, None)
                if command not in self.result_send_map:
                    self.result_send_map[command] = []
                self.result_send_map[command].append(args)
                self.result_send_cnt += 1

                if self.result_send_cnt >= self.result_buf_len:
                    # send datum to server after buffering certain number of datum
                    for command, args_list in self.result_send_map.items():
                        self.server_conn.send((command, args_list))
                        self.server_conn.recv()
                    self.result_send_map = {}
                    self.result_send_cnt = 0


def gather_loop(args, conn, gaid):
    try:
        gather = Gather(args, conn, gaid)
        gather.run()
    finally:
        gather.shutdown()


class WorkerCluster(QueueCommunicator):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        if self.args['remote']:
            # prepare listening connections
            def worker_server(port):
                conn_acceptor = accept_socket_connections(port=port, timeout=0.5)
                print('started worker server %d' % port)
                while not self.shutdown_flag:  # use super class's flag
                    conn = next(conn_acceptor)
                    if conn is not None:
                        self.add_connection(conn)
                print('finished worker server')
            # use super class's thread list
            self.threads.append(threading.Thread(target=worker_server, args=(9998,)))
            self.threads[-1].start()
        else:
            # open local connections
            if 'num_gathers' not in self.args['worker']:
                self.args['worker']['num_gathers'] = 1 + max(0, self.args['worker']['num_parallel'] - 1) // 16
            for i in range(self.args['worker']['num_gathers']):
                conn0, conn1 = mp.Pipe(duplex=True)
                mp.Process(target=gather_loop, args=(self.args, conn1, i)).start()
                conn1.close()
                self.add_connection(conn0)


def entry(worker_args):
    conn = connect_socket_connection(worker_args['server_address'], 9999)
    conn.send(worker_args)
    args = conn.recv()
    conn.close()
    return args


def worker_main(args):
    # offline generation worker
    worker_args = args['worker_args']
    worker_args['address'] = gethostname()
    if 'num_gathers' not in worker_args:
        worker_args['num_gathers'] = 1 + max(0, worker_args['num_parallel'] - 1) // 16

    args = entry(worker_args)
    print(args)
    prepare_env(args['env'])

    # open workers
    process = []
    try:
        for i in range(args['worker']['num_gathers']):
            conn = connect_socket_connection(args['worker']['server_address'], 9998)
            p = mp.Process(target=gather_loop, args=(args, conn, i))
            p.start()
            conn.close()
            process.append(p)
        while True:
            time.sleep(100)
    finally:
        for p in process:
            p.terminate()
