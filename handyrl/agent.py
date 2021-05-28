# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# agent classes

import random

import numpy as np

from .util import softmax


class RandomAgent:
    def reset(self, env, show=False):
        pass

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        return random.choice(actions)

    def observe(self, env, player, show=False):
        return 0.0


class RuleBasedAgent(RandomAgent):
    def action(self, env, player, show=False):
        if hasattr(env, 'rule_based_action'):
            return env.rule_based_action(player)
        else:
            return random.choice(env.legal_actions(player))


class RuleBasedAgentSmartGeese(RandomAgent):
    def action(self, env, player, show=False):
        return env.rule_based_action_smart_geese(player)


class RuleBasedAgentAlpha4Aug(RandomAgent):
    def action(self, env, player, show=False):
        return env.rule_based_action_alpha_64_409_4aug(player)


class RuleBasedAgentAlphaCpuct1(RandomAgent):
    def action(self, env, player, show=False):
        return env.rule_based_action_alpha_64_409_cpuct_1(player)


class RuleBasedAgentAlphaCpuct2(RandomAgent):
    def action(self, env, player, show=False):
        return env.rule_based_action_alpha_64_409_cpuct_2(player)


class RuleBasedAgentAlphaCs(RandomAgent):
    def action(self, env, player, show=False):
        return env.rule_based_action_alpha_64_409_cs(player)


def view(env, player=None):
    if hasattr(env, 'view'):
        env.view(player=player)
    else:
        print(env)


def view_transition(env):
    if hasattr(env, 'view_transition'):
        env.view_transition()
    else:
        pass


def print_outputs(env, prob, v):
    if hasattr(env, 'print_outputs'):
        env.print_outputs(prob, v)
    else:
        # print('v = %f' % v)
        print('p = %s' % (prob * 1000).astype(int))


class Agent:
    def __init__(self, model, observation=False, temperature=0.0):
        # model might be a neural net, or some planning algorithm such as game tree search
        self.model = model
        self.hidden = None
        self.observation = observation
        self.temperature = temperature

    def reset(self, env, show=False):
        self.hidden = self.model.init_hidden()

    def plan(self, obs):
        outputs = self.model.inference(obs, self.hidden)
        self.hidden = outputs.pop('hidden', None)
        return outputs

    def action(self, env, player, show=False):
        outputs = self.plan(env.observation(player))
        actions = env.legal_actions(player)
        p = outputs['policy']
        v = outputs.get('value', None)
        mask = np.ones_like(p)
        mask[actions] = 0
        p -= mask * 1e32

        if show:
            view(env, player=player)
            print_outputs(env, softmax(p), v)

        if self.temperature == 0:
            ap_list = sorted([(a, p[a]) for a in actions], key=lambda x: -x[1])
            return ap_list[0][0]
        else:
            return random.choices(np.arange(len(p)), weights=softmax(p / self.temperature))[0]

    def observe(self, env, player, show=False):
        if self.observation:
            outputs = self.plan(env.observation(player))
            v = outputs.get('value', None)
        if show:
            view(env, player=player)
            if self.observation:
                print_outputs(env, None, v)


class EnsembleAgent(Agent):
    def reset(self, env, show=False):
        self.hidden = [model.init_hidden() for model in self.model]

    def plan(self, obs):
        outputs = {}
        for i, model in enumerate(self.model):
            o = model.inference(obs, self.hidden[i])
            for k, v in o:
                if k == 'hidden':
                    self.hidden[i] = v
                else:
                    outputs[k] = outputs.get(k, []) + [o]
        for k, vl in outputs:
            outputs[k] = np.mean(vl, axis=0)
        return outputs


class SoftAgent(Agent):
    def __init__(self, model, observation=False):
        super().__init__(model, observation=observation, temperature=1.0)
