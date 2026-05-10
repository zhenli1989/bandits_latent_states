import numpy as np
import pandas as pd

from src.utils import combine_dt_dummy_belief, LIST_FEATURE_DUMMY_CB


class LinearBanditsBelief(object):
    __LIST_ACTIONS__ = ['no_action', 'email', 'call']
    def __init__(self, dict_params, dt_env_rewards, dt_env_dummy, list_belief_estimated, prefix_sep="_zl_"):
        self.dict_params = dict_params
        self.prefix_sep = prefix_sep

        self.dt_env_rewards = dt_env_rewards

        belief_mat = np.array(list_belief_estimated, dtype=float)
        dt_dummy_cb = combine_dt_dummy_belief(dt_env_dummy, belief_mat)
        self.dt_env_dummy_cb = pd.concat([dt_env_dummy[['index_env', 'ACTION']].reset_index(drop=True),
                                          dt_dummy_cb], axis=1)

        self._prepare_data()
        self._init_constant()

        np.random.seed(self.dict_params["seed"])

    def _prepare_data(self):
        self.list_env_rewards_grp = [
            self.dt_env_rewards.loc[self.dt_env_rewards['index_env'] == t].reset_index(drop=True)
            for t in range(1, self.dict_params["number_rounds"] + 1)
        ]

        self.list_env_dummy_cb_grp = [
            self.dt_env_dummy_cb.loc[self.dt_env_dummy_cb['index_env'] == t].reset_index(drop=True)
            for t in range(1, self.dict_params["number_rounds"] + 1)
        ]
        self.dt_env_rewards = None
        self.dt_env_dummy_cb = None

    def _init_constant(self):
        self.current_iter = 0
        self.current_batch = 0
        self.actions = []
        self.rewards = []
        self.believed_rewards = []
        self.lmd = self.dict_params["lmd"]
        self.simple_bound = self.dict_params["simple_bound"]
        self.union_bound = self.dict_params["union_bound"]
        self.UCB_multiply = self.dict_params["UCB_multiply"]

    def _get_context(self):
        self.current_env_rewards_grp = self.list_env_rewards_grp[self.current_iter]
        self.current_env_dummy_cb_grp = self.list_env_dummy_cb_grp[self.current_iter]

    def _cal_constant_ucb(self):
        if self.simple_bound:
            self.current_constant_UCB = np.sqrt((self.current_batch + 1) * self.dict_params['batch_lengh']) * self.UCB_multiply
        elif self.union_bound:
            self.current_constant_UCB = \
                ((self.current_batch + 1) * np.sqrt((self.current_batch + 1) * self.dict_params['batch_lengh'])) * self.UCB_multiply
        else:
            self.current_constant_UCB = \
                (self.current_batch + 1) * np.sqrt(self.dict_params['batch_lengh']) * self.UCB_multiply

    def _update_weights(self):
        x = self.current_context_dummy.values.ravel()

        if self.current_iter == 0:
            self.vt_raw = np.outer(x, x)
            self.x_r  = x * self.current_reward
        else:
            self.vt_raw += np.outer(x, x)
            self.x_r += x * self.current_reward

        i = x.shape[0]
        self.vt = self.vt_raw + np.identity(i) * self.lmd
        self.reward_weights = np.inner(np.linalg.inv(self.vt), self.x_r).T

    def _update_weights_batch(self):
        self.vt_batch = self.vt.copy()
        self.vt_batch_inv = np.linalg.inv(self.vt_batch)
        self.reward_weights_batch = self.reward_weights.copy()

    def _cal_bonus_ucb(self, action, context):
        current_bonus_ucb =  self.current_constant_UCB * np.linalg.norm(context @ self.vt_batch_inv, axis=1)

        if self.dict_params["verbose"] is True:
            if self.current_iter % 1000 == 0:
                print(f"For action {action}: Average bonus {np.round(np.mean(current_bonus_ucb), 4)}")

        return current_bonus_ucb

    def _take_action(self, random_action = False):
        if random_action:
            action = np.random.choice(self.__LIST_ACTIONS__)
        else:
            action = self._action_max_reward()
        self.actions.append(action)
        self.current_action = action

        return action

    def _update_action_context(self):
        self.current_context_dummy = \
            self.current_env_dummy_cb_grp.loc[self.current_env_dummy_cb_grp['ACTION'] == self.current_action, LIST_FEATURE_DUMMY_CB].reset_index(drop=True)

    def _get_reward(self):
        current_env_rewards = self.current_env_rewards_grp[self.current_env_rewards_grp['ACTION']==self.current_action].reset_index(drop=True)
        self.current_reward = float(np.random.normal(current_env_rewards["expected_reward"][0], 0.2))
        self.current_believed_reward = current_env_rewards["believed_reward"][0]
        self.rewards.append(self.current_reward)
        self.believed_rewards.append(self.current_believed_reward)

    def _action_max_reward(self):
        best_action = None
        best_reward = -np.inf

        for action_ in self.__LIST_ACTIONS__:
            context_dummy_ = \
                self.current_env_dummy_cb_grp.loc[
                    self.current_env_dummy_cb_grp['ACTION'] == action_, LIST_FEATURE_DUMMY_CB].reset_index(drop=True)
            current_bonus_ucb_ = self._cal_bonus_ucb(action=action_, context=context_dummy_.values)

            reward_ucb = \
                np.squeeze(np.inner(context_dummy_.values, self.reward_weights_batch)) + current_bonus_ucb_

            if (reward_ucb > best_reward) | ((reward_ucb == best_reward) & (np.random.rand() > 0.5)):
                best_action = action_
                best_reward = reward_ucb

        return best_action

    def _run_one_iteration(self):
        self._get_context()

        if self.dict_params["verbose"] is True:
            if self.current_iter % 1000 == 0:
                print(f"Iteration {self.current_iter}")

        rounds_random = max(self.dict_params["hot_start"], self.dict_params["batch_lengh"] + 1)
        if self.current_iter <= rounds_random - 1:
            self._take_action(random_action=True)
        else:
            self._take_action(random_action=False)

        if self.dict_params["verbose"] is True:
            if self.current_iter % 1000 == 0:
                print(f"current action: {self.current_action}")

        self._update_action_context()
        self._get_reward()
        self._update_weights()

        if (self.current_iter + 1) % self.dict_params["batch_lengh"] == 0:
            self._update_weights_batch()
            self.current_batch += 1

        self._cal_constant_ucb()

    def run_simulation(self):
        for i_ in range(self.dict_params["number_rounds"]):
            self._run_one_iteration()
            if self.dict_params["verbose"] is True:
                if self.current_iter % 1000 == 0:
                    print(f"Cumulative Rewards: {np.round(np.sum(self.rewards), 4)}")
                    print(f"Cumulative Believed Rewards: {np.round(np.sum(self.believed_rewards), 4)}")
                    print("------------------------------------")
            self.current_iter += 1

    def clear_for_dump(self):
        self.list_env_rewards_grp = None
        self.list_env_dummy_cb_grp = None
        self.current_env_rewards_grp = None
        self.current_env_dummy_cb_grp = None
        self.current_context_dummy = None