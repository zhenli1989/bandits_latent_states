import numpy as np

from src.utils import LIST_FEATURE_DUMMY


class LinearBandits(object):
    __LIST_ACTIONS__ = ['no_action', 'email', 'call']
    def __init__(self, dict_params, dt_env_rewards, dt_env_dummy, prefix_sep="_zl_"):
        self.dict_params = dict_params
        self.prefix_sep = prefix_sep

        self.dt_env_rewards = dt_env_rewards
        self.dt_env_dummy = dt_env_dummy
        self._prepare_data()
        self._init_constant()

        np.random.seed(self.dict_params["seed"])

    def _prepare_data(self):
        self.list_env_rewards_grp = [
            self.dt_env_rewards.loc[
                self.dt_env_rewards['index_env'] == t
            ].reset_index(drop=True)
            for t in range(1, self.dict_params["number_rounds"] + 1)
        ]

        self.list_env_dummy_grp = [
            self.dt_env_dummy.loc[
                self.dt_env_dummy['index_env'] == t
            ].reset_index(drop=True)
            for t in range(1, self.dict_params["number_rounds"] + 1)
        ]

        self.dt_env_rewards = None
        self.dt_env_dummy = None

    def _init_constant(self):
        self.current_iter = 0
        self.actions = []
        self.rewards = []
        self.believed_rewards = []
        self.lmd = self.dict_params["lmd"]
        self.UCB_multiply = self.dict_params["UCB_multiply"]

    def _get_context(self):
        self.current_env_rewards_grp = self.list_env_rewards_grp[self.current_iter]
        self.current_env_dummy_grp = self.list_env_dummy_grp[self.current_iter]

    def _cal_constant_ucb(self):
        self.current_constant_UCB = (np.log(self.current_iter + 1)) * self.UCB_multiply

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
        self.vt_inv = np.linalg.inv(self.vt)

        self.reward_weights = np.inner(self.vt_inv, self.x_r).T

    def _cal_bonus_ucb(self, action, context, matrix_context):
        current_bonus_ucb = \
            self.current_constant_UCB * \
            np.sqrt(np.sum(np.array(matrix_context * self.vt_inv) * context, axis=1))

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
            self.current_env_dummy_grp.loc[self.current_env_dummy_grp['ACTION'] == self.current_action, LIST_FEATURE_DUMMY].reset_index(drop=True)

    def _get_reward(self):
        current_env_rewards = self.current_env_rewards_grp[self.current_env_rewards_grp['ACTION']==self.current_action].reset_index(drop=True)
        self.current_reward =  float(np.random.normal(current_env_rewards["expected_reward"][0], 0.2))
        self.current_believed_reward = current_env_rewards["believed_reward"][0]
        self.rewards.append(self.current_reward)
        self.believed_rewards.append(self.current_believed_reward)

    def _action_max_reward(self):
        best_action = None
        best_reward = -np.inf

        for action_ in self.__LIST_ACTIONS__:
            context_dummy_ = \
                self.current_env_dummy_grp.loc[
                    self.current_env_dummy_grp['ACTION'] == action_, LIST_FEATURE_DUMMY].reset_index(drop=True)

            matrix_context_ = np.matrix(context_dummy_.values)

            current_bonus_ucb_ = self._cal_bonus_ucb(action=action_, context=context_dummy_.values, matrix_context=matrix_context_)

            reward_ucb = \
                np.squeeze(np.inner(context_dummy_.values, self.reward_weights)) + current_bonus_ucb_

            if (reward_ucb > best_reward) | ((reward_ucb == best_reward) & (np.random.rand() > 0.5)):
                best_action = action_
                best_reward = reward_ucb

        return best_action

    def _run_one_iteration(self):
        self._get_context()

        if self.dict_params["verbose"] is True:
            if self.current_iter % 1000 == 0:
                print(f"Iteration {self.current_iter}")

        if self.current_iter <= self.dict_params["hot_start"] - 1:
            self._take_action(random_action=True)
        else:
            self._take_action(random_action=False)

        if self.dict_params["verbose"] is True:
            if self.current_iter % 1000 == 0:
                print(f"current action: {self.current_action}")
        self._update_action_context()
        self._get_reward()
        self._update_weights()
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
        self.list_env_dummy_grp = None
        self.current_env_rewards_grp = None
        self.current_env_dummy_grp = None
        self.current_context_dummy = None
