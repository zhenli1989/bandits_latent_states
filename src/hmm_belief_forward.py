import numpy as np
from src.utils import update_belief_numba

class HMMBeliefForward(object):
    def __init__(self, vec_init: np.ndarray, transition_matrix: np.ndarray, emission_matrix: np.ndarray):
        self.vec_init = vec_init
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self._init_constant()

    def _init_constant(self):
        self.n_contexts, self.n_regimes = self.emission_matrix.shape
        self.index_context = list(np.arange(self.n_contexts))
        self.current_belief = self.vec_init.copy()
        self.current_iter = 1
        self.hist_belief = []
        self.hist_context = []

    def _update_belief_regime(self, context: int):
        belief_ = []

        prob_cond = 0
        for regime_ in range(self.n_regimes):
            prob_cond += self.emission_matrix[context, regime_] * np.sum(self.transition_matrix[regime_, :] * self.current_belief)

        for regime_ in range(self.n_regimes):
            prob_ = self.emission_matrix[context, regime_] * np.sum(self.transition_matrix[regime_, :] * self.current_belief)
            belief_.append(prob_ / prob_cond)

        return belief_

    def _update_belief_regime_vec(self, context: int):
        prob = self.emission_matrix[context, :] * (self.transition_matrix @ self.current_belief)
        return prob / prob.sum()

    def run_one_iteration(self, context: int):
        self.hist_context.append(context)

        if len(self.hist_context) == 1:
            prob = self.emission_matrix[context, :] * self.current_belief
            self.current_belief = prob / prob.sum()
        else:
            self.current_belief = self._update_belief_regime_vec(context)

        self.hist_belief.append(self.current_belief.copy())

    def run(self, list_contexts):
        for context_ in list_contexts:
            self.run_one_iteration(context_)
            self.current_iter += 1

    def run_numba(self, list_contexts):
        list_contexts = np.asarray(list_contexts, dtype=np.int64)
        hist_belief = update_belief_numba(self.transition_matrix, self.emission_matrix, self.vec_init, list_contexts)
        return hist_belief