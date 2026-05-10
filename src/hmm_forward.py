import numpy as np

class HMMForward(object):
    def __init__(self, vec_init: np.ndarray, transition_matrix: np.ndarray,
                 emission_matrix: np.ndarray, random_seed: int = 1989):
        self.vec_init = vec_init
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.random_seed = random_seed
        self._init_constant()

    def _init_constant(self):
        np.random.seed(self.random_seed)
        self.n_contexts, self.n_regimes = self.emission_matrix.shape
        self.index_context = list(np.arange(self.n_contexts))
        self.current_context = None
        self.current_regime = None
        self.current_regime_distribution = self.vec_init
        self.current_context_distribution = None
        self.current_iter = 1
        self.hist_regime_distribution = []
        self.hist_context_distribution =  []
        self.hist_regime = []
        self.hist_context = []

    def _update_regime_distribution(self):
        return self.transition_matrix[:, self.current_regime]

    def _update_context_distribution(self):
        return self.emission_matrix[:, self.current_regime]

    def _sample_regime(self):
        return np.random.choice(np.arange(self.n_regimes), p=self.current_regime_distribution)

    def _sample_context(self):
        return np.random.choice(np.arange(self.n_contexts), p=self.current_context_distribution)

    def run_one_iteration(self):
        if self.current_iter > 1:
            self.current_regime_distribution = self._update_regime_distribution()
        self.current_regime = self._sample_regime()
        self.current_context_distribution = self._update_context_distribution()
        self.current_context = self._sample_context()
        self.hist_regime_distribution += [self.current_regime_distribution]
        self.hist_context_distribution += [self.current_context_distribution]
        self.hist_regime += [self.current_regime]
        self.hist_context += [self.current_context]
        self.current_iter += 1

    def run(self, n_t: int):
        for _ in range(n_t):
            self.run_one_iteration()
