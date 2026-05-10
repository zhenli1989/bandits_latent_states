import numpy as np
from itertools import permutations
from scipy.stats import special_ortho_group
from src.hmm_belief_forward import HMMBeliefForward

class HMMSpectralDirectEstimation(object):
    def __init__(self,
                 n_contexts: int,
                 n_regimes: int,
                 hot_start: int = 1000,
                 random_seed: int = 1989):
        self.n_contexts = n_contexts
        self.n_regimes = n_regimes
        self.hot_start = hot_start
        self.belief_estimator = HMMBeliefForward
        self.random_seed = random_seed
        self._init_constant()

    def _init_constant(self):
        np.random.seed(self.random_seed)
        self.index_context = list(np.arange(self.n_contexts))
        self.curr_iter = 1

        self.current_P31 = np.zeros((self.n_contexts, self.n_contexts))
        self.current_P32 = np.zeros((self.n_contexts, self.n_contexts))
        self.current_P312 = np.zeros((self.n_contexts, self.n_contexts, self.n_contexts))

        self.current_P31_sum = np.zeros((self.n_contexts, self.n_contexts))
        self.current_P32_sum = np.zeros((self.n_contexts, self.n_contexts))
        self.current_P312_sum = np.zeros((self.n_contexts, self.n_contexts, self.n_contexts))

        self.current_counter = 0
        self.current_U1 = np.zeros((self.n_contexts, self.n_regimes))
        self.current_U2 = np.zeros((self.n_contexts, self.n_regimes))
        self.current_U3 = np.zeros((self.n_contexts, self.n_regimes))
        self.current_R1 = np.zeros((self.n_regimes, self.n_regimes))
        self.current_L = np.zeros((self.n_regimes, self.n_regimes))

        self.current_transition = np.ones((self.n_regimes, self.n_regimes))
        self.current_transition /= self.current_transition.sum(axis=0, keepdims=True)

        self.current_emission = np.ones((self.n_contexts, self.n_regimes))
        self.current_emission /= self.current_emission.sum(axis=0, keepdims=True)

        self.current_belief = np.ones(self.n_regimes)
        self.current_belief /= self.current_belief.sum()

        self.hist_context = []
        self.hist_transition = []
        self.hist_emission = []
        self.hist_belief = []
        self.dict_hist_belief = {}

        self._sample_theta()

    def update_p(self):
        if self.curr_iter < 3:
            return

        self.current_counter += 1

        context_t = self.hist_context[-1]
        context_tm1 = self.hist_context[-2]
        context_tm2 = self.hist_context[-3]

        self.current_P31_sum[context_t, context_tm2] += 1.0
        self.current_P32_sum[context_t, context_tm1] += 1.0
        self.current_P312_sum[context_t, context_tm2, context_tm1] += 1.0

        inv_counter = 1.0 / self.current_counter
        self.current_P31 = self.current_P31_sum * inv_counter
        self.current_P32 = self.current_P32_sum * inv_counter
        self.current_P312 = self.current_P312_sum * inv_counter

    def _update_u(self):
        U, _, Vh = np.linalg.svd(self.current_P31, full_matrices=False)
        self.current_U3 = U[:, :self.n_regimes]
        self.current_U1 = Vh.T[:, :self.n_regimes]

        _, _, Vh2 = np.linalg.svd(self.current_P32, full_matrices=False)
        self.current_U2 = Vh2.T[:, :self.n_regimes]

    def _sample_theta(self):
        self.current_THETA = special_ortho_group.rvs(dim=self.n_regimes, random_state=self.random_seed)

    def _compute_l_comp(self, index_row: int, base_inv: np.ndarray):
        theta_ = self.current_THETA[index_row, :]
        RL_comp = self.current_U3.T @ (self.current_P312 @ (self.current_U2 @ theta_)) @ self.current_U1
        return RL_comp @ base_inv

    def _update_r1(self, base_inv: np.ndarray):
        L_comp_1 = self._compute_l_comp(index_row=0, base_inv=base_inv)
        eig_val, eig_vec = np.linalg.eig(L_comp_1)

        idx = (-eig_val).argsort()[::-1]
        eig_vec = eig_vec[:, idx]
        eig_vec /= np.linalg.norm(eig_vec, axis=0, keepdims=True)

        self.current_R1 = eig_vec

    def _update_l(self, base_inv: np.ndarray):
        current_L = np.zeros((self.n_regimes, self.n_regimes))
        R1_inv = np.linalg.inv(self.current_R1)

        for i in range(self.n_regimes):
            L_comp_i = self._compute_l_comp(index_row=i, base_inv=base_inv)
            element_l = np.diag(R1_inv @ L_comp_i @ self.current_R1)
            current_L[i, :] = element_l

        self.current_L = current_L

    def _apply_state_permutation(self, emission_matrix, transition_matrix, perm):
        perm = np.asarray(perm, dtype=np.int64)
        emission_matrix = emission_matrix[:, perm]
        transition_matrix = transition_matrix[np.ix_(perm, perm)]
        return emission_matrix, transition_matrix

    def _align_with_previous_estimate(self, emission_matrix, transition_matrix):
        # First spectral estimate: keep identity ordering.
        if (self.curr_iter <= self.hot_start) or (len(self.hist_emission) == 0):
            return emission_matrix, transition_matrix

        prev_emission = self.hist_emission[-1]

        best_perm = tuple(range(self.n_regimes))
        best_score = np.inf

        for perm in permutations(range(self.n_regimes)):
            aligned_emission = emission_matrix[:, perm]
            score = np.max(np.linalg.norm(prev_emission - aligned_emission, axis=0))
            if score < best_score:
                best_score = score
                best_perm = perm

        return self._apply_state_permutation(emission_matrix, transition_matrix, best_perm)

    def _update_emission_transition(self, theta_inv: np.ndarray):
        current_emission = self.current_U2 @ theta_inv @ self.current_L
        current_emission = np.maximum(current_emission, 0.0)
        current_emission /= np.sum(current_emission, axis=0, keepdims=True)

        current_transition = np.abs(np.linalg.pinv(self.current_U3.T @ current_emission) @ self.current_R1)
        current_transition /= np.sum(current_transition, axis=0, keepdims=True)

        current_emission, current_transition = self._align_with_previous_estimate(
            current_emission,
            current_transition
        )

        self.current_emission = current_emission
        self.current_transition = current_transition

    def update_spectral_estimation(self):
        self._update_u()

        base = self.current_U3.T @ self.current_P31 @ self.current_U1
        base_inv = np.linalg.inv(base)
        theta_inv = np.linalg.inv(self.current_THETA)

        self._update_r1(base_inv=base_inv)
        self._update_l(base_inv=base_inv)
        self._update_emission_transition(theta_inv=theta_inv)

    def run_one_iteration(self, context):
        self.hist_context.append(context)
        self.update_p()

        if self.curr_iter >= self.hot_start:
            self.update_spectral_estimation()

        self.hist_emission.append(self.current_emission.copy())
        self.hist_transition.append(self.current_transition.copy())

    def update_belief_one_iteration(self, vec_init, number_round):
        belief_estimator = self.belief_estimator(
            vec_init=vec_init,
            transition_matrix=self.hist_transition[number_round - 1],
            emission_matrix=self.hist_emission[number_round - 1]
        )
        hist_belief = belief_estimator.run_numba(self.hist_context[:number_round])
        return hist_belief

    def update_belief(self, vec_init):
        self.hist_belief = []
        self.dict_hist_belief = {}

        for curr_iter_ in range(1, len(self.hist_context) + 1):
            print(curr_iter_)
            if curr_iter_ >= self.hot_start:
                hist_belief = self.update_belief_one_iteration(vec_init, curr_iter_)
                self.current_belief = hist_belief[-1].copy()

                if curr_iter_ % 100 == 0:
                    self.dict_hist_belief[curr_iter_] = hist_belief.copy()

            self.hist_belief.append(self.current_belief.copy())

    def run(self, list_contexts):
        for context_ in list_contexts:
            print(self.curr_iter)
            self.run_one_iteration(context_)
            self.curr_iter += 1