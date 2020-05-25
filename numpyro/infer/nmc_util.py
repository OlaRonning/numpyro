from typing import NamedTuple

import jax.numpy as np
from jax import ops
from jax.lax import scan


class LimitedMemoryBFGS:

    def __init__(self, init_position, init_grad, history_size=5, learning_rate=1e-4):
        """ Limited Memory Broyden–Fletcher–Goldfarb–Shanno algorithm as described in [1]

        *** Reference ***
        [1]

        :param init_position: Column vector with initial position;
        :param init_grad: Column vector with gradient at initial position;
        :param history_size: Number of previous steps for estimating newton direction from.
        """

        self.history_size = history_size
        self.learning_rate = learning_rate

        self.position_history = np.zeros((self.history_size, init_position.shape[0]))  # p_k+1 - p_k
        self.gradient_history = np.zeros((self.history_size, init_grad.shape[0]))  # grad_k+1 - grad_k
        self.current_index = 0

        self._update_history(init_position, init_grad)

    def warmup(self, pos, grad, lr):
        # TODO: conjugated gradient descent

        if not np.array_equal(pos, self.position_history[0]):
            self._update_history(pos, grad)

        pos = pos - lr * grad

        return pos, len(self.position_history) < self.history_size

    def step(self, position, gradient):

        if position.shape != gradient.shape:
            raise Exception()

        m = position.shape[0]
        if m != self.gradient_history.shape[1]:
            raise Exception()
        if m != self.position_history.shape[1]:
            raise Exception()

        if self.current_index < self.history_size:
            # first history_size steps use CGD
            # TODO: conjugated gradient descent
            newton_direction = self.learning_rate * gradient
        else:
            curr_order = (np.arange(self.history_size) + self.current_index) % self.history_size
            reorder = np.arange(self.history_size)
            reorder.at[curr_order].set(np.arange(self.history_size))

            rhos = self._rho()[reorder]

            q, alphas = scan(LimitedMemoryBFGS._update_q,
                             gradient,
                             (self.position_history[reorder, :], self.gradient_history[reorder, :], rhos))

            init_ihes = self._gamma() * np.eye(m)  # initial inverse hessian

            init_dir = init_ihes @ q  # approximate newton direction

            newton_direction, _ = scan(LimitedMemoryBFGS._update_approx_newton_dir,
                                       init_dir,
                                       (self.position_history[reorder, :][::-1],
                                        self.gradient_history[reorder, :][::-1],
                                        rhos[::-1],
                                        alphas[::-1]))

        if not np.array_equal(position, self.position_history[(self.current_index - 1) % self.history_size, :]):
            self._update_history(position, gradient)

        return newton_direction

    def _update_history(self, pos, grd):
        prev_idx = (self.current_index - 1) % self.history_size
        cur_idx = self.current_index % self.history_size
        self.current_index += 1

        self.position_history = ops.index_update(self.position_history,
                                                 ops.index[cur_idx, ...],
                                                 pos - self.position_history[prev_idx, :])
        self.gradient_history = ops.index_update(self.gradient_history,
                                                 ops.index[cur_idx, ...],
                                                 grd - self.gradient_history[prev_idx, :])

    def _gamma(self):
        start_idx = self.current_index % self.history_size
        x = self.position_history[start_idx, :]
        g = self.gradient_history[start_idx, :]

        return x.T @ g / (g.T @ g)

    def _rho(self):
        return np.reciprocal((np.array(self.position_history) * np.array(self.gradient_history)).sum(1))

    @staticmethod
    def _update_q(prev_q, params):
        x, g, rho = params

        alpha = rho * x.T @ prev_q
        new_q = prev_q - alpha * g

        return new_q, alpha

    @staticmethod
    def _update_approx_newton_dir(prev_dir, params):
        x, g, rho, alpha = params

        beta = rho * g.T @ prev_dir
        new_dir = prev_dir + x * (alpha - beta)

        return new_dir, ()


class LBFGSResults(NamedTuple):
    # TODO: fix this (copied from https://github.com/google/jax/pull/3101/files)
    converged: bool  # bool, True if minimization converges
    failed: bool  # bool, True if line search fails
    k: int  # The number of iterations of the BFGS update.
    nfev: int  # The total number of objective evaluations performed.
    ngev: int  # total number of jacobian evaluations
    nhev: int  # total number of hessian evaluations
    x_k: np.ndarray
    # A tensor containing the last argument value found during the search. If the search converged, then
    # this value is the argmin of the objective function.
    f_k: np.ndarray  # A tensor containing the value of the objective
    # function at the `position`. If the search
    # converged, then this is the (local) minimum of
    # the objective function.
    g_k: np.ndarray  # A tensor containing the gradient of the
    # objective function at the
    # `final_position`. If the search converged
    # the max-norm of this tensor should be
    # below the tolerance.
    H_k: np.ndarray  # A tensor containing the inverse of the estimated Hessian.


def l_bfgs(func, init_x):
    pass


def ol_bfgs():
    """ Online Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm

    **References:**

    1. * A stochastic quasi-newton method for online convex optimization *
        (http://proceedings.mlr.press/v2/schraudolph07a/schraudolph07a.pdf)


    """
