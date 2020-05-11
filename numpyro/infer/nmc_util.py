import jax.numpy as np
from jax.lax import scan


class LimitedMemoryBFGS:

    def __init__(self, init_position, init_grad, history_size=5):
        """ Limited Memory Broyden–Fletcher–Goldfarb–Shanno algorithm as described in [1]

        *** Reference ***
        [1]

        :param init_position: Column vector with initial position;
        :param init_grad: Column vector with gradient at initial position;
        :param history_size: Number of previous steps for estimating newton direction from.
        """

        self.history_size = history_size
        self.position_history = np.expand_dims(init_position, 0)  # p_k+1 - p_k
        self.gradient_history = np.expand_dims(init_grad, 0)  # grad_k+1 - grad_k

    def step(self, position, grad):
        if position != grad.shape:
            raise Exception()

        m = position.shape
        if m != self.gradient_history.shape[1]:
            raise Exception()
        if m != self.position_history.shape[1]:
            raise Exception()

        rhos = self._rho()

        q, alphas = scan(LimitedMemoryBFGS._update_q,
                         grad,
                         (self.position_history, self.gradient_history, rhos))

        init_ihes = self._gamma() * np.eye(m)  # initial inverse hessian
        init_dir = init_ihes * q  # approximate newton direction

        newton_direction, _ = scan(LimitedMemoryBFGS._update_approx_newton_dir,
                                   init_dir,
                                   (self.position_history, self.gradient_history, rhos, alphas))

        self._update_history(position, grad)

        return -newton_direction

    def _update_history(self, pos, grad):
        n = self.position_history.shape[0]

        self.position_history = np.concatenate((self.position_history[0, :] - pos, self.position_history))
        self.gradient_history = np.concatenate((self.gradient_history[0, :] - grad, self.gradient_history))

        if n == self.history_size:
            self.position_history = np.delete(self.position_history, -1, 1)
            self.gradient_history = np.delete(self.gradient_history, -1, 1)

    def _gamma(self):
        g = self.gradient_history[-1, :]
        x = self.position_history[-1, :]
        return x.T @ g / g.T @ g

    def _rho(self):
        return np.reciprocal(self.position_history @ self.gradient_history)

    @staticmethod
    def _update_approx_newton_dir(prev_dir, params):
        x, g, rho, alpha = params

        beta = rho * g.T @ prev_dir
        new_dir = prev_dir + x * (alpha - beta)

        return new_dir, ()

    @staticmethod
    def _update_q(prev_q, params):

        x, g, rho = params
        alpha = rho @ x.T * prev_q
        new_q = prev_q - alpha * prev_q

        return new_q, alpha


def ol_bfgs():
    """ Online Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm

    **References:**

    1. * A stochastic quasi-newton method for online convex optimization *
        (http://proceedings.mlr.press/v2/schraudolph07a/schraudolph07a.pdf)


    """
