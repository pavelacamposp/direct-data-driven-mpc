from typing import Callable
import numpy as np

class NonlinearSystem:
    """
    A class representing a Nonlinear Dynamical system.

    The system is defined by its dynamics and output functions, in the form:
        x(k+1) = f(x(k), u(k)) = f_0(x(k)) + B * u(k)
        y(k) = h(x(k), u(k)) = h_0(x(k)) + D * u(k)

    Attributes:
        f (Callable[[np.ndarray, np.ndarray], np.ndarray]): The function
            representing the system's dynamics.
        h (Callable[[np.ndarray, np.ndarray], np.ndarray]): The function
            representing the system's output.
        n (int): The number of system states.
        m (int): The number of inputs to the system.
        p (int): The number of outputs of the system.
        eps_max (float): The upper bound of the system measurement noise.
        x (np.ndarray): The internal state vector of the system.
    """
    def __init__(
        self,
        f: Callable[[np.ndarray, np.ndarray], np.ndarray],
        h: Callable[[np.ndarray, np.ndarray], np.ndarray],
        n: int,
        m: int,
        p: int,
        eps_max: float = 0.0):
        """
        Initialize a Nonlinear Dynamical system with a dynamics function `f`
        and an output function `h`.

        Args:
            f (Callable[[np.ndarray, np.ndarray], np.ndarray]): The function
                representing the system's dynamics.
            h (Callable[[np.ndarray, np.ndarray], np.ndarray]): The function
                representing the system's output.
            n (int): The number of system states.
            m (int): The number of inputs to the system.
            p (int): The number of outputs of the system.
            eps_max (float): The upper bound of the system measurement noise.
                Defaults to 0.0.
        """
        self.f = f
        self.h = h
        self.n = n
        self.m = m
        self.p = p
        self.eps_max = eps_max

        # System state
        self.x = np.zeros(self.n)

    def simulate_step(self, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Simulate a single time step of the Nonlinear system with a given
        input `u`.

        The system simulation follows the state-space equations:
            x(k+1) = f(x(k), u(k)) = f_0(x(k)) + B * u(k)
            y(k) = h(x(k), u(k)) = h_0(x(k)) + D * u(k)

        Args:
            u (np.ndarray): The input vector of shape `(m,)` at the current
                time step, where `m` is the number of inputs.
            w (np.ndarray): The measurement noise vector of shape `(p,)` at
                the current time step, where `p` is the number of outputs.

        Returns:
            np.ndarray: The output vector `y` of shape `(p,)` at the current
                time step, where `p` is the number of outputs.
        
        Note:
            This method updates the `x` attribute, which represents the
            internal state vector of the system, after simulation.
        """
        # Compute output using the output function
        y = self.h(self.x, u) + w
        # Update state based on the dynamics function
        self.x = self.f(self.x, u)
        
        return y

    def simulate(
        self,
        U: np.ndarray,
        W: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """
        Simulate the Nonlinear system over multiple time steps.

        Args:
            U (np.ndarray): An input matrix of shape `(steps, m)` where
                `steps` is the number of time steps and `m` is the number of
                inputs.
            W (np.ndarray): A noise matrix of shape `(steps, p)` where `steps`
                is the number of time steps and `p` is the number of outputs.
            steps (int): The number of simulation steps.

        Returns:
            np.ndarray: The output matrix `Y` of shape `(steps, p)` containing
                the simulated system outputs at each time step.

        Note:
            This method updates the `x` attribute, which represents the
            internal state vector of the system, after each simulation step.
        """
        # Initialize system output
        Y = np.zeros((steps, self.p))

        for k in range(steps):
            Y[k, :] = self.simulate_step(U[k, :], W[k, :])
        
        return Y
