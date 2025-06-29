from enum import Enum

import cvxpy as cp
import numpy as np

from direct_data_driven_mpc.utilities.hankel_matrix import (
    evaluate_persistent_excitation,
    hankel_matrix,
)


# Define the regularization types of `alpha`, considering
# with respect to what variable it is regularized
class AlphaRegType(Enum):
    # Alpha regularized w.r.t. an approximation of alpha_Lin^sr(D_t),
    # based on Remark 1 of [2].
    APPROXIMATED = 0
    # Alpha regularized w.r.t. the previous optimal alpha value to
    # encourage stationary behavior. Refer to Section V of [2].
    PREVIOUS = 1
    # Alpha regularized w.r.t. zero.
    ZERO = 2


class NonlinearDataDrivenMPCController:
    """
    A class that implements a Data-Driven Model Predictive Control (MPC)
    controller for Nonlinear systems. The implementation is based on research
    by J. Berberich et al., as described in [2].

    Attributes:
        n (int): The estimated order of the system.
        m (int): The number of control inputs.
        p (int): The number of system outputs.
        u (np.ndarray): The input trajectory applied to the system.
        y (np.ndarray): The system's output response to `u`.
        du (np.ndarray): The input increment trajectory for a controller that
            uses input increments (`ext_out_incr_in = True`).
        N (int): The length of the initial input (`u`) and output (`y`)
            trajectories.
        L (int): The prediction horizon length.
        Q (np.ndarray): The output weighting matrix for the MPC formulation.
        R (np.ndarray): The input weighting matrix for the MPC formulation.
        S (np.ndarray): The output setpoint weighting matrix for the MPC
            formulation.
        y_r (np.ndarray): The system output setpoint.
        lamb_alpha (float): The ridge regularization weight for `alpha`.
        lamb_sigma (float): The ridge regularization weight for `sigma`.
        U (np.ndarray): An array of shape (`m`, 2) containing the bounds for
            the `m` predicted inputs. Each row specifies the `[min, max]`
            bounds for a single input.
        Us (np.ndarray): An array of shape (`m`, 2) containing the bounds for
            the `m` predicted input setpoints. `Us` must be a subset of `U`.
            Each row specifies the `[min, max]` bounds for a single input.
        alpha_reg_type (AlphaRegType): The alpha regularization type for
            the Nonlinear Data-Driven MPC formulation.
        lamb_alpha_s (float): The ridge regularization weight for
            `alpha_s` for a controller that uses an approximation of
            `alpha_Lin^sr(D_t)` for the regularization of `alpha`.
        lamb_sigma_s (float): The ridge regularization weight for
            `sigma_s` for a controller that uses an approximation of
            `alpha_Lin^sr(D_t)` for the regularization of `alpha`.
        ext_out_incr_in (bool): The controller structure (uses an extended
            output representation and input increments, or operates as a
            standard controller with direct control inputs and without system
            state extensions).
        update_cost_threshold (float): The tracking cost value threshold.
            Online input-output data updates are disabled when the tracking
            cost value is less than this value.
        n_mpc_step (int): The number of consecutive applications of the
            optimal input for an n-Step Data-Driven MPC Scheme (multi-step).
        HLn1_u (np.ndarray): The Hankel matrix constructed from the input data
            `u` (or `du` if the controller uses an extended output
            representation and input increments).
        HLn1_y (np.ndarray): The Hankel matrix constructed from the output
            data `y`.
        alpha (cp.Variable): The optimization variable for `alpha`.
        ubar (cp.Variable): The predicted control input variable.
        ybar (cp.Variable): The predicted system output variable.
        u_s (cp.Variable): The predicted equilibrium input variable.
        y_s (cp.Variable): The predicted equilibrium output variable.
        sigma (cp.Variable): The optimization variable for `sigma`.
        sigma_ubar (Vstack | None): A CVXPY `Vstack` object that stacks the
            `sigma` values corresponding to the input. Defined only if the
            controller uses an extended output representation.
        u_s_tiled (Vstack): A CVXPY `Vstack` object that stacks the predicted
            input setpoint variables for use in the cost function formulation.
        y_s_tiled (Vstack): A CVXPY `Vstack` object that stacks the predicted
            output setpoint variables for use in the cost function
            formulation.
        alpha_s (cp.Variable | None): The optimization variable for `alpha_s`.
            Defined only if `alpha` is regularized with respect to an
            approximation of `alpha_Lin^sr(D_t)`.
        sigma_s (cp.Variable | None): The optimization variable for `sigma_s`.
            Defined only if `alpha` is regularized with respect to an
            approximation of `alpha_Lin^sr(D_t)`.
        sigma_s_ubar (Vstack | None): A CVXPY `Vstack` object that stacks the
            `sigma_s` values corresponding to the input. Defined only if the
            controller uses an extended output representation and if `alpha` is
            regularized with respect to an approximation of
            `alpha_Lin^sr(D_t)`.
        dynamics_constraint (list[cp.Constraint]): The system dynamics
            constraints for a Data-Driven MPC formulation.
        internal_state_constraint (list[cp.Constraint]): The internal state
            constraints for a Data-Driven MPC formulation.
        terminal_constraint (list[cp.Constraint]): The terminal state
            constraints for a Data-Driven MPC formulation.
        input_constraints (list[cp.Constraint]): The input constraints for a
            Data-Driven MPC formulation.
        constraints (list[cp.Constraint]): The combined constraints for the
            Data-Driven MPC formulation.
        cost (cp.Expression): The cost function for the Data-Driven MPC
            formulation.
        problem (cp.Problem): The quadratic programming problem for the
            Data-Driven MPC.
        optimal_u (np.ndarray): The optimal control input derived from the
            Data-Driven MPC solution.
        optimal_du (np.ndarray | None): The optimal control input increments
            derived from the Data-Driven MPC solution. Defined only for
            controllers with an extended output representation and input
            increments.
        prev_alpha_val (np.ndarray | None): The previous value of `alpha` used
            for when `alpha` is regularized with respect to the previous
            optimal `alpha` value. Defined only if `alpha` is regularized with
            respect to its previous optimal alpha value.
        ones_NLn (np.ndarray): A vector of ones of shape (1, N - L - n).
        ones_Ln1 (np.ndarray): A vector of ones of shape (L + n + 1, 1).
        ones_n1 (np.ndarray): A vector of ones of shape (n + 1, 1).
        U_const_low (np.ndarray): A tiled lower bound for the input trajectory
            constraints over the prediction horizon.
        U_const_up (np.ndarray): A tiled upper bound for the input trajectory
            constraints over the prediction horizon.
        Us_const_low (np.ndarray): The lower bound for the equilibrium input
            constraints.
        Us_const_up (np.ndarray): The upper bound for the equilibrium input
            constraints.

    References:
        [2] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer, "Linear
            Tracking MPC for Nonlinear Systems—Part II: The Data-Driven Case,"
            in IEEE Transactions on Automatic Control, vol. 67, no. 9, pp.
            4406-4421, Sept. 2022, doi: 10.1109/TAC.2022.3166851.
    """

    def __init__(
        self,
        n: int,
        m: int,
        p: int,
        u: np.ndarray,
        y: np.ndarray,
        L: int,
        Q: np.ndarray,
        R: np.ndarray,
        S: np.ndarray,
        y_r: np.ndarray,
        lamb_alpha: float,
        lamb_sigma: float,
        U: np.ndarray,
        Us: np.ndarray,
        alpha_reg_type: AlphaRegType = AlphaRegType.ZERO,
        lamb_alpha_s: float | None = None,
        lamb_sigma_s: float | None = None,
        ext_out_incr_in: bool = False,
        update_cost_threshold: float | None = None,
        n_mpc_step: int = 1,
    ):
        """
        Initialize a Direct Nonlinear Data-Driven MPC with specified system
        model parameters, an initial input-output data trajectory measured
        from the system, and Nonlinear Data-Driven MPC parameters.

        Note:
            The input data `u` used to excite the system to get the initial
            output data must be persistently exciting of order (L + n + 1), as
            defined in the Data-Driven MPC formulation in [2].

        Args:
            n (int): The estimated order of the system.
            m (int): The number of control inputs.
            p (int): The number of system outputs.
            u (np.ndarray): A persistently exciting input sequence.
            y (np.ndarray): The system's output response to `u`.
            L (int): The prediction horizon length.
            Q (np.ndarray): The output weighting matrix for the MPC
                formulation.
            R (np.ndarray): The input weighting matrix for the MPC
                formulation.
            S (np.ndarray): The output setpoint weighting matrix for the MPC
                formulation.
            y_r (np.ndarray): The system output setpoint.
            lamb_alpha (float): The ridge regularization weight for
                `alpha`.
            lamb_sigma (float): The ridge regularization weight for
                `sigma`.
            U (np.ndarray): An array of shape (`m`, 2) containing the bounds
                for the `m` predicted inputs. Each row specifies the
                `[min, max]` bounds for a single input.
            Us (np.ndarray): An array of shape (`m`, 2) containing the bounds
                for the `m` predicted input setpoints. `Us` must be a subset
                of `U`. Each row specifies the `[min, max]` bounds for a
                single input.
            alpha_reg_type (AlphaRegType): The `alpha` regularization type
                for the Nonlinear Data-Driven MPC formulation.
            lamb_alpha_s (float): The ridge regularization weight for
                `alpha_s` for a controller that uses an approximation of
                `alpha_Lin^sr(D_t)` for the regularization of `alpha`.
            lamb_sigma_s (float): The ridge regularization weight for
                `sigma_s` for a controller that uses an approximation of
                `alpha_Lin^sr(D_t)` for the regularization of `alpha`.
            ext_out_incr_in (bool): The controller structure:
                If `True`, the controller uses an extended output
                representation (y_ext[k] = [y[k], u[k]]) and input increments
                (u[k] = u[k-1] + du[k-1]). If `False`, the controller operates
                as a standard controller with direct control inputs and
                without system state extensions. Defaults to `False`.
            update_cost_threshold (float | None): The tracking cost value
                threshold. Online input-output data updates are disabled when
                the tracking cost value is less than this value. If `None`,
                input-output data is always updated online. Defaults to
                `None`.
            n_mpc_step (int): The number of consecutive applications of the
                optimal input for an n-Step Data-Driven MPC Scheme
                (multi-step). Defaults to 1.

        References:
            [2] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer, "Linear
                Tracking MPC for Nonlinear Systems—Part II: The Data-Driven
                Case," in IEEE Transactions on Automatic Control, vol. 67, no.
                9, pp. 4406-4421, Sept. 2022, doi: 10.1109/TAC.2022.3166851.
        """
        # Define controller structure:
        # - If `True`: The controller uses an extended output representation
        #              (y_ext[k] = [y[k], u[k]]) and input increments where
        #              the control input is updated incrementally as:
        #              u[k] = u[k-1] + du[k-1]. Based on Section V of [2].
        # - If `False`: Standard controller operation. The controller uses
        #               direct control inputs without system state extensions.
        self.ext_out_incr_in = ext_out_incr_in

        # Define system model
        self.n = n  # Estimated system order
        self.m = m  # Number of inputs
        self.p = p  # Number of outputs

        # Input-Output trajectory data
        self.u = u.copy()  # Input trajectory data
        if self.ext_out_incr_in:
            # Incremental Input trajectory data
            # du[k] = u[k+1] - u[k]
            du_last = np.zeros((self.m))
            self.du = np.vstack([u[1:, :] - u[:-1, :], [du_last]])
        # Output trajectory data
        if self.ext_out_incr_in:
            # Extended Output trajectory data
            self.y = np.hstack([y, u])
        else:
            self.y = y.copy()

        self.N = u.shape[0]  # Initial input-output trajectory length

        # Define Nonlinear Data-Driven MPC parameters
        self.L = L  # Prediction horizon
        self.Q = Q  # Output weighting matrix
        self.R = R  # Input weighting matrix
        self.S = S  # Output setpoint weighting matrix

        self.y_r = y_r  # System output setpoint

        self.lamb_alpha = lamb_alpha  # Ridge regularization weight for alpha
        self.lamb_sigma = lamb_sigma  # Ridge regularization weight for sigma

        self.U = U  # Bounds for the predicted input
        self.Us = Us  # Bounds for the predicted input setpoint
        # Note: Us must be a subset of U.

        # Alpha regularization type for Nonlinear MPC
        self.alpha_reg_type = alpha_reg_type
        # Nonlinear MPC parameters for the approximation of alpha_Lin^sr(D_t).
        # Alpha is regularized w.r.t. this parameter, based on Remark 1
        # of [2].
        if alpha_reg_type == AlphaRegType.APPROXIMATED:
            # Ridge regularization weight for alpha_s
            self.lamb_alpha_s = lamb_alpha_s
            # Ridge regularization weight for sigma_s
            self.lamb_sigma_s = lamb_sigma_s
        elif alpha_reg_type == AlphaRegType.PREVIOUS:
            # Previous alpha value initialized with 0
            self.prev_alpha_val = np.zeros((self.N - self.L - self.n, 1))

        # Online input-output data updates
        self.update_cost_threshold = (
            update_cost_threshold if update_cost_threshold is not None else 0.0
        )
        self.update_data = True

        # n-Step Data-Driven MPC Scheme parameters
        self.n_mpc_step = n_mpc_step  # Number of consecutive applications
        # of the optimal input

        # Define bounds for the predicted inputs and predicted input setpoints
        self.U_const_low = np.tile(self.U[:, 0:1], (self.L + 1, 1))
        self.U_const_up = np.tile(self.U[:, 1:2], (self.L + 1, 1))

        self.Us_const_low = self.Us[:, 0:1]
        self.Us_const_up = self.Us[:, 1:2]

        # Define helper constants for MPC constraints definition
        self.ones_1 = np.ones((1, 1))  # 1
        self.ones_NLn = np.ones((1, self.N - self.L - self.n))  # 1_(N-L-n)^T
        self.ones_Ln1 = np.ones((self.L + self.n + 1, 1))  # 1_(L+n+1)
        self.ones_n1 = np.ones((n + 1, 1))  # 1_(n+1)

        # Evaluate if input trajectory data is persistently exciting of
        # order (L + n + 1)
        self.evaluate_input_persistent_excitation()

        # Check correct prediction horizon length and cost matrix dimensions
        self.check_prediction_horizon_length()
        self.check_weighting_matrices_dimensions()

        # Initialize Data-Driven MPC controller
        self.initialize_data_driven_mpc()

    def evaluate_input_persistent_excitation(self) -> None:
        """
        Evaluate whether the input data is persistently exciting of order
        (L + n + 1).

        This method first verifies that the length of the elements in the
        input data matches the number of inputs of the system. Then, it
        evaluates the rank of the Hankel matrix induced by the input sequence
        to determine if the input sequence is persistently exciting of order
        (L + n + 1), as described in Definition 1 [2].

        Raises:
            ValueError: If the length of the elements in the data sequence
                does not match the number of inputs of the system, or if the
                input data is not persistently exciting of order (L + n + 1).

        References:
            [2]: See class-level docstring for full reference details.
        """
        # Get the length of the elements of the data sequence
        u_n = self.u.shape[1]  # m - Number of inputs

        # Check if the number of inputs matches the expected
        # number of inputs of the system
        if u_n != self.m:
            raise ValueError(
                f"The length of the elements of the data sequence ({u_n}) "
                f"should match the number of inputs of the system ({self.m})."
            )

        # Evaluate if input data is persistently exciting of order (L + n + 1)
        # based on the rank of its induced Hankel matrix
        expected_order = self.L + self.n + 1
        in_hankel_rank, in_pers_exc = evaluate_persistent_excitation(
            X=self.u, order=expected_order
        )

        if not in_pers_exc:
            raise ValueError(
                "Initial input trajectory data is not persistently exciting "
                "of order (L + n + 1). The rank of its induced Hankel matrix "
                f"({in_hankel_rank}) does not match the expected rank ("
                f"{u_n * expected_order})."
            )

    def check_prediction_horizon_length(self) -> None:
        """
        Check if the prediction horizon length, `L`, satisfies the MPC system
        design, as described in [2].

        Raises:
            ValueError: If the prediction horizon `L` is less than the
                required threshold based on the controller type.

        References:
            [2]: See class-level docstring for full reference details.
        """
        if self.L < self.n:
            raise ValueError(
                "The prediction horizon (`L`) must be greater than or equal "
                "to the estimated system order `n`."
            )

    def check_weighting_matrices_dimensions(self) -> None:
        """
        Check if the dimensions of the output (`Q`), input (`R`), and setpoint
        (`S`) weighting matrices are correct for an MPC formulation based on
        their order.

        Raises:
            ValueError: If the dimensions of the `Q`, `R`, or `S` matrices are
                incorrect.
        """
        if self.ext_out_incr_in:
            expected_output_shape = (
                (self.m + self.p) * (self.L + self.n + 1),
                (self.m + self.p) * (self.L + self.n + 1),
            )
        else:
            expected_output_shape = (
                self.p * (self.L + self.n + 1),
                self.p * (self.L + self.n + 1),
            )
        expected_input_shape = (
            self.m * (self.L + self.n + 1),
            self.m * (self.L + self.n + 1),
        )
        expected_setpoint_shape = (self.p, self.p)

        if self.Q.shape != expected_output_shape:
            if self.ext_out_incr_in:
                raise ValueError(
                    "Output weighting square matrix Q should be of order "
                    "((m + p) * (L + n + 1))."
                )
            else:
                raise ValueError(
                    "Output weighting square matrix Q should be of order "
                    "(p * (L + n + 1))."
                )
        if self.R.shape != expected_input_shape:
            raise ValueError(
                "Input weighting square matrix R should be of order "
                "(m * (L + n + 1))."
            )
        if self.S.shape != expected_setpoint_shape:
            raise ValueError(
                "Output setpoint weighting square matrix S should be of "
                "order (p)."
            )

    def initialize_data_driven_mpc(self) -> None:
        """
        Initialize the Data-Driven MPC controller.

        This method performs the following tasks:
        1. Constructs Hankel matrices from the initial input-output trajectory
           data (`u`, `y`). These matrices are used for the data-driven
           characterization of the unknown system, as defined by the system
           dynamic constraints in the Nonlinear Data-Driven MPC formulation
           of [2].
        2. Defines the optimization variables for the Data-Driven MPC problem.
        3. Defines the constraints for the MPC problem, which include the
           system dynamics, internal state, terminal state, and input
           constraints.
        4. Defines the cost function for the MPC problem.
        5. Formulates the MPC problem as a Quadratic Programming (QP) problem.
        6. Solves the initialized MPC problem to ensure the formulation is
           valid and retrieve the optimal control input for the initial
           setup.

        This initialization process ensures that all necessary components for
        the Data-Driven MPC are correctly defined and that the MPC problem is
        solvable with the provided initial data.

        References:
            [2]: See class-level docstring for full reference details.
        """
        # Construct Hankel Matrices from the initial input-output trajectory
        # for the data-driven characterization of the unknown system.
        # Used for the system dynamic constraints defined by Equation (22b).
        if self.ext_out_incr_in:
            # For a controller that uses an extended output representation
            # and input increments, the input Hankel matrix corresponds to
            # input increments instead of absolute inputs.
            # H_{L+n+1}(du)
            self.HLn1_u = hankel_matrix(self.du, self.L + self.n + 1)
        else:
            # H_{L+n+1}(u)
            self.HLn1_u = hankel_matrix(self.u, self.L + self.n + 1)
        # H_{L+n+1}(y)
        self.HLn1_y = hankel_matrix(self.y, self.L + self.n + 1)

        # Define the Data-Driven MPC problem
        self.define_optimization_variables()
        self.define_optimization_parameters()

        if self.alpha_reg_type == AlphaRegType.APPROXIMATED:
            # Define the QP problem for computing an approximation of
            # `alpha_Lin^sr(D_t)` if `alpha` is regularized with respect
            # to this approximation
            self.define_alpha_sr_Lin_Dt_prob()

        self.update_optimization_parameters()
        self.define_mpc_constraints()
        self.define_cost_function()
        self.define_mpc_problem()

        # Validate the Data-Driven MPC formulation with an initial solution
        self.solve_mpc_problem()
        self.get_optimal_control_input()

    def update_and_solve_data_driven_mpc(
        self, warm_start: bool = False
    ) -> None:
        """
        Update the Data-Driven MPC optimization parameters, solve the problem,
        and store the optimal control input.

        This method performs the following tasks:
        1. Constructs Hankel matrices using the latest measured input-output
           data. If the tracking cost value from the previous solution is
           small enough (less than `update_cost_threshold`), omits this step
           and the previously defined matrices are used.
        2. Updates the MPC optimization parameters to use the latest
           input-output measurements. Additionally, it updates the value of
           `alpha_Lin^sr(D_t)` if `alpha` is not regularized with respect to
           zero.

           Note: The value of `alpha_Lin^sr(D_t)` is computed during the
           optimization parameter update.
        3. Solves the MPC problem and stores the resulting optimal control
           input.
        4. Stores the optimal value of `alpha` if `alpha` is regularized
           with respect to its previous optimal value (see Section V of [2]).
        5. Toggles online data updates based on the current tracking cost
           value.

        References:
            [2]: See class-level docstring for full reference details.
        """
        # Update data-driven constants online if data updates are enabled
        # (current tracking cost value is not small enough)
        if self.update_data:
            if self.ext_out_incr_in:
                # For a controller that uses an extended output representation
                # and input increments, the input Hankel matrix corresponds to
                # input increments instead of absolute inputs.
                # H_{L+n+1}(du)
                self.HLn1_u = hankel_matrix(self.du, self.L + self.n + 1)
            else:
                # H_{L+n+1}(u)
                self.HLn1_u = hankel_matrix(self.u, self.L + self.n + 1)
            # H_{L+n+1}(y)
            self.HLn1_y = hankel_matrix(self.y, self.L + self.n + 1)

        # Update MPC optimization parameters
        #
        # Note: The value of `alpha_Lin^sr(D_t)` is computed
        # during the optimization parameter update.
        self.update_optimization_parameters()

        # Solve MPC problem and store the optimal input
        self.solve_mpc_problem(warm_start=warm_start)
        self.get_optimal_control_input()

        # Update previous alpha value if the alpha
        # regularization type is `PREVIOUS`
        if self.alpha_reg_type == AlphaRegType.PREVIOUS:
            self.store_previous_alpha_value()

        # Toggle online data updates based on the tracking cost value
        if self.tracking_cost.value <= self.update_cost_threshold:
            self.update_data = False
        else:
            self.update_data = True

    def define_optimization_variables(self) -> None:
        """
        Define the optimization variables for the Data-Driven MPC controller.

        This method defines data-driven MPC optimization variables as
        described in the Nonlinear Data-Driven MPC formulation in [2].

        Note:
            This method initializes attributes that define the MPC
            optimization variables for the controller. Additional attributes
            are initialized if the MPC controller uses an extended output
            representation and input increments or if the `alpha`
            regularization type is `APPROXIMATED`.

        References:
            [2]: See class-level docstring for full reference details.
        """
        # alpha(t)
        self.alpha = cp.Variable((self.N - self.L - self.n, 1))

        # sigma(t)
        if self.ext_out_incr_in:
            self.sigma = cp.Variable(
                ((self.L + self.n + 1) * (self.m + self.p), 1)
            )
        else:
            self.sigma = cp.Variable(((self.L + self.n + 1) * self.p, 1))

        # ubar[-n, L](t)
        self.ubar = cp.Variable(((self.L + self.n + 1) * self.m, 1))
        # For a controller that uses an extended output representation and
        # input increments, `ubar` corresponds to input increments (du[k])
        # and not absolute control inputs (u[k]).

        # ybar[-n, L](t)
        if self.ext_out_incr_in:
            self.ybar = cp.Variable(
                ((self.L + self.n + 1) * (self.m + self.p), 1)
            )
        else:
            self.ybar = cp.Variable(((self.L + self.n + 1) * self.p, 1))

        # u_s(t)
        self.u_s = cp.Variable((self.m, 1))

        # y_s(t)
        if self.ext_out_incr_in:
            self.y_s = cp.Variable((self.m + self.p, 1))
        else:
            self.y_s = cp.Variable((self.p, 1))

        # Define variable segments for constraint definitions

        # Internal state constraints: Get initial `n` input-output predictions
        # ubar[-n, -1]
        self.ubar_state = self.ubar[: self.n * self.m]

        # ybar[-n, -1]
        if self.ext_out_incr_in:
            self.ybar_state = self.ybar[: self.n * (self.m + self.p)]
        else:
            self.ybar_state = self.ybar[: self.n * self.p]

        # Terminal state constraints:
        # Get terminal segments of input-output predictions
        # ubar[L-n, L]
        self.ubar_terminal = self.ubar[self.L * self.m :]

        # ybar[L-n, L]
        if self.ext_out_incr_in:
            self.ybar_terminal = self.ybar[-(self.n + 1) * (self.m + self.p) :]
        else:
            self.ybar_terminal = self.ybar[self.L * self.p :]

        # Input constraints:
        # Get absolute input variable indices (u[k])
        self.ubar_pred: cp.Expression
        if self.ext_out_incr_in:
            # Get `ubar` predicted input considering the extended output
            # representation: ybar_ext = [ybar, ubar]
            start_indices = np.arange(
                self.n * (self.m + self.p) + self.p,
                self.ybar.shape[0],
                self.p + self.m,
            )
            slices = [
                self.ybar[start : start + self.m] for start in start_indices
            ]
            self.ubar_pred = cp.vstack(slices)  # ubar[0,L] excluding output
        else:
            self.ubar_pred = self.ubar[self.n * self.m :]  # ubar[0,L]

        # sigma_ubar: The sigma values corresponding to the input when using a
        # controller with an extended output representation (ybar_ext =
        # [ybar, ubar]). Used to constrain the input sigma values to 0, since
        # the sigma values should only affect the output and not the input.
        if self.ext_out_incr_in:
            start_indices = np.arange(
                self.p, self.sigma.shape[0], self.p + self.m
            )
            slices = [
                self.sigma[start : start + self.m] for start in start_indices
            ]
            self.sigma_ubar = cp.vstack(slices)

        # Cost function related variables:
        # Define predicted input-output setpoint variable arrays
        # by repetition for cost function
        self.u_s_tiled = cp.vstack(
            [self.u_s for _ in range(self.L + self.n + 1)]
        )
        self.y_s_tiled = cp.vstack(
            [self.y_s for _ in range(self.L + self.n + 1)]
        )

        # Define optimization variables for computing the approximation of
        # alpha_Lin^sr(D_t) by solving Equation (23) of [2] if the alpha
        # regularization type is APPROXIMATED (regularization w.r.t. an
        # approximated alpha_Lin^sr(D_t) value).
        if self.alpha_reg_type == AlphaRegType.APPROXIMATED:
            # alpha_s
            self.alpha_s = cp.Variable((self.N - self.L - self.n, 1))

            # sigma_s
            if self.ext_out_incr_in:
                self.sigma_s = cp.Variable(
                    ((self.L + self.n + 1) * (self.m + self.p), 1)
                )
            else:
                self.sigma_s = cp.Variable(((self.L + self.n + 1) * self.p, 1))

            # sigma_s_ubar: The sigma_s values corresponding to the input.
            # Refer to sigma_ubar definition.
            if self.ext_out_incr_in:
                start_indices = np.arange(
                    self.p, self.sigma_s.shape[0], self.p + self.m
                )
                slices = [
                    self.sigma_s[start : start + self.m]
                    for start in start_indices
                ]
                self.sigma_s_ubar = cp.vstack(slices)

    def define_optimization_parameters(self) -> None:
        """
        Define MPC optimization parameters that are updated at every step
        iteration.

        This method initializes the following MPC parameters:

        - Output setpoint: `y_r_param`.
        - Hankel matrices: `HLn1_u_param` and `HLn1_y_param`.
        - Past inputs and outputs: `u_past_param` and `y_past_param`.
        - Past input increments: `du_past_param` (if applicable).
        - Computed value of `alpha_Lin^sr(D_t)`: `alpha_sr_Lin_D_param` (if
          `alpha` is not regularized with respect to zero).

        The value of `y_r_param` is initialized to `y_r`.

        These parameters are updated at each MPC iteration, except for
        `y_r_param`, which must be manually updated when setting a new
        controller setpoint.

        Using CVXPY `Parameter` objects allows efficient updates without the
        need of reformulating the MPC problem at every step.
        """
        # Define the parameter for y_r and initialize its value
        self.y_r_param = cp.Parameter(self.y_r.shape, name="y_r")
        self.y_r_param.value = self.y_r

        # H_{L+n+1}(u) - H_{L+n+1}(du)
        self.HLn1_u_param = cp.Parameter(self.HLn1_u.shape, name="HLn1_u")

        # H_{L+n+1}(y)
        self.HLn1_y_param = cp.Parameter(self.HLn1_y.shape, name="HLn1_y")

        # u[t-n, t-1]
        self.u_past_param = cp.Parameter((self.n * self.m, 1), name="u_past")

        # du[t-n, t-1]
        if self.ext_out_incr_in:
            self.du_past_param = cp.Parameter(
                (self.n * self.m, 1), name="du_past"
            )

        # y[t-n, t-1]
        if self.ext_out_incr_in:
            self.y_past_param = cp.Parameter(
                (self.n * (self.m + self.p), 1), name="y_past"
            )
        else:
            self.y_past_param = cp.Parameter(
                (self.n * self.p, 1), name="y_past"
            )

        # alpha_sr_Lin_D
        if self.alpha_reg_type != AlphaRegType.ZERO:
            self.alpha_sr_Lin_D_param = cp.Parameter(
                (self.N - self.L - self.n, 1), name="alpha_sr_Lin_D"
            )

    def update_optimization_parameters(self) -> None:
        """
        Update MPC optimization parameters.

        This method updates the MPC optimization parameters using the most
        recent input-output measurement data.

        Additionally, it updates the computed `alpha_Lin^sr(D_t)` value based
        on the alpha regularization type:

        - `alpha_reg_type == AlphaRegType.APPROXIMATED`: Computes
          `alpha_Lin^sr(D_t)` solving Equation (23) of [2] and updates its
          value.
        - `alpha_reg_type == AlphaRegType.PREVIOUS`: Updates
          `alpha_Lin^sr(D_t)` with the previous optimal value of `alpha`.

        References:
            [2]: See class-level docstring for full reference details.
        """
        # Update Hankel matrices if data updates are enabled
        if self.update_data:
            # H_{L+n+1}(u) - H_{L+n+1}(du)
            self.HLn1_u_param.value = self.HLn1_u

            # H_{L+n+1}(y)
            self.HLn1_y_param.value = self.HLn1_y

        # y[t-n, t-1]
        self.y_past_param.value = self.y[-self.n :].reshape(-1, 1)

        if self.ext_out_incr_in:
            # du[t-n, t-1]
            self.du_past_param.value = self.du[-self.n :].reshape(-1, 1)
        else:
            # u[t-n, t-1]
            self.u_past_param.value = self.u[-self.n :].reshape(-1, 1)

        # alpha_sr_Lin_D
        if self.alpha_reg_type == AlphaRegType.APPROXIMATED:
            self.alpha_sr_Lin_D_param.value = self.solve_alpha_sr_Lin_Dt()
        elif self.alpha_reg_type == AlphaRegType.PREVIOUS:
            self.alpha_sr_Lin_D_param.value = self.prev_alpha_val

    def define_mpc_constraints(self) -> None:
        """
        Define the constraints for the Data-Driven MPC formulation.

        This method defines the following constraints, as described in the
        Nonlinear Data-Driven MPC formulation in [2]:
        - **System dynamics**: Ensures input-output predictions are possible
            trajectories of the system based on a data-driven characterization
            of all its input-output trajectories. Defined by Equation (22b).
        - **Internal state**: Ensures predictions align with the internal
            state of the system's trajectory. This constrains the first `n`
            input-output predictions to match the past `n` input-output
            measurements of the system, guaranteeing that the predictions
            consider the initial state of the system. Defined by Equation
            (22c).
        - **Terminal state**: Aims to stabilize the internal state of the
            system so it aligns with the steady-state that corresponds to the
            input-output equilibrium pair (predicted equilibrium setpoints
            `u_s`, `y_s`) in any minimal realization (last `n` input-output
            predictions, as considered in [2]). Defined by Equation (22d).
        - **Input**: Constrains both the equilibrium input (predicted input
            setpoint `u_s`) and the input trajectory (`ubar`). Defined by
            Equation (22e).

        Note:
            This method initializes the `dynamics_constraints`,
            `internal_state_constraints`, `terminal_constraints`,
            `input_constraints`, and `constraints` attributes to define the
            MPC constraints based on the MPC controller type.

        References:
            [2]: See class-level docstring for full reference details.
        """
        # Define System Dynamic, Internal State,
        # Terminal State, and Input Constraints
        self.dynamics_constraints = self.define_system_dynamic_constraints()
        self.internal_state_constraints = (
            self.define_internal_state_constraints()
        )
        self.terminal_constraints = self.define_terminal_state_constraints()
        self.input_constraints = self.define_input_constraints()

        # Combine constraints
        self.constraints = (
            self.dynamics_constraints
            + self.internal_state_constraints
            + self.terminal_constraints
            + self.input_constraints
        )

    def define_system_dynamic_constraints(self) -> list[cp.Constraint]:
        """
        Define the system dynamic constraints for the Data-Driven MPC
        formulation.

        These constraints use a data-driven characterization of all the
        input-output trajectories of a system, as defined by Theorem 1 [2], to
        ensure predictions are possible system trajectories. This is analogous
        to the system dynamics constraints in a typical MPC formulation.

        These constraints are defined according to Equation (22b) of [2].

        Returns:
            list[cp.Constraint]: A list containing the CVXPY system dynamic
                constraints for the Data-Driven MPC controller, corresponding
                to the specified MPC controller type.

        References:
            [2]: See class-level docstring for full reference details.
        """
        dynamics_constraints = [
            cp.vstack(
                [self.ubar, self.ybar + self.sigma, cp.Constant(self.ones_1)]
            )
            == cp.vstack(
                [
                    self.HLn1_u_param,
                    self.HLn1_y_param,
                    cp.Constant(self.ones_NLn),
                ]
            )
            @ self.alpha
        ]

        return dynamics_constraints

    def define_internal_state_constraints(self) -> list[cp.Constraint]:
        """
        Define the internal state constraints for the Data-Driven MPC
        formulation.

        These constraints ensure predictions align with the internal state of
        the system's trajectory. This way, the first `n` input-output
        predictions are constrained to match the past `n` input-output
        measurements of the system, guaranteeing that the predictions consider
        the initial state of the system.

        These constraints are defined according to Equation (22c) of [2].

        Returns:
            list[cp.Constraint]: A list containing the CVXPY internal state
                constraints for the Data-Driven MPC controller.

        Note:
            It is essential to update the system's input-output measurements,
            `u`, `y`, and `du`, at each MPC iteration.

        References:
            [2]: See class-level docstring for full reference details.
        """
        if self.ext_out_incr_in:
            internal_state_constraints = [
                cp.vstack([self.ubar_state, self.ybar_state])
                == cp.vstack([self.du_past_param, self.y_past_param])
            ]
        else:
            internal_state_constraints = [
                cp.vstack([self.ubar_state, self.ybar_state])
                == cp.vstack([self.u_past_param, self.y_past_param])
            ]

        return internal_state_constraints

    def define_terminal_state_constraints(self) -> list[cp.Constraint]:
        """
        Define the terminal state constraints for the Data-Driven MPC
        formulation.

        These constraints aim to stabilize the internal state of the system so
        it aligns with the steady-state that corresponds to the input-output
        pair (predicted equilibrium setpoints `u_s`, `y_s`) in any minimal
        realization, specifically the last `n` input-output predictions, as
        considered in [2].

        These constraints are defined according to Equation (22d) of [2].

        Returns:
            list[cp.Constraint]: A list containing the CVXPY terminal state
                constraints for the Data-Driven MPC controller.

        References:
            [2]: See class-level docstring for full reference details.
        """
        # Define terminal state constraints for the
        # Nonlinear MPC based on Equation (22d) of [2]
        terminal_constraints = [
            cp.vstack([self.ubar_terminal, self.ybar_terminal])
            == cp.vstack(
                [
                    cp.kron(self.ones_n1, self.u_s),
                    cp.kron(self.ones_n1, self.y_s),
                ]
            )
        ]

        return terminal_constraints

    def define_input_constraints(self) -> list[cp.Constraint]:
        """
        Define the input constraints for the Data-Driven MPC formulation.

        These constraints are defined according to Equation (22e) of [2].

        Returns:
            list[cp.Constraint]: A list containing the CVXPY input constraints
                for the Data-Driven MPC controller.
        """
        # Define input constraints
        if self.ext_out_incr_in:
            # Input constraints considering the extended output setpoint
            # y_s_ext = [y_s, u_s]
            input_constraints = [
                self.ubar_pred >= self.U_const_low,
                self.ubar_pred <= self.U_const_up,
                self.y_s[self.p :] >= self.Us_const_low,
                self.y_s[self.p :] <= self.Us_const_up,
                self.sigma_ubar == 0,
            ]
            # For a controller that uses an extended output representation,
            # the sigma values corresponding to the input (sigma_ubar) are
            # constrained to 0, since the sigma values should only affect the
            # output and not the input.
        else:
            input_constraints = [
                self.ubar_pred >= self.U_const_low,
                self.ubar_pred <= self.U_const_up,
                self.u_s >= self.Us_const_low,
                self.u_s <= self.Us_const_up,
            ]

        return input_constraints

    def define_cost_function(self) -> None:
        """
        Define the cost function for the Data-Driven MPC formulation.

        This method defines the MPC cost function as described in the
        Nonlinear Data-Driven MPC formulation in Section IV of [2]. The value
        of `alpha_Lin^sr(D_t)` is computed during the cost function
        definition.

        Note:
            This method initializes the `cost` attribute to define the MPC
            cost function.

        References:
            [2]: See class-level docstring for full reference details.
        """
        # Define tracking cost
        self.tracking_cost = cp.quad_form(
            self.ubar - self.u_s_tiled, self.R
        ) + cp.quad_form(self.ybar - self.y_s_tiled, self.Q)

        if self.ext_out_incr_in:
            # Add input-related cost if an extended output representation
            # and input increments are considered for the controller. Refer to
            # Section V of [2].
            self.tracking_cost += cp.norm(self.ubar) ** 2

        # Define control-related cost
        control_cost = self.tracking_cost + cp.quad_form(
            self.y_s[: self.p] - self.y_r_param, self.S
        )

        # Define alpha-related cost
        if self.alpha_reg_type == AlphaRegType.ZERO:
            alpha_cost = self.lamb_alpha * cp.norm(self.alpha, 2) ** 2
        else:
            alpha_cost = (
                self.lamb_alpha
                * cp.norm(self.alpha - self.alpha_sr_Lin_D_param, 2) ** 2
            )

        # Define sigma-related cost
        sigma_cost = self.lamb_sigma * cp.norm(self.sigma) ** 2

        # Define cost
        self.cost = control_cost + alpha_cost + sigma_cost

    def define_mpc_problem(self) -> None:
        """
        Define the optimization problem for the Data-Driven MPC formulation.

        Note:
            This method initializes the `problem` attribute to define the MPC
            problem of the Data-Driven MPC controller, which is formulated as
            a Quadratic Programming (QP) problem. It assumes that the `cost`
            (objective function) and `constraints` attributes have already
            been defined.
        """
        # Define QP problem
        objective = cp.Minimize(self.cost)
        self.problem = cp.Problem(objective, self.constraints)

    def solve_mpc_problem(self, warm_start: bool = False) -> str:
        """
        Solve the optimization problem for the Data-Driven MPC formulation.

        Returns:
            str: The status of the optimization problem after attempting to
                solve it (e.g., "optimal", "optimal_inaccurate", "infeasible",
                "unbounded").

        Note:
            This method assumes that the MPC problem has already been defined.
            It solves the problem and updates the `problem` attribute with the
            solution status.
        """
        self.problem.solve(warm_start=warm_start)

        return self.problem.status

    def get_problem_solve_status(self) -> str:
        """
        Get the solve status of the optimization problem of the Data-Driven MPC
        formulation.

        Returns:
            str: The status of the optimization problem after attempting to
                solve it (e.g., "optimal", "optimal_inaccurate", "infeasible",
                "unbounded").
        """
        return self.problem.status

    def get_optimal_cost_value(self) -> float:
        """
        Get the cost value corresponding to the solved optimization problem of
        the Data-Driven MPC formulation.

        Returns:
            float: The optimal cost value of the solved MPC optimization
                problem.
        """
        return self.problem.value

    def get_optimal_control_input(self) -> np.ndarray:
        """
        Retrieve and store either the optimal control input or the optimal
        control input increments from the MPC solution.

        Returns:
            np.ndarray: The predicted optimal control input from time step 0
                to L. If the controller uses an extended output representation
                and input increments, returns the predicted optimal control
                input increments instead.

        Raises:
            ValueError: If the MPC problem solution status was not "optimal"
                or "optimal_inaccurate".

        Note:
            This method should be called after the MPC problem has been
            solved. It stores the predicted optimal control input in the
            `optimal_u` attribute, or the predicted optimal control input
            increments in the `optimal_du` attribute if the controller uses an
            extended output representation and input increments.
        """
        # Get optimal control input prediction value
        ubar_pred_val = self.ubar[self.n * self.m :].value

        # Store the optimal control input ubar*[0,L] if the MPC problem
        # solution had an "optimal" or "optimal_inaccurate" status
        if self.problem.status in {"optimal", "optimal_inaccurate"}:
            assert ubar_pred_val is not None  # Prevent mypy [union-attr] error

            if self.ext_out_incr_in:
                # For a controller that uses an extended output representation
                # and input increments, the optimal control variables are input
                # increments, not absolute control inputs.
                self.optimal_du = ubar_pred_val.flatten()
                return self.optimal_du
            else:
                self.optimal_u = ubar_pred_val.flatten()
                return self.optimal_u
        else:
            raise ValueError("MPC problem was not solved optimally.")

    def get_optimal_control_input_at_step(self, n_step: int = 0) -> np.ndarray:
        """
        Get the optimal control input (`u`) from the MPC solution
        corresponding to a specified time step in the prediction horizon
        [0, L].

        Args:
            n_step (int): The time step of the optimal control input to
                retrieve. It must be within the range [0, L].

        Returns:
            np.ndarray: An array containing the optimal control input for the
                specified prediction time step.

        Note:
            This method assumes that the optimal control input from the MPC
            solution has been stored in the `optimal_u` attribute. For a
            controller that uses an extended output representation and input
            increments, the last `du` value should contain the optimal control
            input increment computed from the previous MPC solution
            (`optimal_du[:m]`).

        Raises:
            ValueError: If `n_step` is not within the range [0, L].
        """
        # Ensure n_step is within prediction range [0,L]
        if not 0 <= n_step <= self.L:
            raise ValueError(
                f"The specified prediction time step ({n_step}) is out of "
                f"range. It should be within [0, {self.L}]."
            )

        if self.ext_out_incr_in:
            # For a controller that uses an extended output representation and
            # input increments, the optimal value computed in the current
            # step corresponds to the input increment `du[k]`. This value is
            # stored in `du` and is used in the next iteration, not the
            # current one. This is because, in our formulation, input
            # increments are defined as du[k] = u[k+1] - u[k].

            # Calculate the optimal input step considering the control input
            # u[k] = u[k-1] + du[k-1]
            u_k1 = self.u[-1:]
            du_k1 = self.du[-1:]
            optimal_u_step_n = u_k1 + du_k1
        else:
            optimal_u_step_n = self.optimal_u[
                n_step * self.m : (n_step + 1) * self.m
            ]

        return optimal_u_step_n

    def store_previous_alpha_value(self) -> None:
        """
        Store the previous optimal value of `alpha` for regularization in
        subsequent MPC iterations.
        """
        self.prev_alpha_val = self.alpha.value  #  type: ignore[assignment]
        # Note:
        # mypy [assignment] is ignored since `alpha.value` could only be `None`
        # if `alpha` were a sparse matrix, which is not the case in our system

    def get_du_value_at_step(self, n_step: int = 0) -> np.ndarray | None:
        """
        Get the optimal control input increment (`du`) from the MPC solution
        corresponding to a specified time step in the prediction horizon
        [0, L].

        Args:
            n_step (int): The time step of the optimal control input to
                retrieve. It must be within the range [0, L].

        Returns:
            np.ndarray | None: An array containing the optimal control
                input increment for the specified prediction time step if the
                controller uses an extended output representation and input
                increments. Otherwise, returns `None`.

        Note:
            This method assumes that the `optimal_du` attribute contains the
            optimal control input increments from the MPC solution.
        """
        if self.ext_out_incr_in:
            return self.optimal_du[n_step * self.m : (n_step + 1) * self.m]
        else:
            return None

    def store_input_output_measurement(
        self,
        u_current: np.ndarray,
        y_current: np.ndarray,
        du_current: np.ndarray | None = None,
    ) -> None:
        """
        Store an input-output measurement pair for the current time step in
        the input-output storage variables. If the controller uses an extended
        output representation and input increments, the input increment
        corresponding to the current input measurement is also stored.

        This method updates the input-output storage variables `u`, `y` and
        `du` by shifting the arrays and replacing the oldest measurements with
        the current ones.

        Args:
            u_current (np.ndarray): The control input (u[k]) for the current
                time step, expected to match the dimensions of prior inputs.
            y_current (np.ndarray): The measured system output for the current
                time step, expected to match the dimensions of prior outputs.
                This output should correspond to the system's response to
                `u_current`, as both represent a trajectory of the system.
            du_current (np.ndarray): The control input increment (du[k] =
                u[k+1] - u[k]) for the current time step, expected to match
                the dimensions of prior inputs.

        Raises:
            ValueError: If `u_current`, `y_current`, or `du_current` do not
                match the expected dimensions.

        Note:
            This method updates the `u`, `y`, and `du` arrays.

        References:
            [2]: See class-level docstring for full reference details.
        """
        # Check measurement dimensions
        expected_u0_dim = (self.m,)
        expected_y0_dim = (self.p,)
        if (
            u_current.shape != expected_u0_dim
            or y_current.shape != expected_y0_dim
        ):
            raise ValueError(
                "Incorrect dimensions. Expected dimensions are "
                f"{expected_u0_dim} for u_current and {expected_y0_dim} for "
                f"y_current, but got {u_current.shape} and {y_current.shape} "
                "instead."
            )

        if self.ext_out_incr_in:
            if du_current is None:
                raise ValueError(
                    "A valid `du_current` value is required for a controller "
                    "that uses an extended output representation and input "
                    "increments."
                )

            if du_current.shape != expected_u0_dim:
                raise ValueError(
                    "Incorrect dimensions for `du_current`. Expected "
                    f"dimensions are {expected_u0_dim}, but got "
                    f"{du_current.shape} instead."
                )

        # Shift and update control inputs
        self.u[:-1] = self.u[1:]
        self.u[-1:] = u_current

        # Shift and update input increments if used
        if self.ext_out_incr_in:
            self.du[:-1] = self.du[1:]
            self.du[-1:] = du_current

        # Shift and update outputs
        self.y[:-1] = self.y[1:]
        if self.ext_out_incr_in:
            # Update output considering the extended output
            # representation (y_ext[k] = [y[k], u[k]])
            self.y[-1:, : self.p] = y_current  # Store system output
            self.y[-1:, self.p :] = u_current  # Store control input
        else:
            self.y[-1:] = y_current

    def set_input_output_data(
        self,
        u: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Set the variables for the current input-output measurements.

        This method assigns the provided input-output measurements to the
        arrays storing the current input-output measurements, `u` and `y`.

        Args:
            u (np.ndarray): An array containing the current measured control
                input. Expected to have a shape of (N, m), where 'N' is the
                trajectory length and 'm' is the dimension of the input.
            y (np.ndarray): An array containing the current measured system
                output. Expected to have a shape of (N, p) where 'N' is the
                trajectory length and 'p' is the dimension of the output.

        Raises:
            ValueError: If `u` or `y` do not have correct dimensions.

        Note:
            This method sets the values of the `u` and `y` attributes with the
            provided new input-output data.
        """
        # Validate input types and dimensions
        expected_u_dim = (self.N, self.m)
        expected_y_dim = (self.N, self.p)

        if u.shape != expected_u_dim:
            raise ValueError(
                f"Incorrect dimensions. `u` must have shape {expected_u_dim}, "
                f"but got {u.shape} instead."
            )
        if y.shape != expected_y_dim:
            raise ValueError(
                f"Incorrect dimensions. `y` must have shape {expected_y_dim}, "
                f"but got {y.shape} instead."
            )

        # Update input-output trajectory data
        self.u = u.copy()  # Input trajectory data
        if self.ext_out_incr_in:
            # Incremental Input trajectory data
            du_last = np.zeros((self.m))
            self.du = np.vstack([u[1:, :] - u[:-1, :], [du_last]])
        # Output trajectory data
        if self.ext_out_incr_in:
            # Extended Output trajectory data
            self.y = np.hstack([y, u])
        else:
            self.y = y.copy()

    def set_output_setpoint(self, y_r: np.ndarray) -> None:
        """
        Set the system output setpoint of the Data-Driven MPC controller.

        Args:
            y_r (np.ndarray): The setpoint for system outputs.

        Raises:
            ValueError: If `y_r` does not have the expected dimensions.

        Note:
            This method sets the values of the `y_r` attribute with the
            provided new setpoint and updates the value of `y_r_param`
            to update the data-driven MPC controller setpoint.
        """
        # Validate input types and dimensions
        if y_r.shape != self.y_r.shape:
            raise ValueError(
                "Incorrect dimensions. `y_r` must have shape "
                f"{self.y_r.shape}, but got {y_r.shape} instead."
            )

        # Update output setpoint and its parameter value
        self.y_r = y_r
        self.y_r_param.value = y_r

    def define_alpha_sr_Lin_Dt_prob(self) -> None:
        """
        Define a Quadratic Programming (QP) problem for computing an
        approximation of `alpha_Lin^sr(D_t)` using the latest input-output
        system measurements for the current iteration. The QP formulation is
        based on Equation (23) of [2].

        References:
            [2]: See class-level docstring for full reference details.
        """
        # Define constraints
        if self.ext_out_incr_in:
            constraints = [
                cp.vstack(
                    [
                        self.HLn1_u_param,
                        self.HLn1_y_param,
                        cp.Constant(self.ones_NLn),
                    ]
                )
                @ self.alpha_s
                == cp.vstack(
                    [
                        cp.kron(self.ones_Ln1, self.u_s),
                        cp.kron(self.ones_Ln1, self.y_s) + self.sigma_s,
                        cp.Constant(self.ones_1),
                    ]
                ),
                self.y_s[self.p :] >= self.Us_const_low,
                self.y_s[self.p :] <= self.Us_const_up,
                self.sigma_s_ubar == 0,
            ]
        else:
            constraints = [
                cp.vstack(
                    [
                        self.HLn1_u_param,
                        self.HLn1_y_param,
                        cp.Constant(self.ones_NLn),
                    ]
                )
                @ self.alpha_s
                == cp.vstack(
                    [
                        cp.kron(self.ones_Ln1, self.u_s),
                        cp.kron(self.ones_Ln1, self.y_s) + self.sigma_s,
                        cp.Constant(self.ones_1),
                    ]
                ),
                self.u_s >= self.Us_const_low,
                self.u_s <= self.Us_const_up,
            ]

        # Define objective
        objective = cp.Minimize(
            cp.quad_form(self.y_s[: self.p] - self.y_r_param, self.S)
            + self.lamb_alpha_s * cp.norm(self.alpha_s, 2) ** 2
            + self.lamb_sigma_s * cp.norm(self.sigma_s, 2) ** 2
        )

        # Define the optimization problem
        self.alpha_sr_Lin_Dt_prob = cp.Problem(objective, constraints)

    def solve_alpha_sr_Lin_Dt(self) -> np.ndarray:
        """
        Compute the approximation of `alpha_Lin^sr(D_t)` using the latest
        input-output system measurements for the current iteration.

        Returns:
            np.ndarray: The computed approximation of `alpha_Lin^sr(D_t)`.

        Raises:
            ValueError: If the QP solver fails to converge to an optimal
                solution.

        Note:
            This method assumes that the Quadratic Programming (QP) problem
            for computing the approximation of `alpha_Lin^sr(D_t)`
            (`alpha_sr_Lin_Dt_prob`) has already been defined.
        """
        # Solve the optimization problem
        self.alpha_sr_Lin_Dt_prob.solve()

        # Get the robust approximation of alpha_sr_Lin(Dt) from solution
        alpha_sr_Lin_D = None
        if self.alpha_sr_Lin_Dt_prob.status in {
            "optimal",
            "optimal_inaccurate",
        }:
            alpha_sr_Lin_D = self.alpha_s.value
        else:
            raise ValueError(
                "Failed to compute a robust approximation of "
                "`alpha_sr_Lin(D_t)`: The optimization problem did not "
                "converge to a solution."
            )

        assert alpha_sr_Lin_D is not None  # Prevent mypy [return-value] error

        return alpha_sr_Lin_D
