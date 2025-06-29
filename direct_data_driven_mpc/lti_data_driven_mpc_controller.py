from enum import Enum

import cvxpy as cp
import numpy as np

from direct_data_driven_mpc.utilities.hankel_matrix import (
    evaluate_persistent_excitation,
    hankel_matrix,
)


# Define Direct Data-Driven MPC Controller Types
class LTIDataDrivenMPCType(Enum):
    NOMINAL = 0  # Nominal Data-Driven MPC
    ROBUST = 1  # Robust Data-Driven MPC


# Define Slack Variable Constraint Types for Robust Data-Driven MPC
class SlackVarConstraintType(Enum):
    # Non-Convex slack variable constraint
    NON_CONVEX = 0
    # Convex slack variable constraint
    CONVEX = 1
    # Omit an explicit constraint. The slack variable
    # constraint is implicitly satisfied
    NONE = 2


class LTIDataDrivenMPCController:
    """
    A class that implements a Data-Driven Model Predictive Control (MPC)
    controller for Linear Time-Invariant (LTI) systems. This controller can be
    configured as either a Nominal or a Robust controller. The implementation
    is based on research by J. Berberich et al., as described in [1].

    Attributes:
        controller_type (DataDrivenMPCType): The Data-Driven MPC controller
            type.
        n (int): The estimated order of the system.
        m (int): The number of control inputs.
        p (int): The number of system outputs.
        u_d (np.ndarray): A persistently exciting input sequence.
        y_d (np.ndarray): The system's output response to `u_d`.
        N (int): The length of the initial input (`u_d`) and output (`y_d`)
            trajectories.
        u_past (np.ndarray): The past `n` input measurements (u[t-n, t-1]).
        y_past (np.ndarray): The past `n` output measurements (y[t-n, t-1]).
        L (int): The prediction horizon length.
        Q (np.ndarray): The output weighting matrix for the MPC formulation.
        R (np.ndarray): The input weighting matrix for the MPC formulation.
        u_s (np.ndarray): The setpoint for control inputs.
        y_s (np.ndarray): The setpoint for system outputs.
        eps_max (float | None): The estimated upper bound of the system
            measurement noise.
        lamb_alpha (float | None]): The ridge regularization base weight for
            `alpha`, scaled by `eps_max`.
        lamb_sigma (float | None]): The ridge regularization weight for
            `sigma`.
        U (np.ndarray | None): An array of shape (`m`, 2) containing the
            bounds for the `m` predicted inputs. Each row specifies the
            `[min, max]` bounds for a single input. If `None`, no input bounds
            are applied.
        c (float | None): A constant used to define a Convex constraint for
            the slack variable `sigma` in a Robust MPC formulation.
        slack_var_constraint_type (SlackVarConstraintTypes): The constraint
            type for the slack variable `sigma` in a Robust MPC formulation.
        n_mpc_step (int): The number of consecutive applications of the
            optimal input for an n-Step Data-Driven MPC Scheme (multi-step).
        use_terminal_constraints (bool): If `True`, include terminal equality
            constraints in the Data-Driven MPC formulation. If `False`, the
            controller will not enforce these constraints.
        HLn_ud (np.ndarray): The Hankel matrix constructed from the input data
            `u_d`.
        HLn_yd (np.ndarray): The Hankel matrix constructed from the output
            data `y_d`.
        alpha (cp.Variable): The optimization variable for a data-driven
            input-output trajectory characterization of the system.
        ubar (cp.Variable): The predicted control input variable.
        ybar (cp.Variable): The predicted system output variable.
        sigma (cp.Variable | None): The slack variable to account for noisy
            measurements in a Robust Data-Driven MPC.
        dynamics_constraints (list[cp.Constraint]): The system dynamics
            constraints for a Data-Driven MPC formulation.
        internal_state_constraints (list[cp.Constraint]): The internal state
            constraints for a Data-Driven MPC formulation.
        terminal_constraints (list[cp.Constraint]): The terminal state
            constraints for a Data-Driven MPC formulation.
        slack_var_constraint (list[cp.Constraint]): The slack variable
            constraints for a Robust Data-Driven MPC formulation.
        constraints (list[cp.Constraint]): The combined constraints for the
            Data-Driven MPC formulation.
        cost (cp.Expression): The cost function for the Data-Driven MPC
            formulation.
        problem (cp.Problem): The quadratic programming problem for the
            Data-Driven MPC.
        optimal_u (np.ndarray): The optimal control input derived from the
            Data-Driven MPC solution.

    References:
        [1] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer,
            "Data-Driven Model Predictive Control With Stability and
            Robustness Guarantees," in IEEE Transactions on Automatic Control,
            vol. 66, no. 4, pp. 1702-1717, April 2021,
            doi: 10.1109/TAC.2020.3000182.
    """

    def __init__(
        self,
        n: int,
        m: int,
        p: int,
        u_d: np.ndarray,
        y_d: np.ndarray,
        L: int,
        Q: np.ndarray,
        R: np.ndarray,
        u_s: np.ndarray,
        y_s: np.ndarray,
        eps_max: float | None = None,
        lamb_alpha: float | None = None,
        lamb_sigma: float | None = None,
        U: np.ndarray | None = None,
        c: float | None = None,
        slack_var_constraint_type: SlackVarConstraintType = (
            SlackVarConstraintType.CONVEX
        ),
        controller_type: LTIDataDrivenMPCType = (LTIDataDrivenMPCType.NOMINAL),
        n_mpc_step: int = 1,
        use_terminal_constraints: bool = True,
    ):
        """
        Initialize a Direct LTI Data-Driven MPC with specified system model
        parameters, an initial input-output data trajectory measured from the
        system, and LTI Data-Driven MPC parameters.

        Note:
            The input data `u_d` used to excite the system to get the initial
            output data must be persistently exciting of order (L + 2 * n), as
            defined in the Data-Driven MPC formulations in [1].

        Args:
            n (int): The estimated order of the system.
            m (int): The number of control inputs.
            p (int): The number of system outputs.
            u_d (np.ndarray): A persistently exciting input sequence.
            y_d (np.ndarray): The system's output response to `u_d`.
            L (int): The prediction horizon length.
            Q (np.ndarray): The output weighting matrix for the MPC
                formulation.
            R (np.ndarray): The input weighting matrix for the MPC
                formulation.
            u_s (np.ndarray): The setpoint for control inputs.
            y_s (np.ndarray): The setpoint for system outputs.
            eps_max (float | None): The estimated upper bound of the system
                measurement noise.
            lamb_alpha (float | None): The ridge regularization base weight
                for `alpha`. It is scaled by `eps_max`.
            lamb_sigma (float | None): The ridge regularization weight for
                `sigma`.
            U (np.ndarray | None): An array of shape (`m`, 2) containing
                the bounds for the `m` predicted inputs. Each row specifies
                the `[min, max]` bounds for a single input. If `None`, no
                input bounds are applied. Defaults to `None`.
            c (float | None): A constant used to define a Convex constraint
                for the slack variable `sigma` in a Robust MPC formulation.
            slack_var_constraint_type (SlackVarConstraintTypes): The
                constraint type for the slack variable `sigma` in a Robust MPC
                formulation.
            controller_type (DataDrivenMPCType): The Data-Driven MPC
                controller type.
            n_mpc_step (int): The number of consecutive applications of the
                optimal input for an n-Step Data-Driven MPC Scheme
                (multi-step). Defaults to 1.
            use_terminal_constraints (bool): If `True`, include terminal
                equality constraints in the Data-Driven MPC formulation. If
                `False`, the controller will not enforce these constraints.

        References:
            [1] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer,
                "Data-Driven Model Predictive Control With Stability and
                Robustness Guarantees," in IEEE Transactions on Automatic
                Control, vol. 66, no. 4, pp. 1702-1717, April 2021, doi:
                10.1109/TAC.2020.3000182.
        """
        # Set controller type
        self.controller_type = controller_type  # Nominal or Robust Controller

        # Validate controller type
        controller_types = {
            LTIDataDrivenMPCType.NOMINAL,
            LTIDataDrivenMPCType.ROBUST,
        }
        if controller_type not in controller_types:
            raise ValueError("Unsupported controller type.")

        # Define system model
        self.n = n  # Estimated system order
        self.m = m  # Number of inputs
        self.p = p  # Number of outputs

        # Initial Input-Output trajectory data
        self.u_d = u_d  # Input trajectory data
        self.y_d = y_d  # Output trajectory data
        self.N = u_d.shape[0]  # Initial input-output trajectory length

        # Initialize storage variables for past `n` input-output measurements
        # (used for the internal state constraints that ensure predictions
        # align with the internal state of the system's trajectory)
        self.u_past: np.ndarray = u_d[-n:, :].reshape(-1, 1)  # u[t-n, t-1]
        self.y_past: np.ndarray = y_d[-n:, :].reshape(-1, 1)  # y[t-n, t-1]

        # Define Data-Driven MPC parameters
        self.L = L  # Prediction horizon
        self.Q = Q  # Output weighting matrix
        self.R = R  # Input weighting matrix

        self.u_s = u_s  # Control input setpoint
        self.y_s = y_s  # System output setpoint

        self.eps_max = eps_max  # Upper limit of bounded measurement noise
        self.lamb_alpha = lamb_alpha  # Ridge regularization base weight for
        # alpha. It is scaled by `eps_max`.

        self.lamb_sigma = lamb_sigma  # Ridge regularization weight for sigma
        # (If large enough, can neglect noise constraints)

        self.U = U  # Bounds for the predicted input

        self.c = c  # Convex slack variable constraint:
        # ||sigma||_inf <= c * eps_max

        self.slack_var_constraint_type = slack_var_constraint_type  # Slack
        # variable constraint type

        # Validate slack variable constraint type
        slack_var_constraint_types = {
            SlackVarConstraintType.NON_CONVEX,
            SlackVarConstraintType.CONVEX,
            SlackVarConstraintType.NONE,
        }
        if slack_var_constraint_type not in slack_var_constraint_types:
            raise ValueError("Unsupported slack variable constraint type.")

        # Ensure correct parameter definition for Robust MPC controller
        if self.controller_type == LTIDataDrivenMPCType.ROBUST:
            if None in (eps_max, lamb_alpha, lamb_sigma, c):
                raise ValueError(
                    "All robust MPC parameters (`eps_max`, `lamb_alpha`, "
                    "`lamb_sigma`, `c`) must be provided for a 'ROBUST' "
                    "controller."
                )

        # n-Step Data-Driven MPC Scheme parameters
        self.n_mpc_step = n_mpc_step  # Number of consecutive applications
        # of the optimal input

        # Define bounds for the predicted inputs and predicted input setpoints
        if self.U is not None:
            self.U_const_low = np.tile(self.U[:, 0:1], (self.L, 1))
            self.U_const_up = np.tile(self.U[:, 1:2], (self.L, 1))

        # Terminal constraints use in Data-Driven MPC formulation
        self.use_terminal_constraints = use_terminal_constraints

        # Evaluate if input trajectory data is persistently exciting of
        # order (L + 2 * n)
        self.evaluate_input_persistent_excitation()

        # Check correct prediction horizon length and cost matrix dimensions
        self.check_prediction_horizon_length()
        self.check_weighting_matrices_dimensions()

        # Initialize Data-Driven MPC controller
        self.initialize_data_driven_mpc()

    def evaluate_input_persistent_excitation(self) -> None:
        """
        Evaluate whether the input data is persistently exciting of order
        (L + 2 * n).

        This method first verifies that the length of the elements in the
        input data matches the number of inputs of the system. Then, it
        ensures that the length of the input sequence meets a minimum
        threshold, as defined in Remark 1 [1]. Finally, it evaluates the rank
        of the Hankel matrix induced by the input sequence to determine if the
        input sequence is persistently exciting, as described in Definition 1
        [1].

        Raises:
            ValueError: If the length of the elements in the data sequence
                does not match the number of inputs of the system, or if the
                input data is not persistently exciting of order (L + 2 * n).

        References:
            [1]: See class-level docstring for full reference details.
        """
        # Get the length of the elements of the data sequence
        u_d_n = self.u_d.shape[1]  # m - Number of inputs

        # Check if the number of inputs matches the expected
        # number of inputs of the system
        if u_d_n != self.m:
            raise ValueError(
                f"The length of the elements of the data sequence ({u_d_n}) "
                f"should match the number of inputs of the system ({self.m})."
            )

        # Compute the minimum required length of the input sequence
        # based on Remark 1 of [1]
        N_min = self.m * (self.L + 2 * self.n) + self.L + 2 * self.n - 1

        # Check if the length of the input sequence is sufficient
        if self.N < N_min:
            raise ValueError(
                "Initial input trajectory data is not persistently exciting "
                "of order (L + 2 * n). It does not satisfy the inequality: "
                "N - L - 2 * n + 1 ≥ m * (L + 2 * n). The required minimum N "
                f"is {N_min}, but got {self.N}."
            )

        # Evaluate if input data is persistently exciting of order (L + 2 * n)
        # based on the rank of its induced Hankel matrix
        expected_order = self.L + 2 * self.n
        in_hankel_rank, in_pers_exc = evaluate_persistent_excitation(
            X=self.u_d, order=expected_order
        )

        if not in_pers_exc:
            raise ValueError(
                "Initial input trajectory data is not persistently exciting "
                "of order (L + 2 * n). The rank of its induced Hankel matrix "
                f"({in_hankel_rank}) does not match the expected rank ("
                f"{u_d_n * expected_order})."
            )

    def check_prediction_horizon_length(self) -> None:
        """
        Check if the prediction horizon length, `L`, satisfies the MPC system
        design restriction based on the type of the Direct Data-Driven MPC
        controller. For a `Nominal` type, it must be greater than or equal to
        the estimated system order `n`. For a `Robust` controller, it must be
        greater than or equal to two times the estimated system order `n`.

        These restrictions are defined by Assumption 3, for the Nominal MPC
        scheme, and Assumption 4, for the Robust one, as described in [1].

        Raises:
            ValueError: If the prediction horizon `L` is less than the
                required threshold based on the controller type.

        References:
            [1]: See class-level docstring for full reference details.
        """
        if self.controller_type == LTIDataDrivenMPCType.NOMINAL:
            if self.L < self.n:
                raise ValueError(
                    "The prediction horizon (`L`) must be greater than or "
                    "equal to the estimated system order `n`."
                )
        elif self.controller_type == LTIDataDrivenMPCType.ROBUST:
            if self.L < 2 * self.n:
                raise ValueError(
                    "The prediction horizon (`L`) must be greater than or "
                    "equal to two times the estimated system order `n`."
                )

    def check_weighting_matrices_dimensions(self) -> None:
        """
        Check if the dimensions of the output and input weighting matrices, Q
        and R, are correct for an MPC formulation based on their order.

        Raises:
            ValueError: If the dimensions of the Q or R matrices are incorrect.
        """
        expected_output_shape = (self.p * self.L, self.p * self.L)
        expected_input_shape = (self.m * self.L, self.m * self.L)

        if self.Q.shape != expected_output_shape:
            raise ValueError(
                "Output weighting square matrix Q should be of order (p * L)."
            )
        if self.R.shape != expected_input_shape:
            raise ValueError(
                "Input weighting square matrix R should be of order (m * L)."
            )

    def initialize_data_driven_mpc(self) -> None:
        """
        Initialize the Data-Driven MPC controller.

        This method performs the following tasks:
        1. Constructs Hankel matrices from the initial input-output trajectory
           data (`u_d`, `y_d`). These matrices are used for the data-driven
           characterization of the unknown system, as defined by the system
           dynamic constraints in the Robust and Nominal Data-Driven MPC
           formulations of [1].
        2. Defines the optimization variables for the Data-Driven MPC problem.
        3. Defines the constraints for the MPC problem, which include the
           system dynamics, internal state, terminal state, and input
           constraints. For a Robust MPC controller, it also includes the
           slack variable constraint.
        4. Defines the cost function for the MPC problem.
        5. Formulates the MPC problem as a Quadratic Programming (QP) problem.
        6. Solves the initialized MPC problem to ensure the formulation is
           valid and retrieve the optimal control input for the initial
           setup.

        This initialization process ensures that all necessary components for
        the Data-Driven MPC are correctly defined and that the MPC problem is
        solvable with the provided initial data.

        References:
            [1]: See class-level docstring for full reference details.
        """
        # Construct Hankel Matrices from the initial input-output trajectory
        # for the data-driven characterization of the unknown system.
        # Used for the system dynamic constraints, as defined by Equations
        # (3b) (Nominal) and (6a) (Robust) of [1].
        self.HLn_ud = hankel_matrix(self.u_d, self.L + self.n)  # H_{L+n}(u^d)
        self.HLn_yd = hankel_matrix(self.y_d, self.L + self.n)  # H_{L+n}(y^d)

        # Define the Data-Driven MPC problem
        self.define_optimization_variables()
        self.define_optimization_parameters()
        self.update_optimization_parameters()
        self.define_mpc_constraints()
        self.define_cost_function()
        self.define_mpc_problem()

        # Validate the Data-Driven MPC formulation with an initial solution
        self.solve_mpc_problem()
        self.get_optimal_control_input()

    def update_and_solve_data_driven_mpc(self) -> None:
        """
        Update the Data-Driven MPC optimization parameters, solve the problem,
        and store the optimal control input.

        This method updates the MPC optimization parameters to incorporate the
        latest `n` input-output measurements of the system. It then solves the
        MPC problem and stores the resulting optimal control input.

        References:
            [1]: See class-level docstring for full reference details.
        """
        # Update MPC optimization parameters
        self.update_optimization_parameters()

        # Solve MPC problem and store the optimal input
        self.solve_mpc_problem()
        self.get_optimal_control_input()

    def define_optimization_variables(self) -> None:
        """
        Define the optimization variables for the Data-Driven MPC formulation
        based on the specified MPC controller type.

        This method defines data-driven MPC optimization variables as
        described in the Nominal and Robust MPC formulations in [1]:
        - **Nominal MPC**: Defines the variable `alpha` for a data-driven
            input-output trajectory characterization of the system, and the
            predicted input (`ubar`) and output (`ybar`) variables, as
            described in Equation (3).
        - **Robust MPC**: In addition to the optimization variables defined
            for a Nominal MPC formulation, defines the `sigma` variable to
            account for noisy measurements, as described in Equation (6).

        Note:
            This method initializes the `alpha`, `ubar`, `ybar`, and `sigma`
            attributes to define the MPC optimization variables based on the
            MPC controller type. The `sigma` variable is only initialized for
            a Robust MPC controller.

        References:
            [1]: See class-level docstring for full reference details.
        """
        # alpha(t)
        self.alpha = cp.Variable((self.N - self.L - self.n + 1, 1))
        # ubar[-n, L-1](t)
        self.ubar = cp.Variable(((self.L + self.n) * self.m, 1))
        # ybar[-n, L-1](t)
        self.ybar = cp.Variable(((self.L + self.n) * self.p, 1))
        # The time indices of the predicted input and output start at k = −n,
        # since the last `n` inputs and outputs are used to invoke a unique
        # initial state at time `t`, as described in Definition 3 of [1].

        if self.controller_type == LTIDataDrivenMPCType.ROBUST:
            # sigma(t)
            self.sigma = cp.Variable(((self.L + self.n) * self.p, 1))

    def define_optimization_parameters(self) -> None:
        """
        Define MPC optimization parameters that are updated at every step
        iteration.

        This method initializes the input setpoint (`u_s_param`), output
        setpoint (`y_s_param`), past inputs (`u_past_param`), and past outputs
        (`y_past_param`) MPC parameters.

        The values of `u_s_param` and `y_s_param` are initialized to `u_s` and
        `y_s`.

        These parameters are updated at each MPC iteration, except for
        `u_s_param` and `y_s_param`, which must be manually updated when
        setting new controller setpoint pairs.

        Using CVXPY `Parameter` objects allows efficient updates without the
        need of reformulating the MPC problem at every step.
        """
        # Define input-output setpoint parameters and initialize their values
        self.u_s_param = cp.Parameter(self.u_s.shape, name="u_s")
        self.u_s_param.value = self.u_s

        self.y_s_param = cp.Parameter(self.y_s.shape, name="y_s")
        self.y_s_param.value = self.y_s

        # u[t-n, t-1]
        self.u_past_param = cp.Parameter((self.n * self.m, 1), name="u_past")

        # y[t-n, t-1]
        self.y_past_param = cp.Parameter((self.n * self.p, 1), name="y_past")

    def update_optimization_parameters(self) -> None:
        """
        Update MPC optimization parameters.

        This method updates MPC parameters with the latest input-output
        measurement data.
        """
        # u[t-n, t-1]
        self.u_past_param.value = self.u_past

        # y[t-n, t-1]
        self.y_past_param.value = self.y_past

    def define_mpc_constraints(self) -> None:
        """
        Define the constraints for the Data-Driven MPC formulation based on
        the specified MPC controller type.

        This method defines the following constraints, as described in the
        Nominal and Robust MPC formulations in [1]:
        - **System dynamics**: Ensures input-output predictions are possible
            trajectories of the system based on a data-driven characterization
            of all its input-output trajectories. In a Robust MPC scheme, adds
            a slack variable to account for noisy measurements. Defined by
            Equations (3b) (Nominal) and (6a) (Robust).
        - **Internal state**: Ensures predictions align with the internal
            state of the system's trajectory. This constrains the first `n`
            input-output predictions to match the past `n` input-output
            measurements of the system, guaranteeing that the predictions
            consider the initial state of the system. Defined by Equations
            (3c) (Nominal) and (6b) (Robust).
        - **Terminal state**: Aims to stabilize the internal state of the
            system so it aligns with the steady-state that corresponds to the
            input-output pair (`u_s`, `y_s`) in any minimal realization (last
            `n` input-output predictions, as considered in [1]). Defined by
            Equations (3d) (Nominal) and (6c) (Robust).
        - **Input**: Constrains the predicted input (`ubar`). Defined by
            Equation (6c).
        - **Slack Variable**: Bounds a slack variable that accounts
            for noisy online measurements and for noisy data used for
            prediction (used to construct the Hankel matrices). Defined by
            Equation (6d), for a Non-Convex constraint, and Remark 3, for a
            Convex constraint and an implicit alternative.

        Note:
            This method initializes the `dynamics_constraints`,
            `internal_state_constraints`, `terminal_constraints`,
            `input_constraints`, `slack_var_constraint`, and `constraints`
            attributes to define the MPC constraints based on the MPC
            controller type.

        References:
            [1]: See class-level docstring for full reference details.
        """
        # Define System Dynamic, Internal State and Terminal State Constraints
        self.dynamics_constraints = self.define_system_dynamic_constraints()
        self.internal_state_constraints = (
            self.define_internal_state_constraints()
        )
        self.terminal_constraints = (
            self.define_terminal_state_constraints()
            if self.use_terminal_constraints
            else []
        )

        # Define Input constraints if U is provided
        self.input_constraints = (
            self.define_input_constraints() if self.U is not None else []
        )

        # Define Slack Variable Constraint if controller type is Robust
        self.slack_var_constraint = (
            self.define_slack_variable_constraint()
            if self.controller_type == LTIDataDrivenMPCType.ROBUST
            else []
        )

        # Combine constraints
        self.constraints = (
            self.dynamics_constraints
            + self.internal_state_constraints
            + self.terminal_constraints
            + self.input_constraints
            + self.slack_var_constraint
        )

    def define_system_dynamic_constraints(self) -> list[cp.Constraint]:
        """
        Define the system dynamic constraints for the Data-Driven MPC
        formulation corresponding to the specified MPC controller type.

        These constraints use a data-driven characterization of all the
        input-output trajectories of a system, as defined by Theorem 1 [1], to
        ensure predictions are possible system trajectories. This is analogous
        to the system dynamics constraints in a typical MPC formulation.

        In a Robust MPC scheme, these constraints include a slack variable to
        account for noisy online measurements and for noisy data used for
        prediction (used to construct the Hankel matrices).

        These constraints are defined according to the following equations
        from the Nominal and Robust MPC formulations in [1]:
        - Nominal MPC: Equation (3b).
        - Robust MPC: Equation (6a).

        Returns:
            list[cp.Constraint]: A list containing the CVXPY system dynamic
                constraints for the Data-Driven MPC controller, corresponding
                to the specified MPC controller type.

        References:
            [1]: See class-level docstring for full reference details.
        """
        if self.controller_type == LTIDataDrivenMPCType.NOMINAL:
            # Define system dynamic constraints for Nominal MPC
            # based on Equation (3b) of [1]
            dynamics_constraints = [
                cp.vstack([self.ubar, self.ybar])
                == cp.vstack([self.HLn_ud, self.HLn_yd]) @ self.alpha
            ]
        elif self.controller_type == LTIDataDrivenMPCType.ROBUST:
            # Define system dynamic constraints for Robust MPC
            # including a slack variable to account for noise,
            # based on Equation (6a) of [1]
            dynamics_constraints = [
                cp.vstack([self.ubar, self.ybar + self.sigma])
                == cp.vstack([self.HLn_ud, self.HLn_yd]) @ self.alpha
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

        These constraints are defined according to Equations (3c) (Nominal)
        and (6b) (Robust) from the Nominal and Robust MPC formulations in [1].

        Returns:
            list[cp.Constraint]: A list containing the CVXPY internal state
                constraints for the Data-Driven MPC controller.

        Note:
            It is essential to update the past `n` input-output measurements
            of the system, `u_past` and `y_past`, at each MPC iteration.

        References:
            [1]: See class-level docstring for full reference details.
        """
        # Define internal state constraints for Nominal and Robust MPC
        # based on Equations (3c) and (6b) of [1], respectively
        ubar_state = self.ubar[: self.n * self.m]  # ubar[-n, -1]
        ybar_state = self.ybar[: self.n * self.p]  # ybar[-n, -1]
        internal_state_constraints = [
            cp.vstack([ubar_state, ybar_state])
            == cp.vstack([self.u_past_param, self.y_past_param])
        ]

        return internal_state_constraints

    def define_terminal_state_constraints(self) -> list[cp.Constraint]:
        """
        Define the terminal state constraints for the Data-Driven MPC
        formulation.

        These constraints aim to stabilize the internal state of the system so
        it aligns with the steady-state that corresponds to the input-output
        pair (`u_s`, `y_s`) in any minimal realization, specifically the last
        `n` input-output predictions, as considered in [1].

        These constraints are defined according to Equations (3d) (Nominal) and
        (6c) (Robust) from the Nominal and Robust MPC formulations in [1].

        Returns:
            list[cp.Constraint]: A list containing the CVXPY terminal state
                constraints for the Data-Driven MPC controller.

        References:
            [1]: See class-level docstring for full reference details.
        """
        # Get terminal segments of input-output predictions
        # ubar[L-n, L-1]
        ubar_terminal = self.ubar[self.L * self.m : (self.L + self.n) * self.m]
        # ybar[L-n, L-1]
        ybar_terminal = self.ybar[self.L * self.p : (self.L + self.n) * self.p]

        # Replicate steady-state vectors to match minimum realization
        # dimensions for constraint comparison
        u_sn = cp.vstack([self.u_s_param] * self.n)
        y_sn = cp.vstack([self.y_s_param] * self.n)

        # Define terminal state constraints for Nominal and Robust MPC
        # based on Equations (3d) and (6c) of [1], respectively.
        terminal_constraints = [
            cp.vstack([ubar_terminal, ybar_terminal])
            == cp.vstack([u_sn, y_sn])
        ]

        return terminal_constraints

    def define_input_constraints(self) -> list[cp.Constraint]:
        """
        Define the input constraints for the Data-Driven MPC formulation.

        These constraints are defined according to Equation (6c) of [1].

        Returns:
            list[cp.Constraint]: A list containing the CVXPY input constraints
                for the Data-Driven MPC controller.
        """
        # Define input constraints
        ubar_pred = self.ubar[self.n * self.m :]  # ubar[0,L-1]
        input_constraints = [
            ubar_pred >= self.U_const_low,
            ubar_pred <= self.U_const_up,
        ]

        return input_constraints

    def define_slack_variable_constraint(self) -> list[cp.Constraint]:
        """
        Define the slack variable constraint for a Robust Data-Driven MPC
        formulation based on the specified slack variable constraint type.

        This constraint bounds a slack variable (`sigma`) that accounts for
        noisy online measurements and for noisy data used for prediction (used
        to construct the Hankel matrices for the system dynamic constraint).

        As described in [1], this constraint can be defined in three different
        ways, achieving the same theoretical guarantees:
        - **Non-Convex**: Defines a non-convex constraint (Equation (6d)).
        - **Convex**: Defines a convex constraint using a sufficiently large
            coefficient `c` (Remark 3).
        - **None**: Omits an explicit constraint definition. The slack variable
            constraint is implicitly met, relying on a high `lamb_sigma` value
            (Remark 3).

        Returns:
            list[cp.Constraint]: A list containing the CVXPY slack variable
                constraint for the Robust Data-Driven MPC controller,
                corresponding to the specified slack variable constraint type.
                The list is empty if the `NONE` constraint type is selected.

        References:
            [1]: See class-level docstring for full reference details.
        """
        # Get prediction segments of sigma variable
        sigma_pred = self.sigma[self.n * self.p :]  # sigma[0,L-1]

        # Define slack variable constraint for Robust MPC based
        # on the noise constraint type
        slack_variable_constraint = []
        if self.slack_var_constraint_type == SlackVarConstraintType.NON_CONVEX:
            # Raise NotImplementedError for NON-CONVEX constraint
            raise NotImplementedError(
                "Robust Data-Driven MPC with a Non-Convex slack variable "
                "constraint is not currently implemented, since it cannot "
                "be efficiently solved."
            )
        elif self.slack_var_constraint_type == SlackVarConstraintType.CONVEX:
            # Prevent mypy [operator] error
            assert self.c is not None and self.eps_max is not None

            # Define slack variable constraint considering
            # a CONVEX constraint based on Remark 3 [1]
            slack_variable_constraint = [
                cp.norm(sigma_pred, "inf") <= self.c * self.eps_max
            ]

        return slack_variable_constraint

    def define_cost_function(self) -> None:
        """
        Define the cost function for the Data-Driven MPC formulation based on
        the specified MPC controller type.

        This method defines the MPC cost function as described in the Nominal
        and Robust MPC formulations in [1]:
        - **Nominal MPC**: Implements a quadratic stage cost that penalizes
            deviations of the predicted control inputs (`ubar`) and outputs
            (`ybar`) from the desired equilibrium (`u_s`, `y_s`), as described
            in Equation (3).
        - **Robust MPC**: In addition to the quadratic stage cost, adds ridge
            regularization terms for `alpha` and `sigma` variables to account
            for noisy measurements, as described in Equation (6).

        Note:
            This method initializes the `cost` attribute to define the MPC
            cost function based on the MPC controller type.

        References:
            [1]: See class-level docstring for full reference details.
        """
        # Get segments of input-output predictions from time step 0 to (L - 1)
        # ubar[0,L-1]
        ubar_pred = self.ubar[self.n * self.m : (self.L + self.n) * self.m]
        # ybar[0,L-1]
        ybar_pred = self.ybar[self.n * self.p : (self.L + self.n) * self.p]

        # Define control-related cost
        control_cost = cp.quad_form(
            ubar_pred - cp.vstack([self.u_s_param] * self.L), self.R
        ) + cp.quad_form(
            ybar_pred - cp.vstack([self.y_s_param] * self.L), self.Q
        )

        # Define noise-related cost if controller type is Robust
        if self.controller_type == LTIDataDrivenMPCType.ROBUST:
            # Prevent mypy [operator] error
            assert (
                self.lamb_alpha is not None
                and self.eps_max is not None
                and self.lamb_sigma is not None
            )

            noise_cost = (
                self.lamb_alpha * self.eps_max * cp.norm(self.alpha, 2) ** 2
                + self.lamb_sigma * cp.norm(self.sigma, 2) ** 2
            )

            # Define MPC cost function for a Robust MPC controller
            self.cost = control_cost + noise_cost
        else:
            # Define MPC cost function for a Nominal MPC controller
            self.cost = control_cost

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
        Retrieve and store the optimal control input from the MPC solution.

        Returns:
            np.ndarray: The predicted optimal control input from time step 0
                to (L - 1).

        Raises:
            ValueError: If the MPC problem solution status was not "optimal"
                or "optimal_inaccurate".

        Note:
            This method should be called after the MPC problem has been
            solved. It stores the predicted optimal control input in the
            `optimal_u` attribute.
        """
        # Get segment of the input prediction from time step 0 to (L - 1)
        # ubar[0,L-1]
        ubar_pred = self.ubar[self.n * self.m : (self.L + self.n) * self.m]

        # Store the optimal control input ubar*[0,L-1] if the MPC problem
        # solution had an "optimal" or "optimal_inaccurate" status
        if self.problem.status in {"optimal", "optimal_inaccurate"}:
            # Prevent mypy [union-attr] error
            assert ubar_pred.value is not None

            self.optimal_u = ubar_pred.value.flatten()
            return self.optimal_u
        else:
            raise ValueError("MPC problem was not solved optimally.")

    def get_optimal_control_input_at_step(self, n_step: int = 0) -> np.ndarray:
        """
        Get the optimal control input from the MPC solution corresponding
        to a specified time step in the prediction horizon [0, L-1].

        Args:
            n_step (int): The time step of the optimal control input to
                retrieve. It must be within the range [0, L-1].

        Returns:
            np.ndarray: An array containing the optimal control input for the
                specified prediction time step.

        Note:
            This method assumes that the optimal control input from the MPC
            solution has been stored in the `optimal_u` attribute.

        Raises:
            ValueError: If `n_step` is not within the range [0, L-1].
        """
        # Ensure n_step is within prediction range [0,L-1]
        if not 0 <= n_step < self.L:
            raise ValueError(
                f"The specified prediction time step ({n_step}) is out of "
                f"range. It should be within [0, {self.L - 1}]."
            )

        optimal_u_step_n = self.optimal_u[
            n_step * self.m : (n_step + 1) * self.m
        ]

        return optimal_u_step_n

    def store_input_output_measurement(
        self,
        u_current: np.ndarray,
        y_current: np.ndarray,
    ) -> None:
        """
        Store an input-output measurement pair for the current time step in
        the input-output storage variables.

        This method updates the input-output storage variables `u_past` and
        `y_past` by appending the current input-output measurements and
        removing the oldest measurements located at the first position. This
        ensures these variables only store the past `n` measurements, as
        required for the internal state constraints defined by Equations (3c)
        (Nominal) and (6b) (Robust) of [1].

        Args:
            u_current (np.ndarray): The control input for the current
                time step, expected to match the dimensions of prior inputs.
            y_current (np.ndarray): The measured system output for the current
                time step, expected to match the dimensions of prior outputs.
                This output should correspond to the system's response to
                `u_current`, as both represent a trajectory of the system.

        Raises:
            ValueError: If `u_current` or `y_current` do not match the
                expected dimensions.

        Note:
            This method modifies the `u_past` and `y_past` arrays directly to
            ensure that only the most recent `n` measurements are retained.

        References:
            [1]: See class-level docstring for full reference details.
        """
        # Check measurement dimensions
        expected_u0_dim = (self.m, 1)
        expected_y0_dim = (self.p, 1)
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

        # Shift input-output storage arrays to discard the oldest
        # measurements and append the new ones
        # u[t-n, t-1]
        self.u_past = np.vstack([self.u_past[self.m :], u_current])
        # y[t-n, t-1]
        self.y_past = np.vstack([self.y_past[self.p :], y_current])

    def set_past_input_output_data(
        self,
        u_past: np.ndarray,
        y_past: np.ndarray,
    ) -> None:
        """
        Set the storage variables for past input-output measurements.

        This method assigns the provided input-output measurements to the
        arrays storing past input-output measurements, `u_past` and `y_past`.
        It is intended to be used for setting the historical data used in
        the MPC problem formulation.

        Args:
            u_past (np.ndarray): An array containing past control inputs.
                Expected to have a shape of (n * m, 1), where 'n' is the
                estimated system order and 'm' is the dimension of the input.
            y_past (np.ndarray): An array containing past measured system
                outputs. Expected to have a shape of (n * p, 1) where 'n' is
                the estimated system order and 'p' is the dimension of the
                output.

        Raises:
            ValueError: If `u_past` or `y_past` do not have correct
                dimensions.

        Note:
            This method sets the values of the `u_past` and `y_past`
            attributes with the provided new historical data.
        """
        # Validate input types and dimensions
        expected_u_dim = (self.n * self.m, 1)
        expected_y_dim = (self.n * self.p, 1)
        if u_past.shape != expected_u_dim:
            raise ValueError(
                "Incorrect dimensions. `u_past` must have shape "
                f"{expected_u_dim}, but got {u_past.shape} instead."
            )
        if y_past.shape != expected_y_dim:
            raise ValueError(
                "Incorrect dimensions. `y_past` must have shape "
                f"{expected_y_dim}, but got {y_past.shape} instead."
            )

        # Update past input-output data
        # u[t-n, t-1]
        self.u_past = u_past
        # y[t-n, t-1]
        self.y_past = y_past

    def set_input_output_setpoints(
        self, u_s: np.ndarray, y_s: np.ndarray
    ) -> None:
        """
        Set the control and system setpoints of the Data-Driven MPC controller.

        Args:
            u_s (np.ndarray): The setpoint for control inputs.
            y_s (np.ndarray): The setpoint for system outputs.

        Raises:
            ValueError: If `u_s` or `y_s` do not have the expected
                dimensions.

        Note:
            This method sets the values of the `u_s` and `y_s` attributes with
            the provided new setpoints and updates the values of `u_s_param`
            and `y_r_param` to update the data-driven MPC controller setpoint.
        """
        # Validate input types and dimensions
        if u_s.shape != self.u_s.shape:
            raise ValueError(
                "Incorrect dimensions. `u_s` must have shape "
                f"{self.u_s.shape}, but got {u_s.shape} instead."
            )
        if y_s.shape != self.y_s.shape:  # Replace with actual expected shape
            raise ValueError(
                "Incorrect dimensions. `y_s` must have shape "
                f"{self.y_s.shape}, but got {y_s.shape} instead."
            )

        # Update input-output setpoints and their parameter values
        self.u_s = u_s
        self.y_s = y_s
        self.u_s_param.value = u_s
        self.y_s_param.value = y_s
