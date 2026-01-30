import numpy as np
from scipy.optimize import linprog

class NashSolver:
    """Solves Nash equilibria for 2-player games"""
    
    @staticmethod
    def solve_zero_sum(payoff_matrix):
        """Solve zero-sum game using linear programming"""
        n_a, n_b = payoff_matrix.shape
        
        # Player A's problem (maximizer)
        c = np.zeros(n_a + 1)
        c[0] = -1  # Maximize v
        
        # Constraints: payoff^T * p >= v
        A_ub = np.zeros((n_b, n_a + 1))
        for j in range(n_b):
            A_ub[j, 0] = 1
            for i in range(n_a):
                A_ub[j, i + 1] = -payoff_matrix[i, j]
        
        b_ub = np.zeros(n_b)
        A_eq = np.zeros((1, n_a + 1))
        A_eq[0, 1:] = 1
        b_eq = np.array([1])
        
        bounds = [(None, None)] + [(0, 1)] * n_a
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                     bounds=bounds, method='highs')
        
        if res.success:
            value = res.x[0]
            policy_a = res.x[1:]
        else:
            value = 0
            policy_a = np.ones(n_a) / n_a
        
        # Player B's best response (minimizer)
        expected = payoff_matrix.T @ policy_a
        policy_b = np.zeros(n_b)
        policy_b[np.argmin(expected)] = 1
        
        return policy_a, policy_b, value, -value
    
    @staticmethod
    def compute_payoff_matrix(state, actions_a, actions_b, get_q_value_func, agent_id):
        """Build payoff matrix from Q-values"""
        n_a, n_b = len(actions_a), len(actions_b)
        payoff = np.zeros((n_a, n_b))
        
        for i, a_i in enumerate(actions_a):
            for j, a_j in enumerate(actions_b):
                if agent_id == 0:
                    payoff[i, j] = get_q_value_func(state, a_i, a_j)
                else:
                    payoff[i, j] = get_q_value_func(state, a_j, a_i)
        
        return payoff