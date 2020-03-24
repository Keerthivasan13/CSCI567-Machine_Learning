from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        # Base case
        for s in range(S):
            alpha[s, 0] = self.pi[s] * self.B[s, self.obs_dict[Osequence[0]]]

        # Recurrence
        for t in range(1, L):
            for s in range(S):
                alpha[s, t] = self.B[s, self.obs_dict[Osequence[t]]] * np.sum(self.A[:, s] * alpha[:, t-1])
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        # Base case
        for s in range(S):
            beta[s, L-1] = 1

        # Recurrence
        for t in range(L-2, -1, -1):
            for s in range(S):
                beta[s, t] = np.sum(self.A[s, :] * self.B[:, self.obs_dict[Osequence[t+1]]] * beta[:, t+1])
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        prob = np.sum(self.forward(Osequence)[:, -1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)

        for s in range(S):
            prob[s, ] = alpha[s, :] * beta[s, :]
        prob = np.divide(prob, self.sequence_prob(Osequence))

        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)

        for t in range(L-1):
            for s in range(S):
                for s_ in range(S):
                    prob[s, s_, t] = alpha[s, t] * self.A[s, s_] * self.B[s_, self.obs_dict[Osequence[t+1]]] * beta[s_, t+1]

        prob = np.divide(prob, self.sequence_prob(Osequence))
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        S = len(self.pi)
        L = len(Osequence)

        delta = np.zeros([S, L])
        precede = np.zeros([S, L-1])

        ###################################################
        # Q3.3 Edit here
        # Base case
        for s in range(S):
            delta[s, 0] = self.pi[s] * self.B[s, self.obs_dict[Osequence[0]]]

        for t in range(1, L):
            for s in range(S):
                temp = self.A[:, s] * delta[:, t-1]
                delta[s, t] = self.B[s, self.obs_dict[Osequence[t]]] * np.max(temp)
                precede[s, t-1] = np.argmax(temp)

        # Backtracking
        subs = {}
        for key, val in self.state_dict.items():
            subs[val] = key

        path.append(np.argmax(delta[:, L-1]))

        for t in range(L-2, -1, -1):
            path.append(int(precede[path[-1], t]))

        # Return as Path
        return [subs[x] for x in path[::-1]]

        ###################################################
