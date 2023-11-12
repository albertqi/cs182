# -*- coding: utf-8 -*-
"""
CS 182 Problem Set 3: Python Coding Questions - Fall 2023
Due November 15, 2023 at 11:59pm
"""

### Package Imports ###
import numpy as np
import matplotlib.pyplot as plt

### Package Imports ###

#### Coding Problem Set General Instructions - PLEASE READ ####
# 1. Unlike previous psets, this code does not need to be submitted; there is no autograder.
# 2. This code goes with Problem 2: Employment Status, on this pset.
# 3. This starter code has been provided to you, feel free to use it (or not, if you want to code something different) however you
#    see fit. Change variables, solve the question in another way, however you need to best understand the question and your code.
#    This coding problem can be written in a variety of different ways, this code is only a rough sketch of what your code could
#    look like. We encourage you to change it if you need to.
# 4. Make sure you write the optimal policies your code determines and copy your code and graphs onto your written submission.


n_states = 3  #  0 is Safely Employed (SE), 1 is PIP, 2 is Unemployed (UE).
n_actions = 2  # 0 is Code, 1 is Netflix.

t_p = np.zeros((n_states, n_actions, n_states))
# Transition Probabilities: These are represented as a 3-dimensional array.
# t_p[s_1, a, s_2] = p indicates that beginning from state s_1 and taking action a will result in state s_2 with probability p.
t_p[0, 0, 0] = 1  #     t_p[SE, Code, SE]      = 1
t_p[0, 1, 0] = 1 / 4  # t_p[SE, Netflix, SE]   = 1 / 4
t_p[0, 1, 1] = 3 / 4  # t_p[SE, Code, PIP]     = 3 / 4
t_p[1, 0, 0] = 1 / 4  # t_p[PIP, Code, SE]     = 1 / 4
t_p[1, 0, 1] = 3 / 4  # t_p[PIP, Code, PIP]    = 3 / 4
t_p[1, 1, 1] = 7 / 8  # t_p[PIP, Netflix, PIP] = 7 / 8
t_p[1, 1, 2] = 1 / 8  # t_p[PIP, Netflix, UE]  = 1 / 8
# All other transition probabilities are 0.

r = np.zeros((n_states, n_actions))
# Reward Values: These are represented as a 2-dimensional array.
# r[s, a] = val indicates that taking at state s, taking action a will give a reward of val.
r[0, 0] = 4  #  r[SE, Code]       = 4
r[0, 1] = 10  # r[SE, Netflix]    = 10
r[1, 0] = 4  #  r[PIP, Code]      = 4
r[1, 1] = 10  # r[PIP, Netflix]   = 10


def Q(state, action, gamma, V):
    val = 0
    for new_state in range(n_states):
        val += t_p[state, action, new_state] * (r[state, action] + gamma * V[new_state])
    return val


def value_iteration(policy, gamma, nsteps=float("inf")):
    V = np.zeros(n_states, dtype=float)
    vals = []

    i = 0
    while i < nsteps:
        vals.append(np.sum(V))

        V_next = np.zeros(n_states, dtype=float)
        for state in range(n_states):
            action = policy[state]
            V_next[state] = Q(state, action, gamma, V)
        V = V_next
        i += 1

    return vals


def policy_iteration(gamma):
    """
    You should find the optimal policy for Liz under the constrants of discount factor gamma, which is given as a parameter.
    Relevant variables and the transition probabilities are defined above, feel free to use them and change them how you want.
    What this function returns is up to you and how you want to determine the sum of utilities at each iteration in the plots.
    """

    theta = 1e-5  # Define a theta that determines if the change in utilities from iteration to iteration is "small enough".

    policy = np.zeros(
        n_states, dtype=int
    )  # Define your policy, which begins as Netflix regardless of state.

    while True:
        # Policy Evaluation
        V = np.zeros(n_states, dtype=float)
        while True:
            V_next = np.zeros(n_states, dtype=float)
            for state in range(n_states):
                action = policy[state]
                for new_state in range(n_states):
                    V_next[state] += t_p[state, action, new_state] * (
                        r[state, action] + gamma * V[new_state]
                    )
            if np.max(np.abs(V - V_next)) <= theta:
                V = V_next
                break
            V = V_next

        # Policy Iteration
        policy_next = np.zeros(n_states, dtype=int)
        for state in range(n_states):
            action_vals = np.zeros(n_actions, dtype=float)
            for action in range(n_actions):
                for new_state in range(n_states):
                    action_vals[action] += t_p[state, action, new_state] * (
                        r[state, action] + gamma * V[new_state]
                    )
            policy_next[state] = np.argmax(action_vals)

        # Policy Change Check
        if np.all(policy == policy_next):
            return policy
        policy = policy_next


def value_plots(p1_vals, p2_vals):
    """
    Your plots should indicate the cumulative utility summed across all states across iterations. More specifically, your y-val
    should indicate the total amount of utility acumulated across the states and actions as the iterations progress. This means
    you likely will have to keep track of what policies you have at every iteration, or some other method that will allow you to
    determine the cumulative sum of utilities as iterations continue.
    """

    iterations = range(0, 50)

    # You will need to find a way to calculate the cumulative utility values for policy 1 and policy 2.
    plt.plot(iterations, p1_vals, label="Policies for gamma = 0.9")
    plt.plot(iterations, p2_vals, label="Policies for gamma = 0.8")
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Utility Value")
    plt.legend()
    plt.title("Cumulative Utility Values over Time")
    plt.show()


if __name__ == "__main__":
    # Policy iteration to verify your answer from problem 2 part c, with gamma = 0.9.
    p1 = policy_iteration(0.9)
    p1_vals = value_iteration(p1, 0.9, 50)
    print(p1)

    # Policy iteration for problem 2 part d, with gamma = 0.8.
    p2 = policy_iteration(0.8)
    p2_vals = value_iteration(p2, 0.8, 50)
    print(p2)

    value_plots(p1_vals, p2_vals)
    # You will need to find some way to get the total utility values to the value_plots function. For example, you could pass
    # in the policies or you could pass in the cumulative sum of utility values.
