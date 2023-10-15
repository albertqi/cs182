# -*- coding: utf-8 -*-
"""
CS 182 Problem Set 2: Python Coding Questions - Fall 2023
Due October 18, 2023 at 11:59pm
"""
### Package Imports ###
from scipy.optimize import linprog
import numpy as np

### Package Imports

#### Coding Problem Set General Instructions - PLEASE READ ####
# 1. All code should be written in python 3.7 or higher to be compatible with the autograder
# 2. Your submission file must be named "pset2a.py" exactly
# 3. Submit this file to Pset2 - Coding Submission A
# 4. No additional outside packages can be referenced or called, they will result in an import error on the autograder
# 5. Function/method/class/attribute names should not be changed from the default starter code provided
# 6. All helper functions and other supporting code should be wholly contained in the default starter code declarations provided.
#    Functions and objects from your submission are imported in the autograder by name, unexpected functions will not be included in the import sequence


def linear_programming(pm):
    """
    You are given a payoff matrix of the form pm[i][j][k]. i represents Fiona's choice
    and j represents Paul's choice, both indicated by 0, 1, 2, for red, blue, green respectively.
    k = 0 will give Fiona's payoff for a certain selection of cards, and k = 1 will give Paul's payoff.

    Return a tuple comprised of two lists, both of the form [prob_r, prob_g, prog_b, value]. Fiona's
    strategy should be first in the tuple, and the value in each list should return Paul's expected value
    in both cases. More specifically, you should return something of the form ([f_r, f_g, f_b, val],
    [p_r, p_g, p_b, val]).

    To solve the LP, utilize the linprog function from SciPy. To ensure that the results match the
    staff results, please pass the argument method="highs" into your linprog call.

    Your code should not be dependent on the exact values in the payoff matrix, but should correctly
    solve the linear programming regardless of what (3x3) payoff matrix is passed in as a parameter.

    Do not attempt to hard code the solution and return a static list--you will fail our hidden
    autograder tests.
    """

    """*** YOUR CODE HERE ***"""

    raise NotImplementedError


if __name__ == "__main__":
    payoff = [
        [[8, 2], [10, 3], [7, 4]],
        [[9, 11], [8, 6], [8, 2]],
        [[8, 3], [9, 4], [9, -4]],
    ]

    strategies = linear_programming(payoff)
    fiona = strategies[0]
    paul = strategies[1]
    print(f"val = {fiona[3]}")
    print(f"(f_r, f_g, f_b) = ({fiona[0]}, {fiona[1]}, {fiona[2]})")
    print(f"(p_r, p_g, p_b) = ({paul[0]}, {paul[1]}, {paul[2]})")
    assert fiona[0] + fiona[1] + fiona[2] == 1
    assert paul[0] + paul[1] + paul[2] == 1
