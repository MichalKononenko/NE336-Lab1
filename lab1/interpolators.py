"""
Contains evaluation interpolators
"""
from functools import reduce
from itertools import count
import numpy as np
import operator as op

__all__ = ['lagrange_interpolant', 'lagrange_evaluate']

def _vandermonde_matrix(x: np.ndarray) -> np.ndarray:
    r"""
    For an array of polynomial coefficients x, it formulates the
    Vandermonde matrix. For a list of x coefficients 
    :math:`[x_0, x_1, x_2, ... x_n]`, the Vandermonde matrix has the 
    form

    .. math::
        \begin{bmatrix}
            1      & x_0    & x_0^2 & \cdots & x_0^{n - 1} \\
            1      & x_1    & x_1^2 & \cdots & x_1^{n - 1} \\
            1      & x_2    & x_2^2 & \cdots & x_2^{n - 1} \\
            \vdots & \ddots &       & \cdots & \vdots      \\
            1      & x_n    & x_n^2 & \cdots & x_n^{n - 1}
        \end{bmatrix}

    :param np.ndarray x: The array for which the Vandermonde matrix
        is to be calculated
    
    """
    return np.array(
        [[x_value ** power for power in range(0, len(x))] for x_value in x]
    )

def lagrange_interpolant(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    """
    return np.linalg.solve(_vandermonde_matrix(x), y)[::-1]

def lagrange_evaluate(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    For an array of input coefficients and an array of values at which
    to evaluate the interpolant, this method returns the value of the
    Lagrange interpolant formulated from the coefficients

    :param np.ndarray a: The coefficients that 
    """
    return np.polyval(a, x) 

def _permutations(n: int, k: int) -> int:
    r"""
    Returns the permutations according to the formula

    .. math::
        P = \frac{n!}{(n - k)!}

    """
    if k > n: raise ValueError("k > n. This is not allowed")
    return reduce(op.mul, [(n - k_i) for k_i in range(0, k)], 1)

def lagrange_differentiate(a: np.ndarray, n: int, x: np.ndarray) -> np.ndarray:
    """
    """
    ### A must be reversed
    reversed_a = a[::-1]
    return np.array(sum((
        _permutations(index, n) * reversed_a[index] * x ** (index - n) 
        for index in range(n, len(a))
    )))

def lagrange_integrate(a: np.ndarray, x0: float, x1: float) -> float:
    """
    """
    rev_a = a[::-1]

    return sum((
        rev_a[index]/(index + 1) * (x1 ** (index + 1) - x0 ** (index + 1))
        for index in range(len(rev_a))
    ))
