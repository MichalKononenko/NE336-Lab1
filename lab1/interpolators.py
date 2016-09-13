"""
Contains evaluation interpolators
"""
import numpy as np

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
