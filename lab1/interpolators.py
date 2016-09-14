"""
Contains functions for calculating Lagrangian interpolators for a
set of (x, y) points, or a given set of polynomial coefficients.
Functions for calculating the nth order derivative and first-order definite
integral of a Lagrangian interpolant.

As is Python convention, functions beginning with an underscore `_`, are private.
This means they're not an intended part of this module's API.

.. note::
    In this module, an array of polynomial coefficients is assumed to have
    the value of its highest power on the left-most side of an array. 
    For example, an array ``a = [4, 4, -1]`` corresponds to the polynomial
    
    .. math::
        y(x) = 4x^2 + 4x - 1

    not 

    .. math::
        y(x) = -x^2 + 4x + 4
"""
from functools import reduce
from itertools import count
import numpy as np
import operator as op

__all__ = [
    'lagrange_interpolant', 'lagrange_evaluate',
    'lagrange_differentiate', 'lagrange_integrate'
]

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

    The Vandermonde matrix is generated using a nested list comprehension.

    :param :class:`numpy.ndarray` x: The array for which the Vandermonde 
        matrix is to be calculated
    :return: A square matrix of the same length and width as the longest
        dimension of the input vector of x values. This array is the
        Vandermonde matrix of the system
    :rtype: :class:`numpy.ndarray`
    """
    return np.array(
        [[x_value ** power for power in range(0, len(x))] for x_value in x]
    )

def lagrange_interpolant(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Using :meth:`_vandermonde_matrix`, this method finds the Lagrange
    interpolant by inverting the Vandermonde matrix and premultiplying it
    by the vector of ``y`` values provided for the interpolant. 

    If the Vandermonde matrix is :math:`V`, then this method solves the
    system :math:`Va = y`. 

    :param :class:`numpy.ndarray` x : The list of `x` coordinates that are 
        to be used when determining the interpolant
    :param :class:`numpy.ndarray` y : The list of `y` coordinates to be used
        in drawing the interpolant
    :return: An array with the inerpolant's coefficients
    :rtype: :class:`numpy.ndarray`
    """
    return np.linalg.solve(_vandermonde_matrix(x), y)[::-1]

def lagrange_evaluate(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    For an array of input coefficients and an array of values at which
    to evaluate the interpolant, this method returns the value of the
    Lagrange interpolant formulated from the coefficients

    :param :class:`numpy.ndarray` a: The array of coefficients for the 
        interpolating polynomial
    :param :class:`numpy.ndarray` x: The array of x values at which the
        polynomial is to be evaluated
    :return: An array of values cotnaing each value at which the 
        interpolating polynomial was evaluated
    :rtype: :class:`numpy.ndarray`
    """
    return np.polyval(a, x) 

def _permutations(n: int, k: int) -> int:
    r"""
    Returns the permutations according to the formula

    .. math::
        P = \frac{n!}{(n - k)!}

    The actual implementation uses a reducer acting on a generator
    to compute the product :math:`n(n-1)(n-2)\cdots(n-k)`.

    :param int n: The cardinality of the set from which the
        elements for calculating the number of permutations are
        drawn.
    :param int k: The number of elements in a subset of the
        set represented by `n`, such that a tuple of this length
        constitutes a valid permutation of the set to be
        permuted
    :return: The number of permutations of length `k` that can
        be made from a set of `n` members
    :rtype int:
    :raises: :exc:`ValueError` if `k` is greater than `n`. The
        permutation function is not defined in these cases.
    """
    if k > n: raise ValueError("k > n. This is not allowed")
    return reduce(op.mul, ((n - k_i) for k_i in range(0, k)), 1)

def lagrange_differentiate(a: np.ndarray, n: int, x: np.ndarray) -> np.ndarray:
    r"""
    Uses the formula

    .. math::
        y^{(n)}(x) = \sum\limits_{n = k}^N \frac{n!}{(n - k)!} a_n x^{n - k}

    To compute the nth order derivative of the Lagrangian interpolant. This
    formula was developed as a generalization of the first-order derivative
    for an arbitrary polynomial.

    :param :class:`numpy.ndarray` a: The array of polynomial coefficients
        that define an interpolating polynomial.
    :param int n: The order of the derivative of the interpolating
        polynomial is to be calculated.
    :param :class:`numpy.ndarray` x: The values for which the derivative is
        to be evaluated
    """
    reversed_a = a[::-1]
    return np.array(sum((
        _permutations(index, n) * reversed_a[index] * x ** (index - n) 
        for index in range(n, len(a))
    )))

def lagrange_integrate(a: np.ndarray, x0: float, x1: float) -> float:
    """
    Calculate the first-order definite integral of the interpolant from
    :var:`x0` to :var:`x1`. This function calculates

    .. math::
        \int_{x_0}^{x_1} \sum\limits_{n = 1}^{N} a_n x^n \, \mathrm{d}x

    :param :class:`numpy.ndarray` a: The interpolant coefficients
    :param float x0: The lower limit of the definite integral
    :param float x1: The upper limit of the integral
    :return: The definite integral
    :rtype: float
    """
    rev_a = a[::-1]

    return sum((
        rev_a[index]/(index + 1) * (x1 ** (index + 1) - x0 ** (index + 1))
        for index in range(len(rev_a))
    ))
