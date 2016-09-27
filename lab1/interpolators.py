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
    'lagrange_differentiate', 'lagrange_integrate',
    'ArraysNotEqualError'
]

class ArraysNotEqualError(ValueError):
    """
    Thrown if two arrays that are supposed to be of equal length are
    not equal in length
    """
    pass

def lagrange_interpolant(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""
    Performs Lagrange interpolation on an array of ``x`` and ``y``
    values. Returns the coefficients for the Lagrange interpolating
    polynomial. The Lagrange interpolating polynomial is defined
    as 

    .. math::
        
        L(x) := \sum\limits_{j = 0}^k y_j 
            \prod\limits_{\substack{0 \leq m \leq k \\ m \neq j}}^k
            \frac{x - x_m}{x_j - x_m}

    The names of variables used in the method match those in the 
    definition above. 

    :param numpy.ndarray x: The list of `x` coordinates that are 
        to be used when determining the interpolant
    :param numpy.ndarray y: The list of `y` coordinates to be used
        in drawing the interpolant
    :return: An array with the inerpolant's coefficients
    :rtype: :class:`numpy.ndarray`
    :raises: :exc:`ArraysNotEqualError` if the length of `x` and `y` do not
        match
    """
    if len(x) != len(y): 
        raise ArraysNotEqualError("""
            The length of the x array of %d is not equal to the length
            of the y array %d
            """ % (len(x), len(y))
        )

    def basis_polynomial(j: int) -> np.array:
        r"""
        Returns a Lagrange Basis Polynomial for a given index j.
        The Lagrange basis polynomial is defined as

        .. math::
            
            l_j(x) := \prod\limits_{\substack{0 \leq m \leq k \\ m \neq j}}^k 
                \frac{x - x_m}{x_j - x_m}

        This method makes use of numpy's convolution method to multiply a set
        of linear polynomials. The values used in the code, as well as in
        this method's API, are derived from the Lagrange interpolation formula.


        .. note::
            This method must be inside :func:`lagrange_interpolant`, as it
            scopes the variables ``x`` and ``y`` for Lagrange interpolation,
            that are located below the stack frame from which this method
            is called. I don't see a use for ``basis_polynomial`` anywhere
            outside Lagrange interpolation, so I'm confident in keeping
            it as a helper method within the ``lagrange_interpolant`` function

        :param int j: The index of the x value for which to skip over
            Lagrange interpolation

        """
        return reduce(
            np.convolve,
            ([1, -x[m]]/(x[j] - x[m]) for m in range(0, len(y)) if m != j)
        )

    return sum((y[j] * basis_polynomial(j) for j in range(0, len(y))))

def divided_difference(
        y_values: np.ndarray, 
        x_values: np.ndarray
    ) -> np.ndarray:
    r"""
    Returns the divided difference between a set of ``(x, y)`` coordinates.

    The divided difference :math:`[y_0, y_1, ..., y_j]` is defined as

    .. math::
        [y_0] = y_0 \\
        [y_0, y_1] = \frac{y_1 - y_0}{x_1 - x_0} \\
        [y_0, y_1, ..., y_{j - 1}, y_j] = 
            \frac{[y_0, ..., y_{j-1}] - [y_1, ..., y_j]}{x_j - x_0}
    
    The calculation is done assuming four cases

    1. The divided difference of an empty list is 0
    2. The divided difference of a list with a single entry is that entry
    3. The divided difference of a list with two entries is calculated
        using the definition given in the formula
    4. The divided difference of any other list is calculated recursively
        using the third definition given above

     :param numpy.ndarray x_values: The list of `x` coordinates that are 
        to be used when determining the divided difference
    :param numpy.ndarray y_values: The list of `y` coordinates to be used
        in drawing the divided difference
    :return: An array with the inerpolant's coefficients
    :rtype: :class:`numpy.ndarray`
    :raises: :exc:`ArraysNotEqualError` if the length of `x` and `y` do not
        match
    """
    if len(x_values) != len(y_values):
        raise ArraysNotEqualError(
        """The arrays for divided_difference are
            not of equal length. len(x) == %d while len(y) == %d"""
            % (len(x), len(y))
        )
    if len(y_values) == 0:
        return np.array([0])
    elif len(y_values) == 1:
        return y_values
    elif len(y_values) == 2:
        return (y_values[1] - y_values[0])/(x_values[1] - x_values[0])
    else:
        return (
                divided_difference(y_values[1:], x_values[1:]) - \
                divided_difference(y_values[:-1], x_values[:-1])
        )/(x_values[-1] - x_values[0])

def newton_interpolant(x: np.ndarray, y: np.ndarray) -> np.ndarray: 
    r"""
    Returns the Newton interpolant of a series of points with x and y.
    Returns the coefficients for the interpolating polynomial, using
    the same API as :meth:`lagrange_interpolant`

    A Newton interpolant is defined as

    .. math::
        N(x) := \sum\limits_{j=0}^k a_j n_j(x)

    Where :math:`a_j = [y_0, y_1, ..., y_j]` is the divided difference 
    between the y coordinate values, and :math:`n_j(x)` are the Newton 
    basis polynomials. The divided difference is defined recursively
    as

    .. math::
        [y_0] = y_0 \\
        [y_0, y_1] = \frac{y_1 - y_0}{x_1 - x_0} \\
        [y_0, y_1, ..., y_{j - 1}, y_j] = 
            \frac{[y_0, ..., y_{j-1}] - [y_1, ..., y_j]}{x_j - x_0}

    The Newton basis polynomials are defined as

    .. math::
        n_j(x) = \product\limits_{i = 0}^{j - 1} (x - x_j)

    Where possible, notation in the function definitions matches
    with that of the definitions given above.


    :param numpy.ndarray x: The list of `x` coordinates that are 
        to be used when determining the interpolant
    :param numpy.ndarray y: The list of `y` coordinates to be used
        in drawing the interpolant
    :return: An array with the inerpolant's coefficients
    :rtype: :class:`numpy.ndarray`
    :raises: :exc:`ArraysNotEqualError` if the length of `x` and `y` do not
        match
    """
    if len(x) != len(y):
        raise ArraysNotEqualError(
        """The arrays for divided_difference are
            not of equal length. len(x) == %d while len(y) == %d"""
            % (len(x), len(y))
        )
    
    def n(j: int) -> np.ndarray:
        return reduce(
            np.convolve, 
            ([1, -x[i]] for i in range(0,j)), 
            np.array([1])
        )

    def a(j: int) -> np.ndarray:
        return divided_difference(y[0:j+1], x[0:j+1])

    coeffs = reduce(
        np.polynomial.polynomial.polyadd,
        ((a(j) * n(j))[::-1] for j in range(0, len(x)))
    )

    return coeffs[::-1]

def lagrange_evaluate(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    For an array of input coefficients and an array of values at which
    to evaluate the interpolant, this method returns the value of the
    Lagrange interpolant formulated from the coefficients

    :param numpy.ndarray a: The array of coefficients for the 
        interpolating polynomial
    :param numpy.ndarray x: The array of x values at which the
        polynomial is to be evaluated
    :return: An array of values cotnaing each value at which the 
        interpolating polynomial was evaluated
    :rtype: :class:`numpy.ndarray`
    """
    rev_a = a[::-1]
    return sum((rev_a[index] * x ** index for index in range(0, len(rev_a))))

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
        y^{(k)}(x) = \sum\limits_{n = k}^N \frac{n!}{(n - k)!} a_n x^{n - k}

    To compute the nth order derivative of the Lagrangian interpolant. This
    formula was developed as a generalization of the first-order derivative
    for an arbitrary polynomial.

    :param numpy.ndarray a: The array of polynomial coefficients
        that define an interpolating polynomial.
    :param int n: The order of the derivative of the interpolating
        polynomial is to be calculated.
    :param numpy.ndarray x: The values for which the derivative is
        to be evaluated
    """
    reversed_a = a[::-1]
    return np.array(sum((
        _permutations(index, n) * reversed_a[index] * x ** (index - n) 
        for index in range(n, len(a))
    )))

def lagrange_integrate(a: np.ndarray, x0: float, x1: float) -> float:
    r"""
    Calculate the first-order definite integral of the interpolant from
    `x0` to `x1`. This function calculates

    .. math::
        \int_{x_0}^{x_1} \sum\limits_{n = 0}^{N} a_n x^n \, \mathrm{d}x

    using the formula

    .. math::
        \int_{x_0}^{x_1} \sum\limits_{n = 0}^{N} a_n x^n \, \mathrm{d}x = 
            \sum\limits_{n = 0}^N \frac{a_n}{n + 1} x^{n + 1}

    :param numpy.ndarray a: The interpolant coefficients
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

