import numpy as np
from typing import Callable, Union
from collections import namedtuple
from lab1.interpolators import lagrange_interpolant
from lab1.interpolators import lagrange_evaluate

array_or_number = Union[int, float, np.ndarray]

def diff_1st_order_finite_difference(
    f: Callable[[array_or_number], array_or_number], 
    center: array_or_number, 
    spread: array_or_number) -> array_or_number:
    """
    Calculate the 1st order derivative using finite differences
    """
    return (f(center + spread) - f(center - spread))/(2 * spread)

def diff_2nd_order_finite_difference(
    f: Callable[[array_or_number], array_or_number],
    center: array_or_number,
    spread: array_or_number) -> array_or_number:
    """
    Calculate the 2nd order derivative using first-order finite
    difference
    """
    return (
        (f(center + spread) - 2 * f(center) + f(center - spread))\
        /\
        ((spread)**2)
    )

def diff_1st_order_four_points(
    f: Callable[[array_or_number], array_or_number],
    center: array_or_number,
    spread: array_or_number) -> array_or_number:
    """
    """
    f1 = f(center + 2 * spread)
    f2 = f(center + spread)
    f3 = f(center - spread)
    f4 = f(center - 2 * spread)

    return (- f1 + 8 * f2 - 8 * f3 + f4)/(12 * spread)

def diff_2nd_order_four_points(
    f: Callable[[array_or_number], array_or_number],
    center: array_or_number,
    spread: array_or_number) -> array_or_number:
    """
    """
    f1 = f(center + 2 * spread)
    f2 = f(center + spread)
    f3 = f(center)
    f4 = f(center - spread)
    f5 = f(center - 2 * spread)

    return (-f1 + 16 * f2 - 30 * f3 + 16 * f4 - f5)/(12 * (spread ** 2))

def interpolation_error():
    points = (5 * value for value in range(1, 10))

    Result = namedtuple('Result', ['n', 'error'])

    results = []

    for point in points:
        x = np.linspace(0, 2*np.pi, point)
        y_n = np.sin(x)

        coeffs = lagrange_interpolant(x, y_n)

        y = lagrange_evaluate(coeffs, x)

        error = y_n - y

        results.append(Result(n=point, error=error))

    return results

