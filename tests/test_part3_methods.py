import unittest
import numpy as np
from lab1 import interpolators as interpolators
from lab1 import part3_methods as p3m

class TestLagrangeInterpolation(unittest.TestCase):
    @staticmethod
    def func(x):
        return x*(x - 1)

class TestThreePoints(TestLagrangeInterpolation):
    def setUp(self):
        self.x_values = np.linspace(0, 1, 3)
        self.y_values = self.func(self.x_values)

        self.center = np.array([0.5])
        self.spread = 0.2

        self.interpolating_coefficients = interpolators.lagrange_interpolant(
            self.x_values, self.y_values
        )

    def test_derivative(self):
        derivative_from_interpolant = interpolators.lagrange_differentiate(
            self.interpolating_coefficients, 1, self.center 
        )
        derivative_from_difference = p3m.diff_1st_order_finite_difference(
            self.func, self.center, self.spread
        )

        np.testing.assert_array_almost_equal(
            derivative_from_interpolant, derivative_from_difference
        )

class TestFivePoints(TestLagrangeInterpolation):
    def setUp(self):
        self.x_values = np.linspace(0, 1, 5)
        self.y_values = self.func(self.x_values)
        self.spread = 0.1

        self.interpolating_coefficients = interpolators.lagrange_interpolant(
            self.x_values, self.y_values
        )

    def test_derivative(self):
        derivative_from_difference = p3m.diff_1st_order_four_points(
            self.func, self.x_values, self.spread
        )
        derivative_from_interpolant = interpolators.lagrange_differentiate(
            self.interpolating_coefficients, 1, self.x_values
        )

        np.testing.assert_array_almost_equal(
            derivative_from_interpolant, derivative_from_difference
        )

class TestIterator(TestLagrangeInterpolation):
    def test_interpolator(self):
        result = p3m.interpolation_error()

        assert result

