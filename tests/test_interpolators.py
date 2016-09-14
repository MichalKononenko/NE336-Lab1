import numpy as np
import unittest
from lab1 import interpolators

class TestVandermondeMatrix(unittest.TestCase):
    def setUp(self):
        self.x = [2, 3, 4]

        self.expected_result = np.array([
            [1, 2, 4],
            [1, 3, 9],
            [1, 4, 16]
        ])

    def test_vandermonde(self):
        np.testing.assert_array_almost_equal(
            self.expected_result,
            interpolators._vandermonde_matrix(self.x)
        )

class TestLagrangeInterpolant(unittest.TestCase):
    def test_given_test_case(self):
        x = np.array([0., .5, 1.])
        y = np.array([1., 0., 1.])

        expected_result = np.array([4., -4., 1.])

        np.testing.assert_array_almost_equal(
            expected_result,
            interpolators.lagrange_interpolant(x, y)
        )

class TestLagrangeEvaluate(unittest.TestCase):
    def setUp(self):
        self.coefficients = np.array([5, 3, 1, 3, 2])
        self.x_value = np.array([2])

    def test_lagrange_evaluate_given_case(self):
        a = np.array([4., -4., 1.])
        x = np.array([0., .5, 1.])

        expected_result = np.array([1., 0., 1.])

        np.testing.assert_array_almost_equal(
            expected_result,
            interpolators.lagrange_evaluate(a, x)
        )

    def test_lagrange_evaluate(self):
        expected_result = np.array([116])
        np.testing.assert_array_almost_equal(
            expected_result,
            interpolators.lagrange_evaluate(
                self.coefficients, self.x_value)
        )

class TestLagrangeDifferentiate(unittest.TestCase):
    
    def test_first_given_case(self):
        a = np.array([4., -4., 1.])
        n = 1
        x = np.array([0., .5, 1.])

        expected_result = np.array([-4., 0., 4.])

        np.testing.assert_array_almost_equal(
            expected_result,
            interpolators.lagrange_differentiate(
                a, n, x
            )
        )

    def test_second_given_case(self):
        a = np.array([4., -4., 1.])
        n = 1
        x = np.array([1./3., 2./3.])

        expected_result = np.array([-1.33333333, 1.333333333])

        np.testing.assert_array_almost_equal(
            expected_result,
            interpolators.lagrange_differentiate(
                a, n, x
            )
        )

class TestPermutations(unittest.TestCase):

    def test_permutations_zero(self):
        self.assertEqual(1, interpolators._permutations(3, 0))

    def test_permutation_all_factors(self):
        self.assertEqual(6, interpolators._permutations(3, 3))

    def test_permutation_k_bigger_than_n(self):
        with self.assertRaises(ValueError):
            interpolators._permutations(3, 5)

class TestLagrangeIntegrate(unittest.TestCase):
    def setUp(self):
        self.x0 = 0.
        self.x1 = 1.
        self.a = [4., -4., 1.]

    def test_integrate(self):
        expected_result = 0.33333333

        self.assertAlmostEqual(
            expected_result,
            interpolators.lagrange_integrate(self.a, self.x0, self.x1)
        )

