"""
Contains unit tests for :mod:`interpolators`
"""
import numpy as np
import unittest
from lab1 import interpolators

class TestLagrangeInterpolant(unittest.TestCase):
    """
    Contains unit tests for :meth:`interpolators.lagrange_interpolant`
    """
    def test_unequal_x_and_y_array(self):
        """
        Tests that :exc:`interpolators.ArraysNotEqualError` is thrown
        if the input arrays `x` and `y` are not of equal length
        """
        x = np.array([0., 0.5])
        y = np.array([1., 0., 1.])

        assert len(x) != len(y)

        with self.assertRaises(interpolators.ArraysNotEqualError):
            interpolators.lagrange_interpolant(x, y)

    def test_given_test_case(self):
        """
        Tests that the values given in the lab description yield the
        expected value given in the lab.
        """
        x = np.array([0., .5, 1.])
        y = np.array([1., 0., 1.])

        expected_result = np.array([4., -4., 1.])

        np.testing.assert_array_almost_equal(
            expected_result,
            interpolators.lagrange_interpolant(x, y)
        )

    def test_sine_interpolant(self):
        """
        A bug was encountered while interpolating 
        a function for sine. This bug must be fixed
        """
        x = np.linspace(0, 2*np.pi, 5)
        y = np.sin(x)

        coeffs = interpolators.lagrange_interpolant(x, y)

        assert coeffs.any()

        result = interpolators.lagrange_evaluate(coeffs, x)

        np.testing.assert_array_almost_equal(y, result)

class TestDividedDifferences(unittest.TestCase):
    def test_single_case(self):
        x = np.array([0])
        y = np.array([1])

        np.testing.assert_array_almost_equal(
            y, interpolators.divided_difference(y, x)
        )

    def test_double_case(self):
        x = np.array([0, 1])
        y = np.array([1, 2])

        expected_result = (y[1] - y[0])/(x[1] - x[0])

        np.testing.assert_array_almost_equal(
            expected_result, interpolators.divided_difference(y, x)
        )

    def test_recursive_case(self):
        x = np.array([0, 1, 2])
        y = np.array([1, 2, 3])

        expected_result = np.array([0])
        np.testing.assert_array_almost_equal(
            expected_result,
            interpolators.divided_difference(y, x)
        )

    def test_empty_list(self):
        x = []
        y = []

        expected_result = np.array([0])

        np.testing.assert_array_almost_equal(
            expected_result,
            interpolators.divided_difference(y, x)
        )

class TestNewtonInterpolant(unittest.TestCase):
    """
    Contains unit tests for :meth:`interpolators.newton_interpolant`
    """
    def test_unequal_x_and_y_array(self):
        x = np.array([0., 0.5])
        y = np.array([1., 0., 1.])
        assert len(x) != len(y)

        with self.assertRaises(interpolators.ArraysNotEqualError):
            interpolators.newton_interpolant(x, y)

    def test_given_test_case(self):
        x = np.array([0, .5, 1.])
        y = np.array([1., 0., 1.])

        expected_result = np.array([4., -4., 1.])

        np.testing.assert_array_almost_equal(
            expected_result,
            interpolators.newton_interpolant(x, y)
        )

class TestLagrangeEvaluate(unittest.TestCase):
    """
    Contains unit tests for :meth:`interpolators.lagrange_evaluate`
    """
    def setUp(self):
        """
        Set up a series of poylynomial coefficients and x values
        for which to evaluate the interpolant
        """
        self.coefficients = np.array([5, 3, 1, 3, 2])
        self.x_value = np.array([2])

    def test_lagrange_evaluate_given_case(self):
        """
        Tests that the method provides the expected result given in the 
        lab description when the parameters from the lab are passed into
        the function
        """
        a = np.array([4., -4., 1.])
        x = np.array([0., .5, 1.])

        expected_result = np.array([1., 0., 1.])

        np.testing.assert_array_almost_equal(
            expected_result,
            interpolators.lagrange_evaluate(a, x)
        )

    def test_lagrange_evaluate(self):
        """
        Contains a custom test case for the method using the values
        in :meth:`setUp`
        """
        expected_result = np.array([116])
        np.testing.assert_array_almost_equal(
            expected_result,
            interpolators.lagrange_evaluate(
                self.coefficients, self.x_value)
        )

class TestLagrangeDifferentiate(unittest.TestCase):
    """
    Contains unit tests for :meth:`interpolators.lagrange_differentiate`
    """ 
    def test_first_given_case(self):
        """
        Tests that the first given test case returns the correct answer
        """
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
        """
        Tests the second test case from the lab
        """
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

    def test_differentiate_part_3(self):
        x_values = np.array([0, 1, 5])
        y_values = np.array([(lambda x: x*(x-1))(x) for x in x_values])
        
        interpolating_coefficients = interpolators.lagrange_interpolant(    
            x_values, y_values
        )

        center = np.array([0.5])
        order = 1

        np.testing.assert_array_almost_equal(
            np.array([0]), interpolators.lagrange_differentiate(
                interpolating_coefficients, 1, center
            )
        )

class TestPermutations(unittest.TestCase):
    """
    Contains unit tests for :meth:`interpolators._permutations`
    """
    def test_all_values_zero(self):
        """
        Tests that the method returns 1 if both :math:`n = 0`
        and :math:`k = 0`
        """
        self.assertEqual(1, interpolators._permutations(0, 0))

    def test_permutations_zero(self):
        """
        Tests that the method returns 1 if :math:`k = 0` for the
        permutations. No matter what the set, the empty set (with
        cardinality :math:`k = 0`) is always a permutation of
        the set
        """
        self.assertEqual(1, interpolators._permutations(3, 0))

    def test_permutation_all_factors(self):
        """
        Tests that if :math:`k = n` in the permutations, then
        the number of permutations is :math:`n!`
        """
        self.assertEqual(6, interpolators._permutations(3, 3))

    def test_permutation_k_bigger_than_n(self):
        """
        Tests that a :exc:`ValueError` is thrown if the function
        is called with :math:`k > n`
        """
        with self.assertRaises(ValueError):
            interpolators._permutations(3, 5)

class TestLagrangeIntegrate(unittest.TestCase):
    """
    Contains unit tests for :meth:`interpolators.lagrange_integrate`
    """
    def setUp(self):
        """
        Sets up the parameters for the given test case
        """
        self.x0 = 0.
        self.x1 = 1.
        self.a = [4., -4., 1.]

    def test_integrate(self):
        """
        Tests that the expected result matches the required test case
        """
        expected_result = 0.33333333

        self.assertAlmostEqual(
            expected_result,
            interpolators.lagrange_integrate(self.a, self.x0, self.x1)
        )

