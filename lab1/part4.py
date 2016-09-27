import csv
from collections import namedtuple
from itertools import compress
import numpy as np
from lab1.interpolators import lagrange_differentiate
from lab1.interpolators import lagrange_interpolant
from lab1.interpolators import lagrange_integrate

DataPoint = namedtuple('DSCPoint', ['temperature', 'heat_flux'])
Derivative = namedtuple('Derivative', ['temperature', 'change_in_flux'])

class DSCCurve(object):
    def __init__(self, file_path):
        with open(file_path, mode='r') as csv_file:
            if csv.Sniffer().has_header(csv_file.readline()):
                next(csv_file)
            reader = csv.reader(csv_file, delimiter=',')
            self.data = [DataPoint(
                temperature=float(row[0]), heat_flux=float(row[1])
            ) for row in reader]

    @property
    def crystallization_temperature(self):
        return max(self.data, key=(lambda point: point.heat_flux))

    @property
    def melting_temperature(self):
        return min(self.data, key=(lambda point: point.heat_flux))

    def __getitem__(self, index):
        return self.data[index]

    def derivative(self, order, number_of_points=4):
        """
        Return the first derivative value for a point using
        the given order for the interpolating polynomial
        """
        if order > number_of_points:
            raise ValueError("""
                The order of the derivative is greater than the number
                of points. Cannot unambiguously determine the derivative
            """)

        derivative = []
        for index in range(
            number_of_points - 1, len(self.data) - number_of_points
        ):
            points = [self.data[picker]
                for picker in range(index - number_of_points, index + number_of_points)
            ]

            coeffs = lagrange_interpolant(
                np.array([point.temperature for point in points]),
                np.array([point.heat_flux for point in points])
            )

            change_in_flux = lagrange_differentiate(
                coeffs, order, self.data[index].temperature
            )

            result = Derivative(temperature=self.data[index].temperature,
                    change_in_flux=change_in_flux
            )

            derivative.append(result)


        return derivative

    def glass_transition_temperature(self, lower_limit, upper_limit):
        derivative = self.derivative(order=1)
        points = (
            point for point in derivative 
            if lower_limit < point.temperature < upper_limit
        )
        return max(points, key=(lambda point: abs(point.change_in_flux)))

    @property
    def phase_changes(self):
        derivative = self.derivative(order=2)
        signs = np.sign([point.change_in_flux for point in derivative])
        sign_change = ((np.roll(signs, 1) - signs != 0)).astype(bool)

        return [point for point in compress(derivative, sign_change)]

    def integral(self, lower_limit, upper_limit):
        points_from_curve = [
            point for point in self.data
            if lower_limit < point.temperature < upper_limit
        ]

        x = [point.temperature for point in points_from_curve]
        y = [point.heat_flux for point in points_from_curve]

        coeffs = lagrange_interpolant(x, y)

        return lagrange_integrate(coeffs, lower_limit, upper_limit)
    
    def interpolant(self, lower_limit, upper_limit): 
        points_from_curve = [
            point for point in self.data
            if lower_limit <= point.temperature <= upper_limit
        ]

        x = np.array([point.temperature for point in points_from_curve])
        y = np.array([point.heat_flux for point in points_from_curve])

        coeffs = lagrange_interpolant(x, y)

        return coeffs

