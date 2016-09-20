import csv
from collections import namedtuple
import numpy as np
from lab1.interpolators import lagrange_differentiate
from lab1.interpolators import lagrange_interpolant

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

    def first_derivative(self, number_of_points=4):
        """
        Return the first derivative value for a point using
        the given order for the interpolating polynomial
        """
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
                coeffs, 1, self.data[index].temperature
            )

            result = Derivative(temperature=self.data[index].temperature,
                    change_in_flux=change_in_flux
            )

            derivative.append(result)


        return derivative

    @property
    def glass_transition_temperature(self):
        return max(self.first_derivative(), key=(lambda point: abs(point.change_in_flux)))

