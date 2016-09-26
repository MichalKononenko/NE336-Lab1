from lab1 import part4
import unittest
import os

DIRECTORY = os.path.abspath(os.path.curdir)

DSC_CURVE_PATH = os.path.join(DIRECTORY, 'lab1_data.csv') 

class TestDSCCurve(unittest.TestCase):
    def setUp(self):
        self.curve = part4.DSCCurve(DSC_CURVE_PATH)

    def test_crystallization_temperature(self):
        self.assertIsNotNone(self.curve.crystallization_temperature)

    def test_melting_temperature(self):
        self.assertIsNotNone(self.curve.melting_temperature)

    def test_derivative(self):
        self.assertIsNotNone(self.curve.derivative(order=1))

    def test_glass_transition_temperature(self):
        self.assertIsNotNone(self.curve.glass_transition_temperature(40, 100)[0])

    def test_phase_changes(self):
        self.assertIsNotNone(self.curve.phase_changes[0])
