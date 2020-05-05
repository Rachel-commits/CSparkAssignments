import unittest
from intro_numpy import cauchy
from more_numpy import find_element, filter_matrix, largest_sum
import numpy as np


class TestCauchy(unittest.TestCase):

    def test_cauchy_example(self):

        input_x = np.array([1, 2, 3, 4])
        input_y = np.array([7, 5, 10])
        result = cauchy(input_x, input_y)
        expected = np.array([[-1/6, -1/4, -1/9],
                             [-1/5, -1/3, -1/8],
                             [-1/4, -1/2, -1/7],
                             [-1/3, -1.0, -1/6]])

        self.assertEqual(expected.tolist(), result.tolist())

    def test_cauchy_empty(self):

        input_x = np.array([])
        input_y = np.array([])
        result = cauchy(input_x, input_y)
        expected = np.empty([0, 0])

        self.assertEqual(expected.tolist(), result.tolist())

    def test_cauchy_divisor_zero(self):

        input_x = np.array([1, 6])
        input_y = np.array([6, 5, 10])

        self.assertRaises(ValueError, cauchy, input_x, input_y)


class TestFind(unittest.TestCase):

    def test_example_find_element(self):

        sq_mat = np.array([[10, 7, 5],
                           [9, 4, 2],
                           [5, 2, 1]])
        val = 4
        result = find_element(sq_mat, val)
        expected = {(1, 1)}

        self.assertEqual(expected, result)

    def test_example_find_element_2values(self):

        sq_mat = np.array([[10, 7, 5],
                           [9, 4, 10],
                           [5, 2, 10]])
        val = 10
        result = find_element(sq_mat, val)
        expected = {(0, 0), (1, 2), (2, 2)}

        self.assertEqual(expected, result)

    def test_example_find_element_empty(self):

        sq_mat = np.array([[10, 7, 5],
                           [9, 4, 10],
                           [5, 2, 10]])
        val = 3
        self.assertRaises(ValueError, find_element, sq_mat, val)


class TestFilterMatrix(unittest.TestCase):

    def test_example_filtermatrix(self):

        mat = np.array([[0, 7, 5],
                        [9, 4, 2],
                        [5, 2, 1]])

        result = filter_matrix(mat)
        expected = np.array([[0, 0, 0],
                             [0, 4, 2],
                             [0, 2, 1]])

        self.assertEqual(expected.tolist(), result.tolist())

    def test_example2_filtermatrix(self):

        mat = np.array([[100, 7, 5],
                        [9, 0, 2],
                        [5, 0, 0]])

        result = filter_matrix(mat)
        expected = np.array([[100, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]])

        self.assertEqual(expected.tolist(), result.tolist())

    def test_nozero_filtermatrix(self):

        mat = np.array([[100, 7, 5, 2],
                        [9, 1, 2, 6],
                        [5, 1, 3, 8]])

        result = filter_matrix(mat)
        expected = np.array([[100, 7, 5, 2],
                             [9, 1, 2, 6],
                             [5, 1, 3, 8]])

        self.assertEqual(expected.tolist(), result.tolist())


class TestLargestSum(unittest.TestCase):

    def test_largest_sum_all_pos(self):

        input = [1, 5, 8, 4, 7]
        result = largest_sum(input)
        expected = 25
        self.assertEqual(expected, result)

    def test_largest_sum_mix(self):

        input = [1, 5, -100, 4, 7]
        result = largest_sum(input)
        expected = 11
        self.assertEqual(expected, result)

    def test_largest_sum_mix2(self):

        input = [1, 5, -100, 4, 1]
        result = largest_sum(input)
        expected = 6
        self.assertEqual(expected, result)

    def test_largest_sum_neg(self):

        input = [-1, -3, -4]
        result = largest_sum(input)
        expected = -1
        self.assertEqual(expected, result)

    
    def test_largest_empty(self):

        input = []
        result = largest_sum(input)
        expected = 0
        self.assertEqual(expected, result)
