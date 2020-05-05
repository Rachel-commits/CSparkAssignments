import unittest
from applied_numpy import build_sequences, moving_averages, block_matrix
from pandas_agg import get_prices_for_heaviest_item
from intro_pca import distance_in_n_dimensions, find_outliers_pca
from more_pandas import return_post_codes, return_location
from postcodes import new_pc

import numpy as np
import pandas as pd


class TestBuildSeq(unittest.TestCase):

    def test_seq1_example(self):

        min_value = 3
        max_value = 17
        sequence_number = 1

        result = build_sequences(min_value, max_value, sequence_number)
        expected = np.array([3, 5, 7, 9, 11, 13, 15, 17])

        self.assertEqual(expected.tolist(), result.tolist())

    def test_seq2_example(self):

        min_value = 28
        max_value = 35
        sequence_number = 2

        result = build_sequences(min_value, max_value, sequence_number)
        expected = np.array([35, 30])

        self.assertEqual(expected.tolist(), result.tolist())

    def test_seq3_example(self):

        min_value = 9
        max_value = 24
        sequence_number = 3

        result = build_sequences(min_value, max_value, sequence_number)
        expected = np.array([16])

        self.assertEqual(expected.tolist(), result.tolist())

    def test_seq3_example2(self):

        min_value = 0
        max_value = 100
        sequence_number = 3

        result = build_sequences(min_value, max_value, sequence_number)
        expected = np.array([1, 2, 4, 8, 16, 32, 64])

        self.assertEqual(expected.tolist(), result.tolist())


class TestMovingAverage(unittest.TestCase):

    def test_ma_ex1(self):

        x = np.array([1, 2, 3, 4, 5, 6])
        k = 5

        result = moving_averages(x, k)
        expected = np.array([3, 4])

        self.assertEqual(expected.tolist(), result.tolist())

    def test_ma_ex2(self):

        x = np.array([1, 2, 3, 4])
        k = 3

        result = moving_averages(x, k)
        expected = np.array([2, 3])

        self.assertEqual(expected.tolist(), result.tolist())

    def test_ma_ex3(self):

        x = np.array([1, 0, 0, 4, 5])
        k = 2

        result = moving_averages(x, k)
        expected = np.array([0.5, 0, 2, 4.5])

        self.assertEqual(expected.tolist(), result.tolist())


class TestBlockMatrix(unittest.TestCase):

    def test_block_mat_ex1(self):

        A = np.array([[1]])
        B = np.array([[9]])

        result = block_matrix(A, B)
        expected = np.array([[1, 0],
                             [0, 9]])

        self.assertEqual(expected.tolist(), result.tolist())

    def test_block_mat_ex2(self):

        A = np.array([[1, 4, 5],
                      [2, 2, 2],
                      [4, 5, 6]])

        B = np.array([[7, 7, 7],
                      [2, 2, 2],
                      [4, 4, 9]])

        result = block_matrix(A, B)
        expected = np.array([[1, 4, 5, 0, 0, 0],
                             [2, 2, 2, 0, 0, 0],
                             [4, 5, 6, 0, 0, 0],
                             [0, 0, 0, 7, 7, 7],
                             [0, 0, 0, 2, 2, 2],
                             [0, 0, 0, 4, 4, 9]])

        self.assertEqual(expected.tolist(), result.tolist())

    def test_block_mat_ex3(self):

        A = np.array([[1, 2, 5, 5],
                      [3, 7, 7, 8],
                      [6, 9, 1, 8]])

        B = np.array([[5, 3],
                      [7, 8],
                      [2, 5]])

        result = block_matrix(A, B)
        expected = np.array([[1, 2, 5, 5, 0, 0],
                             [3, 7, 7, 8, 0, 0],
                             [6, 9, 1, 8, 0, 0],
                             [0, 0, 0, 0, 5, 3],
                             [0, 0, 0, 0, 7, 8],
                             [0, 0, 0, 0, 2, 5]])

        self.assertEqual(expected.tolist(), result.tolist())


class TestGetPrices(unittest.TestCase):

    def test_get_prices_example(self):

        data = [['electronics', 400,   740,     False],
                ['health',        5,    100,    False],
                ['electronics', 300,    6000,    True],
                ['books',        20,    300,     True]]

        col_names = ['category', 'price', 'weight', 'in_stock']
        inventory = pd.DataFrame(data, columns=col_names)

        result = get_prices_for_heaviest_item(inventory)
        expected = pd.Series({'electronics': 300,
                              'books': 20})

        self.assertEqual(expected.tolist(), result.tolist())

    def test_get_prices_example2(self):

        data = [['electronics', 400,   740,     False],
                ['health',        5,    100,    True],
                ['electronics', 300,    6000,    True],
                ['health',      60,    600,      False],
                ['electronics', 500,    9000,    True],
                ['books',        20,    300,     True]]

        col_names = ['category', 'price', 'weight', 'in_stock']
        inventory = pd.DataFrame(data, columns=col_names)
        result = get_prices_for_heaviest_item(inventory)
        expected = pd.Series({'electronics': 500,
                              'books': 20,
                              'health': 5})

        self.assertEqual(expected.tolist(), result.tolist())

    def test_get_prices_missing_cats(self):

        data = [['electronics', 400,   740,     False],
                ['health',        5,    100,    False],
                ['electronics', 300,    6000,    False],
                ['health',      60,    600,      False],
                ['electronics', 500,    9000,    False],
                ['books',        20,    300,     False]]

        col_names = ['category', 'price', 'weight', 'in_stock']
        inventory = pd.DataFrame(data, columns=col_names)

        result = get_prices_for_heaviest_item(inventory)
        expected = pd.Series([])

        self.assertEqual(expected.tolist(), result.tolist())


class TestPostcodes(unittest.TestCase):

    def test_postcode_example(self):

        data = [['Great Doddington, Wellingborough NN29 7TA, UK\nTaylor, Leeds LS14 6JA, UK'],
                ['This is some text, and here is a postcode CB4    9NE']]

        col_names = ['text']
        df = pd.DataFrame(data, columns=col_names)

        result = return_post_codes(df)
        expected = pd.DataFrame([['NN29 7TA | LS14 6JA'],
                                 ['CB4    9NE']], columns=['postcodes'])

        from pandas.testing import assert_frame_equal
        assert_frame_equal(expected, result)

    def test_postcode_example_l(self):

        data = [['Great Doddington, Wellingborough NN29 7TA, UK\nTaylor, Leeds LS14 6JA, UK'],
                ['This is some text, and here is a postcode CB4    9NE']]

        col_names = ['text']
        df = pd.DataFrame(data, columns=col_names)


        result = new_pc(df)
        expected = pd.DataFrame([['NN29 7TA | LS14 6JA'],
                                 ['CB4    9NE']], columns=['postcodes'])

        from pandas.testing import assert_frame_equal
        assert_frame_equal(expected, result)


class TestLocation(unittest.TestCase):

    def test_location_example(self):

        data = [['{"short_name": "Detroit, MI", "id": 2391585}'],
                [' {"short_name": "Tracy, CA", "id": 2507550}']]

        col_names = ['locations']
        df = pd.DataFrame(data, columns=col_names)

        result = return_location(df)
        expected = pd.DataFrame([['Detroit, MI'],
                                 ['Tracy, CA']], columns=['short_name'])

        from pandas.testing import assert_frame_equal
        assert_frame_equal(expected, result)


class TestDirectionDistance(unittest.TestCase):

    def test_direction_example(self):

        df = pd.DataFrame({'alcohol': {19: 13.64, 45: 14.21, 140: 12.93, 30: 13.73, 67: 12.37, 16: 14.3, 119: 12.0, 174: 13.4, 109: 11.61, 141: 13.36},
                           'malic_acid': {19: 3.1, 45: 4.04, 140: 2.81, 30: 1.5, 67: 1.17, 16: 1.92, 119: 3.43, 174: 3.91, 109: 1.35, 141: 2.56},
                           'ash': {19: 2.56, 45: 2.44, 140: 2.7, 30: 2.7, 67: 1.92, 16: 2.72, 119: 2.0, 174: 2.48, 109: 2.7, 141: 2.35},
                           'alcalinity_of_ash': {19: 15.2, 45: 18.9, 140: 21.0, 30: 22.5, 67: 19.6, 16: 20.0, 119: 19.0, 174: 23.0, 109: 20.0, 141: 20.0},
                           'magnesium': {19: 116.0, 45: 111.0, 140: 96.0, 30: 101.0, 67: 78.0, 16: 120.0, 119: 87.0, 174: 102.0, 109: 94.0, 141: 89.0}})

        point_a = np.array([12.29, 1.61, 2.21, 20.4, 103.0])
        point_b = np.array([14.22, 1.7, 2.3, 16.3, 118.0])
        n = 2
        scale = True

        result = distance_in_n_dimensions(df, point_a, point_b, n, scale)
        expected = 2.6747550276573233
        self.assertAlmostEqual(result, expected)


class TestOutliers(unittest.TestCase):

    def test_outliers_example(self):

        df = pd.DataFrame({'alcohol': {157: 12.45, 168: 13.58, 55: 13.56, 37: 13.05, 70: 12.29, 43: 13.24, 44: 13.05, 81: 12.72},
                           'malic_acid': {157: 3.03, 168: 2.58, 55: 1.73, 37: 1.65, 70: 1.61, 43: 3.98, 44: 1.77, 81: 1.81}, 
                           'ash': {157: 2.64, 168: 2.69, 55: 2.46, 37: 2.55, 70: 2.21, 43: 2.29, 44: 2.1, 81: 2.2}, 
                           'alcalinity_of_ash': {157: 27.0, 168: 24.5, 55: 20.5, 37: 18.0, 70: 20.4, 43: 17.5, 44: 17.0, 81: 18.8}, 
                           'magnesium': {157: 97.0, 168: 105.0, 55: 116.0, 37: 98.0, 70: 103.0, 43: 103.0, 44: 107.0, 81: 86.0}})

        n = 1
        scale = True

        result = find_outliers_pca(df, n, scale)
        expected = pd.DataFrame({'alcohol': {157: 12.45, 168: 13.58, 44: 13.05, 81: 12.72},
                                 'malic_acid': {157: 3.03, 168: 2.58, 44: 1.77, 81: 1.81}, 
                                 'ash': {157: 2.64, 168: 2.69, 44: 2.1, 81: 2.2}, 
                                 'alcalinity_of_ash': {157: 27.0, 168: 24.5, 44: 17.0, 81: 18.8}, 
                                 'magnesium': {157: 97.0, 168: 105.0, 44: 107.0, 81: 86.0}})

        from pandas.testing import assert_frame_equal
        assert_frame_equal(expected, result)


        