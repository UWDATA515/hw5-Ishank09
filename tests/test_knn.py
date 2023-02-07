#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable= W1503, E0401
#E0401: Unable to import 'knn.knn' (import-error),
#W1503: Redundant use of assertTrue with constant value True (redundant-unittest-assert)
"""
testing module for knn.py
"""
import unittest
import numpy as np
from knn.knn import knn_regression


class KnnRegressionTests(unittest.TestCase):
    """
    testing module for knn.py containing 15 tests.
    """

    # ************************************************************************
    # Module 1: smoke test - 1=2
    # the function should return something and the data type for return should be float or int
    # In step 2, nothing is returned thus it should fail

    def test_smoke_test_check_result_exist(self):
        """
        smoke test to check the result validity and bounding it to float and int return type
        arguments: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        n_neighbors = 3
        data = np.array(
            [[3, 1, 230], [6, 2, 745], [6, 6, 1080], [4, 3, 495], [2, 5, 260]]
        )

        query = np.array([5, 4])

        result = knn_regression(n_neighbors, data, query)
        self.assertIsInstance(result, (float, int))

    # program is running seamlessly

    def test_smoke_test(self):
        """
        smoke test to check the result validity
        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        n_neighbors = 3
        data = np.array(
            [[3, 1, 230], [6, 2, 745], [6, 6, 1080], [4, 3, 495], [2, 5, 260]]
        )
        query = np.array([5, 4])
        knn_regression(n_neighbors, data, query)
        self.assertTrue(True)

    # ************************************************************************
    # Module 2: one shot test - 2

    def test_one_shot_test_check_result_correct_1(self):
        """
        one shot test to check if the program is returning correct data
        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        # Test with only one sample in the data

        n_neighbors = 3
        data = np.array(
            [[3, 1, 230], [6, 2, 745], [6, 6, 1080], [4, 3, 495], [2, 5, 260]]
        )
        query = np.array([5, 4])
        result = knn_regression(n_neighbors, data, query)
        self.assertEqual(773.33, result)

    def test_one_shot_test_check_result_correct_2(self):
        """
        one shot test to check if the program is returning correct data
        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        n_neighbors = 1
        data = np.array([[3, 1, 230], [6, 2, 745]])
        query = np.array([5, 4])
        result = knn_regression(n_neighbors, data, query)
        self.assertEqual(result, 745)

    def test_one_shot_test_check_result_correct_3(self):
        """
        one shot test to check if the program is returning correct data
        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        n_neighbors = 5
        data = np.array(
            [
                [3, 4, 230],
                [6, 2, 745],
                [6, 6, 1080],
                [4, 3, 495],
                [2, 5, 260],
                [1, 7, 180],
            ]
        )
        query = np.array([5, 2])

        result = knn_regression(n_neighbors, data, query)
        self.assertEqual(result, 562)

    def test_one_shot_test_check_result_correct_4(self):
        """
        one shot test to check if the program is returning correct data
        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        n_neighbors = 4
        data = np.array(
            [[3, 1, 230], [6, 2, 745], [6, 6, 1080], [4, 3, 495], [2, 5, 260]]
        )
        query = np.array([5, 8])
        result = knn_regression(n_neighbors, data, query)
        self.assertEqual(result, 645)

    # ************************************************************************
    # Module 3: edge test - 10

    # Edge test 1

    def test_edge_test_check_n_neighbors_datatype(self):
        """
        edge test to test the function when n_neighbors parameter is not int
        (the defined/required data type)

        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        # test: n_neighbors data type

        data = np.array(
            [[3, 1, 230], [6, 2, 745], [6, 6, 1080], [4, 3, 495], [2, 5, 260]]
        )

        query = np.array([5, 4])
        with self.assertRaises(ValueError):
            knn_regression("String", data, query)

    # Edge test 2

    def test_edge_test_check_query_datatype(self):
        """
        edge test to test the function when query parameter is not np array
        (the defined/required data type)

        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        # test: query data type

        n_neighbors = 3
        data = np.array(
            [[3, 1, 230], [6, 2, 745], [6, 6, 1080], [4, 3, 495], [2, 5, 260]]
        )
        with self.assertRaises(ValueError):
            knn_regression(n_neighbors, data, "String")

    # Edge test 3
    def test_edge_test_check_data_datatype_1(self):
        """
        edge test to test the function when data parameter is not np array
        (the defined/required data type)

        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        # test: data data type

        n_neighbors = 3
        query = np.array([5, 4])
        with self.assertRaises(ValueError):
            knn_regression(n_neighbors, "String", query)

    # Edge test 4

    def test_edge_test_check_data_datatype_2(self):
        """
        edge test to test the function when data parameter is not np array
        (the defined/required data type)

        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        # test: data dimention

        n_neighbors = 3
        data = [[3, 1], [6, 2, 7], [6, 6, 1080], [4, 3, 495], [2, 5, 260]]

        query = np.array([5, 4])
        with self.assertRaises(ValueError):
            knn_regression(n_neighbors, data, query)

    # Edge test 5

    def test_edge_test_check_data_dimention_1(self):
        """
        edge test to test the function when data parameter have incorrect dimention
        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        # test: data dimention

        n_neighbors = 3
        data = np.array([[3, 1], [6, 2], [6, 6], [4, 3], [2, 5]])

        query = np.array([5, 4])
        with self.assertRaises(ValueError):
            knn_regression(n_neighbors, data, query)

    # Edge test 6

    def test_edge_test_check_data_dimention_3(self):
        """
        edge test to test the function when data parameter have incorrect dimention
        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        # test: data dimention

        n_neighbors = 3
        data = np.array([[3, 1, 230]])

        query = np.array([5, 4])

        with self.assertRaises(ValueError):
            knn_regression(n_neighbors, data, query)

    # Edge test 7

    def test_edge_test_check_n_neighbors_value(self):
        """
        edge test to test the function when n_neighbors parameter have incorrect value
        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        # test: invalid value of n_n

        n_neighbors = 0
        data = np.array([[3, 1, 230], [6, 2, 745]])

        query = np.array([5, 4])

        with self.assertRaises(ValueError):
            knn_regression(n_neighbors, data, query)

    # Edge test 8

    def test_edge_test_check_query_dimention_1(self):
        """
        edge test to test the function when query parameter have incorrect dimention
        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        # test: query dimention

        n_neighbors = 3
        data = np.array([[3, 1, 230], [6, 2, 745]])

        query = np.array([5])

        with self.assertRaises(ValueError):
            knn_regression(n_neighbors, data, query)

    # Edge test 9

    def test_edge_test_check_query_data_empty(self):
        """
        edge test to test the function when data parameter is empty
        parameter: self
        return: nothing
        assert/Exception: error/exception when the test fails
        """

        with self.assertRaises(ValueError):
            knn_regression(1, np.array([]), np.array([]))


if __name__ == "__main__":

    unittest.main()
