#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=R0912
# R0912: Too many branches (17/12) (too-many-branches)
"""
Module to calculate knn
imports: numpy
"""

import numpy as np


def knn_regression(n_neighbors, data, query):
    """
    Function to calculate knn algorithm
    knn_regression arguments:
    n_neighbors-- this is a first param
    data-- 2 dimensional numpy array of shape (m, n+1)). m denotes the number of samples and
        n is the number of variables in each sample. +1 is for the labels in each sample - the last
        column in the sample. m must be at least as large as n_neighbors. n must be at least 1 (so
        the 2-dimensional array must be at least (m, 2)). All samples must have the same value of n.
        All samples and labels must be numeric.
    query-- 1 dimensional numpy array, shape (n,). n must be the same as in the data argument.
    return: mean of n_neighbors lables
    raise ValueError: raises an exception when requirements arent satisfied
        (requirements given in arguments)
    """

    try:

        # Checking the validiy of parameters 
        if not isinstance(n_neighbors, int):
            raise ValueError("n_neighbors is not an integer.")
        if n_neighbors <= 0:
            raise ValueError("n_neighbors should be greater than 0.")
        if not isinstance(data, np.ndarray):
            raise ValueError("data should be a numpy array.")
        if not isinstance(query, np.ndarray):
            raise ValueError("query should be a numpy array.")
        (rows, cols) = data.shape
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("all samples and labels should be numeric.")
        if not np.issubdtype(query.dtype, np.number):
            raise ValueError("query should be numeric.")
        if query.ndim != 1:
            raise ValueError("query should be a 1-dimensional numpy array.")
        if query.shape[0] != cols - 1:
            raise ValueError(
                "number of variables in query should be the same as in data."
            )

        if data.ndim != 2:
            raise ValueError("data should be a 2-dimensional numpy array.")

        if rows < n_neighbors:
            raise ValueError(
                "number of samples in data should be at least as large as n_neighbors."
            )
        if cols < 2:
            raise ValueError("n+1 should be at least 2.(n should be at least 1)")
        if not all(cols == len(row) for row in data):
            raise ValueError(
                "all sarowsples in data should have the same number of variables"
            )

        var = True
        for single_obj in data[:, :-1]:
            if single_obj.shape[0] != cols - 1:
                var = False
        if not var:
            raise ValueError("all samples must have the same value of n")

        # KNN

        def calculate_distance(x_cordinate, y_cordinate):
            return np.sqrt(np.sum((x_cordinate - y_cordinate) ** 2))

        distances = []
        for data_row in data:
            dist = calculate_distance(query, data_row[:-1])
            distances.append(dist)
        distances = np.array(distances)
        distances = np.argsort(distances)
        nearest_neighbors = distances[:n_neighbors]
        knn_values = [data[i][-1] for i in nearest_neighbors]
        knn_response = np.mean(knn_values)
        return round(knn_response, 2)
    except ValueError as error:
        raise error
