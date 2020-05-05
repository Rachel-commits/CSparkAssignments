"""
This file contains a set of functions to practice your
linear algebra skills.

It needs to be completed using "vanilla" Python, without
help from any library.
"""


def gradient(w1, w2, x):
    """
    Given the following function f(x) = w1 * x1^2 + w2 * x2
    where x is a valid vector with coordinates [x1, x2]
    evaluate the gradient of the function at the point x

    :param w1: first coefficient
    :param w2: second coefficient
    :param x: a point represented by a valid tuple (x1, x2)
    :return: the two coordinates of gradient of f
    at point x
    :rtype: float, float
    """
    if len(x) != 2:
        raise ValueError
    grad = (2*w1*x[0], w2)

    return grad


def metrics(u, v):
    """
    Given two vectors u and v, compute the following distances/norm between
    the two and return them.
    - l1 Distance (norm)
    - l2 Distance (norm)

    If the two vectors have different dimensions,
    you should raise a ValueError

    :param u: first vector (list)
    :param v: second vector (list)
    :return: l1 distance, l2 distance
    :rtype: float, float
    :raise ValueError:
    """

    dist_l1 = 0
    dist_l2_temp = 0

    if len(u) != len(v):
        raise ValueError

    for i, val in enumerate(u):
        i_dist = v[i]-val
        dist_l1 += abs(i_dist)
        dist_l2_temp += (i_dist**2)
    dist_l2 = dist_l2_temp**(1/2)

    return dist_l1, dist_l2


def list_mul(u, v):
    """
    Given two vectors, calculate and return the following quantities:
    - element-wise sum
    - element-wise product
    - dot product

    If the two vectors have different dimensions,
    you should raise a ValueError

    :param u: first vector (list)
    :param v: second vector (list)
    :return: the three quantities above
    :rtype: list, list, float
    :raise ValueError:
    """

    el_sum = []
    el_prod = []
    dot_prod = 0

    if len(u) != len(v):
        raise ValueError

    for i, val in enumerate(u):
        el_sum.append(v[i]+val)
        el_prod.append(v[i]*val)
        dot_prod += v[i]*val

    return el_sum, el_prod, dot_prod


def matrix_mul(A, B):
    """
    Given two valid matrices A and B represented as a list of lists,
    implement a function to multiply them together (A * B). Your solution
    can either be a pure mathematical one or a more pythonic one where you
    make use of list comprehensions.

    For example:
    A = [[1, 2, 3],
         [4, 5, 6]]
    is a matrix with two rows and three columns.

    If the two matrices have incompatible dimensions or are not valid meaning that
    not all rows in the matrices have the same length you should raise a ValueError.

    :param A: first matrix (list of lists)
    :param B: second matrix (list of lists)
    :return: resulting matrix (list of lists)
    :rtype: list of lists
    :raise ValueError:
    """

    a_rows = len(A)
    b_rows = len(B)
    a_cols = len(A[0])   
    b_cols = len(B[0])

    # check that all rows in the matrix are of the same length
    if a_cols != b_rows:
        raise ValueError

    for i in range(a_rows):
        if len(A[i]) != a_cols:
            raise ValueError
    for j in range(b_rows):
        if len(B[j]) != b_cols:
            raise ValueError

        # New matrix will be of dimension A+rowsxBCols
    c_matrix = [[0 for x in range(b_cols)] for y in range(a_rows)]
    for i in range(a_rows):
        for j in range(b_cols):
            for k in range(b_rows):
                c_matrix[i][j] += A[i][k]*B[k][j]

    return c_matrix



