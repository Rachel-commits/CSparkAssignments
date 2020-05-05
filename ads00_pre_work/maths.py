"""
This file contains a set of functions to practice your
maths skills.

It needs to be completed with "vanilla" Python, without
help from any library.
"""


def derivative(w1, w2, x):
    """
    Given the following function f(x) = w1 * x^3 + w2 * x - 1
    evaluate the derivative of the function on x

    :param w1: first coefficient
    :param w2: second coefficient
    :param x: point on which to evaluate derivative (float)
    :return: value of the derivative on point x
    :rtype: float
    """

    deriv = 3*w1*x**2 + w2
    return deriv


def abs_dist(x):
    """
    Return the absolute value of x

    :param x: a number (float)
    :return: absolute value of x
    :rtype: float
    """

    return abs(x)


def fact(x):
    """
    Return the factorial of x.
    Your function should raise a ValueError
    if x is negyative

    :param x: a number (int)
    :return: the factorial of x
    :rtype: float
    :raise ValueError:
    """
    if x < 0:
        raise ValueError

    factorial = 1

    for i in range(1, x+1):
        factorial *= i

    return factorial


def combination(n, r):
    """
    Given n total number of items,
    what is the number of possible ways
    to choose r items from it?

    :param n: total number of items (integer)
    :param r: number of items to arrange (int)
    :return: number of combinations
:rtype: integer
    """
    if n < 0 or r < 0 or r > n:
        raise ValueError

    combi = fact(n)/(fact(n-r)*fact(r))
    return combi
