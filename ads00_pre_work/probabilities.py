"""
This file contains a set of functions to practice your
probabilities skills.

It needs to be completed with "vanilla" Python, without
help from any library -- except for the bin_dist function
for which you can use scipy.
"""
from scipy.special import comb


def head_tails(p, n):
    """
    Given a coin that have probability p of giving a heads
    in each toss independently, what is the probability of
    having n heads consecutively in a row?

    :param p: probability of a head
    :param n: number of heads in a row (int)
    :return: probability of having n heads in a row
    :rtype: float
    """

    return p**n


def bin_dist(n, p, x):
    """
    Given n number of trials, p the probability of success,
    what is the probability of having x successes?

    Your function should raise a ValueError if x is higher
    than n.
    

    If you need to compute combinations, you can import the
    function "comb" from the package "scipy.special"

    :param n: number of trials (int)
    :param p: probability of success
    :param x: number of successes (int)
    :return: probability of having x successes
    :rtype: float
    :raise ValueError: if x > n
    """

    if x > n:
        raise ValueError
    x_success = comb(n, x)*(p**x) * (1-p)**(n-x)
    return x_success


bin_dist(4, 0.2, 2)


def bin_cdf(n, p, x):
    """
    Given n number of trials, p the probability of successes,
    what is the probability of having less than or equal to x successes?

    Your function should raise a ValueError if x is higher
    than n.

    :param n: number of trials (int)
    :param p: probability of success
    :param x: number of successes (int)
    :return: probability of having less than or
    equal to x successes
    :rtype: float
    :raise ValueError: if x > n
    """
    prob = 0
    if x > n:
        raise ValueError
    for i in range(x+1):
        prob += bin_dist(n, p, i)
    return prob

print(bin_cdf(3, 0.7, 1))

# assert bin_cdf(15, 0.2, 2) == 0.167

print(bin_cdf(16, 0.2, 2))
