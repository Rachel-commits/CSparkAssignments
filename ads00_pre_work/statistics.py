"""
This file contains a set of functions to practice your
statistics skills.

It needs to be completed with "vanilla" Python, without
help from any library.
"""


def calculate_mean(data):
    """
    Return the mean of a python list

    If data is empty raise a ValueError

    :param data: a list of numbers
    :return: the mean of the list
    :rtype: float
    :raise ValueError:
    """
    if not data:
        raise ValueError

    sum_val = 0
    for val in data:
        sum_val += val
    mean = sum_val/len(data)
    return mean

# print(calculate_mean([3,5,7,8]))


def calculate_standard_deviation(data):
    """
    Return the standard deviation of a python list using the
    population size (N) in order to calculate the variance.

    If data is empty raise a ValueError

    :param data: list of numbers
    :return: the standard deviation of the list
    :rtype: float
    :raise ValueError:
    """
    if not data:
        raise ValueError

    sum_val_sq = 0
    mean = calculate_mean(data)
    for val in data:
        sum_val_sq += (val-mean)**2

    std_dev = (sum_val_sq/len(data))**(1/2)

    return std_dev


print(calculate_standard_deviation([3, 6, 15]))


def remove_outliers(data):
    """
    Given a list of numbers, find outliers and return a new
    list that contains all points except outliers
    We consider points lying outside 2 standard
    deviations from the mean.

    Make sure that you do not modify the original list!

    If data is empty raise a ValueError

    :param data: list of numbers
    :return: a new list without outliers
    :rtype: list
    :raise ValueError:
    """
    if not data:
        raise ValueError

    new_list = []
    std_dev = calculate_standard_deviation(data)
    mean = calculate_mean(data)
    lower = mean - 2*std_dev
    upper = mean + 2*std_dev
    for val in data:
        if lower <= val <= upper:
            new_list.append(val)
    return new_list
