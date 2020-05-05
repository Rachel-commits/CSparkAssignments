"""This fit pca on scaled data contains a set of functions to implement using
 PCA.

All of them take at least a dataframe df as argument. To test your functions
locally, we recommend using the wine dataset that you can load from sklearn by
importing sklearn.datasets.load_wine"""

# Import all libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_cumulated_variance(df, scale):
    """Apply PCA on a DataFrame and return a new DataFrame containing
    the cumulated explained variance from with only the first component,
    up to using all components together. Values should be expressed as
    a percentage of the total variance explained.

    The DataFrame will have one row and each column should correspond to a
    principal component.

    Example:
             PC1        PC2        PC3        PC4    PC5
    0  36.198848  55.406338  66.529969  73.598999  100.0

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param scale: boolean, whether to scale or not
    :return: a new DataFrame with cumulated variance in percent
    """
    # Create instance of pca
    pca = PCA()

    # scale if required scale then fit pca on scaled data
    if scale:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        pca.fit_transform(scaled_data)
    # else fit
    else:
        pca.fit_transform(df)

    # Calculate cumalative sum of variance
    cum_var_sum = np.cumsum(100 * pca.explained_variance_ratio_)

    # Create column names varaiable PC1 ... PCn
    col_names = ['PC' + str(i+1) for i in range(df.columns.size)]

    # Convert to a dataframe, transpose and set column names
    cum_var_df = pd.DataFrame(cum_var_sum, index=col_names).T

    return cum_var_df


def get_coordinates_of_first_two(df, scale):
    """Apply PCA on a given DataFrame df and return a new DataFrame
    containing the coordinates of the first two principal components
    expressed in the original basis (with the original columns).

    Example:
    if the original DataFrame was:

          A    B
    0   1.3  1.2
    1  27.0  2.1
    2   3.3  6.8
    3   5.1  3.2

    we want the components PC1 and PC2 expressed as a linear combination
    of A and B, presented in a table as:

              A      B
    PC1    0.99  -0.06
    PC2    0.06   0.99

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param scale: boolean, whether to scale or not
    :return: a new DataFrame with coordinates of PC1 and PC2
    """

    # Create instance of pca
    pca = PCA()

    # scale if required scale then fit pca on scaled data
    if scale:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        pca.fit_transform(scaled_data)
    # else fit
    else:
        pca.fit_transform(df)

    # Get first two components
    top_cmp = pca.components_[0:2]

    # Convert to dataframe and set row index and col names
    top_two = pd.DataFrame(
        top_cmp, index=['PC1', 'PC2'], columns=df.columns.values)

    return top_two


def get_most_important_two(df, scale):
    """Apply PCA on a given DataFrame df and use it to determine the
    'most important' features in your dataset. To do so we will focus
    on the principal component that exhibits the highest explained
    variance (that's PC1).

    PC1 can be expressed as a vector with weight on each of the original
    columns. Here we want to return the names of the two features that
    have the highest weights in PC1 (in absolute value).

    Example:
        if the original DataFrame was:

          A    B     C
     0  1.3  1.2   0.1
     1  2.0  2.1   1.2
     2  3.3  6.8  23.4
     3  5.1  3.2   4.5

    and PC1 can be written as [0.05, 0.22, 0.97] in [A, B, C].

    Then you should return C, B as the two most important features.

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param scale: boolean, whether to scale or not
    :return: names of the two most important features as a tuple
    """

    # call function to scale if required and fit model which returns
    # PC1 and PC2 coordeinates in order to access that PC1 data
    top_two = get_coordinates_of_first_two(df, scale)

    #  Get  absolute value of coordinates of pc1
    pc1 = np.abs(top_two.iloc[0])

    # Retuen indices corresponding to the sort
    # flip to get descending order and select the
    # indices of the first 2 (max) values
    max_indices = np.flip(np.argsort(pc1))[0:2]

    # extract columsn relating to indices return as a tuple
    return tuple(df.columns[max_indices])


def distance_in_n_dimensions(df, point_a, point_b, n, scale):
    """Write a function that applies PCA on a given DataFrame df in order
    to find a new subspace of dimension n.

    Transform the two points point_a and point_b to be represented into that
    n dimensions space, compute the Euclidean distance between the points in
    that space and return it.

    Example:
        if the original DataFrame was:

          A    B     C
     0  1.3  1.2   0.1
     1  2.0  2.1   1.2
     2  3.3  6.8  23.4
     3  5.1  3.2   4.5

    and n = 2, you can learn a new subspace with two columns [PC1, PC2].

    Then given two points:

    point_a = [1, 2, 3]
    point_b = [2, 3, 4]
    expressed in [A, B, C]

    Transform them to be expressed in [PC1, PC2], here we would have:
    point_a -> [-4.57, -1.74]
    point_b -> [-3.33, -0.65]

    and return the Euclidean distance between the points
    in that space.

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param point_a: a numpy vector expressed in the same basis as df
    :param point_b: a numpy vector expressed in the same basis as df
    :param n: number of dimensions of the new space
    :param scale: whether to scale data or not
    :return: distance between points in the subspace
    """

    # call function to scale if required and fit model
    # Create instance of pca
    pca = PCA(n)

    # scale if required scale then fit pca on scaled data
    if scale:
        scaler = StandardScaler()
        transformed_data = scaler.fit_transform(df)
        # converts to a 2d array and then transforms
        transformed_point_a = scaler.transform(point_a.reshape(1, -1))
        transformed_point_b = scaler.transform(point_b.reshape(1, -1))
    # if scale is false then set to use the original data
    else:
        transformed_data = df
        transformed_point_a = point_a.reshape(1, -1)
        transformed_point_b = point_b.reshape(1, -1)

    # Apply PCA
    pca.fit_transform(transformed_data)
    # Transform points a and b
    a_transformed = pca.transform(transformed_point_a)
    b_transformed = pca.transform(transformed_point_b)

    # Calculate the L2 norm
    distance = np.linalg.norm(a_transformed - b_transformed, 2)
    return distance


def find_outliers_pca(df, n, scale):
    """Apply PCA on a given DataFrame df and transofmr all the data to be
     expressed on the first principal component (you can discard other
     components)

    With all those points in a one-dimension space, find outliers by looking
    for points that lie at more than n standard deviations from the mean.

    You should return a new dataframe containing all the rows of the
    original datasetthat have been found to be outliers when projected.

    Example:
        if the original DataFrame was:

          A    B     C
     0  1.3  1.2   0.1
     1  2.0  2.1   1.2
     2  3.3  6.8  23.4
     3  5.1  3.2   4.5

    Once projected on PC1 it will be:
          PC1
    0   -7.56
    1   -6.26
    2   16.46
    3   -2.65

    Compute the mean of this one dimensional dataset and find
    all rows that lie at more than n standard deviations from it.

    Here, if n==1, only the row 2 is an outlier.

    So you should return:
         A    B     C
    2  3.3  6.8  23.4


    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param n: number of standard deviations from the mean to be
     considered outlier
    :param scale: whether to scale data or not
    :return: pandas DataFrame containing outliers only
    """

    # only interested in the first component so pass n_compenents = 1 to PCA
    pca = PCA(1)

    # scale if required scale then fit pca on scaled data
    if scale:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        transformed_df = pca.fit_transform(scaled_data)
    # else fit
    else:
        transformed_df = pca.fit_transform(df)

    # Calc mean and sd for PC1
    mean = transformed_df.mean()
    std_dev = np.sqrt(transformed_df.var())
#   Get the boundaries
    x_high = mean + n * std_dev
    x_low = mean - n * std_dev

    # Outliers are above or below n sd from the mean
    outliers = df[(transformed_df > x_high) | (transformed_df < x_low)]

    return outliers

    
