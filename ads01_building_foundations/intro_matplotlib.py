"""A set of exercises with matplotlib"""
import numpy as np
import matplotlib.pyplot as plt


def draw_co2_plot():
    """
    Here is some chemistry data

      Time (decade): 0, 1, 2, 3, 4, 5, 6
      CO2 concentration (ppm): 250, 265, 272, 260, 300, 320, 389

    Create a line graph of CO2 versus time, the line should be a blue dashed
    line.

    The title of the plot should be 'Chemistry data'
    The label of the x axis should be 'Time (decade)'
    The label of the y axis should be 'CO2 concentration (ppm)'

    Your function does not need to return the plot -
    the final line of the function should be
    (assuming you have imported pyplot as plt):
    plt.show()
    """

    time = np.array([0, 1, 2, 3, 4, 5, 6])
    co2 = np.array([250, 265, 272, 260, 300, 320, 389])

    plt.plot(time, co2, ls='--')
    plt.title('Chemistry data')
    plt.xlabel('Time (decade)')
    plt.ylabel('CO2 concentration (ppm)')
    plt.show()


def draw_equations_plot():
    """
    Plot the following lines on the same plot

      y=cos(x) coloured in red with dashed lines
      y=x^2 coloured in blue with linewidth 3
      y=exp(-x^2) coloured in black

    Add a legend, title for the x-axis and a title to the curve, the x-axis
    should range from -4 to 4 (with 50 points) and the y axis should range
    from 0 to 2. The figure should have a size of 8x6 inches.

    NOTE: Make sure you create the figure at the beginning as doing it at the
    end will reset any plotting you have done, and again finish with plt.show()
    """

    x = np.linspace(-4, 4, 50)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.xlim([-4, 4])
    plt.ylim([0, 2])
    ax.plot(x, np.cos(x), ls='--', color='r', label='y=cos(x)')
    ax.plot(x, x**2, color='b', lw=3, label='y=x^2')
    ax.plot(x, np.exp(-(x**2)), color='black', label='y=exp(-x^2)')
    ax.set_title('3 example graphs')
    ax.set_xlabel('x Values')
    ax.legend()
    plt.show()
