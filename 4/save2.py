import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)

    polynomial = np.poly1d(0)
    base_functions = []

    # TODO: Generate Lagrange base polynomials and interpolation polynomial
    polynomial = np.poly1d(0)
    n = len(x)
    l = np.poly1d([1])
    xp = np.poly1d([1, 0])
    for i in range(n):
        for j in range(n):
            if i != j:
                l *= (xp - x[j]) / (x[i] - x[j])
        base_functions.append(l)
        polynomial += base_functions[i] * y[i]
        l = np.poly1d([1])

    return polynomial, base_functions


def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []
    # TODO compute piecewise interpolating cubic polynomials
    n = len(x)

    for i in range(n - 1):
        entry = (np.linalg.inv(np.array([[1, x[i], x[i] ** 2, x[i] ** 3],
                                         [1, x[i + 1], x[i + 1] ** 2, x[i + 1] ** 3],
                                         [0, 1, 2 * x[i], 3 * x[i] ** 2],
                                         [0, 1, 2 * x[i + 1], 3 * x[i + 1] ** 2]])))
        entry = entry.transpose()
        entry[0] = entry[0] * y[i]
        entry[1] = entry[1] * y[i + 1]
        entry[2] = entry[2] * yp[i]
        entry[3] = entry[3] * yp[i + 1]
        entry = np.flip(entry)
        spline.append(np.poly1d(entry[0]) + np.poly1d(entry[1]) + np.poly1d(entry[2]) + np.poly1d(entry[3]))

    return spline


####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """
    assert (x.size == y.size)
    # TODO construct linear system with natural boundary conditions

    n = 4 * x.size - 4
    matrix = np.zeros((n, n))
    my = np.zeros(n)
    j = 0
    for i in range(n - 2):
        # 1 Zeile erster Matrix
        matrix[4 * i][4 * j] = 1
        matrix[4 * i][4 * j + 1] = x[i]
        matrix[4 * i][4 * j + 2] = x[i] ** 2
        matrix[4 * i][4 * j + 3] = x[i] ** 3

        # 2 Zeile erster Matrix
        matrix[4 * i + 1][4 * j] = 1
        matrix[4 * i + 1][4 * j + 1] = x[i + 1]
        matrix[4 * i + 1][4 * j + 2] = x[i + 1] ** 2
        matrix[4 * i + 1][4 * j + 3] = x[i + 1] ** 3

        my[4 * i] = y[i]
        my[4 * i + 1] = y[i + 1]
        my[4 * i + 2] = 0
        my[4 * i + 3] = 0

        if i == x.size - 2:
            break

        # 3 Zeile erster Matrix
        matrix[4 * i + 2][4 * i] = 0
        matrix[4 * i + 2][4 * j + 1] = 1
        matrix[4 * i + 2][4 * j + 2] = 2 * x[i + 1]
        matrix[4 * i + 2][4 * j + 3] = 3 * x[i + 1] ** 2

        # 4 Zeile erster Matrix
        matrix[4 * i + 3][4 * j] = 0
        matrix[4 * i + 3][4 * j + 1] = 0
        matrix[4 * i + 3][4 * j + 2] = 2
        matrix[4 * i + 3][4 * j + 3] = 6 * x[i + 1]

        # erste Zeile zweiter Matrix
        matrix[4 * i + 2][4 * j + 4] = 0
        matrix[4 * i + 2][4 * j + 5] = -1
        matrix[4 * i + 2][4 * j + 6] = -2 * x[i + 1]
        matrix[4 * i + 2][4 * j + 7] = -3 * x[i + 1] ** 2

        # zweite Zeile zweiter Matrix
        matrix[4 * i + 3][4 * j + 4] = 0
        matrix[4 * i + 3][4 * j + 5] = 0
        matrix[4 * i + 3][4 * j + 6] = -2
        matrix[4 * i + 3][4 * j + 7] = -6 * x[i + 1]

        j = j + 1

    # natürliche randbedingungen zeilen zeilen
    matrix[n - 2][0] = 0
    matrix[n - 2][1] = 0
    matrix[n - 2][2] = 2
    matrix[n - 2][3] = 6 * x[0]

    matrix[n - 1][n - 4] = 0
    matrix[n - 1][n - 3] = 0
    matrix[n - 1][n - 2] = 2
    matrix[n - 1][n - 1] = 6 * x[x.size - 1]

    # TODO solve linear system for the coefficients of the spline

    # if not np.any(my):
    #     u, s, v = np.linalg.svd(matrix)
    #     coefficients = v.transpose()[:, -1]
    #
    # else:
    coefficients = np.linalg.solve(matrix, my)

    spline = []
    # TODO extract local interpolation coefficients from solution
    #for i in range(int((n/4))):
     #   spline.append(np.poly1d((coefficients[4 * i + 3], coefficients[4 * i + 2], coefficients[4 * i + 1], coefficients[4 * i])))
    for i in range(0, n, 4):
        spline.append(np.poly1d((coefficients[i+3], coefficients[i + 2], coefficients[i + 1], coefficients[i])))
    print(len(spline))

    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary oefficients = np.linalg.solve(matrix, my)
        cconditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO: construct linear system with periodic boundary conditions

    # TODO solve linear system for the coefficients of the spline

    spline = []
    # TODO extract local interpolation coefficients from solution

    return spline


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
