import numpy
import numpy as np

from lib import timedcall, plot_2d


def matrix_multiplication(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate product of two matrices a * b.

    Arguments:
    a : first matrix
    b : second matrix
import numpy as np
ModuleNotFoundError: No module named 'numpy'
    Return:
    c : matrix product a * b

    Raised Exceptions:
    ValueError: if matrix sizes are incompatible

    Side Effects:
    -

    Forbidden: numpy.dot, numpy.matrix
    """

    n, m_a = a.shape
    m_b, p = b.shape

    # TODO: test if shape of matrices is compatible and raise error if not
    # soll testen ob m_a == m_b
    if m_a != m_b:
        raise ValueError()

    # Initialize result matrix with zeros
    c = np.zeros((n, p))

    # TODO: Compute matrix product without the usage of numpy.dot()
    # iterate through rows of X
    for i in range(len(a)): #height of a
        # iterate through columns of Y
        for j in range(len(b[0])): # width of b
            # iterate through rows of Y
            for k in range(len(b)):
                c[i][j] += a[i][k] * b[k][j]
    return c


def compare_multiplication(nmax: int, n: int) -> dict:
    """
    Compare performance of numpy matrix multiplication (np.dot()) and matrix_multiplication.

    Arguments:
    nmax : maximum matrix size to be tested
    n : step size for matrix sizes

    Return:
    tr_dict : numpy and matrix_multiplication timings and results {"timing_numpy": [numpy_timings],
    "timing_mat_mult": [mat_mult_timings], "results_numpy": [numpy_results], "results_mat_mult": [mat_mult_results]}

    Raised Exceptions:
    -

    Side effects:
    Generates performance plots.
    """

    x, y_mat_mult, y_numpy, r_mat_mult, r_numpy = [], [], [], [], []
    tr_dict = dict(timing_numpy=y_numpy, timing_mat_mult=y_mat_mult, results_numpy=r_numpy, results_mat_mult=r_mat_mult)
    # TODO: Can be removed if matrices a and b are created in loop
   # a = np.ones((2, 2))
   # b = np.ones((2, 2))

    for m in range(2, nmax, n):

        # TODO: Create random mxm matrices a and b
        a = np.random.rand(m, m)
        b = np.random.rand(m, m)

        # Execute functions and measure the execution time
        time_mat_mult, result_mat_mult = timedcall(matrix_multiplication, a, b)
        time_numpy, result_numpy = timedcall(np.dot, a, b)

        # Add calculated values to lists
        x.append(m) #matrix size
        y_numpy.append(time_numpy) #our time for m
        y_mat_mult.append(time_mat_mult)#our result for a*b
        r_numpy.append(result_numpy)#numpy time for m
        r_mat_mult.append(result_mat_mult)#numpy result for a*b

    # Plot the computed data
    plot_2d(x_data=x, y_data=[y_mat_mult, y_numpy], labels=["matrix_mult", "numpy"],
            title="NumPy vs. for-loop matrix multiplication",
            x_axis="Matrix size", y_axis="Time", x_range=[2, nmax])

    return tr_dict


def machine_epsilon(fp_format: np.dtype) -> np.number:
    """
    Calculate the machine precision for the given floating point type.

    Arguments:
    fp_format: floating point format, e.g. float32 or float64

    Return:
    eps : calculated machine precision

    Raised Exceptions:
    -

    Side Effects:
    Prints out iteration values.

    Forbidden: numpy.finfo
    """

    # TODO: create epsilon element with correct initial value and data format fp_format
    eps = fp_format.type(0.0)
    eps = fp_format.type(1.0)

    # Create necessary variables for iteration
    one = fp_format.type(1.0)
    two = fp_format.type(2.0)
    i = 0

    print('  i  |       2^(-i)        |  1 + 2^(-i)  ')
    print('  ----------------------------------------')

    # TODO: determine machine precision without the use of numpy.finfo()
    while (one + (eps / two)) != one:
        eps /= two
        i += 1

    print('{0:4.0f} |  {1:16.8e}   | equal 1'.format(i, eps))
    return eps


def rotation_matrix(theta: float) -> np.ndarray:
    """
    Create 2x2 rotation matrix around angle theta.

    Arguments:
    theta : rotation angle (in degrees)

    Return:
    r : rotation matrix

    Raised Exceptions:
    -

    Side Effects:
    -
    """

    # create empty matrix
    r = np.zeros((2, 2))

    # TODO: convert angle to radians
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)

    # TODO: calculate diagonal terms of matrix
    r = np.array(((c, -s), (s, c)))


    # TODO: off-diagonal terms of matrix


    return r


def inverse_rotation(theta: float) -> np.ndarray:
    """
    Compute inverse of the 2d rotation matrix that rotates a 
    given vector by theta.
    
    Arguments:
    theta: rotation angle
    
    Return:
    Inverse of the rotation matrix

    Forbidden: numpy.linalg.inv, numpy.linalg.solve
    """

    # TODO: compute inverse rotation matrix

    m = np.zeros((2, 2))
    mr = rotation_matrix(theta)

    # iterate through rows
    for i in range(len(mr)):
        # iterate through columns
        for j in range(len(mr[0])):
            m[j][i] = mr[i][j]

    return m


if __name__ == '__main__':
    compare_multiplication(100, 10)
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")

