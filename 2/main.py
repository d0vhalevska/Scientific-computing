import numpy
import numpy as np


import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not

    m_a = len(A[0])
    m_b = len(b)

    if m_a != m_b:
        raise ValueError('Matrix and vector are not compatible!')

    if A.shape[0] != A.shape[1]:                                     # check if square
        raise ValueError('matrix is not square!')

   # TODO: Perform gaussian elimination

    #check if any 0 on diagonal, if yes, then no pivoting
    if any(np.diag(A) == 0):
        raise ValueError("Zero division error!")

    if any(np.diag(A) == 0) and use_pivoting:
            i = 0
            for x in A:
                x.append(b[i])
                i += 1

            for k in range(m_a):
                for i in range(k+1, m_a):
                    if abs(A[i][k]) > abs(A[k][k]):
                        A[k], A[i] = A[i], A[k]
                    else:
                        pass

                for j in range(k + 1, m_a):
                    q = float(A[i][k]) / A[k][k]
                    for p in range(k, m_a + 1):
                        A[j][p] -= q * A[k][p]

    else:
        # Gauß without pivoting
        for row in range(0, m_a - 1):
            for i in range(row + 1, m_a):
                f = A[i, row] / A[row, row]

                for j in range(row, m_a):
                    A[i, j] = A[i, j] - f * A[row, j]
                b[i] = b[i] - f * b[row]
    return A, b



def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    m_a = len(A[0])
    m_b = len(b)
    if m_a != m_b:
        raise ValueError()

    # TODO: Initialize solution vector with proper size
    x = np.zeros(1)
    x = np.zeros((m_a, ))

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist

    if any(np.diag(A) == 0) and b != 0:
        raise ValueError('Keine Lösung!')

    if any(np.diag(A) == 0) and b == 0:
        raise ValueError('Unendlich viele Lösungen!')

    x[m_a-1] = b[m_a-1] / A[m_a-1, m_a-1]
    for row in range(m_a-2, -1, -1):
        var = b[row]
        for i in range(row+1, m_a):
            var = var - A[row, i] * x[i]
        x[row] = var / A[row, row]

    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # TODO check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape

    if M.shape[0] != M.shape[1]:                                     # check if square
        raise ValueError('matrix is not square!')

    if np.allclose(M, np.tril(M)):                                   # check if lower triangular
        raise ValueError('Matrix is lower triangular')

    if np.allclose(M, np.triu(M)):                                   # check if upper triangular
        raise ValueError("Matrix is upper triangular")

    if np.allclose(M, np.diag(np.diag(M))):                          # check if diagonal
        raise ValueError('Matrix is diagonal')

    if not np.allclose(M, np.transpose(M)):
        raise ValueError('The Matrix ist not symmetric!')             #check if symmetrisch

    # TODO build the factorization and raise a ValueError in case of a non-positive definite input matrix
    L = np.zeros((n, n))

    m = M.shape[0]
    for i in range(0, m):
        L[i, i] = M[i, i]                                             #zuerst Diagonale
        for j in range(0, i):
            L[i, i] = L[i, i] - L[i, j] * L[i, j]
            if (L[i, i] - L[i, j] * L[i, j]) < 0:                     #prüfe PSD
                raise ValueError('Matrix is not PSD')
        L[i, i] = numpy.sqrt(L[i, i])

        for k in range(i + 1, m):                                     #dann alle anderen Spalten
            L[k, i] = M[k, i]
            for j in range(0, i):
                L[k, i] = L[k, i] - L[k, j] * L[i, j]
            L[k, i] = L[k, i] / L[i, i]                               #dividiere und i,i kann bei mir nciht negativ sein
    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:

    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # TODO Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape

    m_l = len(L[0])
    m_b = len(b)
    if m_l != m_b:
        raise ValueError()

    if not np.allclose(L, np.tril(L)):                              # check if lower triangular
        raise ValueError("Matrix is upper triangular")

    if L.shape[0] != L.shape[1]:                                     # check if square
        raise ValueError('matrix is not square!')

     # TODO Solve the system by forward- and backsubstitution

    x = np.zeros(m)
    y = np.zeros(m)

    for i in range(n):
        y[i] = (b[i]-sum([L[i][j] * y[j] for j in range(i)])) / L[i][i]

    x = back_substitution(np.transpose(L), y)
    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # TODO: Initialize system matrix with proper size
    #L = np.zeros((1, 1))
    L = np.zeros((n_rays * n_shots, n_grid * n_grid))

    # TODO: Initialize intensity vector
    #g = np.zeros(1)
    g = np.zeros(n_shots * n_rays)

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    theta = 0
    # Take a measurement with the tomograph from direction r_theta.
    # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
    # ray_indices: indices of rays that intersect a cell
    # isect_indices: indices of intersected cells
    # lengths: lengths of segments in intersected cells
    # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.

    #winkel = np.pi/n_shots
    winkel_array = numpy.linspace(0, np.pi, n_shots, endpoint=False)
    incrementation = 0

    for step in winkel_array:
        theta = step
        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)
        for i, j, k in zip (ray_indices, isect_indices, lengths):             #itterieren alle Richtungen
            L[i + incrementation*n_rays][j] = k
            #g[i + incrementation*n_rays] = l
            g[i + incrementation*n_rays] = intensities[i]
        incrementation += 1

    return [L, g]


def compute_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)



    # TODO: Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)
    rechte_seite = np.transpose(L).dot(g)
    matrizen_zusammen = np.transpose(L).dot(L)
    x = np.linalg.solve(matrizen_zusammen, rechte_seite)

    # TODO: Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))
    tim = np.reshape(x, n_grid*n_grid)

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
    A = np.zeros((3, 3))
    A = np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]])
    g = compute_cholesky(A)
    n_shots = 64  # 128
    n_rays = 64  # 128
    n_grid = 32  # 64
    L, g = setup_system_tomograph(4, 4, 2)
    tim = compute_tomograph(4, 4, 2)
    print(tim)