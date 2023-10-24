import numpy as np
import lib
import matplotlib as mpl


####################################################################################################
# Exercise 1: Power Iteration

def power_iteration(M: np.ndarray, epsilon: float = -1.0) -> (np.ndarray, list):
    """
    Compute largest eigenvector of matrix M using power iteration. It is assumed that the
    largest eigenvalue of M, in magnitude, is well separated.

    Arguments:
    M: matrix, assumed to have a well separated largest eigenvalue
    epsilon: epsilon used for convergence (default: 10 * machine precision)

    Return:
    vector: eigenvector associated with largest eigenvalue
    residuals : residual for each iteration step

    Raised Exceptions:
    ValueError: if matrix is not square

    Forbidden:
    numpy.linalg.eig, numpy.linalg.eigh, numpy.linalg.svd
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix not nxn")

    # TODO: set epsilon to default value if not set by user
    if epsilon == -1.0:
        epsilon = 10.0 * np.finfo(M.dtype).eps

    # TODO: random vector of proper size to initialize iteration
    vector = np.random.rand(M.shape[1])

    # Initialize residual list and residual of current eigenvector estimate
    residuals = []
    residual = 2.0 * epsilon

    # Perform power iteration
    while residual > epsilon:
        # TODO: implement power iteration
        vector = np.dot(M, vector)
        vector = np.divide(vector, np.amax(np.abs(vector)))

        residuals.append(np.linalg.norm(vector))
        if len(residuals) > 1:
            residual = np.abs(residuals[-1] - residuals[-2])

        vector = np.divide(vector, np.linalg.norm(vector))

    print("result vector " + str(vector))
    return vector, residuals


####################################################################################################
# Exercise 2: Eigenfaces

def load_images(path: str, file_ending: str = ".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions
    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction
    """

    images = []

    # TODO read each image in path as numpy.ndarray and append to images
    # Useful functions: lib.list_directory(), matplotlib.image.imread(), numpy.asarray()
    # print(lib.list_directory("./"))
    # print("hello")
    for file in sorted(lib.list_directory(path)):
        #  print("reading...:" +  path+file)
        # simple check for file type... might not work for "asda.zip.gzip" files as they have multiple dots..
        # vergleicht ob  .png == .png
        if (file.split(".")[-1] == file_ending.split(".")[1]):
            images.append(np.asarray(mpl.image.imread(path + file), dtype="float64"))

    dimension_x = images[0].shape[1]  # 1 ist wohl x...
    dimension_y = images[0].shape[0]  # 0 ist wohl y... warum auch immer

    # print("shape:" + str(images[0].shape))
    return images, dimension_x, dimension_y


def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    Arguments:
    images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    Return:
    D: data matrix that contains the flattened images as rows
    """
    # TODO: initialize data matrix with proper size and data type
    # reshape makes array out of matrix
    D = []  # <- will contain all images as vector rows..
    # TODO: add flattened images to data matrix
    for image in images:
        D.append(image.reshape(-1))
    return np.array(D)  # make numpy array out of it...


def calculate_pca(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform principal component analysis for given data matrix.

    Arguments:
    D: data matrix of size m x n where m is the number of observations and n the number of variables

    Return:
    pcs: matrix containing principal components as rows
    svals: singular values associated with principle components
    mean_data: mean that was subtracted from data
    """

    # TODO: subtract mean from data / center data at origin
    mean_data = np.zeros((1, 1))
    mean_data = np.mean(D, axis=0)

    for x in range(D.shape[0]):
       D[x] = D[x] - mean_data

    # TODO: compute left and right singular vectors and singular values
    # Useful functions: numpy.linalg.svd(..., full_matrices=False)
    svals, pcs = [np.ones((1, 1))] * 2

    U, Sigma, V = np.linalg.svd(D, full_matrices=False)
    svals = Sigma
    pcs = V

    return pcs, svals, mean_data


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    Arguments:
    singular_values: vector containing singular values
    threshold: threshold for determining k (default = 0.8)

    Return:
    k: threshold index
    """

    # TODO: Normalize singular value magnitudes

    k = 0

    # TODO: Determine k that first k singular values make up threshold percent of magnitude
    sum_of_singular = np.sum(singular_values)
    sum_of_percentage = 0
    for i in singular_values:
        sum_of_percentage = i / sum_of_singular + sum_of_percentage
        if sum_of_percentage < threshold:
            k += 1

    return k+1


def project_faces(pcs: np.ndarray, images: list, mean_data: np.ndarray) -> np.ndarray:
    """
    Project given image set into basis.

    Arguments:
    pcs: matrix containing principal components / eigenfunctions as rows
    images: original input images from which pcs were created
    mean_data: mean data that was subtracted before computation of SVD/PCA

    Return:
    coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """

    # TODO: initialize coefficients array with proper size
    #coefficients = np.zeros((1, 1))
    D = setup_data_matrix(images)
    coefficients = np.zeros((D.shape[0], pcs.shape[0]))

    for x in range(D.shape[0]):
       D[x] = D[x] - mean_data

    # TODO: iterate over images and project each normalized image into principal component basis

    for i in range(D.shape[0]):
        coefficients[i] = np.dot(pcs, D[i])
    return coefficients


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: str) -> (
        np.ndarray, list, np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    Arguments:
    coeffs_train: coefficients for training images, each image is represented in a row
    path_test: path to test image data

    Return:
    scores: Matrix with correlation between all train and test images, train images in rows, test images in columns
    imgs_test: list of test images
    coeffs_test: Eigenface coefficient of test images
    """

    # TODO: load test data set
    imgs_test = []
    imgs_test = load_images(path_test)[0]

    # TODO: project test data set into eigenbasis
    #coeffs_test = np.zeros(coeffs_train.shape)
    coeffs_test = project_faces(pcs, imgs_test, mean_data)

    # TODO: Initialize scores matrix with proper size
    scores = np.zeros((1, 1))
    scores = np.zeros((coeffs_test.shape[0], coeffs_train.shape[0]))
    # TODO: Iterate over all images and calculate pairwise correlation

    for i in range (coeffs_train.shape[0]):
        for j in range(coeffs_test.shape[0]):
            cos = np.dot(coeffs_train[i], coeffs_test[j]) / np.dot(np.linalg.norm(coeffs_train[i]), np.linalg.norm(coeffs_test[j]))
            winkel = np.arccos(cos)
            scores[j][i] = winkel
    scores = scores.T
    return scores, imgs_test, coeffs_test


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
