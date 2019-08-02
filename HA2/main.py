
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
    #------------------------------------------------------------------------------------
    b_row = b.size
    A_row, A_col = A.shape

    if A_row != b_row:
        raise ValueError('ERROR!!!! A_row != b_row')

    if A_row != A_col:
        raise ValueError('ERROR!!!! A_row != A_col')
    #------------------------------------------------------------------------------------

    # TODO: Perform gaussian elimination
    # ------------------------------------------------------------------------------------
    for k in range(0, A_row):
        if use_pivoting == True:
            max_abs_i = k

            # ишем строку с максимальным значением в столбце k матрицы А
            for p in range(k, A_row):
                if np.abs(A[p][k]) > np.abs(A[max_abs_i][k]):
                    max_abs_i = p

            # меняем местами строки в матрице A
            tmp_row = A[max_abs_i].copy()
            A[max_abs_i] = A[k].copy()
            A[k] = tmp_row.copy()

            # меняем местами строки в векторе b
            tmp_row = b[max_abs_i].copy()
            b[max_abs_i] = b[k].copy()
            b[k] = tmp_row.copy()

        for i in range(k + 1, A_row):
            # Возвращает значение True, если a[k][k] == 0  делитель не может быть равным 0
            if np.allclose(A[k][k], [0.0]) == True:
                raise ValueError('ERROR!!!!  A[k][k] == 0.0')

            # множитель который надо отнять что бы получить 0 значение в заданном столбце
            t = (-1) * A[i][k] / A[k][k]

            b[i] = b[i] + t * b[k]
            for j in range(k + 1, A_col):
                A[i][j] = A[i][j] + t * A[k][j]

            # явным образом приравниваем к 0 чтобы избежать ошибки из-за неточного вычисления при работе с малыми числами
            A[i][k] = 0.0

    return A, b
#------------------------------------------------------------------------------------

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
    b_row = b.size
    A_row, A_col = A.shape

    if A_row != b_row:
        raise ValueError('ERROR!!!! A_row != b_row')

    if A_row != A_col:
        raise ValueError('ERROR!!!! A_row != A_col')


    # TODO: Initialize solution vector with proper size
    x = np.zeros(1)

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist

    for i in range(0, A_row):
        for j in range(i, A_col):
            # если внизу есть хотя бы один ненулевой элемент выходим из цикла
            if A[i][j] != 0:
                break

            if j == A_col - 1:
                raise ValueError('ERROR!!!! j == A_col - 1')


    x[A_row - 1] = b[A_row - 1] / A[A_row - 1][A_row - 1]

    for i in range(A_row - 2, -1, -1):
        subtract = 0.0

        for j in range(A_row - 1, -1, -1):
            subtract += A[i][j] * x[j]

        x[i] = (b[i] - subtract) / A[i][i]

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



    # TODO build the factorization and raise a ValueError in case of a non-positive definite input matrix

    L = np.zeros((n, n))


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


    # TODO Solve the system by forward- and backsubstitution
    x = np.zeros(m)


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
    L = np.zeros((1, 1))
    # TODO: Initialize intensity vector
    g = np.zeros(1)

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    theta = 0
    # Take a measurement with the tomograph from direction r_theta.
    # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
    # ray_indices: indices of rays that intersect a cell
    # isect_indices: indices of intersected cells
    # lengths: lengths of segments in intersected cells
    # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.
    intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)


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

    # TODO: Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
