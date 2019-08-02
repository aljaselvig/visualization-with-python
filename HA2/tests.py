
import numpy as np
import matplotlib.pyplot as plt
import datetime

import unittest
import tomograph
from main import compute_tomograph, gaussian_elimination, back_substitution, compute_cholesky


class Tests(unittest.TestCase):
    def test_gaussian_elimination(self):
        A = np.random.randn(4, 4)
        x = np.random.rand(4)
        b = np.dot(A, x)
        A_elim, b_elim = gaussian_elimination(A, b)

        self.assertTrue(np.allclose(A_elim, np.triu(A_elim)))  # Check if matrix is upper triangular
        self.assertTrue(np.allclose(np.linalg.solve(A_elim, b_elim), x))  # Check if system is still solvable

    def test_back_substitution(self):
        pass
        """
        A = np.array( [ [ 11, 44, 1 ], [ 0.1, 0.4, 3 ], [ 0, 1, (-1) ] ] )
        b = np.array( [ 1, 1, 1 ] )

        A_gaus, b_gaus  = gaussian_elimination( A, b )

        x = np.array( [ (-1) * 1732/329, 438/329, 109/329 ] )

        print( x )

        print( A_gaus )

        print( b_gaus )

        print ( back_substitution( A_gaus, b_gaus ) )

        self.assertTrue( np.allclose( x, back_substitution( A_gaus, b_gaus ) ) )
        """

    def test_cholesky_decomposition(self):
        A = np.array( [ [ 1, 3, 0 ], [ 3, 2, 6 ], [ 0, 6, 5] ] )

        #self.assertTrue(  )

    def test_solve_cholesky(self):
        pass
        # TODO

    def test_compute_tomograph(self):
        t = datetime.datetime.now()
        print("Start time: " + str(t.hour) + ":" + str(t.minute) + ":" + str(t.second))

        # Compute tomographic image
        n_shots = 64  # 128
        n_rays = 64  # 128
        n_grid = 32  # 64
        tim = compute_tomograph(n_shots, n_rays, n_grid)

        t = datetime.datetime.now()
        print("End time: " + str(t.hour) + ":" + str(t.minute) + ":" + str(t.second))

        # Visualize image
        plt.imshow(tim, cmap='gist_yarg', extent=[-1.0, 1.0, -1.0, 1.0],
                   origin='lower', interpolation='nearest')
        plt.gca().set_xticks([-1, 0, 1])
        plt.gca().set_yticks([-1, 0, 1])
        plt.gca().set_title('%dx%d' % (n_grid, n_grid))

        plt.show()


if __name__ == '__main__':
    unittest.main()

