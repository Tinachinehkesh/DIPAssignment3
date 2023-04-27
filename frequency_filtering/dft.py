# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries

import numpy as np
import math

class Dft:
    def __init__(self):
        pass

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        (rows, columns) = matrix.shape
        res = np.zeros((rows, columns), dtype=complex)

        for i in range (rows):
            for j in range (columns):
                for u in range(rows):
                    for k in range(columns):
                        res[i][j] += matrix[u][k] * complex(math.cos(2 * np.pi * (i * u + j * k) / rows) - complex(0, 1) * math.sin(2 * np.pi * (i * u + j * k) / rows)) / 15**2


        return res

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        You can implement the inverse transform formula with or without the normalizing factor.
        Both formulas are accepted.
        takes as input:
        matrix: a 2d matrix (DFT) usually complex
        returns a complex matrix representing the inverse fourier transform"""

        (rows, columns) = matrix.shape
        res = np.zeros((rows, columns), dtype=complex)

        for i in range (rows):
            for j in range (columns):
                for u in range(rows):
                    for k in range(columns):
                        res[i][j] += matrix[u][k] * \
                            complex(math.cos(2 * np.pi * (i * u + j * k) / rows) + complex(0, 1) * math.sin(2 * np.pi * (i * u + j * k) / rows))

        return res

    def magnitude(self, matrix):
        """Computes the magnitude of the input matrix (iDFT)
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the complex matrix"""

        (rows, columns) = matrix.shape
        res = np.zeros((rows, columns))

        for i in range(rows):
            for j in range(columns):
                res[i][j] = math.sqrt(math.pow(np.real(matrix[i][j]), 2) + math.pow(np.imag(matrix[i][j]), 2))

        return res