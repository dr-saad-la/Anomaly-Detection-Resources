"""
RPCA Module

This module implements the Robust Principal Component Analysis (RPCA) algorithm.
RPCA is used for separating low-rank and sparse components in a matrix, making it
effective for handling contaminated datasets with outliers and noise.

Python 3.6 or higher is required for this module.

Optional Dependencies:
    - pyarrow: If installed, pyarrow will be used for array operations as an alternative to numpy.

Attributes:
    __author__ (str): The name of the module author.
    __version__ (str): The current version of the module.
    __license__ (str): The license under which the module is distributed.
    __status__ (str): The current status of the module (e.g., "Development", "Production").
    __maintainer__ (str): The name and contact information of the current maintainer.
    __email__ (str): The contact email for the module author or maintainer.
    __credits__ (list): List of people or organizations that contributed to the module.
    __url__ (str): URL for more information or documentation.
    __date__ (str): Date of the last update or release.
"""

__author__ = "Dr. Saad Laouadi"  
__version__ = "1.0.0"  
__license__ = "MIT"  
__status__ = "Development"  
__maintainer__ = "Saad Laouadi"  
__email__ = "dr.saad.laouadi@gmail.com" 
__credits__ = [""]
__url__ = "https://github.com/dr-saad-la/rpca"
__date__ = "2024-08-19"


try:
    import pyarrow as pa
    import pyarrow.compute as pc
    USE_PYARROW = True
except ImportError:
    import numpy as np
    USE_PYARROW = False

import matplotlib.pyplot as plt


class RPCAResult:
    """
    A class to store the results of the RPCA fitting process.
    """
    def __init__(self, L, S, history):
        self.L = L                      # Low-rank matrix
        self.S = S                      # Sparse matrix
        self.history = history          # Convergence history

class R_pca:
    """
    Robust Principal Component Analysis (RPCA) implementation.

    This class implements the Principal Component Pursuit (PCP) algorithm for
    robust PCA, which decomposes a given matrix D into a low-rank matrix L and
    a sparse matrix S.

    Attributes:
        D (numpy.ndarray): The input data matrix.
        S (numpy.ndarray): The sparse matrix (initialized as zeros).
        Y (numpy.ndarray): The Lagrange multiplier matrix (initialized as zeros).
        mu (float): The step size parameter for the algorithm.
        mu_inv (float): The inverse of the step size parameter.
        lmbda (float): The regularization parameter.
    """

    def __init__(self, D, mu=None, lmbda=None):
        """
        Initialize the R_pca instance.

        Args:
            D (numpy.ndarray): The input data matrix to be decomposed.
            mu (float, optional): The step size parameter. If None, it is calculated automatically.
            lmbda (float, optional): The regularization parameter. If None, it is calculated automatically.
        """
        self.D = D
        if USE_PYARROW:
            self.S = pa.zeros_like(self.D)
            self.Y = pa.zeros_like(self.D)
            self.D_np = self.D.to_numpy()  # Convert to numpy array for certain calculations
        else:
            self.S = np.zeros(self.D.shape)
            self.Y = np.zeros(self.D.shape)

        # Calculate mu if not provided
        if mu is None:
            if USE_PYARROW:
                self.mu = np.prod(self.D_np.shape) / (4 * np.linalg.norm(self.D_np, ord=1))
            else:
                self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))
        else:
            self.mu = mu
        self.mu_inv = 1 / self.mu

        # Calculate lambda if not provided
        if lmbda is None:
            if USE_PYARROW:
                self.lmbda = 1 / np.sqrt(np.max(self.D_np.shape))
            else:
                self.lmbda = 1 / np.sqrt(np.max(self.D.shape))
        else:
            self.lmbda = lmbda

    @staticmethod
    def frobenius_norm(M):
        """
        Compute the Frobenius norm of a matrix.

        Args:
            M (array-like): The input matrix.

        Returns:
            float: The Frobenius norm of the matrix.
        """
        if USE_PYARROW:
            M_np = M.to_numpy()
            return np.linalg.norm(M_np, ord='fro')
        else:
            return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        """
        Apply the shrinkage operator to the matrix.

        Args:
            M (array-like): The input matrix.
            tau (float): The shrinkage threshold.

        Returns:
            array-like: The matrix after applying the shrinkage operator.
        """
        if USE_PYARROW:
            sign_M = pc.sign(M)
            abs_M = pc.abs(M)
            max_result = pc.subtract(abs_M, tau)
            max_result = pc.maximum(max_result, pa.zeros_like(M))
            return pc.multiply(sign_M, max_result)
        else:
            return np.sign(M) * np.maximum(np.abs(M) - tau, 0)

    def svd_threshold(self, M, tau):
        """
        Apply singular value thresholding to the matrix.

        Args:
            M (array-like): The input matrix.
            tau (float): The threshold parameter.

        Returns:
            array-like: The matrix after applying singular value thresholding.
        """
        if USE_PYARROW:
            M_np = M.to_numpy()
            U, S, V = np.linalg.svd(M_np, full_matrices=False)
            return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))
        else:
            U, S, V = np.linalg.svd(M, full_matrices=False)
            return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        """
        Fit the model to the input data using the PCP algorithm.

        Args:
            tol (float, optional): The tolerance for convergence. If None, it is set automatically.
            max_iter (int): The maximum number of iterations.
            iter_print (int): The interval at which to print progress.

        Returns:
            tuple: A tuple (L, S) where L is the low-rank matrix and S is the sparse matrix.
        """
        iter = 0
        err = np.inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        # Set tolerance if not provided
        _tol = tol if tol is not None else 1E-7 * self.frobenius_norm(self.D)

        # PCP algorithm loop
        while err > _tol and iter < max_iter:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk, self.mu_inv)  # Step 3
            Sk = self.shrink(self.D - Lk + self.mu_inv * Yk, self.mu_inv * self.lmbda)  # Step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)  # Step 5

            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1

            if iter % iter_print == 0 or iter == 1 or iter >= max_iter or err <= _tol:
                print(f'Iteration: {iter}, Error: {err:.6f}')

        self.L = Lk
        self.S = Sk

        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):
        """
        Plot the results of the RPCA fit, showing the original data (D), the low-rank approximation (L),
        and the sparse component (S) for comparison.

        Args:
            size (tuple, optional): The size of the plot grid (nrows, ncols). If not provided, it will be determined automatically.
            tol (float): The tolerance for adjusting the plot's y-axis limits.
            axis_on (bool): Whether to show the axis for each subplot.
        """
        if USE_PYARROW:
            D_np = self.D.to_numpy()
            L_np = self.L.to_numpy()
            S_np = self.S.to_numpy()
        else:
            D_np = self.D
            L_np = self.L
            S_np = self.S

        n, d = D_np.shape

        # Determine the number of rows and columns for the subplots
        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        # Determine the y-axis limits
        ymin = np.nanmin(D_np)
        ymax = np.nanmax(D_np)
        print(f'ymin: {ymin}, ymax: {ymax}')

        # Number of plots to create
        numplots = min(n, nrows * ncols)
        plt.figure()

        # Generate subplots for the data
        for i in range(numplots):
            plt.subplot(nrows, ncols, i + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(L_np[i, :] + S_np[i, :], 'r', label='L + S')
            plt.plot(L_np[i, :], 'b', label='L')
            if not axis_on:
                plt.axis('off')

        plt.show()

    def reconstruction_error(self):
        """
        Calculate the reconstruction error as the Frobenius norm of the difference
        between the original matrix D and the sum of L and S.
    
        Returns:
            float: The reconstruction error.
        """
        if USE_PYARROW:
            D_np = self.D.to_numpy()
            L_np = self.L.to_numpy()
            S_np = self.S.to_numpy()
        else:
            D_np = self.D
            L_np = self.L
            S_np = self.S
    
        error = self.frobenius_norm(D_np - (L_np + S_np))
        return error

    
    def fit_with_convergence(self, tol=None, max_iter=1000, iter_print=100):
        """
        Fit the model and track the convergence history of the error.

        Args:
            tol (float, optional): The tolerance for convergence. If None, it is set automatically.
            max_iter (int): The maximum number of iterations.
            iter_print (int): The interval at which to print progress.

        Returns:
            RPCAResult: An object containing the low-rank matrix, sparse matrix, and history of errors.
        """
        iter = 0
        err = np.inf
        history = []
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        _tol = tol if tol is not None else 1E-7 * self.frobenius_norm(self.D)

        while err > _tol and iter < max_iter:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(self.D - Lk + self.mu_inv * Yk, self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)

            err = self.frobenius_norm(self.D - Lk - Sk)
            history.append(err)
            iter += 1

            if iter % iter_print == 0 or iter == 1 or iter >= max_iter or err <= _tol:
                print(f'Iteration: {iter}, Error: {err:.6f}')

        self.L = Lk
        self.S = Sk

        return RPCAResult(L=Lk, S=Sk, history=history)
        

    def save_model(self, filename):
        """
        Save the model components L and S to a file.
    
        Args:
            filename (str): The name of the file to save the model.
        """
        np.savez(filename, L=self.L, S=self.S)

    def load_model(self, filename):
        """
        Load the model components L and S from a file.
    
        Args:
            filename (str): The name of the file to load the model from.
        """
        data = np.load(filename)
        self.L = data['L']
        self.S = data['S']


    def predict(self, D_new):
        """
        Predict the low-rank approximation for new data using the fitted model.
    
        Args:
            D_new (array-like): The new data matrix.
    
        Returns:
            array-like: The predicted low-rank matrix.
        """
        if USE_PYARROW:
            D_new_np = D_new.to_numpy()
        else:
            D_new_np = D_new
    
        L_new = self.svd_threshold(D_new_np, self.mu_inv)
    
        if USE_PYARROW:
            return pa.array(L_new)
        else:
            return L_new

    def set_mu(self, mu):
        """
        Set a new value for the mu hyperparameter.
    
        Args:
            mu (float): The new mu value.
        """
        self.mu = mu
        self.mu_inv = 1 / self.mu
    
    def set_lambda(self, lmbda):
        """
        Set a new value for the lambda hyperparameter.
    
        Args:
            lmbda (float): The new lambda value.
        """
        self.lmbda = lmbda

    def cross_validate(self, data_splits, mu_values, lmbda_values):
        """
        Perform cross-validation to find the best mu and lambda values.
    
        Args:
            data_splits (list of tuples): A list of (train, test) data splits.
            mu_values (list of floats): A list of mu values to try.
            lmbda_values (list of floats): A list of lambda values to try.
    
        Returns:
            tuple: The best mu and lambda values found during cross-validation.
        """
        best_mu = None
        best_lmbda = None
        best_score = np.inf
    
        for mu in mu_values:
            for lmbda in lmbda_values:
                scores = []
                for train, test in data_splits:
                    self.__init__(train, mu, lmbda)
                    self.fit()
                    score = self.reconstruction_error()
                    scores.append(score)
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_mu = mu
                    best_lmbda = lmbda
    
        return best_mu, best_lmbda

    def plot_singular_values(self):
        """
        Plot the singular values of the data matrix D.
    
        This plot helps in visualizing the rank structure of the matrix.
        """
        if USE_PYARROW:
            D_np = self.D.to_numpy()
        else:
            D_np = self.D
    
        U, S, V = np.linalg.svd(D_np, full_matrices=False)
        plt.plot(S, 'o-', label='Singular values')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Singular Values of the Data Matrix')
        plt.legend()
        plt.show()




    