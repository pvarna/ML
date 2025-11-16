import unittest
import numpy as np

from kernels import linear, polynomial, rbf, sigmoid


class TestLinearKernel(unittest.TestCase):

    def test_when_feature_sizes_differ_then_throws_runtime_error(self):
        # Arrange
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = np.array([[1, 2], [3, 4]])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            linear(X, Y)

    def test_when_known_small_matrices_then_returns_expected_kernel_matrix(
            self):
        # Arrange
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[5, 6], [7, 8]])
        expected = np.array([[1 * 5 + 2 * 6, 1 * 7 + 2 * 8],
                             [3 * 5 + 4 * 6, 3 * 7 + 4 * 8]])

        # Act
        actual = linear(X, Y)

        # Assert
        np.testing.assert_array_equal(actual, expected)

    def test_when_same_object_then_returns_gram_matrix(self):
        # Arrange
        X = np.array([[1, 0], [0, 1]])
        expected = np.array([[1, 0], [0, 1]])

        # Act
        actual = linear(X, X)

        # Assert
        np.testing.assert_array_equal(actual, expected)

    def test_when_empty_matrices_then_returns_empty_kernel_matrix(self):
        # Arrange
        X = np.empty((0, 3))
        Y = np.empty((0, 3))

        # Act
        actual = linear(X, Y)

        # Assert
        self.assertEqual(actual.shape, (0, 0))


class TestPolynomialKernel(unittest.TestCase):

    def test_when_feature_sizes_differ_then_throws_runtime_error(self):
        # Arrange
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = np.array([[1, 2], [3, 4]])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            polynomial(X, Y)

    def test_when_known_small_matrices_default_params_then_expected_values(
            self):
        # Arrange
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[1, 2], [3, 4]])

        dot = X @ Y.T
        gamma = 1.0 / X.shape[1]
        expected = (gamma * dot + 1.0)**3

        # Act
        actual = polynomial(X, Y)

        # Assert
        np.testing.assert_allclose(actual, expected)

    def test_when_custom_params_then_expected_values(self):
        # Arrange
        X = np.array([[1, 0], [0, 1]])
        Y = np.array([[2, 3], [4, 5]])
        degree = 2
        gamma = 2.0
        coef0 = 0.5
        dot = X @ Y.T
        expected = (gamma * dot + coef0)**degree

        # Act
        actual = polynomial(X, Y, degree=degree, gamma=gamma, coef0=coef0)

        # Assert
        np.testing.assert_allclose(actual, expected)

    def test_when_empty_matrices_then_returns_empty_kernel_matrix(self):
        # Arrange
        X = np.empty((0, 3))
        Y = np.empty((0, 3))

        # Act
        actual = polynomial(X, Y)

        # Assert
        self.assertEqual(actual.shape, (0, 0))


class TestRBFKernel(unittest.TestCase):

    def test_when_feature_sizes_differ_then_throws_runtime_error(self):
        # Arrange
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = np.array([[1, 2], [3, 4]])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            rbf(X, Y)

    def test_when_known_small_vectors_custom_gamma_then_expected_values(self):
        # Arrange
        X = np.array([[0, 0], [1, 0]])
        Y = np.array([[0, 0], [2, 0]])
        gamma = 0.5
        expected = np.array([[1.0, np.exp(-2.0)], [np.exp(-0.5),
                                                   np.exp(-0.5)]])

        # Act
        actual = rbf(X, Y, gamma=gamma)

        # Assert
        np.testing.assert_allclose(actual, expected)

    def test_when_same_object_then_matrix_is_symmetric_and_diagonal_ones(self):
        # Arrange
        X = np.array([[0, 0], [1, 1], [2, 0]])

        # Act
        K = rbf(X, X, gamma=0.5)

        np.testing.assert_allclose(K, K.T)
        np.testing.assert_allclose(np.diag(K), np.ones(X.shape[0]))

    def test_when_empty_matrices_then_returns_empty_kernel_matrix(self):
        # Arrange
        X = np.empty((0, 3))
        Y = np.empty((0, 3))

        # Act
        actual = rbf(X, Y)

        # Assert
        self.assertEqual(actual.shape, (0, 0))


class TestSigmoidKernel(unittest.TestCase):

    def test_when_feature_sizes_differ_then_throws_runtime_error(self):
        # Arrange
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = np.array([[1, 2], [3, 4]])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            sigmoid(X, Y)

    def test_when_known_small_matrices_default_params_then_expected_values(self):
        # Arrange
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[5, 6], [7, 8]])
        gamma = 1.0 / X.shape[1]
        coef0 = 1.0
        expected = np.tanh(gamma * (X @ Y.T) + coef0)

        # Act
        actual = sigmoid(X, Y)

        # Assert
        np.testing.assert_allclose(actual, expected)

    def test_when_custom_params_then_expected_values(self):
        # Arrange
        X = np.array([[1, 0], [0, 1]])
        Y = np.array([[2, 3], [4, 5]])
        gamma = 0.5
        coef0 = -0.2
        expected = np.tanh(gamma * (X @ Y.T) + coef0)

        # Act
        actual = sigmoid(X, Y, gamma=gamma, coef0=coef0)

        # Assert
        np.testing.assert_allclose(actual, expected)

    def test_when_same_object_then_kernel_is_symmetric(self):
        # Arrange
        X = np.array([[1, 2], [3, 4], [5, 6]])
        gamma = 1.0 / X.shape[1]
        coef0 = 0.75

        # Act
        K = sigmoid(X, X, gamma=gamma, coef0=coef0)

        # Assert
        np.testing.assert_allclose(K, K.T)
        diag_expected = np.tanh(gamma * np.sum(X * X, axis=1) + coef0)
        np.testing.assert_allclose(np.diag(K), diag_expected)

    def test_when_empty_matrices_then_returns_empty_kernel_matrix(self):
        # Arrange
        X = np.empty((0, 3))
        Y = np.empty((0, 3))

        # Act
        actual = sigmoid(X, Y)

        # Assert
        self.assertEqual(actual.shape, (0, 0))
