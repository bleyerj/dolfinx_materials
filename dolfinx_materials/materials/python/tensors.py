import numpy as np
import scipy.linalg as sl


def Identity(d):
    """2nd rank identity tensor of dimension d"""
    return np.eye(d * (d + 1) // 2)


def J():
    A = 1 / 3.0 * np.ones((3, 3))
    return sl.block_diag(A, np.zeros((3, 3)))


def K():
    A = np.array(
        [
            [2 / 3.0, -1 / 3.0, -1 / 3.0],
            [-1 / 3.0, 2 / 3.0, -1 / 3.0],
            [-1 / 3.0, -1 / 3.0, 2 / 3.0],
        ]
    )
    return sl.block_diag(A, np.eye(3))


def vectorized_outer(A, B):
    """Outer product on the first axis, applied elementwise on the second axis.
    A x B: (ik) x (jk) = ijk
    """
    return np.einsum("ik, jk -> ijk", A, B)


def expand_size(A, batch_size):
    """Expands by repeating A along a new axis in last position of length batch_size"""
    return np.tile(A, [1 for i in A.shape] + [batch_size])
