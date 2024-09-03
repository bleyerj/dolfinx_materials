import jax.numpy as jnp
import scipy.linalg as sl


def Identity(d):
    """2nd rank identity tensor of dimension d"""
    return jnp.eye(d * (d + 1) // 2)


# Projector onto isotropic space
A = 1 / 3.0 * jnp.ones((3, 3))
J = sl.block_diag(A, jnp.zeros((3, 3)))

# Projector onto deviatoric space
B = jnp.array(
    [
        [2 / 3.0, -1 / 3.0, -1 / 3.0],
        [-1 / 3.0, 2 / 3.0, -1 / 3.0],
        [-1 / 3.0, -1 / 3.0, 2 / 3.0],
    ]
)
K = sl.block_diag(B, jnp.eye(3))


def tr(x):
    if len(x.shape) == 1:
        return sum(x[:3])
    else:
        return x[0, 0] + x[1, 1] + x[2, 2]


def dev(x):
    if len(x.shape) == 1:
        return K @ x
    else:
        return x - tr(x) / 3 * jnp.eye(3)


def to_mat(x):
    if len(x) == 6:
        return jnp.array(
            [
                [x[0], x[3] / jnp.sqrt(2), x[4] / jnp.sqrt(2)],
                [x[3] / jnp.sqrt(2), x[1], x[5] / jnp.sqrt(2)],
                [x[4] / jnp.sqrt(2), x[5] / jnp.sqrt(2), x[2]],
            ]
        )
    else:
        return jnp.array(
            [
                [x[0], x[3], x[5]],
                [x[4], x[1], x[7]],
                [x[6], x[8], x[2]],
            ]
        )


def to_vect(X, symmetry=False):
    if symmetry:
        return jnp.array(
            [
                X[0, 0],
                X[1, 1],
                X[2, 2],
                jnp.sqrt(2) * X[0, 1],
                jnp.sqrt(2) * X[0, 2],
                jnp.sqrt(2) * X[1, 2],
            ]
        )
    else:
        return jnp.array(
            [
                X[0, 0],
                X[1, 1],
                X[2, 2],
                X[0, 1],
                X[1, 0],
                X[0, 2],
                X[2, 0],
                X[1, 2],
                X[2, 1],
            ]
        )


def transpose(X):
    symmetric = len(X) == 6
    if not symmetric:
        return jnp.array(
            [
                X[0, 0],
                X[1, 1],
                X[2, 2],
                X[1, 0],
                X[0, 1],
                X[2, 0],
                X[0, 2],
                X[2, 1],
                X[1, 2],
            ]
        )
    else:
        return X


def det(X):
    if len(X.shape) == 1:
        symmetric = len(X) == 6
        return jnp.linalg.det(to_mat(X, symmetric))
    else:
        return jnp.linalg.det(X)


def inv(X):
    symmetric = len(X) == 6
    return to_vect(jnp.linalg.inv(to_mat(X)), symmetric)


def dot(A, B):
    symmetric_A = len(A) == 6
    symmetric_B = len(B) == 6
    assert (
        symmetric_A == symmetric_B
    ), "Tensors should be both symmetric or non-symmetric"
    return to_vect(to_mat(transpose(A)) @ to_mat(B), symmetric_A)
