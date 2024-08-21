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
    return sum(x[:3])


def dev(x):
    return K @ x


def to_mat(x):
    return jnp.array(
        [
            [x[0], x[3] / jnp.sqrt(2), x[4] / jnp.sqrt(2)],
            [x[3] / jnp.sqrt(2), x[1], x[5] / jnp.sqrt(2)],
            [x[4] / jnp.sqrt(2), x[5] / jnp.sqrt(2), x[2]],
        ]
    )
