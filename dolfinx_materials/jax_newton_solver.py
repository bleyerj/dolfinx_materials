#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole des Nationale des Ponts et Chaussées, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   21/08/2024
"""

import jax
import jax.numpy as jnp
from functools import partial
from jaxopt import GaussNewton
from typing import NamedTuple, Any


class SolverParameters(NamedTuple):
    """Parameters of Newton solver"""

    tol: float
    niter_max: int


class SolverState(NamedTuple):
    """Current state of NewtonSolver"""

    iter: int
    residual: Any
    residual_norm: float
    value: Any


def newton_solve(x, r, dr_dx, params):
    niter = 0

    res = r(x)
    norm_res0 = jnp.linalg.norm(res)

    @jax.jit
    def _solve_linear_system(x, b):
        if jnp.isscalar(x):
            return b / dr_dx(x)
        else:
            J = dr_dx(x)
            dx = jax.scipy.linalg.solve(J, b)
        return dx

    @jax.jit
    def cond_fun(state):
        norm_res, niter, _ = state
        return jnp.logical_and(
            norm_res > params.tol * norm_res0, niter < params.niter_max
        )

    @jax.jit
    def body_fun(state):
        norm_res, niter, history = state

        x, res = history

        dx = _solve_linear_system(x, -res)
        x += dx

        res = r(x)
        norm_res = jnp.linalg.norm(res)
        history = x, res
        niter += 1

        return (norm_res, niter, history)

    history = (x, res)

    norm_res, niter_total, history = jax.lax.while_loop(
        cond_fun, body_fun, (norm_res0, niter, history)
    )
    return history


class JAXNewton:
    """A tiny Newton solver implemented in JAX."""

    def __init__(self, r, dr_dx=None, tol=1e-8, niter_max=200):
        """Newton solver for a vector of residual r(x).

        Parameters
        ----------
        r : callable
            Residual to solve for. r(x) is a function of R^n to R^n, n>=1.
        dr_dx : callable, optional
            The jacobian of the residual. dr_dx(x) is a function of R^n to R^{n}xR^n. If None (default),
            JAX computes the residual using forward-mode automatic differentiation.
        tol :float, optional
            Relative tolerance for the Newton method, by default 1e-8
        niter_max : int, optional
            Maximum number of allowed iterations, by default 200
        """
        self.params = SolverParameters(tol, niter_max)
        # residual
        self.r = r
        if dr_dx is None:
            self.dr_dx = jax.jit(jax.jacfwd(r))
        else:
            self.dr_dx = dr_dx

    @property
    def jacobian(self):
        return self.dr_dx

    @partial(jax.jit, static_argnums=(0,))
    def solve(self, x):
        solve = lambda f, x: newton_solve(x, f, jax.jacfwd(f), self.params)[0]

        if jnp.ndim(x) == 0:
            tangent_solve = lambda g, y: y / g(1.0)
        else:
            tangent_solve = lambda g, y: jax.scipy.linalg.solve(jax.jacfwd(g)(y), y)
        x_sol = jax.lax.custom_root(self.r, x, solve, tangent_solve)
        return x_sol, self.r(x_sol)
