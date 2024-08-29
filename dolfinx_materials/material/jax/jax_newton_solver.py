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
from typing import NamedTuple


class SolverParameters(NamedTuple):
    """Parameters of Newton solver"""

    rtol: float
    atol: float
    niter_max: int


@jax.jit
def _solve_linear_system(x, J, b):
    if jnp.isscalar(x):
        return b / J
    else:
        dx = jnp.linalg.solve(J, b)
    return dx


def newton_solve(x, r, dr_dx, params):
    niter = 0

    res = r(x)
    norm_res0 = jnp.linalg.norm(res)

    @jax.jit
    def cond_fun(state):
        norm_res, niter, _ = state
        return jnp.logical_and(
            jnp.logical_and(norm_res > params.atol, norm_res > params.rtol * norm_res0),
            niter < params.niter_max,
        )

    @jax.jit
    def body_fun(state):
        norm_res, niter, history = state

        x, res = history

        dx = _solve_linear_system(x, dr_dx(x), -res)
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
    x_sol, res_sol = history
    data = (niter_total, norm_res, res_sol)
    return x_sol, data


class JAXNewton:
    """A tiny Newton solver implemented in JAX. Derivatives are computed via custom implicit differentiation."""

    def __init__(self, r, dr_dx=None, rtol=1e-8, atol=1e-8, niter_max=2000):
        """Newton solver for a vector of residual r(x).

        Parameters
        ----------
        r : callable, list, tuple
            Residual to solve for. r(x) is a function of R^n to R^n. Alternatively, r can be a list/tuple
            of functions with the same signature. The resulting system corresponds to a concatenation of all
            individual residuals.
        dr_dx : callable, optional
            The jacobian of the residual. dr_dx(x) is a function of R^n to R^{n}xR^n. If None (default),
            JAX computes the residual using forward-mode automatic differentiation.
        rtol : float, optional
            Relative tolerance for the Newton method, by default 1e-8
        atol : float, optional
            Absolute tolerance for the Newton method, by default 1e-8
        niter_max : int, optional
            Maximum number of allowed iterations, by default 200
        """
        self.params = SolverParameters(rtol, atol, niter_max)
        # residual
        if isinstance(r, list) or isinstance(r, tuple):
            self.r = lambda x: jnp.concatenate([jnp.atleast_1d(ri(x)) for ri in r])
        else:
            self.r = r
        if dr_dx is None:
            self.dr_dx = jax.jit(jax.jacfwd(self.r))
        else:
            self.dr_dx = dr_dx

    @partial(jax.jit, static_argnums=(0,))
    def solve(self, x):
        solve = lambda f, x: newton_solve(x, f, jax.jacfwd(f), self.params)

        tangent_solve = lambda g, y: _solve_linear_system(x, jax.jacfwd(g)(y), y)
        x_sol, data = jax.lax.custom_root(self.r, x, solve, tangent_solve, has_aux=True)
        return x_sol, data
