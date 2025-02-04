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


def _initial_check(norm_res, x, tol):
    if norm_res > tol or jnp.isnan(norm_res):
        raise ValueError(f"{x}, norm = {norm_res}")


def _inside_check(norm_res, x, dx, r, dr_dx):
    if jnp.isnan(norm_res):
        raise ValueError(f"x={x} dx={dx} r={r} dr_dx={dr_dx}, norm = {norm_res}")


def _final_check(norm_res, cond):
    if cond:
        _convergence_info(norm_res, "Residual has not converged")


def _convergence_info(norm_res, tol, string):
    if norm_res > tol or jnp.isnan(norm_res):
        raise ValueError(f"{string}, norm = {norm_res}")


def _solve_linear_system(x, J, b):
    if jnp.isscalar(x):
        return b / J
    else:
        dx = jnp.linalg.solve(J, b)
    return dx


def newton_solve(x, r, dr_dx, params):
    def run_newton(x, params):
        niter = 0

        res = r(x)
        norm_res0 = jnp.linalg.norm(res)
        # jax.debug.callback(_initial_check, norm_res0, res, params.atol)

        def cond_fun(state):
            norm_res, niter, _ = state
            return jnp.logical_and(
                jnp.logical_and(
                    norm_res > params.atol, norm_res > params.rtol * norm_res0
                ),
                niter < params.niter_max,
            )

        def body_fun(state):
            norm_res, niter, history = state

            x, res = history

            dx = _solve_linear_system(x, dr_dx(x), -res)
            x += dx

            res = r(x)
            norm_res = jnp.linalg.norm(res)

            # jax.debug.callback(_inside_check, norm_res, x, dx, res, dr_dx(x))

            history = x, res
            niter += 1

            return (norm_res, niter, history)

        history = (x, res)

        state = jax.lax.while_loop(cond_fun, body_fun, (norm_res0, niter, history))
        norm_res, niter_total, history = state
        # jax.debug.callback(_final_check, norm_res, cond_fun(state))
        x_sol, res_sol = history
        data = (niter_total, norm_res0, norm_res, res_sol)
        return x_sol, data

    x_sol, data = run_newton(x, params)
    return x_sol, data


class JAXNewton:
    """A tiny Newton solver implemented in JAX. Derivatives are computed via custom implicit differentiation."""

    def __init__(self, rtol=1e-8, atol=1e-8, niter_max=2000):
        """Newton solver

        Parameters
        ----------
        rtol : float, optional
            Relative tolerance for the Newton method, by default 1e-8
        atol : float, optional
            Absolute tolerance for the Newton method, by default 1e-8
        niter_max : int, optional
            Maximum number of allowed iterations, by default 200
        """
        self.params = SolverParameters(rtol, atol, niter_max)

    def set_residual(self, r, dr_dx=None):
        """Set the residual  vector r(x) and its jacobian

        Parameters
        ----------
        r : callable, list, tuple
            Residual to solve for. r(x) is a function of R^n to R^n. Alternatively, r can be a list/tuple
            of functions with the same signature. The resulting system corresponds to a concatenation of all
            individual residuals.
        dr_dx : callable, optional
            The jacobian of the residual. dr_dx(x) is a function of R^n to R^{n}xR^n. If None (default),
            JAX computes the residual using forward-mode automatic differentiation.
        """
        # residual
        if isinstance(r, list) or isinstance(r, tuple):
            self.r = lambda x: jnp.concatenate([jnp.atleast_1d(ri(x)) for ri in r])
        else:
            self.r = r
        if dr_dx is None:
            self.dr_dx = jax.jacfwd(self.r)
        else:
            self.dr_dx = dr_dx

    @partial(jax.jit, static_argnums=(0,))
    def solve(self, x):
        solve = lambda f, x: newton_solve(x, f, jax.jacfwd(f), self.params)

        tangent_solve = lambda g, y: _solve_linear_system(x, jax.jacfwd(g)(y), y)
        x_sol, data = jax.lax.custom_root(self.r, x, solve, tangent_solve, has_aux=True)

        # x_sol, data = newton_solve(x, self.r, self.dr_dx, self.params)
        return x_sol, data
