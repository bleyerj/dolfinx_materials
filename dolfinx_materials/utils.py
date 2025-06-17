#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Utility functions.

@Author  :   Jeremy Bleyer, Ecole Nationale des Ponts et Chaussées, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   15/05/2023
"""
from dolfinx import fem
import ufl
from petsc4py import PETSc
import numpy as np
from dolfinx.common import Timer
from functools import lru_cache
import basix


# The function performs a manual projection of an original_field function onto a target_field space
def project(
    original_field,
    target_field: fem.Function,
    dx: ufl.Measure = ufl.dx,
    bcs=[],
    smooth=None,
    entity_maps=None,
):
    """Performs a manual projection of an original function onto a target space.

    Parameters
    ----------
    original_field : fem.Function/fem.Expression
        Original function/expression to project.
    target_field : fem.Function
        Receiver function.
    dx : ufl.Measure, optional
        Volume measure used for projection, by default ufl.dx
    bcs : list, optional
        Boundary conditions, by default []
    smooth : float, optional
        Performs a smoothed projection with a Helmholtz filter of size smooth
    entity_maps: dict
        Entity maps in case of mixed subdomains
    """
    # Ensure we have a mesh and attach to measure
    V = target_field.function_space

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = ufl.inner(Pv, w) * dx
    if smooth is not None:
        a += smooth**2 * ufl.inner(ufl.grad(Pv), ufl.grad(w)) * dx
    a = fem.form(a, entity_maps=entity_maps)
    L = fem.form(ufl.inner(original_field, w) * dx, entity_maps=entity_maps)

    # Assemble linear system
    A = fem.petsc.assemble_matrix(a, bcs)
    A.assemble()
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_field.x.petsc_vec)
    target_field.x.scatter_forward()


def to_mat(array):
    M = ufl.as_matrix(array)
    shape = ufl.shape(M)
    if shape == (1, 1):
        return M[0, 0]
    elif shape[0] == 1:
        return ufl.as_vector(array[0])
    elif shape[1] == 1:
        return ufl.as_vector([a[0] for a in array])
    else:
        return M


def create_quadrature_functionspace(domain, deg_quad, shape):
    if shape == 0:
        shape = ()
    elif isinstance(shape, int) or isinstance(shape, np.integer):
        shape = (shape,)
    We = basix.ufl.quadrature_element(
        domain.topology.cell_name(),
        value_shape=shape,
        scheme="default",
        degree=deg_quad,
    )
    return fem.functionspace(domain, We)


def get_vals(fun):
    """Get values of a function in reshaped form N x dim where dim is the function space dimension"""
    if ufl.shape(fun) == ():
        dim = 1
    else:
        dim = len(fun)
    return fun.x.array.reshape((-1, dim))


def cell_to_dofs(cells, V):
    with Timer("dx_mat:cell_to_dofs"):
        dofs = fem.locate_dofs_topological(V, V.mesh.geometry.dim, cells)
        block_size = V.dofmap.bs
        return cell_to_dofs_cached(tuple(dofs), block_size)


def cacheRef(f):
    cache = {}

    def g(*args):
        # use `id` to get memory address for function argument.
        cache_key = "-".join(list(map(lambda e: str(id(e)), args)))
        if cache_key in cache:
            return cache[cache_key]
        v = f(*args)
        cache[cache_key] = v
        return v

    return g


@lru_cache
def cell_to_dofs_cached(dofs, block_size):
    return np.asarray(
        np.kron(np.asarray(dofs), block_size * np.ones(block_size))
        + np.kron(np.ones(len(dofs)), np.arange(0, block_size)),
        dtype=np.int32,
    )


def update_vals(fun, array, cells=None):
    if cells is None:
        fun.x.array[:] = array.ravel()
    else:
        dofs = cell_to_dofs(cells, fun.function_space)
        fun.x.array[dofs] = array.ravel()


def symmetric_tensor_to_vector(T, T22=0):
    """Return symmetric tensor components in vector form notation following MFront conventions
    T22 can be specified when T is only (2,2)"""
    if ufl.shape(T) == (2, 2):
        return ufl.as_vector([T[0, 0], T[1, 1], T22, np.sqrt(2) * T[0, 1]])
    elif ufl.shape(T) == (3, 3):
        return ufl.as_vector(
            [
                T[0, 0],
                T[1, 1],
                T[2, 2],
                np.sqrt(2) * T[0, 1],
                np.sqrt(2) * T[0, 2],
                np.sqrt(2) * T[1, 2],
            ]
        )
    elif len(ufl.shape(T)) == 1:
        return T
    else:
        raise NotImplementedError


def nonsymmetric_tensor_to_vector(T, T22=0):
    """Return nonsymmetric tensor components in vector form notation following MFront conventions
    T22 can be specified when T is only (2,2)"""
    if ufl.shape(T) == (2, 2):
        return ufl.as_vector([T[0, 0], T[1, 1], T22, T[0, 1], T[1, 0]])
    elif ufl.shape(T) == (3, 3):
        return ufl.as_vector(
            [
                T[0, 0],
                T[1, 1],
                T[2, 2],
                T[0, 1],
                T[1, 0],
                T[0, 2],
                T[2, 0],
                T[1, 2],
                T[2, 1],
            ]
        )
    elif len(ufl.shape(T)) == 1:
        return T
    else:
        raise NotImplementedError


def vector_to_tensor(T):
    """Return vector following MFront conventions as a tensor"""
    if ufl.shape(T) == (4,):
        return ufl.as_matrix([[T[0], T[3] / np.sqrt(2)], T[3] / np.sqrt(2), T[1]])
    elif ufl.shape(T) == (6,):
        return ufl.as_matrix(
            [
                [T[0], T[3] / np.sqrt(2), T[4] / np.sqrt(2)],
                [T[3] / np.sqrt(2), T[1], T[5] / np.sqrt(2)],
                [T[4] / np.sqrt(2), T[5] / np.sqrt(2), T[2]],
            ]
        )
    elif ufl.shape(T) == (5,):
        return ufl.as_matrix([[T[0], T[3]], T[4], T[1]])
    elif ufl.shape(T) == (9,):
        return ufl.as_matrix(
            [[T[0], T[3], T[5]], [T[4], T[1], T[7]], [T[6], T[8], T[2]]]
        )
    else:
        raise NotImplementedError


def axi_grad(r, v):
    """
    Axisymmetric gradient in cylindrical coordinate (er, etheta, ez) for:
    * a scalar v(r, z)
    * a 2d-vectorial (vr(r,z), vz(r, z))
    * a 3d-vectorial (vr(r,z), 0, vz(r, z))
    """
    if ufl.shape(v) == (3,):
        return ufl.as_matrix(
            [
                [v[0].dx(0), -v[1] / r, v[0].dx(1)],
                [v[1].dx(0), v[0] / r, v[1].dx(1)],
                [v[2].dx(0), 0, v[2].dx(1)],
            ]
        )
    elif ufl.shape(v) == (2,):
        return ufl.as_matrix(
            [[v[0].dx(0), 0, v[0].dx(1)], [0, v[0] / r, 0], [v[1].dx(0), 0, v[1].dx(1)]]
        )
    elif ufl.shape(v) == ():
        return ufl.as_vector([v.dx(0), 0, v.dx(1)])
    else:
        raise NotImplementedError


def grad_3d(u):
    return ufl.as_matrix(
        [[u[0].dx(0), u[0].dx(1), 0], [u[1].dx(0), u[1].dx(1), 0], [0, 0, 0]]
    )


def symmetric_gradient(g):
    """Return symmetric gradient components in vector form"""
    return symmetric_tensor_to_vector(ufl.sym(g))


def transformation_gradient(g, dim=3):
    """Return transformation gradient components in vector form"""
    return nonsymmetric_tensor_to_vector(ufl.Identity(dim) + g, T22=1)


def gradient(g):
    """Return displacement gradient components in vector form"""
    return nonsymmetric_tensor_to_vector(g)
