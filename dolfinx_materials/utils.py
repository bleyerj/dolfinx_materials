#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Utility functions.

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   15/05/2023
"""
from dolfinx import fem
import ufl
from petsc4py import PETSc


# The function performs a manual projection of an original_field function onto a target_field space
def project(
    original_field, target_field: fem.Function, dx: ufl.Measure = ufl.dx, bcs=[]
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
    """
    # Ensure we have a mesh and attach to measure
    V = target_field.function_space

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = fem.form(ufl.inner(Pv, w) * dx)
    L = fem.form(ufl.inner(original_field, w) * dx)

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
    solver.solve(b, target_field.vector)
    target_field.x.scatter_forward()


def get_function_space_type(x):
    shape = x.ufl_shape
    if len(shape) == 0:
        return ("scalar", None)
    elif len(shape) == 1:
        return ("vector", shape[0])
    elif len(shape) == 2:
        return ("tensor", shape)
    else:
        raise NotImplementedError


function_space_dict = {
    "scalar": ufl.FiniteElement,
    "vector": ufl.VectorElement,
    "tensor": ufl.TensorElement,
}


def to_mat(array):
    M = ufl.as_matrix(array)
    shape = ufl.shape(M)
    if shape[0] == 1:
        return ufl.as_vector(array[0])
    elif shape[1] == 1:
        return ufl.as_vector([a[0] for a in array])
    else:
        return M


def create_quadrature_space(mesh, degree, type, shape):
    We = function_space_dict[type](
        "Quadrature", mesh.ufl_cell(), degree, shape, quad_scheme="default"
    )
    return fem.FunctionSpace(mesh, We)


def create_scalar_quadrature_space(mesh, degree):
    return create_vector_quadrature_space(mesh, degree, 1)


def create_vector_quadrature_space(mesh, degree, dim):
    if dim > 0:
        We = ufl.VectorElement(
            "Quadrature",
            mesh.ufl_cell(),
            degree=degree,
            dim=dim,
            quad_scheme="default",
        )
        return fem.FunctionSpace(mesh, We)
    else:
        raise ValueError("Vector dimension should be at least 1.")


def create_tensor_quadrature_space(mesh, degree, shape):
    We = ufl.TensorElement(
        "Quadrature",
        mesh.ufl_cell(),
        degree=degree,
        shape=shape,
        quad_scheme="default",
    )

    return fem.FunctionSpace(mesh, We)


def get_vals(fun):
    """Get values of a function in reshaped form N x dim where dim is the function space dimension"""
    dim = len(fun)
    return fun.vector.array.reshape((-1, dim))


def update_vals(fun, array):
    fun.vector.array[:] = array.ravel()
