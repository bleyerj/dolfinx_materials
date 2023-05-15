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
