#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
QuadratureFunction class and utility functions

@Author  :   Jeremy Bleyer, Ecole Nationale des Ponts et Chaussées, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   17/05/2023
"""
import ufl
import numpy as np
from dolfinx import fem
from dolfinx.common import Timer
from .utils import build_cell_to_dofs_map, create_quadrature_functionspace


def create_quadrature_function(name, shape, mesh, quadrature_degree):
    if shape in [0, 1]:
        shape = ()
    function_space = create_quadrature_functionspace(mesh, quadrature_degree, shape)
    return fem.Function(function_space, name=name)


class QuadratureExpression:
    def __init__(self, name, expression, mesh, quadrature_degree):
        self.ufl_shape = expression.ufl_expression.ufl_shape
        self.name = name
        self.expression = expression
        self.shape = expression.ufl_expression.ufl_shape
        self.initialize_function(mesh, quadrature_degree)

        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        self._mesh_cells = np.arange(0, num_cells, dtype=np.int32)
        self.total_dofs = build_cell_to_dofs_map(self._function_space)

    def initialize_function(self, mesh, quadrature_degree):
        self.quadrature_degree = quadrature_degree
        self.mesh = mesh
        self._function_space = create_quadrature_functionspace(
            self.mesh, self.quadrature_degree, self.shape
        )
        self.function = fem.Function(self._function_space, name=self.name)

    def eval(self, cells):
        if cells is None:
            cells = self._mesh_cells
        with Timer("dx_mat:Function eval"):
            expr_eval = self.expression.eval(self.mesh, cells)
        with Timer("dx_mat:Prepare dofs"):
            dofs = self.total_dofs[cells].ravel()
        self.function.x.array[dofs] = expr_eval.flatten()[:]

    def variation(self, u, v):
        deriv = sum(
            [
                ufl.derivative(self.expression.ufl_expression, var, v_)
                for (var, v_) in zip(ufl.split(u), ufl.split(v))
            ]
        )
        return ufl.algorithms.expand_derivatives(deriv)

    def set_values(self, x):
        self.function.x.array[:] = x
