#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
QuadratureFunction class and utility functions

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   17/05/2023
"""
import ufl
import numpy as np
from dolfinx import fem
from .utils import project, get_function_space_type, create_quadrature_space


def create_quadrature_function(name, shape, mesh, quadrature_degree):
    function_space = create_quadrature_space(mesh, quadrature_degree, "vector", shape)
    return fem.Function(function_space, name=name)


class QuadratureExpression:
    def __init__(self, name, expression, mesh, quadrature_degree):
        self.ufl_shape = expression.ufl_expression.ufl_shape
        self.name = name
        self.expression = expression
        self.type, self.shape = get_function_space_type(expression.ufl_expression)
        self.initialize_function(mesh, quadrature_degree)

    def initialize_function(self, mesh, quadrature_degree):
        self.quadrature_degree = quadrature_degree
        self.mesh = mesh
        self._function_space = create_quadrature_space(
            self.mesh, self.quadrature_degree, self.type, self.shape
        )
        self.function = fem.Function(self._function_space, name=self.name)

    def eval(self, cells):
        expr_eval = self.expression.eval(cells)
        self.function.vector.array[:] = expr_eval.flatten()[:]

    def variation(self, u, v):
        deriv = sum(
            [
                ufl.derivative(self.expression.ufl_expression, var, v_)
                for (var, v_) in zip(ufl.split(u), ufl.split(v))
            ]
        )
        return ufl.algorithms.expand_derivatives(deriv)

    def set_values(self, x):
        self.function.vector.array[:] = x


# class QuadratureFunction:
#     """An abstract class for functions defined at quadrature points"""

#     def __init__(self, name, shape, hypothesis):
#         self.shape = shape
#         self.name = name
#         self.hypothesis = hypothesis

#     def initialize_function(self, mesh, quadrature_degree):
#         self.quadrature_degree = quadrature_degree
#         self.mesh = mesh
#         self.dx = Measure("dx", metadata={"quadrature_degree": quadrature_degree})
#         We = get_quadrature_element(mesh.ufl_cell(), quadrature_degree, self.shape)
#         self.function_space = FunctionSpace(mesh, We)
#         self.function = Function(self.function_space, name=self.name)

#     def update(self, x):
#         self.function.vector().set_local(x)
#         self.function.vector().apply("insert")

#     def project_on(self, space, degree, as_tensor=False, **kwargs):
#         """
#         Returns the projection on a standard CG/DG space
#         Parameters
#         ----------
#         space: str
#             FunctionSpace type ("CG", "DG",...)
#         degree: int
#             FunctionSpace degree
#         as_tensor: bool
#             Returned as a tensor if True, in vector notation otherwise
#         """
#         fun = self.function
#         if as_tensor:
#             fun = vector_to_tensor(self.function)
#             shape = ufl.shape(fun)
#             V = TensorFunctionSpace(self.mesh, space, degree, shape=shape)
#         elif self.shape == 1:
#             V = FunctionSpace(self.mesh, space, degree)
#         else:
#             V = VectorFunctionSpace(self.mesh, space, degree, dim=self.shape)
#         v = Function(V, name=self.name)
#         v.assign(
#             project(
#                 fun,
#                 V,
#                 form_compiler_parameters={"quadrature_degree": self.quadrature_degree},
#                 **kwargs
#             )
#         )
#         return v
