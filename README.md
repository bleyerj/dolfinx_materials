# `dolfinx_materials` : A Python package for advanced material modelling

Author: Jeremy Bleyer, jeremy.bleyer@enpc.fr, Laboratoire Navier, Ecole des Ponts ParisTech

## About

`dolfinx_materials` is a Python add-on package to the `dolfinx` interface to the [FEniCSx project](https://fenicsproject.org/).
It enables the user to define **complex material constitutive behaviours** which are not expressible using classical [UFL](https://fenics.readthedocs.io/projects/ufl/en/latest/) operators.

The library supports in particular:
- [MFront](https://tfel.sourceforge.net/) constitutive behaviours compiled with the `generic` interface, relying on the [MFrontGenericInterfaceSupport](https://github.com/thelfer/MFrontGenericInterfaceSupport) project
- Python-based constitutive relations, using `numpy/scipy` for instance
- constitutive relations based on inference of trained Neural Networks
- constitutive relations solved using external libraries including convex optimization libraries such as [cvxpy](http://cvxpy.org/)


