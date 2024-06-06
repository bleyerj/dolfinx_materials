# `dolfinx_materials` : A Python package for advanced material modelling

Author: Jeremy Bleyer, jeremy.bleyer@enpc.fr, Laboratoire Navier, Ecole des Ponts ParisTech

![Navier](https://navier-lab.fr/wp-content/uploads/2021/11/NAVIER-LOGO-COULEUR-RVB-ECRAN-72ppp_pour-site.png)


## About

`dolfinx_materials` is a Python add-on package to the `dolfinx` interface to the [FEniCSx project](https://fenicsproject.org/).
It enables the user to define **complex material constitutive behaviours** which are not expressible using classical [UFL](https://fenics.readthedocs.io/projects/ufl/en/latest/) operators.

Current version supports FEniCSx version `0.8.0`.

The library supports in particular:
- [MFront](https://tfel.sourceforge.net/) constitutive behaviours compiled with the `generic` interface, relying on the [MFrontGenericInterfaceSupport](https://github.com/thelfer/MFrontGenericInterfaceSupport) project
- Python-based constitutive relations, using `numpy/scipy` for instance
- constitutive relations based on inference of trained Neural Networks
- constitutive relations solved using external libraries including convex optimization libraries such as [cvxpy](http://cvxpy.org/)

## Extensibility

*Disclaimer: The following functionalities are not currently available but should in theory be possible to implement relatively easily. Contributions are most welcome!*

This library should also help you in:
- writing additional interface to other material libraries such as UMATs of Abaqus for instance
- performing multi-scale simulations (FE²) where constitutive update is obtained from the solution of a problem formulated on a RVE.
- implementing data-driven constitutive models

## Installation

Simply clone the [`dolfinx_materials` public repository](https://github.com/bleyerj/dolfinx_materials.git)

and install the package by typing

```bash
pip install dolfinx_materials/ --user
```

> Note: With the latest pip versions, you might need to add `--break-system-packages` to install it in your system Python environment. Or use `pipx`.