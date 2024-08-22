# `dolfinx_materials` : A Python package for advanced material modelling


`dolfinx_materials` is a Python add-on package to the `dolfinx` interface to the [FEniCSx project](https://fenicsproject.org/).
It enables the user to define **complex material constitutive behaviors** which are not expressible using classical [UFL](https://fenics.readthedocs.io/projects/ufl/en/latest/) operators.


* Github repository: https://github.com/bleyerj/dolfinx_materials
* Online documentation: https://bleyerj.github.io/dolfinx_materials/

## Features

The library supports in particular:

- [JAX](https://jax.readthedocs.io)-based implementations of constitutive relations
- Python-based constitutive relations, using `numpy/scipy` (*slow*)
- [MFront](https://tfel.sourceforge.net/) constitutive behaviors compiled with the `generic` interface, relying on the [MFrontGenericInterfaceSupport](https://github.com/thelfer/MFrontGenericInterfaceSupport) project
- constitutive relations based on inference of trained Neural Networks
- constitutive relations solved using external libraries including convex optimization libraries such as [cvxpy](http://cvxpy.org/)

*Disclaimer: The following functionalities are not currently available but should in theory be possible to implement relatively easily. Contributions are most welcome!*

This library should also help you in:

- writing additional interface to other material libraries such as UMATs of Abaqus for instance
- performing multi-scale simulations (FE²) where constitutive update is obtained from the solution of a problem formulated on a RVE.
- implementing data-driven constitutive models

## Prerequisites
**dolfinx_materials** requires: 
* **FEniCSx** (v.0.8), see [installation instructions here](https://fenicsproject.org/download/).
* **jax** for  JAX-based materials. JAX can be simply installed via `pip`:

```
pip install jax --user
```
See [JAX installation instructions](https://jax.readthedocs.io/en/latest/installation.html) for more details about GPU acceleration.

### Optional for MFront behaviors

- You must install the [TFEL project](https://github.com/thelfer/tfel) to be able to compile MFront behaviors (`.mfront` files). In particular, the project requires Boost and Boost-Python libraries.

- The [MFrontGenericInterfaceSupport](https://github.com/thelfer/MFrontGenericInterfaceSupport/) (`mgis`) package with Python binding must then be installed to load such compiled behaviors within `dolfinx`.

## Installation and usage
Simply clone the [`dolfinx_materials` public repository](https://github.com/bleyerj/dolfinx_materials)
```
https://github.com/bleyerj/dolfinx_materials
```
and install the package by typing
```
pip install dolfinx_materials/ --user
```


## License

All this work is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/>) ![license](https://i.creativecommons.org/l/by-sa/4.0/88x31.png).

## Citing

Papers related to the MFront and MGIS projects can be cited as:

[![DOI](https://joss.theoj.org/papers/10.21105/joss.02003/status.svg)](https://doi.org/10.21105/joss.02003)
```
@article{helfer2020mfrontgenericinterfacesupport,
  title={The MFrontGenericInterfaceSupport project},
  author={Helfer, Thomas and Bleyer, Jeremy and Frondelius, Tero and Yashchuk, Ivan and Nagel, Thomas and Naumov, Dmitri},
  journal={Journal of Open Source Software},
  volume={5},
  number={48},
  pages={1--8},
  year={2020},
  publisher={Open Journals}
}

@article{helfer2015introducing,
  title={Introducing the open-source mfront code generator: Application to mechanical behaviours and material knowledge management within the PLEIADES fuel element modelling platform},
  author={Helfer, Thomas and Michel, Bruno and Proix, Jean-Michel and Salvo, Maxime and Sercombe, J{\'e}r{\^o}me and Casella, Michel},
  journal={Computers \& Mathematics with Applications},
  volume={70},
  number={5},
  pages={994--1023},
  year={2015},
  publisher={Elsevier}
}
```



## About the author

[Jeremy Bleyer](https://bleyerj.github.io/) is a researcher in Solid and Structural Mechanics at [Laboratoire Navier](https://navier-lab.fr), a joint research  (UMR 8205) of [Ecole Nationale des Ponts et Chaussées](http://www.enpc.fr),
[Université Gustave Eiffel](https://www.univ-gustave-eiffel.fr/) and [CNRS](http://www.cnrs.fr).

[{fas}`at` jeremy.bleyer@enpc.fr](mailto:jeremy.bleyer@enpc.fr)

[{fab}`linkedin` jeremy-bleyer](http://www.linkedin.com/in/jérémy-bleyer-0aabb531)

<a href="https://orcid.org/0000-0001-8212-9921">
<img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_32x32.png" width="16" height="16" />
 0000-0001-8212-9921
</a>
