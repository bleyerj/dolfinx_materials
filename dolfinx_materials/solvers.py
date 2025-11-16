from typing import Sequence, Callable
from functools import partial
from mpi4py import MPI
from petsc4py import PETSc

import ufl
import dolfinx

from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.petsc import (
    assign,
    assemble_vector,
    apply_lifting,
    set_bc,
    NonlinearProblem,
    _assign_block_data,
    _bcs_by_block,
    _extract_function_spaces,
)
from dolfinx.fem.forms import Form
from dolfinx.fem.function import Function as _Function
from dolfinx.mesh import EntityMap as _EntityMap
from dolfinx_materials.quadrature_map import QuadratureMap

from dolfinx.common import Timer
from dolfinx.mesh import EntityMap as _EntityMap

from dolfinx_materials.quadrature_map import QuadratureMap


def _assemble_residual(
    external_callback: Callable,
    u: _Function | Sequence[_Function],
    residual: Form | Sequence[Form],
    jacobian: Form | Sequence[Sequence[Form]],
    bcs: Sequence[DirichletBC],
    _snes: PETSc.SNES,  # type: ignore[name-defined]
    x: PETSc.Vec,  # type: ignore[name-defined]
    b: PETSc.Vec,  # type: ignore[name-defined]
):
    """Assemble the residual at ``x`` into the vector ``b``.

    A function conforming to the interface expected by ``SNES.setFunction``
    can be created by fixing the first four arguments, e.g.:

    Example::

        snes = PETSc.SNES().create(mesh.comm)
        assemble_residual = functools.partial(
            dolfinx.fem.petsc.assemble_residual,
            u, residual, jacobian, bcs)
        snes.setFunction(assemble_residual, b)

    Args:
        u: Function(s) tied to the solution vector within the residual and
           Jacobian.
        residual: Form of the residual. It can be a sequence of forms.
        jacobian: Form of the Jacobian. It can be a nested sequence of
            forms.
        bcs: List of Dirichlet boundary conditions to lift the residual.
        _snes: The solver instance.
        x: The vector containing the point to evaluate the residual at.
        b: Vector to assemble the residual into.
    """
    # Update input vector before assigning
    dolfinx.la.petsc._ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[attr-defined]

    # Assign the input vector to the unknowns
    assign(x, u)

    # Call external functions, e.g. evaluation of external operators
    external_callback()

    # Assign block data if block assembly is requested
    if isinstance(residual, Sequence) and b.getType() != PETSc.Vec.Type.NEST:  # type: ignore[attr-defined]
        _assign_block_data(residual, b)
        _assign_block_data(residual, x)

    # Assemble the residual
    dolfinx.la.petsc._zero_vector(b)
    assemble_vector(b, residual)  # type: ignore[arg-type]

    # Lift vector
    if isinstance(jacobian, Sequence):
        # Nest and blocked lifting
        bcs1 = _bcs_by_block(_extract_function_spaces(jacobian, 1), bcs)  # type: ignore[arg-type]
        apply_lifting(b, jacobian, bcs=bcs1, x0=x, alpha=-1.0)
        dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
        bcs0 = _bcs_by_block(_extract_function_spaces(residual), bcs)  # type: ignore[arg-type]
        set_bc(b, bcs0, x0=x, alpha=-1.0)
    else:
        # Single form lifting
        apply_lifting(b, [jacobian], bcs=[bcs], x0=[x], alpha=-1.0)
        dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
        set_bc(b, bcs, x0=x, alpha=-1.0)
    dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[attr-defined]


class NonlinearMaterialProblem(NonlinearProblem):
    """
    This class handles the definition of a nonlinear problem containing an abstract `QuadratureMap` object compatible with a dolfinx NewtonSolver.
    """

    def __init__(
        self,
        qmap: QuadratureMap | Sequence[QuadratureMap],
        F: ufl.form.Form | Sequence[ufl.form.Form],
        u: _Function | Sequence[_Function],
        *,
        petsc_options_prefix: str,
        bcs: Sequence[DirichletBC] | None = None,
        J: ufl.form.Form | Sequence[Sequence[ufl.form.Form]] | None = None,
        P: ufl.form.Form | Sequence[Sequence[ufl.form.Form]] | None = None,
        kind: str | Sequence[Sequence[str]] | None = None,
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[_EntityMap] | None = None,
    ):
        """
        Parameters
        ----------
        qmap : dolfinx_materials.quadrature_map.QuadratureMap, list
            The abstract QuadratureMap object, can also be a list.
        F : Form
            Nonlinear residual form
        u : fem.Function
            Unknown function representing the solution
        J : Form
            Associated Jacobian form
        J : Form
            Associated Jacobian form
        bcs : list
            list of ``fem.dirichletbc``
        entity_maps:
            If any trial functions, test functions, or
            coefficients in the form are not defined over the same mesh
            as the integration domain, a corresponding :class:
            `EntityMap<dolfinx.mesh.EntityMap>` must be provided.
        """
        if J is None:
            J = qmap.derivative(u, ufl.TrialFunction(u.function_space))
        super().__init__(
            F,
            u,
            J=J,
            bcs=bcs,
            petsc_options_prefix=petsc_options_prefix,
            kind=kind,
            petsc_options=petsc_options,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        self.bcs = bcs
        if not isinstance(qmap, list):
            self.quadrature_maps = [qmap]
        else:
            self.quadrature_maps = qmap

        self.solver.setFunction(
            partial(
                _assemble_residual,
                self._constitutive_update,
                self.u,
                self.F,
                self.J,
                self.bcs,
            ),
            self.b,
        )

    def _constitutive_update(self):
        with Timer("SNES: constitutive update"):
            for qmap in self.quadrature_maps:
                qmap.update()

    def _constitutive_advance(self):
        for qmap in self.quadrature_maps:
            qmap.advance()

    def solve(self):
        # Copy current iterate into the work array.
        assign(self.u, self.x)

        # Solve problem
        with Timer("SNES: solve"):
            self.solver.solve(None, self.x)
        dolfinx.la.petsc._ghost_update(self.x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[attr-defined]

        # Copy solution back to function
        assign(self.x, self.u)

        self._constitutive_advance()

        return self.u
