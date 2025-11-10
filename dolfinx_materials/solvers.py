from mpi4py import MPI
from ufl import Form
from dolfinx.fem import form, Function
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.petsc import (
    apply_lifting,
    set_bc,
    assemble_vector,
    assemble_matrix,
    create_matrix,
    create_vector,
    NonlinearProblem,
)
from petsc4py import PETSc
from dolfinx_materials.quadrature_map import QuadratureMap

from dolfinx.common import Timer


def mpiprint(s):
    if MPI.COMM_WORLD.rank == 0:
        print(s)


class NonlinearMaterialProblem(NonlinearProblem):
    """
    This class handles the definition of a nonlinear problem containing an abstract `QuadratureMap` object compatible with a dolfinx NewtonSolver.
    """

    def __init__(
        self,
        qmap,
        F,
        J,
        u,
        bcs=None,
        petsc_options_prefix=None,
        petsc_options=None,
    ):
        """
        Parameters
        ----------
        qmap : dolfinx_materials.quadrature_map.QuadratureMap
            The abstract QuadratureMap object
        F : Form
            Nonlinear residual form
        J : Form
            Associated Jacobian form
        u : fem.Function
            Unknown function representing the solution
        bcs : list
            list of fem.dirichletbc
        """
        super().__init__(
            F,
            u,
            J=J,
            bcs=bcs,
            petsc_options_prefix=petsc_options_prefix,
            petsc_options=petsc_options,
        )
        # self.u = u
        self.bcs = bcs
        if not isinstance(qmap, list):
            self.quadrature_maps = [qmap]
        else:
            self.quadrature_maps = qmap

    def _constitutive_update(self):
        with Timer("Constitutive update"):
            for qmap in self.quadrature_maps:
                qmap.update()

    def _constitutive_advance(self):
        for qmap in self.quadrature_maps:
            qmap.advance()

    def form(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self._constitutive_update()

    # def matrix(self):
    #     return create_matrix(self.a)

    # def vector(self):
    #     return create_vector(self.L)

    def solve(self, solver, print_solution=True):
        """Solve the problem

        Parameters
        ----------
        solver :
            Nonlinear solver object
        print_solution : bool, optional
            Print convergence info, by default True

        Returns
        -------
        converged: bool
            Convergence status
        it: int
            Number of iterations to convergence
        """
        solver.setF(self.F, self.b)
        solver.setJ(self.J, self.A)
        solver.set_form(self.form)

        it, converged = solver.solve(self.u.x.petsc_vec)
        self.u.x.scatter_forward()

        if converged:
            # (Residual norm {error_norm})")
            if print_solution:
                mpiprint(f"Solution reached in {it} iterations.")
            self._constitutive_advance()

        else:
            mpiprint(
                f"No solution found after {it} iterations. Revert to previous solution and adjust solver parameters."
            )

        return converged, it


class SNESNonlinearMaterialProblem(NonlinearMaterialProblem):
    """
    This class handles the definition of a nonlinear problem containing an abstract `QuadratureMap` object compatible with a PETSc SNESSolver.
    """

    def residual(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)

        self.u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        self._constitutive_update()

        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self._F)
        apply_lifting(F, [self._J], bcs=[self.bcs], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bcs, x, -1.0)

    def jacobian(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        assemble_matrix(J, self._J, bcs=self.bcs)
        J.assemble()

    def solve(self, solver, print_solution=True):
        """Solve the problem

        Parameters
        ----------
        solver :
            Nonlinear solver object
        print_solution : bool, optional
            Print convergence info, by default True

        Returns
        -------
        converged: bool
            Convergence status
        it: int
            Number of iterations to convergence
        """
        solver.setFunction(self.residual, self.b)
        solver.setJacobian(self.jacobian, self.A)
        with Timer("SNES: solve"):
            solver.solve(None, self.u.x.petsc_vec)
        converged = solver.getConvergedReason() > 0
        it = solver.getIterationNumber()
        if converged:
            # (Residual norm {error_norm})")
            if print_solution:
                mpiprint(f"Solution reached in {it} iterations.")
            self._constitutive_advance()
        else:
            mpiprint(
                f"No solution found after {it} iterations. Revert to previous solution and adjust solver parameters."
            )

        return converged, it
