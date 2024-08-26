from ufl import TrialFunction, derivative
from dolfinx.fem import form, apply_lifting, set_bc, Function
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_matrix,
    create_vector,
    NonlinearProblem,
)
from petsc4py import PETSc
from dolfinx.common import Timer
import numpy as np
from mpi4py import MPI


def mpiprint(s):
    if MPI.COMM_WORLD.rank == 0:
        print(s)


class CustomNewtonProblem:
    """
    Custom implementation of the Newton method for `QuadratureMap` objects.
    """

    def __init__(self, quadrature_map, F, J, u, bcs, max_it=50, rtol=1e-8, atol=1e-8):
        """
        Parameters
        ----------
        quadrature_map : dolfinx_materials.quadrature_map.QuadratureMap
            The abstract QuadratureMap object
        F : Form
            Nonlinear residual form
        J : Form
            Associated Jacobian form
        u : fem.Function
            Unknown function representing the solution
        bcs : list
            list of fem.dirichletbc
        max_it : int, optional
            Maximum number of iterations, by default 50
        rtol : float, optional
            Relative tolerance, by default 1e-8
        atol : float, optional
            Absolute tolerance, by default 1e-8
        """
        self.quadrature_map = quadrature_map
        if isinstance(F, list):
            self.L = [form(f) for f in F]
        else:
            self.L = form(F)
        if isinstance(J, list):
            self.a = [form(j) for j in J]
        else:
            self.a = form(J)
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u
        self.du = Function(self.u.function_space, name="Increment")
        self.it = 0

        if isinstance(self.a, list):
            self.A = create_matrix(self.a[0])
        else:
            self.A = create_matrix(self.a)  # preallocating sparse jacobian matrix
        if isinstance(self.L, list):
            self.b = create_vector(self.L[0])  # preallocating residual vector
        else:
            self.b = create_vector(self.L)  # preallocating residual vector
        self.max_it = max_it
        self.rtol = rtol
        self.atol = atol

    def solve(self, solver, print_steps=True, print_solution=True):
        """Solve method.

        Parameters
        ----------
        solver : KSP object
            PETSc KSP solver for the linear system
        print_steps : bool, optional
            Print iteration info, by default True
        print_solution : bool, optional
            Print convergence info, by default True

        Returns
        -------
        converged: bool
            Convergence status
        it: int
            Number of iterations to convergence
        """
        i = 0  # number of iterations of the Newton solver
        converged = False
        while i < self.max_it:
            with Timer("Constitutive update"):
                self.quadrature_map.update()

            # Assemble Jacobian and residual
            with self.b.localForm() as loc_b:
                loc_b.set(0)
            self.A.zeroEntries()
            if isinstance(self.a, list):
                self.A.assemble()
                for ai in self.a:
                    Ai = assemble_matrix(ai, bcs=self.bcs)
                    Ai.assemble()
                    self.A.axpy(1.0, Ai)
            else:
                assemble_matrix(self.A, self.a, bcs=self.bcs)
                self.A.assemble()

            if isinstance(self.L, list):
                for Li in self.L:
                    bi = assemble_vector(Li)
                    self.b.axpy(1.0, bi)
            else:
                assemble_vector(self.b, self.L)
            self.b.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            self.b.scale(-1)

            # Compute b - J(u_D-u_(i-1))
            if isinstance(self.a, list):
                for ai in self.a:
                    apply_lifting(self.b, [ai], [self.bcs], x0=[self.u.vector], scale=1)
            else:
                apply_lifting(self.b, [self.a], [self.bcs], x0=[self.u.vector], scale=1)
            # Set dx|_bc = u_{i-1}-u_D
            set_bc(self.b, self.bcs, self.u.vector, 1.0)
            self.b.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )

            # Solve linear problem
            solver.setOperators(self.A)
            with Timer("Linear solve"):
                solver.solve(self.b, self.du.vector)
            self.du.x.scatter_forward()

            # Update u_{i+1} = u_i + relaxation_param * delta x_i
            self.u.x.array[:] += self.du.x.array[:]
            i += 1
            # Compute norm of update
            correction_norm = self.du.vector.norm(0)
            error_norm = self.b.norm(0)
            if i == 1:
                error_norm0 = error_norm
            relative_norm = error_norm / error_norm0
            if print_steps:
                print(
                    f"    Iteration {i}:  Residual abs: {error_norm} rel: {relative_norm}"
                )

            if relative_norm < self.rtol or error_norm < self.atol:
                converged = True
                break

        if print_solution:
            if converged:
                # (Residual norm {error_norm})")
                print(f"Solution reached in {i} iterations.")
                self.quadrature_map.advance()
            else:
                print(
                    f"No solution found after {i} iterations. Revert to previous solution and adjust solver parameters."
                )

        return converged, it


class NonlinearMaterialProblem(NonlinearProblem):
    """
    This class handles the definition of a nonlinear problem containing an abstract `QuadratureMap` object compatible with a dolfinx NewtonSolver.
    """

    def __init__(self, qmap, F, J, u, bcs):
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
        super().__init__(F, u, J=J, bcs=bcs)
        self._F = None
        self._J = None
        self.u = u
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

    def matrix(self):
        return create_matrix(self.a)

    def vector(self):
        return create_vector(self.L)

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
        solver.setF(self.F, self.vector())
        solver.setJ(self.J, self.matrix())
        solver.set_form(self.form)

        it, converged = solver.solve(self.u.vector)
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

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)

        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        self._constitutive_update()

        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bcs)
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
        solver.setFunction(self.F, self.vector())
        solver.setJacobian(self.J, self.matrix())

        solver.solve(None, self.u.vector)
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
