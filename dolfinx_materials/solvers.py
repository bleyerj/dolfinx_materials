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


class NonlinearMaterialProblem(NonlinearProblem):
    def __init__(self, qmap, F, J, u, bcs):
        super().__init__(F, u, J=J, bcs=bcs)
        self._F = None
        self._J = None
        self.u = u
        self.quadrature_map = qmap

    def form(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with Timer("Constitutive update"):
            self.quadrature_map.update()

    def matrix(self):
        return create_matrix(self.a)

    def vector(self):
        return create_vector(self.L)

    def solve(self, solver):
        solver.setF(self.F, self.vector())
        solver.setJ(self.J, self.matrix())
        solver.set_form(self.form)

        it, converged = solver.solve(self.u.vector)
        self.u.x.scatter_forward()
        if converged:
            # (Residual norm {error_norm})")
            mpiprint(f"Solution reached in {it} iterations.")
            mpiprint("Constitutive relation update for next time step.")
            self.quadrature_map.advance()
        else:
            mpiprint(
                f"No solution found after {it} iterations. Revert to previous solution and adjust solver parameters."
            )

        return converged, it


class SNESNonlinearMaterialProblem(NonlinearMaterialProblem):
    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)

        self.quadrature_map.update()
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

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

    def solve(self, solver):
        solver.setFunction(self.F, self.vector())
        solver.setJacobian(self.J, self.matrix())

        solver.solve(None, self.u.vector)
        converged = solver.getConvergedReason() > 0
        it = solver.getIterationNumber()
        if converged:
            # (Residual norm {error_norm})")
            mpiprint(f"Solution reached in {it} iterations.")
            mpiprint("Constitutive relation update for next time step.")
            self.quadrature_map.advance()
        else:
            mpiprint(
                f"No solution found after {it} iterations. Revert to previous solution and adjust solver parameters."
            )

        return converged, it


class TAOProblem:
    """Nonlinear problem class compatible with PETSC.TAO solver."""

    def __init__(self, quadrature_map, F, J, u, bcs):
        self.quadrature_map = quadrature_map
        self.L = form(F)
        self.a = form(J)
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u
        self.it = 0
        """This class set up structures for solving a non-linear problem using Newton's method.

        Parameters
        ==========
        f: Objective.
        F: Residual.
        J: Jacobian.
        u: Solution.
        bcs: Dirichlet boundary conditions.
        """

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self.A = create_matrix(self.a)
        self.b = create_vector(self.L)

    def f(self, tao, x: PETSc.Vec):
        """Assemble the objective f.

        Parameters
        ==========
        tao: the tao object
        x: Vector containing the latest solution.
        """

        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        self.quadrature_map.update()
        return 0.0

    def F(self, tao: PETSc.TAO, x: PETSc.Vec, F):
        """Assemble the residual F into the vector b.

        Parameters
        ==========
        tao: the tao object
        x: Vector containing the latest solution.
        b: Vector to assemble the residual into.
        """
        # We need to assign the vector to the function

        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bcs, x, -1.0)

    def J(self, tao: PETSc.TAO, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat):
        """Assemble the Jacobian matrix.

        Parameters
        ==========
        x: Vector containing the latest solution.
        A: Matrix to assemble the Jacobian into.
        """
        A.zeroEntries()
        assemble_matrix(A, self.a, self.bcs)
        A.assemble()
