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
                    apply_lifting(
                        self.b, [ai], [self.bcs], x0=[self.u.x.petsc_vec], scale=1
                    )
            else:
                apply_lifting(
                    self.b, [self.a], [self.bcs], x0=[self.u.x.petsc_vec], scale=1
                )
            # Set dx|_bc = u_{i-1}-u_D
            set_bc(self.b, self.bcs, self.u.x.petsc_vec, 1.0)
            self.b.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )

            # Solve linear problem
            solver.setOperators(self.A)
            with Timer("Linear solve"):
                solver.solve(self.b, self.du.x.petsc_vec)
            self.du.x.scatter_forward()

            # Update u_{i+1} = u_i + relaxation_param * delta x_i
            self.u.x.array[:] += self.du.x.array[:]
            i += 1
            # Compute norm of update
            correction_norm = self.du.x.petsc_vec.norm(0)
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

        return converged, i


class NonlinearMaterialProblem:
    """
    A nonlinear solver for material-aware problems with constitutive integration.
    Performs Newton iterations with material updates at each step.
    """

    def __init__(self, qmap, F, J, u, bcs, petsc_options_prefix="nl_"):
        """
        Parameters
        ----------
        qmap : dolfinx_materials.quadrature_map.QuadratureMap
            The QuadratureMap object
        F : Form
            Nonlinear residual form
        J : Form
            Associated Jacobian form
        u : fem.Function
            Unknown function representing the solution
        bcs : list
            list of fem.dirichletbc
        petsc_options_prefix : str
            PETSc options prefix
        """
        from dolfinx import fem
        
        # Compile forms
        self._F = fem.form(F) if not isinstance(F, fem.Form) else F
        self._J = fem.form(J) if not isinstance(J, fem.Form) else J
        
        self.u = u
        self.bcs = bcs
        
        # Store quadrature maps
        if not isinstance(qmap, list):
            self.quadrature_maps = [qmap]
        else:
            self.quadrature_maps = qmap
        
        # Get function spaces for creating vectors/matrices
        u_space = u.function_space
        
        # Create vectors and matrices for assembly
        self.b = create_vector(u_space)
        self.A = create_matrix(self._J)
        self.du = create_vector(u_space)
        
        # Create KSP solver once and reuse it
        self.ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("lu")

    def _constitutive_update(self):
        with Timer("Constitutive update"):
            for qmap in self.quadrature_maps:
                qmap.update()

    def _constitutive_advance(self):
        for qmap in self.quadrature_maps:
            qmap.advance()

    def solve(self, solver=None, print_solution=True):
        """Solve the problem using Newton iterations with material updates.

        Parameters
        ----------
        solver :
            Nonlinear solver object with rtol, atol, max_it attributes
        print_solution : bool, optional
            Print convergence info, by default True

        Returns
        -------
        converged: bool
            Convergence status
        it: int
            Number of iterations to convergence
        """
        if solver is None:
            from dolfinx.cpp.nls.petsc import NewtonSolver
            solver = NewtonSolver(MPI.COMM_WORLD)
            solver.rtol = 1e-6
            solver.atol = 1e-6
            solver.max_it = 20
        
        # Initial Newton iteration
        i = 0
        converged = False
        res0_norm = 1.0
        
        while i < solver.max_it:
            import time
            t_iter_start = time.time()
            
            # Constitutive update with current displacement before each residual assembly
            t_const_start = time.time()
            self._constitutive_update()
            t_const = time.time() - t_const_start
            
            # Assemble residual
            t_asm_start = time.time()
            with self.b.localForm() as b_local:
                b_local.set(0.0)
            assemble_vector(self.b, self._F)
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.b.scale(-1.0)  # Negate for Newton system
            
            # Assemble Jacobian (assembler handles BCs correctly)
            self.A.zeroEntries()
            assemble_matrix(self.A, self._J, bcs=self.bcs)
            self.A.assemble()
            t_asm = time.time() - t_asm_start
            
            # Apply BC modifications to RHS
            from dolfinx.fem import apply_lifting, set_bc
            apply_lifting(self.b, [self._J], bcs=[self.bcs], x0=[self.u.x.petsc_vec], alpha=1.0)
            set_bc(self.b, self.bcs, self.u.x.petsc_vec, 1.0)
            self.b.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
            
            # Compute residual norm (after BC modifications)
            res_norm = self.b.norm()
            if i == 0:
                res0_norm = max(res_norm, 1.0)
            
            # Check convergence
            if res_norm < solver.atol or res_norm < solver.rtol * res0_norm:
                converged = True
                if print_solution:
                    mpiprint(f"Solution converged in {i} iterations")
                break
            
            # Solve linear system: A * du = b (reuse KSP)
            t_ksp_start = time.time()
            with Timer("KSP Solve"):
                self.ksp.setOperators(self.A)
                self.ksp.solve(self.b, self.du)
            t_ksp = time.time() - t_ksp_start
            
            # Update solution: u += du
            self.u.x.petsc_vec.axpy(1.0, self.du)
            self.u.x.scatter_forward()
            
            i += 1
        
        # Always advance material state after solve (converged or not)
        if converged:
            self._constitutive_advance()
            
        else:
            mpiprint(
                f"No solution found after {it} iterations. Revert to previous solution and adjust solver parameters."
            )
        return converged, i


class SNESNonlinearMaterialProblem(NonlinearMaterialProblem):
    """
    This class handles the definition of a nonlinear problem containing an abstract `QuadratureMap` object compatible with a PETSc SNESSolver.
    """

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)

        self.u.x.petsc_vec.ghostUpdate(
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
        solver.setFunction(self.F, self.x.petsc_vec())
        solver.setJacobian(self.J, self.matrix())

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
