from ufl import TrialFunction, derivative
from dolfinx.fem import form, apply_lifting, set_bc, Function
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_matrix,
    create_vector,
)
from petsc4py import PETSc


### iteration adopted from Custom Newton Solver tutorial ###
class CustomNewton:
    def __init__(self, quadrature_map, F, J, u, bcs, max_it=50, tol=1e-8):
        self.quadrature_map = quadrature_map
        self.L = form(F)
        self.a = form(J)
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u
        self.du = Function(self.u.function_space, name="Increment")
        self.it = 0
        self.A = create_matrix(self.a)  # preallocating sparse jacobian matrix
        self.b = create_vector(self.L)  # preallocating residual vector
        self.max_it = max_it
        self.tol = tol

    def solve(self, solver, print_steps=True, print_solution=True):
        i = 0  # number of iterations of the Newton solver
        converged = False
        while i < self.max_it:
            self.quadrature_map.update()

            # Assemble Jacobian and residual
            with self.b.localForm() as loc_b:
                loc_b.set(0)
            self.A.zeroEntries()
            assemble_matrix(self.A, self.a, bcs=self.bcs)
            self.A.assemble()
            assemble_vector(self.b, self.L)
            self.b.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            self.b.scale(-1)

            # Compute b - J(u_D-u_(i-1))
            apply_lifting(self.b, [self.a], [self.bcs], x0=[self.u.vector], scale=1)
            # Set dx|_bc = u_{i-1}-u_D
            set_bc(self.b, self.bcs, self.u.vector, 1.0)
            self.b.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )

            # Solve linear problem
            solver.setOperators(self.A)
            solver.solve(self.b, self.du.vector)
            self.du.x.scatter_forward()

            # Update u_{i+1} = u_i + relaxation_param * delta x_i
            self.u.x.array[:] += self.du.x.array[:]
            i += 1
            # Compute norm of update
            correction_norm = self.du.vector.norm(0)
            error_norm = self.b.norm(0)
            if print_steps:
                print(
                    f"    Iteration {i}: Correction norm {correction_norm}, Residual {error_norm}"
                )

            if correction_norm < self.tol:
                converged = True
                break

        if print_solution:
            if converged:
                # (Residual norm {error_norm})")
                print(f"Solution reached in {i} iterations.")
                print("Constitutive relation update for next time step.")
                self.quadrature_map.advance()
            else:
                print(
                    f"No solution found after {i} iterations. Revert to previous solution and adjust solver parameters."
                )

        return converged, i


class SNESProblem:
    def __init__(self, quadrature_map, F, J, u, bcs):
        self.quadrature_map = quadrature_map
        self.L = form(F)
        self.a = form(J)
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u
        self.it = 0

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # Update constitutive relation
        print("Current iteration:", self.it)
        self.quadrature_map.update()
        self.it += 1

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
