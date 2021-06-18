from sympy import *
from sympy.printing.numpy import NumPyPrinter
from sympy.utilities.codegen import codegen

import os

init_printing()


class SymbolicKinDyn():

    def __init__(self, DOF=None, gravity_vector=None, ee=None, A=None, B=None, X=None, Y=None, Mb=None):
        self.n = DOF
        self.gravity_vector = gravity_vector
        self.ee = ee
        self.A = A
        self.B = B
        self.X = X
        self.Y = Y
        self.Mb = Mb

        # variables for Code Generation:
        self.fkin = None
        self.J = None
        self.Jb = None
        self.Jh = None
        self.Jdot = None
        self.Vb_ee = None
        self.Vh_ee = None
        self.Jb_ee = None
        self.Jh_ee = None
        self.M = None
        self.C = None
        self.Qgrav = None
        self.Q = None

        self.var_syms = set({})

    def generateCode(self, python=True, C=True, Matlab=True, folder="./generated_code", use_global_vars=True, name="plant", project="Project"):
        all_functions = [self.fkin, self.J, self.Jb, self.Jh, self.Jdot, self.Vb_ee,
                         self.Vh_ee, self.Jb_ee, self.Jh_ee, self.M, self.C, self.Qgrav, self.Q]
        # names = ["fkin", "J", "Jb", "Jh", "Jdot", "Vb_ee", "Vh_ee", "Jb_ee", "Jh_ee", "M", "C", "Qgrav", "Q"]
        all_names = ["forward_kinematics", "system_jacobian_matrix", "body_jacobian_matrix", "hybrid_jacobian_matrix", "system_jacobian_dot", "body_twist_ee", "hybrid_twist_ee",
                     "body_jacobian_matrix_ee", "hybrid_jacobian_matrix_ee", "generalized_mass_inertia_matrix", "coriolis_centrifugal_matrix", "gravity_vector", "inverse_dynamics"]
        functions = []
        names = []
        for i in range(len(all_functions)):
            if all_functions[i] is not None:
                functions.append(all_functions[i])
                names.append(all_names[i])

        all_syms = set()
        for f in functions:
            all_syms.update(f.free_symbols)
        if use_global_vars:
            constant_syms = list(all_syms.difference(self.var_syms))
        else:
            constant_syms = []
        if python:
            p = NumPyPrinter()
            s = ["import numpy\n\n"]
            s.append("class "+name.capitalize()+"():")
            s.append("\tdef __init__(self, %s):" % (
                ", ".join(sorted([str(constant_syms[i]) for i in range(len(constant_syms))]))))
            if len(constant_syms) > 0:
                s.append("\t\t"+", ".join(sorted(["self."+str(constant_syms[i]) for i in range(len(constant_syms))]))
                         + " = " + ", ".join(sorted([str(constant_syms[i]) for i in range(len(constant_syms))])))

            for i in range(len(functions)):
                var_syms = list(self.var_syms.intersection(
                    functions[i].free_symbols))
                const_syms = list(set(constant_syms).intersection(
                    functions[i].free_symbols))
                if len(var_syms) > 0:
                    s.append("\n\tdef "+names[i]+"(self, %s):" % (
                        ", ".join(sorted([str(var_syms[i]) for i in range(len(var_syms))]))))

                else:
                    s.append("\n\tdef "+names[i]+"(self):")
                if len(const_syms) > 0:
                    s.append("\t\t"+", ".join(sorted([str(const_syms[i]) for i in range(len(const_syms))]))
                             + " = " + ", ".join(sorted(["self."+str(const_syms[i]) for i in range(len(const_syms))])))

                s.append("\t\t"+names[i] + " = " + p.doprint(functions[i]))
                s.append("\t\treturn " + names[i])
            s = "\n".join(s)

            with open(os.path.join(folder, name + ".py"), "w+") as f:
                f.write(s)

        if C:
            if use_global_vars:
                [(c_name, c_code), (h_name, c_header)] = codegen([tuple((names[i], functions[i])) for i in range(len(functions))],
                                                                 "C99", name, project, header=False, empty=True, global_vars=constant_syms)
            else:
                [(c_name, c_code), (h_name, c_header)] = codegen([tuple((names[i], functions[i])) for i in range(len(functions))],
                                                                 "C99", name, project, header=False, empty=True)

            with open(os.path.join(folder, c_name), "w+") as f:
                f.write(c_code)
            with open(os.path.join(folder, h_name), "w+") as f:
                f.write(c_header)

        if Matlab:
            for i in range(len(functions)):
                if use_global_vars:
                    [(m_name, m_code)] = codegen((names[i], functions[i]), "Octave",
                                                 project=project, header=False, empty=True, global_vars=constant_syms)
                else:
                    [(m_name, m_code)] = codegen((names[i], functions[i]),
                                                 "Octave", project=project, header=False, empty=True)

                with open(os.path.join(folder, m_name), "w+") as f:
                    f.write(m_code)

    def closed_form_kinematics_body_fixed(self, q, qd, q2d):
        self.var_syms.update(q.free_symbols)
        self.var_syms.update(qd.free_symbols)
        self.var_syms.update(q2d.free_symbols)

        if self.A is not None:
            print("Using absolute configuration (A) of the body frames")
            FK_f = [self.SE3Exp(self.Y[0], q[0])]
            FK_C = [FK_f[0]*self.A[0]]
            for i in range(1, self.n):
                FK_f.append(FK_f[i-1]*self.SE3Exp(self.Y[i], q[i]))
                FK_C.append(FK_f[i]*self.A[i])
        elif self.B is not None:
            print('Using relative configuration (B) of the body frames')
            FK_C = [self.B[0]*self.SE3Exp(self.X[0], q[0])]
            for i in range(1, self.n):
                FK_C.append(FK_C[i-1]*self.B[i]*self.SE3Exp(self.X[i], q[i]))
        else:
            'Absolute (A) or Relative (B) configuration of the bodies should be provided in class!'
            return

        fkin = simplify(FK_C[self.n-1]*self.ee)
        self.fkin = fkin

        # Block diagonal matrix A (6n x 6n) of the Adjoint of body frame
        A = Matrix(Identity(6*self.n))
        for i in range(self.n):
            for j in range(i):
                Crel = self.SE3Inv(FK_C[i])*FK_C[j]
                AdCrel = self.SE3AdjMatrix(Crel)
                r = 6*(i)
                c = 6*(j)
                A[r:r+6, c:c+6] = AdCrel

        # Block diagonal matrix X (6n x n) of the screw coordinate vector associated to all joints in the body frame (Constant)
        X = zeros(6*self.n, self.n)
        for i in range(self.n):
            X[6*i:6*i+6, i] = self.X[i]

        # System level Jacobian
        J = A*X
        J_simplified = simplify(J)

        self.J = J_simplified

        # System twist (6n x 1)
        V = J*qd

        # Different Jacobians
        R_i = Matrix(fkin[:3, :3]).row_join(
            zeros(3, 1)).col_join(Matrix([0, 0, 0, 1]).T)

        # Body fixed Jacobian of last moving body (This may not correspond to end-effector frame)
        Jb = J[-6:, :]

        Vb_BFn = Jb*qd  # Body fixed twist of last moving body
        Vb_BFn = simplify(Vb_BFn)

        # Body fixed twist of end-effector frame
        Vb_ee = self.SE3AdjMatrix(self.SE3Inv(self.ee))*Vb_BFn
        # Hybrid twist of end-effector frame
        Vh_ee = self.SE3AdjMatrix(R_i)*Vb_ee

        Vb_ee_simplified = simplify(Vb_ee)
        Vh_ee_simplified = simplify(Vh_ee)

        self.Vb_ee = Vb_ee_simplified
        self.Vh_ee = Vh_ee_simplified

        # Body fixed Jacobian of end-effector frame
        Jb_ee = self.SE3AdjMatrix(self.SE3Inv(self.ee))*Jb
        # Hybrid Jacobian of end-effector frame
        Jh_ee = self.SE3AdjMatrix(R_i)*Jb_ee
        Jh = self.SE3AdjMatrix(R_i)*Jb  # Hybrid Jacobian of last moving body

        Jh_ee_simplified = simplify(Jh_ee)
        Jb_ee_simplified = simplify(Jb_ee)
        Jh_simplified = simplify(Jh)
        Jb_simplified = simplify(Jb)

        self.Jh_ee = Jh_ee_simplified
        self.Jb_ee = Jb_ee_simplified
        self.Jh = Jh_simplified
        self.Jb = Jb_simplified

        # Acceleration computations

        # Block diagonal matrix a (6n x 6n)
        a = zeros(6*self.n, 6*self.n)
        for i in range(self.n):
            a[6*i:6*i+6, 6*i:6*i+6] = self.SE3adMatrix(self.X[i])*qd[i]

        # System acceleration (6n x 1)
        Jdot = -A*a*J  # Sys-level Jacobian time derivative
        Jdot_simplified = simplify(Jdot)

        self.Jdot = Jdot_simplified

        Vbd = J*q2d - A*a*V

        Vbd_BFn = simplify(Vbd[-6:, :])
        # Body fixed twist of end-effector frame
        Vbd_ee = self.SE3AdjMatrix(self.SE3Inv(self.ee))*Vbd_BFn
        Vhd_ee = self.SE3AdjMatrix(R_i)*Vbd_ee + self.SE3adMatrix(Matrix(
            Vh_ee[:3, :]).col_join(Matrix([0, 0, 0])))*Vb_ee  # Hybrid twist of end-effector frame
        Vhd_ee_simplified = simplify(Vhd_ee)

        px = fkin[0, 3]
        py = fkin[1, 3]

        # jacobian(px,q) * qd in Matlab
        pxd = simplify(Matrix([px.diff(q[i]) for i in range(self.n)]).T * qd)
        pyd = simplify(Matrix([py.diff(q[i]) for i in range(self.n)]).T * qd)
        pxd_res = Vh_ee_simplified[3, 0]
        pyd_res = Vh_ee_simplified[4, 0]

        # Todo: check  equality

        px2d = simplify(Matrix([pxd.diff(q[i]) for i in range(
            self.n)]).T * qd + Matrix([pxd.diff(qd[i]) for i in range(self.n)]).T * q2d)
        px2d = simplify(Matrix([pyd.diff(q[i]) for i in range(
            self.n)]).T * qd + Matrix([pyd.diff(qd[i]) for i in range(self.n)]).T * q2d)
        px2d_res = Vhd_ee_simplified[3, 0]
        py2d_res = Vhd_ee_simplified[4, 0]

        # Todo: check  equality

        T = simplify(FK_C[self.n-1]*self.ee)
        return T

    def closed_form_inv_dyn_body_fixed(self, q, qd, q2d, WEE=zeros(6, 1), simplify_expressions=True):
        self.var_syms.update(q.free_symbols)
        self.var_syms.update(qd.free_symbols)
        self.var_syms.update(q2d.free_symbols)
        self.var_syms.update(WEE.free_symbols)

        if self.A is not None:
            print("Using absolute configuration (A) of the body frames")
            FK_f = [self.SE3Exp(self.Y[0], q[0])]
            FK_C = [FK_f[0]*self.A[0]]
            for i in range(1, self.n):
                FK_f.append(FK_f[i-1]*self.SE3Exp(self.Y[i], q[i]))
                FK_C.append(FK_f[i]*self.A[i])
        elif self.B is not None:
            print('Using relative configuration (B) of the body frames')
            FK_C = [self.B[0]*self.SE3Exp(self.X[0], q[0])]
            for i in range(1, self.n):
                FK_C.append(FK_C[i-1]*self.B[i]*self.SE3Exp(self.X[i], q[i]))
        else:
            'Absolute (A) or Relative (B) configuration of the bodies should be provided in class!'
            return

        # fkin = simplify(FK_C[self.n-1]*self.ee)
        # self.fkin = fkin

        # Block diagonal matrix A (6n x 6n) of the Adjoint of body frame
        A = Matrix(Identity(6*self.n))
        for i in range(self.n):
            for j in range(i):
                Crel = self.SE3Inv(FK_C[i])*FK_C[j]
                AdCrel = self.SE3AdjMatrix(Crel)
                r = 6*(i)
                c = 6*(j)
                A[r:r+6, c:c+6] = AdCrel

        # Block diagonal matrix X (6n x n) of the screw coordinate vector associated to all joints in the body frame (Constant)
        X = zeros(6*self.n, self.n)
        for i in range(self.n):
            X[6*i:6*i+6, i] = self.X[i]

        # System level Jacobian
        J = A*X

        # J_simplified = simplify(J)
        # self.J = J_simplified

        # System twist (6n x 1)
        V = J*qd

        # Acceleration computations

        # Block diagonal matrix a (6n x 6n)
        a = zeros(6*self.n, 6*self.n)
        for i in range(self.n):
            a[6*i:6*i+6, 6*i:6*i+6] = self.SE3adMatrix(self.X[i])*qd[i]

        # System acceleration (6n x 1)
        Vd = J*q2d - A*a*V

        # Block Diagonal Mb (6n x 6n) Mass inertia matrix in body frame (Constant)
        Mb = zeros(6*self.n, 6*self.n)
        for i in range(self.n):
            Mb[i*6:i*6+6, i*6:i*6+6] = self.Mb[i]

        # Block diagonal matrix b (6n x 6n) used in Coriolis matrix
        b = zeros(6*self.n, 6*self.n)
        for i in range(self.n):
            b[i*6:i*6+6, i*6:i*6+6] = self.SE3adMatrix(Matrix(V[6*i:6*i+6]))

        # Block diagonal matrix Cb (6n x 6n)
        Cb = -Mb*A*a - b.T * Mb

        # Lets setup the Equations of Motion

        # Mass inertia matrix in joint space (n x n)
        M = J.T*Mb*J

        # Coriolis-Centrifugal matrix in joint space (n x n)
        C = J.T * Cb * J

        # Gravity Term
        U = self.SE3AdjInvMatrix(FK_C[0])
        for k in range(1, self.n):
            U = U.col_join(self.SE3AdjInvMatrix(FK_C[k]))

        Vd_0 = zeros(6, 1)
        Vd_0[3:6, 0] = self.gravity_vector
        Qgrav = J.T*Mb*U*Vd_0

        # External Wrench
        Wext = zeros(6*self.n, 1)
        # WEE (t) is the time varying wrench on the EE link.
        Wext[-6:, 0] = WEE
        Qext = J.T * Wext

        # Generalized forces Q
        # Q = M*q2d + C*qd   # without gravity
        Q = M*q2d + C*qd + Qgrav + Qext

        if simplify_expressions:
            M = simplify(M)
            C = simplify(C)
            Qgrav = simplify(Qgrav)
            Q = simplify(Q)

        self.M = M
        self.C = C
        self.Q = Q
        self.Qgrav = Qgrav

        return Q

    def SE3AdjInvMatrix(self, C):
        AdInv = Matrix([[C[0, 0], C[1, 0], C[2, 0], 0, 0, 0],
                        [C[0, 1], C[1, 1], C[2, 1], 0, 0, 0],
                        [C[0, 2], C[1, 2], C[2, 2], 0, 0, 0],
                        [-C[2, 3]*C[1, 0]+C[1, 3]*C[2, 0], C[2, 3]*C[0, 0]-C[0, 3]*C[2, 0],
                            (-C[1, 3])*C[0, 0]+C[0, 3]*C[1, 0], C[0, 0], C[1, 0], C[2, 0]],
                        [-C[2, 3]*C[1, 1]+C[1, 3]*C[2, 1], C[2, 3]*C[0, 1]-C[0, 3]*C[2, 1],
                            (-C[1, 3])*C[0, 1]+C[0, 3]*C[1, 1], C[0, 1], C[1, 1], C[2, 1]],
                        [-C[2, 3]*C[1, 2]+C[1, 3]*C[2, 2], C[2, 3]*C[0, 2]-C[0, 3]*C[2, 2],
                            (-C[1, 3])*C[0, 2]+C[0, 3]*C[1, 2], C[0, 2], C[1, 2], C[2, 2]]])
        return AdInv

    def SE3AdjMatrix(self, C):
        Ad = Matrix([[C[0, 0], C[0, 1], C[0, 2], 0, 0, 0],
                     [C[1, 0], C[1, 1], C[1, 2], 0, 0, 0],
                     [C[2, 0], C[2, 1], C[2, 2], 0, 0, 0],
                     [-C[2, 3]*C[1, 0]+C[1, 3]*C[2, 0], -C[2, 3]*C[1, 1]+C[1, 3]*C[2, 1],
                      -C[2, 3]*C[1, 2]+C[1, 3]*C[2, 2], C[0, 0], C[0, 1], C[0, 2]],
                     [C[2, 3]*C[0, 0]-C[0, 3]*C[2, 0],  C[2, 3]*C[0, 1]-C[0, 3]*C[2, 1],
                         C[2, 3]*C[0, 2]-C[0, 3]*C[2, 2], C[1, 0], C[1, 1], C[1, 2]],
                     [-C[1, 3]*C[0, 0]+C[0, 3]*C[1, 0], -C[1, 3]*C[0, 1]+C[0, 3]*C[1, 1],
                      -C[1, 3]*C[0, 2]+C[0, 3]*C[1, 2], C[2, 0], C[2, 1], C[2, 2]]])
        return Ad

    def SE3adMatrix(self, X):
        ad = Matrix([[0, -X[2, 0], X[1, 0], 0, 0, 0],
                     [X[2, 0], 0, -X[0, 0], 0, 0, 0],
                     [-X[1, 0], X[0, 0], 0, 0, 0, 0],
                     [0, -X[5, 0], X[4, 0], 0, -X[2, 0], X[1, 0]],
                     [X[5, 0], 0, -X[3, 0], X[2, 0], 0, -X[0, 0]],
                     [-X[4, 0], X[3, 0], 0, -X[1, 0], X[0, 0], 0]])
        return ad

    def SE3Exp(self, XX, t):
        X = XX.T
        xi = Matrix(X[0:3])
        eta = Matrix(X[3:6])
        xihat = Matrix([[0, -X[2], X[1]],
                        [X[2], 0, -X[0]],
                        [-X[1], X[0], 0]])
        R = Matrix(Identity(3)) + sin(t)*xihat + (1-cos(t))*(xihat*xihat)
        if xi == zeros(1, 3):
            p = eta.T * t
        else:
            p = (Matrix(Identity(3))-R)*(xihat*eta) + xi*(xi.T*eta)*t
        C = R.row_join(p).col_join(Matrix([0, 0, 0, 1]).T)
        return C

    def SE3Inv(self, C):
        CInv = Matrix([[C[0, 0], C[1, 0], C[2, 0], -C[0, 0]*C[0, 3]-C[1, 0]*C[1, 3]-C[2, 0]*C[2, 3]],
                       [C[0, 1], C[1, 1], C[2, 1], -C[0, 1] *
                           C[0, 3]-C[1, 1]*C[1, 3]-C[2, 1]*C[2, 3]],
                       [C[0, 2], C[1, 2], C[2, 2], -C[0, 2] *
                           C[0, 3]-C[1, 2]*C[1, 3]-C[2, 2]*C[2, 3]],
                       [0, 0, 0, 1]])
        return CInv

    def SO3Exp(self, x, t):
        xhat = Matrix([[0, -x[2], x[1]],
                       [x[2], 0, -x[0]],
                       [-x[1], x[0], 0]])
        R = Matrix(Identity(3)) + sin(t) * xhat + (1-cos(t))*(xhat*xhat)
        return R

    def InertiaMatrix(self, Ixx, Ixy, Ixz, Iyy, Iyz, Izz):
        I = Matrix([[Ixx, Ixy, Ixz],
                    [Ixy, Iyy, Iyz],
                    [Ixz, Iyz, Izz]])
        return I

    def MassMatrixMixedData(self, m, Theta, COM):
        M = Matrix([[Theta[0, 0], Theta[0, 1], Theta[0, 2], 0, (-COM[2])*m, COM[1]*m],
                    [Theta[0, 1], Theta[1, 1], Theta[1, 2],
                        COM[2]*m, 0, (-COM[0]*m)],
                    [Theta[0, 2], Theta[1, 2], Theta[2, 2],
                        (-COM[1])*m, COM[0]*m, 0],  # TODO: Code generation

                    [0, COM[2]*m, (-COM[1]*m), m, 0, 0],
                    [(-COM[2])*m, 0, COM[0]*m, 0, m, 0],
                    [COM[1]*m, (-COM[0])*m, 0, 0, 0, m]])
        return M


if __name__ == "__main__":
    s = SymbolicKinDyn()

    # Declaration of symbolic variables
    q1, q2 = symbols("q1 q2")
    dq1, dq2 = symbols("dq1 dq2")
    ddq1, ddq2 = symbols("ddq1 ddq2")

    m1, m2, I1, I2 = symbols("m1 m2 I1 I2", real=1, constant=1)
    cg1, cg2, g = symbols("cg1 cg2 g", real=1, constant=1)
    L1, L2 = symbols("L1 L2", real=1, constant=1)
    pi = symbols("pi", real=1, constant=1)

    s.gravity_vector = Matrix([0, g, 0])

    # Joint screw coordinates in spatial representation

    s.Y = []
    e1 = Matrix([0, 0, 1])
    y1 = Matrix([0, 0, 0])
    s.Y.append(Matrix([e1, y1.cross(e1)]))
    e2 = Matrix([0, 0, 1])
    y2 = Matrix([L1, 0, 0])
    s.Y.append(Matrix([e2, y2.cross(e2)]))

    # Reference configurations of bodies (i.e. of body-fixed reference frames)

    r1 = Matrix([0, 0, 0])
    r2 = Matrix([L1, 0, 0])

    s.A = []
    s.A.append(Matrix(Identity(3)).row_join(
        r1).col_join(Matrix([0, 0, 0, 1]).T))
    s.A.append(Matrix(Identity(3)).row_join(
        r2).col_join(Matrix([0, 0, 0, 1]).T))

    # s.B = []
    # s.B.append(Matrix(Identity(3)).row_join(r1).col_join(Matrix([0,0,0,1]).T))
    # s.B.append(Matrix(Identity(3)).row_join(r2).col_join(Matrix([0,0,0,1]).T))

    # End-effector configuration wrt last link body fixed frame in the chain
    re = Matrix([L2, 0, 0])
    s.ee = Matrix(Identity(3)).row_join(re).col_join(Matrix([0, 0, 0, 1]).T)

    # Joint screw coordinates in body-fixed representation computed from screw coordinates in IFR
    s.X = []
    s.X.append(s.SE3AdjInvMatrix(s.A[0])*s.Y[0])
    s.X.append(s.SE3AdjInvMatrix(s.A[1])*s.Y[1])

    # Joint screw coordinates in body-fixed representation

    # s.X = []
    # s.X.append(Matrix([0,0,1,0,0,0]).T)
    # s.X.append(Matrix([0,0,1,0,0,0]).T)

    # Mass-Inertia parameters
    cg1 = Matrix([L1, 0, 0]).T
    cg2 = Matrix([L2, 0, 0]).T
    I1 = m1*L1*L1
    I2 = m2*L2*L2

    s.Mb = []
    s.Mb.append(s.MassMatrixMixedData(m1, I1*Identity(3), cg1))
    s.Mb.append(s.MassMatrixMixedData(m2, I2*Identity(3), cg2))

    # Declaring generalised vectors
    q = Matrix([q1, q2])
    qd = Matrix([dq1, dq2])
    q2d = Matrix([ddq1, ddq2])
    s.n = len(q)

    # Kinematics
    F = s.closed_form_kinematics_body_fixed(q, qd, q2d)
    Q = s.closed_form_inv_dyn_body_fixed(q, qd, q2d)
    s.generateCode(python=True, C=True, Matlab=True,
                   use_global_vars=True, name="plant", project="Project")
