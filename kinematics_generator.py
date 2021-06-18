from sympy import *
from sympy.printing.numpy import NumPyPrinter
from sympy.utilities.codegen import codegen

import os
import re as regex

init_printing()


class SymbolicKinDyn():

    def __init__(self, n=None, gravity_vector=None, ee=None, A=None, B=None, X=None, Y=None, Mb=None):
        """[summary]

        Args:
            n (int, optional): Degrees of Freedom. Defaults to None.
            gravity_vector (sympy.Matrix, optional): Vector of gravity. Defaults to None.
            ee (sympy.Matrix, optional): End-effector configuration with reference to last link body fixed frame in the chain. Defaults to None.
            A (list of sympy.Matrix, optional): [description]. Defaults to None.
            B (list of sympy.Matrix, optional): [description]. Defaults to None.
            X (list of sympy.Matrix, optional): [description]. Defaults to None.
            Y (list of sympy.Matrix, optional): List of joint screw coordinates in spacial representation. Defaults to None.
            Mb (list of sympy.Matrix, optional): [description]. Defaults to None.
        """
        self.n = n
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
        self.Vh_BFn = None
        self.Vb_BFn = None
        self.Vhd_BFn = None
        self.Vbd_BFn = None
        self.Vhd_ee = None
        self.Vbd_ee = None
        self.Jh_dot = None
        self.Jb_dot = None
        self.Jh_ee_dot = None
        self.Jb_ee_dot = None

        # set of variable symbols to include in generated functions as arguments
        self.var_syms = set({})

    def generateCode(self, python=True, C=True, Matlab=False, folder="./generated_code", use_global_vars=True, name="plant", project="Project"):
        """Generate code of saved Equations. 
        Needs closed_form_inv_dyn_body_fixed and/or closed_form_kinematics_body_fixed to run first.


        Args:
            python (bool, optional): Generate Python code. Defaults to True.
            C (bool, optional): Generate C99 code. Defaults to True.
            Matlab (bool, optional): Generate Matlab/Octav code. Defaults to False.
            folder (str, optional): Folder where to save code. Defaults to "./generated_code".
            use_global_vars (bool, optional): Constant vars like mass etc are no arguments of the functions. Defaults to True.
            name (str, optional): Name of Class and file (for Python and C). Defaults to "plant".
            project (str, optional): Project name in C header. Defaults to "Project".

        """
        all_functions = {"forward_kinematics" : self.fkin, 
                        "system_jacobian_matrix" : self.J, 
                        "body_jacobian_matrix" : self.Jb, 
                        "hybrid_jacobian_matrix" : self.Jh, 
                        "system_jacobian_dot" : self.Jdot, 
                        "body_twist_ee" : self.Vb_ee,
                        "hybrid_twist_ee" : self.Vh_ee,
                        "body_jacobian_matrix_ee" : self.Jb_ee,
                        "hybrid_jacobian_matrix_ee" : self.Jh_ee,
                        "generalized_mass_inertia_matrix" : self.M, 
                        "coriolis_centrifugal_matrix" : self.C, 
                        "gravity_vector" : self.Qgrav, 
                        "inverse_dynamics" : self.Q,
                        "hybrid_acceleration" : self.Vhd_BFn, 
                        "body_acceleration" : self.Vbd_BFn, 
                        "hybrid_acceleration_ee" : self.Vhd_ee,
                        "body_acceleration_ee" : self.Vbd_ee, 
                        "hybrid_jacobian_matrix_dot" : self.Jh_dot, 
                        "body_jacobian_matrix_dot" : self.Jb_dot, 
                        "hybrid_jacobian_matrix_ee_dot" : self.Jh_ee_dot, 
                        "body_jacobian_matrix_ee_dot" : self.Jb_ee_dot} # not included: self.Vh_BFn, self.Vb_BFn, 
        
        
        # all_functions = [self.fkin, self.J, self.Jb, self.Jh, self.Jdot, self.Vb_ee,
        #                  self.Vh_ee, self.Jb_ee, self.Jh_ee, self.M, self.C, self.Qgrav, self.Q,
        #                  self.Vhd_BFn, self.Vbd_BFn, self.Vhd_ee,
        #                  self.Vbd_ee, self.Jh_dot, self.Jb_dot, self.Jh_ee_dot, self.Jb_ee_dot] # not included: self.Vh_BFn, self.Vb_BFn, 
        # names = ["fkin", "J", "Jb", "Jh", "Jdot", "Vb_ee", "Vh_ee", "Jb_ee", "Jh_ee", "M", "C", "Qgrav", "Q",
        #          "Vhd_BFn", "Vbd_BFn", "Vhd_ee","Vbd_ee", "Jh_dot", "Jb_dot", "Jh_ee_dot", "Jb_ee_dot"]
        # all_names = ["forward_kinematics", "system_jacobian_matrix", "body_jacobian_matrix", "hybrid_jacobian_matrix", "system_jacobian_dot", "body_twist_ee", "hybrid_twist_ee",
        #              "body_jacobian_matrix_ee", "hybrid_jacobian_matrix_ee", "generalized_mass_inertia_matrix", "coriolis_centrifugal_matrix", "gravity_vector", "inverse_dynamics", 
        #              "hybrid_acceleration", "body_acceleration", "hybrid_acceleration_ee","body_acceleration_ee", "hybrid_jacobian_matrix_dot", "body_jacobian_matrix_dot", 
        #              "hybrid_jacobian_matrix_ee_dot", "body_jacobian_matrix_ee_dot"]
        functions = []
        names = []
        for i in all_functions:
            if all_functions[i] is not None:
                functions.append(all_functions[i])
                names.append(i)

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
            # change strange var names
            c_code = regex.sub(r"out_\d{19}", "out", c_code)
            c_header = regex.sub(r"out_\d{19}", "out", c_header)
            c_code = regex.sub(r"out_\d{18}", "out", c_code)
            c_header = regex.sub(r"out_\d{18}", "out", c_header)
            
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
        """Position, Velocity and Acceleration Kinematics using Body fixed representation of the twists in closed form.

        The following functions are saved in the class and can be code generated afterwards:
            body_acceleration
            body_acceleration_ee
            body_jacobian_matrix
            body_jacobian_matrix_dot 
            body_jacobian_matrix_ee
            body_jacobian_matrix_ee_dot
            body_twist_ee
            forward_kinematics
            hybrid_acceleration
            hybrid_acceleration_ee
            hybrid_jacobian_matrix
            hybrid_jacobian_matrix_dot
            hybrid_jacobian_matrix_ee
            hybrid_jacobian_matrix_ee_dot
            hybrid_twist_ee

        Args:
            q (sympy.Matrix): (n,1) Genearlized position vector.
            qd (sympy.Matrix): (n,1 )Generalized velocity vector.
            q2d (sympy.Matrix): (n,1) Generalized acceleration vector.

        Returns:
            sympy.Matrix: Forward kinematics (transformation matrix of ee).
        """
        self.var_syms.update(q.free_symbols)
        self.var_syms.update(qd.free_symbols)
        self.var_syms.update(q2d.free_symbols)

        s.n = len(q)

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
        R_BFn = Matrix(FK_C[-1][:3,:3]).row_join(
            zeros(3,1)).col_join(Matrix([0, 0, 0, 1]).T)

        # Body fixed Jacobian of last moving body (This may not correspond to end-effector frame)
        Jb = J[-6:, :]

        Vb_BFn = Jb*qd  # Body fixed twist of last moving body
        Vb_BFn = simplify(Vb_BFn)
        Vh_BFn = self.SE3AdjMatrix(R_BFn)*Vb_BFn

        self.Vb_BFn = Vb_BFn
        self.Vh_BFn = simplify(Vh_BFn)

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

        # Hybrid acceleration of the last body
        Vbd_BFn = simplify(Vbd[-6:, :])
        Vhd_BFn = self.SE3AdjMatrix(R_BFn)*Vbd_BFn + self.SE3adMatrix(Matrix(Vh_BFn[:3,:]).col_join(Matrix([0,0,0])))*self.SE3AdjMatrix(R_BFn)*Vb_BFn # Hybrid twist of end-effector frame
        Vhd_BFn_simplified = simplify(Vhd_BFn)
        
        self.Vbd_BFn = Vbd_BFn
        self.Vhd_BFn = Vhd_BFn_simplified


        # Body fixed twist of end-effector frame
        # Hybrid acceleration of the EE
        Vbd_ee = self.SE3AdjMatrix(self.SE3Inv(self.ee))*Vbd_BFn
        Vhd_ee = self.SE3AdjMatrix(R_i)*Vbd_ee + self.SE3adMatrix(Matrix(
            Vh_ee[:3, :]).col_join(Matrix([0, 0, 0])))*self.SE3AdjMatrix(R_i)*Vb_ee  # Hybrid twist of end-effector frame
        Vbd_ee_simplified = simplify(Vbd_ee)
        Vhd_ee_simplified = simplify(Vhd_ee)

        self.Vbd_ee = Vbd_ee_simplified
        self.Vhd_ee = Vhd_ee_simplified
        
        ## Body Jacobian time derivative
        
        # For the last moving body
        Jb_dot = Jdot_simplified[-6:,:]
        self.Jb_dot = Jb_dot

        # For the EE
        Jb_ee_dot = self.SE3AdjMatrix(self.SE3Inv(self.ee))*Jb_dot
        Jb_ee_dot_simplified = simplify(Jb_ee_dot)
        self.Jb_ee_dot = Jb_ee_dot_simplified

        ## Hybrid Jacobian time derivative
        # For the last moving body
        Jh_dot = self.SE3AdjMatrix(R_BFn)*Jb_dot + self.SE3adMatrix(Matrix(Vh_BFn[:3,:]).col_join(Matrix([0,0,0])))*self.SE3AdjMatrix(R_BFn)*Jb_simplified
        Jh_dot_simplified = simplify(Jh_dot)
        self.Jh_dot = Jh_dot_simplified

        # For the EE
        Jh_ee_dot = self.SE3AdjMatrix(R_i)*Jb_ee_dot + self.SE3adMatrix(Matrix(Vh_ee[:3,:]).col_join(Matrix([0,0,0])))*self.SE3AdjMatrix(R_i)*Jb_ee_simplified
        Jh_ee_dot_simplified = simplify(Jh_ee_dot)
        self.Jh_ee_dot = Jh_ee_dot_simplified



        # For Verification (not implemented):
        if False:
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
        """Inverse Dynamics using Body fixed representation of the twists in closed form. 

        The following functions are saved in the class and can be code generated afterwards:
            coriolis_cntrifugal_matrix
            generalized_mass_inertia_matrix
            gravity_vector
            inverse_dynamics

        Args:
            q (sympy.Matrix): (n,1) Genearlized position vector.
            qd (sympy.Matrix): (n,1 )Generalized velocity vector.
            q2d (sympy.Matrix): (n,1) Generalized acceleration vector.
            WEE (sympy.Matrix, optional): (6,1) WEE (t) is the time varying wrench on the EE link. Defaults to zeros(6, 1).
            simplify_expressions (bool, optional): Use simplify command on saved expressions. Defaults to True.

        Returns:
            sympy.Matrix: Generalized Forces
        """
        self.var_syms.update(q.free_symbols)
        self.var_syms.update(qd.free_symbols)
        self.var_syms.update(q2d.free_symbols)
        self.var_syms.update(WEE.free_symbols)

        s.n = len(q)

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
        """Compute Inverse of (6x6) Adjoint Matrix for SE(3)

        Args:
            C ([type]): [description]

        Returns:
            [type]: [description]
        """
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
        """Compute (6x6) Adjoint Matrix for SE(3)

        Args:
            C ([type]): [description]

        Returns:
            [type]: [description]
        """
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
        """Compute (6x6) adjoint Matrix for SE(3) 
            - also known as spatial cross product in the literature.

        Args:
            X ([type]): [description]

        Returns:
            [type]: [description]
        """
        ad = Matrix([[0, -X[2, 0], X[1, 0], 0, 0, 0],
                     [X[2, 0], 0, -X[0, 0], 0, 0, 0],
                     [-X[1, 0], X[0, 0], 0, 0, 0, 0],
                     [0, -X[5, 0], X[4, 0], 0, -X[2, 0], X[1, 0]],
                     [X[5, 0], 0, -X[3, 0], X[2, 0], 0, -X[0, 0]],
                     [-X[4, 0], X[3, 0], 0, -X[1, 0], X[0, 0], 0]])
        return ad

    def SE3Exp(self, XX, t):
        """compute exponential mapping for SE(3).

        Args:
            XX ([type]): [description]
            t ([type]): [description]

        Returns:
            [type]: [description]
        """
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
        """Compute analytical inverse of exponential mapping for SE(3).

        Args:
            C ([type]): [description]

        Returns:
            [type]: [description]
        """
        CInv = Matrix([[C[0, 0], C[1, 0], C[2, 0], -C[0, 0]*C[0, 3]-C[1, 0]*C[1, 3]-C[2, 0]*C[2, 3]],
                       [C[0, 1], C[1, 1], C[2, 1], -C[0, 1] *
                           C[0, 3]-C[1, 1]*C[1, 3]-C[2, 1]*C[2, 3]],
                       [C[0, 2], C[1, 2], C[2, 2], -C[0, 2] *
                           C[0, 3]-C[1, 2]*C[1, 3]-C[2, 2]*C[2, 3]],
                       [0, 0, 0, 1]])
        return CInv

    def SO3Exp(self, x, t):
        """Compute exponential mapping for SO(3).

        Args:
            x ([type]): [description]
            t ([type]): [description]

        Returns:
            [type]: [description]
        """
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
        """Build mass-inertia matrix in SE(3) from mass, inertia and center of mass information.

        Args:
            m (float): Mass.
            Theta (sympy.Matrix): Inertia.
            COM (sympy.Matrix): Center of Mass.

        Returns:
            sympy.Matrix: Mass-inertia Matrix.
        """
        M = Matrix([[Theta[0, 0], Theta[0, 1], Theta[0, 2], 0, (-COM[2])*m, COM[1]*m],
                    [Theta[0, 1], Theta[1, 1], Theta[1, 2],
                        COM[2]*m, 0, (-COM[0]*m)],
                    [Theta[0, 2], Theta[1, 2], Theta[2, 2],
                        (-COM[1])*m, COM[0]*m, 0],  # TODO: Code generation

                    [0, COM[2]*m, (-COM[1]*m), m, 0, 0],
                    [(-COM[2])*m, 0, COM[0]*m, 0, m, 0],
                    [COM[1]*m, (-COM[0])*m, 0, 0, 0, m]])
        return M

    def dhToScrewCoord(self, DH_param_table):
        """Build screw coordinate paramters (joint axis frames and body reference frames) 
        from a given modified Denavit-Hartenberg (DH) paramter table.
        
        Args:
            DH_param_table (array_like): Table with modified DH parameters (n,5) -> (gamma,alpha,d,theta,r)
        """
        number_of_frames = DH_param_table.shape[0]
        self.B = []
        self.X = []
        for i in range(number_of_frames):
            # Reference configurations of bodies (i.e. of body-fixed reference frames) w.r.t their previous bodies  
            # gamma, alpha, d, theta,r
            frame = DH_param_table[i,:]
            gamma = frame[0]
            alpha = frame[1]
            d = frame[2]
            theta = frame[3]
            r = frame[4]

            self.B.append(self.SO3Exp(Matrix([1,0,0]),alpha).row_join(Matrix([d,0,0])).col_join(Matrix([0,0,0,1]).T)
                          * self.SO3Exp(Matrix([0,0,1], theta)).row_join(Matrix([0,0,r])).col_join(Matrix([0,0,0,1]).T))
 
            #  Joint screw coordinates in body-fixed representation
            if gamma == 0:
                self.X.append(Matrix([0,0,1,0,0,0]))
            else:
                self.X.append(Matrix([0,0,0,0,0,1]))

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
    e1 = Matrix([0, 0, 1]) # joint axis of revolute joint
    y1 = Matrix([0, 0, 0]) # Vector to joint axis from inertial Frame
    s.Y.append(Matrix([e1, y1.cross(e1)])) # Joint screw coordinates in spacial representation
    
    e2 = Matrix([0, 0, 1]) # joint axis of revolute joint
    y2 = Matrix([L1, 0, 0]) # Vector to joint axis from inertial Frame
    s.Y.append(Matrix([e2, y2.cross(e2)])) # Joint screw coordinates in spacial representation

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
    

    # Kinematics
    F = s.closed_form_kinematics_body_fixed(q, qd, q2d)
    Q = s.closed_form_inv_dyn_body_fixed(q, qd, q2d)
    s.generateCode(python=True, C=True, Matlab=True,
                   use_global_vars=True, name="plant", project="Project")
