import queue
from sympy import *
from sympy.printing.numpy import NumPyPrinter
from sympy.utilities.codegen import codegen
from urdfpy import URDF
from multiprocessing import Process, Queue
from time import sleep

# import multiprocessing, logging
# mpl = multiprocessing.log_to_stderr()
# mpl.setLevel(logging.DEBUG)
# mpl.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
# fileHandler = logging.FileHandler("log.log")
# fileHandler.setFormatter(formatter)
# fileHandler.setLevel(logging.DEBUG)
# mpl.addHandler(fileHandler)
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

        # temporary vars
        self._FK_C = None
        self._A = None
        # J
        self._a = None
        self._V = None

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

        # Multiprocessing
        self.queue_dict = {}
        self.process_dict = {}

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
        # create Folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        # dict of function names and functions
        all_functions = {"forward_kinematics": self.fkin,
                         "system_jacobian_matrix": self.J,
                         "body_jacobian_matrix": self.Jb,
                         "hybrid_jacobian_matrix": self.Jh,
                         "system_jacobian_dot": self.Jdot,
                         "body_twist_ee": self.Vb_ee,
                         "hybrid_twist_ee": self.Vh_ee,
                         "body_jacobian_matrix_ee": self.Jb_ee,
                         "hybrid_jacobian_matrix_ee": self.Jh_ee,
                         "generalized_mass_inertia_matrix": self.M,
                         "coriolis_centrifugal_matrix": self.C,
                         "gravity_vector": self.Qgrav,
                         "inverse_dynamics": self.Q,
                         "hybrid_acceleration": self.Vhd_BFn,
                         "body_acceleration": self.Vbd_BFn,
                         "hybrid_acceleration_ee": self.Vhd_ee,
                         "body_acceleration_ee": self.Vbd_ee,
                         "hybrid_jacobian_matrix_dot": self.Jh_dot,
                         "body_jacobian_matrix_dot": self.Jb_dot,
                         "hybrid_jacobian_matrix_ee_dot": self.Jh_ee_dot,
                         "body_jacobian_matrix_ee_dot": self.Jb_ee_dot}  # not included: self.Vh_BFn, self.Vb_BFn,

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
            print("Generate Python code")
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
            print("Done")

        if C:
            print("Generate C code")
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
            print("Done")

        if Matlab:
            print("Generate Matlab code")
            for i in range(len(functions)):
                if use_global_vars:
                    [(m_name, m_code)] = codegen((names[i], functions[i]), "Octave",
                                                 project=project, header=False, empty=True, global_vars=constant_syms)
                else:
                    [(m_name, m_code)] = codegen((names[i], functions[i]),
                                                 "Octave", project=project, header=False, empty=True)

                with open(os.path.join(folder, m_name), "w+") as f:
                    f.write(m_code)
            print("Done")

    def closed_form_kinematics_body_fixed(self, q, qd, q2d, simplify_expressions=True):
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
        print("Forward kinematics calculation")
        self.var_syms.update(q.free_symbols)
        self.var_syms.update(qd.free_symbols)
        self.var_syms.update(q2d.free_symbols)

        self.n = len(q)

        if self._FK_C is not None:
            FK_C = self._FK_C
        elif self.A is not None:
            print("Using absolute configuration (A) of the body frames")
            FK_f = [self.SE3Exp(self.Y[0], q[0])]
            FK_C = [FK_f[0]*self.A[0]]
            for i in range(1, self.n):
                FK_f.append(FK_f[i-1]*self.SE3Exp(self.Y[i], q[i]))
                FK_C.append(FK_f[i]*self.A[i])
            self._FK_C = FK_C
        elif self.B is not None:
            print('Using relative configuration (B) of the body frames')
            FK_C = [self.B[0]*self.SE3Exp(self.X[0], q[0])]
            for i in range(1, self.n):
                FK_C.append(FK_C[i-1]*self.B[i]*self.SE3Exp(self.X[i], q[i]))
            # if simplify_expressions:
            #     FK_C = simplify(FK_C)
            self._FK_C = FK_C
        else:
            'Absolute (A) or Relative (B) configuration of the bodies should be provided in class!'
            return

        fkin = FK_C[self.n-1]*self.ee
        if simplify_expressions:
            fkin = simplify(fkin)
        self.fkin = fkin

        # Block diagonal matrix A (6n x 6n) of the Adjoint of body frame
        if self._A is not None:
            A = self._A
        else:
            A = Matrix(Identity(6*self.n))
            for i in range(self.n):
                for j in range(i):
                    Crel = self.SE3Inv(FK_C[i])*FK_C[j]
                    AdCrel = self.SE3AdjMatrix(Crel)
                    r = 6*(i)
                    c = 6*(j)
                    A[r:r+6, c:c+6] = AdCrel
            # if simplify_expressions:
                # A = simplify(A)
            self._A = A

        if self.J is not None:
            J = self.J
            V = self._V
        else:
            # Block diagonal matrix X (6n x n) of the screw coordinate vector associated to all joints in the body frame (Constant)
            X = zeros(6*self.n, self.n)
            for i in range(self.n):
                X[6*i:6*i+6, i] = self.X[i]

            # System level Jacobian
            J = A*X
            if simplify_expressions:
                J = simplify(J)
            self.J = J

            # System twist (6n x 1)
            V = J*qd
            self._V = V

        # Different Jacobians
        R_i = Matrix(fkin[:3, :3]).row_join(
            zeros(3, 1)).col_join(Matrix([0, 0, 0, 1]).T)
        if simplify_expressions:  # fastens later simmplifications
            R_i = simplify(R_i)

        R_BFn = Matrix(FK_C[-1][:3, :3]).row_join(
            zeros(3, 1)).col_join(Matrix([0, 0, 0, 1]).T)

        # Body fixed Jacobian of last moving body (This may not correspond to end-effector frame)
        Jb = J[-6:, :]
        if simplify_expressions:
            Jb = simplify(Jb)

        Vb_BFn = Jb*qd  # Body fixed twist of last moving body
        if simplify_expressions:
            Vb_BFn = simplify(Vb_BFn)
        Vh_BFn = self.SE3AdjMatrix(R_BFn)*Vb_BFn
        if simplify_expressions:
            Vh_BFn = simplify(Vh_BFn)
        self.Vb_BFn = Vb_BFn
        self.Vh_BFn = Vh_BFn

        # Body fixed twist of end-effector frame
        Vb_ee = self.SE3AdjMatrix(self.SE3Inv(self.ee))*Vb_BFn
        if simplify_expressions:
            Vb_ee = simplify(Vb_ee)
        # Hybrid twist of end-effector frame
        Vh_ee = self.SE3AdjMatrix(R_i)*Vb_ee
        if simplify_expressions:
            Vh_ee = simplify(Vh_ee)

        self.Vb_ee = Vb_ee
        self.Vh_ee = Vh_ee

        # Body fixed Jacobian of end-effector frame
        Jb_ee = self.SE3AdjMatrix(self.SE3Inv(self.ee))*Jb
        if simplify_expressions:
            Jb_ee = simplify(Jb_ee)

        # Hybrid Jacobian of end-effector frame
        Jh_ee = self.SE3AdjMatrix(R_i)*Jb_ee
        Jh = self.SE3AdjMatrix(R_i)*Jb  # Hybrid Jacobian of last moving body

        if simplify_expressions:
            Jh_ee = simplify(Jh_ee)
            Jh = simplify(Jh)

        self.Jh_ee = Jh_ee
        self.Jb_ee = Jb_ee
        self.Jh = Jh
        self.Jb = Jb

        # Acceleration computations
        if self._a is not None:
            a = self._a
        else:
            # Block diagonal matrix a (6n x 6n)
            a = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                a[6*i:6*i+6, 6*i:6*i+6] = self.SE3adMatrix(self.X[i])*qd[i]
            if simplify_expressions:
                a = simplify(a)
            self._a = a

        # System acceleration (6n x 1)
        Jdot = -A*a*J  # Sys-level Jacobian time derivative
        if simplify_expressions:
            Jdot = simplify(Jdot)

        self.Jdot = Jdot

        Vbd = J*q2d - A*a*V

        # Hybrid acceleration of the last body
        Vbd_BFn = Vbd[-6:, :]
        if simplify_expressions:
            Vbd_BFn = simplify(Vbd_BFn)
        Vhd_BFn = self.SE3AdjMatrix(R_BFn)*Vbd_BFn + self.SE3adMatrix(Matrix(Vh_BFn[:3, :]).col_join(
            Matrix([0, 0, 0])))*self.SE3AdjMatrix(R_BFn)*Vb_BFn  # Hybrid twist of end-effector frame

        if simplify_expressions:
            Vhd_BFn = simplify(Vhd_BFn)

        self.Vbd_BFn = Vbd_BFn
        self.Vhd_BFn = Vhd_BFn

        # Body fixed twist of end-effector frame
        # Hybrid acceleration of the EE
        Vbd_ee = self.SE3AdjMatrix(self.SE3Inv(self.ee))*Vbd_BFn
        if simplify_expressions:
            Vbd_ee = simplify(Vbd_ee)
        Vhd_ee = self.SE3AdjMatrix(R_i)*Vbd_ee + self.SE3adMatrix(Matrix(
            Vh_ee[:3, :]).col_join(Matrix([0, 0, 0])))*self.SE3AdjMatrix(R_i)*Vb_ee  # Hybrid twist of end-effector frame
        if simplify_expressions:
            Vhd_ee = simplify(Vhd_ee)

        self.Vbd_ee = Vbd_ee
        self.Vhd_ee = Vhd_ee

        # Body Jacobian time derivative

        # For the last moving body
        Jb_dot = Jdot[-6:, :]
        self.Jb_dot = Jb_dot

        # For the EE
        Jb_ee_dot = self.SE3AdjMatrix(self.SE3Inv(self.ee))*Jb_dot
        if simplify_expressions:
            Jb_ee_dot = simplify(Jb_ee_dot)
        self.Jb_ee_dot = Jb_ee_dot

        # Hybrid Jacobian time derivative
        # For the last moving body
        Jh_dot = self.SE3AdjMatrix(R_BFn)*Jb_dot + self.SE3adMatrix(
            Matrix(Vh_BFn[:3, :]).col_join(Matrix([0, 0, 0])))*self.SE3AdjMatrix(R_BFn)*Jb
        if simplify_expressions:
            Jh_dot = simplify(Jh_dot)
        self.Jh_dot = Jh_dot

        # For the EE
        Jh_ee_dot = self.SE3AdjMatrix(R_i)*Jb_ee_dot + self.SE3adMatrix(
            Matrix(Vh_ee[:3, :]).col_join(Matrix([0, 0, 0])))*self.SE3AdjMatrix(R_i)*Jb_ee
        if simplify_expressions:
            Jh_ee_dot = simplify(Jh_ee_dot)
        self.Jh_ee_dot = Jh_ee_dot

        print("Done")
        return fkin

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
        print("Inverse dynamics calculation")

        self.var_syms.update(q.free_symbols)
        self.var_syms.update(qd.free_symbols)
        self.var_syms.update(q2d.free_symbols)
        self.var_syms.update(WEE.free_symbols)

        self.n = len(q)

        if self._FK_C is not None:
            FK_C = self._FK_C
        elif self.A is not None:
            print("Using absolute configuration (A) of the body frames")
            FK_f = [self.SE3Exp(self.Y[0], q[0])]
            FK_C = [FK_f[0]*self.A[0]]
            for i in range(1, self.n):
                FK_f.append(FK_f[i-1]*self.SE3Exp(self.Y[i], q[i]))
                FK_C.append(FK_f[i]*self.A[i])
            self._FK_C = FK_C
        elif self.B is not None:
            print('Using relative configuration (B) of the body frames')
            FK_C = [self.B[0]*self.SE3Exp(self.X[0], q[0])]
            for i in range(1, self.n):
                FK_C.append(FK_C[i-1]*self.B[i]*self.SE3Exp(self.X[i], q[i]))
            self._FK_C = FK_C
        else:
            'Absolute (A) or Relative (B) configuration of the bodies should be provided in class!'
            return

        # Block diagonal matrix A (6n x 6n) of the Adjoint of body frame
        if self._A is not None:
            A = self._A
        else:
            A = Matrix(Identity(6*self.n))
            for i in range(self.n):
                for j in range(i):
                    Crel = self.SE3Inv(FK_C[i])*FK_C[j]
                    AdCrel = self.SE3AdjMatrix(Crel)
                    r = 6*(i)
                    c = 6*(j)
                    A[r:r+6, c:c+6] = AdCrel
            self._A = A

        if self.J is not None:
            J = self.J
            V = self._V
        else:
            # Block diagonal matrix X (6n x n) of the screw coordinate vector associated to all joints in the body frame (Constant)
            X = zeros(6*self.n, self.n)
            for i in range(self.n):
                X[6*i:6*i+6, i] = self.X[i]

            # System level Jacobian
            J = A*X
            if simplify_expressions:
                J = simplify(J)
            self.J = J

            # System twist (6n x 1)
            V = J*qd
            self._V = V

        # Acceleration computations

        if self._a is not None:
            a = self._a
        else:
            # Block diagonal matrix a (6n x 6n)
            a = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                a[6*i:6*i+6, 6*i:6*i+6] = self.SE3adMatrix(self.X[i])*qd[i]
            self._a = a

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
        if simplify_expressions:
            M = simplify(M)

        # Coriolis-Centrifugal matrix in joint space (n x n)
        C = J.T * Cb * J
        if simplify_expressions:
            C = simplify(C)

        # Gravity Term
        U = self.SE3AdjInvMatrix(FK_C[0])
        for k in range(1, self.n):
            U = U.col_join(self.SE3AdjInvMatrix(FK_C[k]))

        Vd_0 = zeros(6, 1)
        Vd_0[3:6, 0] = self.gravity_vector
        Qgrav = J.T*Mb*U*Vd_0
        if simplify_expressions:
            Qgrav = simplify(Qgrav)

        # External Wrench
        Wext = zeros(6*self.n, 1)
        # WEE (t) is the time varying wrench on the EE link.
        Wext[-6:, 0] = WEE
        Qext = J.T * Wext

        # Generalized forces Q
        # Q = M*q2d + C*qd   # without gravity
        Q = M*q2d + C*qd + Qgrav + Qext

        if simplify_expressions:
            Q = simplify(Q)

        self.M = M
        self.C = C
        self.Q = Q
        self.Qgrav = Qgrav

        print("Done")
        return Q

    def _set_value_as_process(self, name, target):
        """Set return value of taget as value to queue in self.queue_dict with identifier name

        Args:
            name (str): Identifier
            target (function): function, which returns value 
        """
        if name not in self.queue_dict:
            self.queue_dict[name] = Queue()
        if name in self.process_dict:
            print("already there")
        self.process_dict[name] = Process(
            target=lambda: self._set_value(name, target()), args=(), name=name)
        self.process_dict[name].start()

    def _set_value(self, name, var):
        """Set value to queue in self.queue_dict

        Args:
            name (str): Identifier
            var (any): Value to save
        """
        if name not in self.queue_dict:
            self.queue_dict[name] = Queue()
        self.queue_dict[name].put(var)

    def _start_simplificaton_process(self, name):
        """Start Process, which simplifies and overwrites value in queue from self.queue_dict

        Args:
            name (str): Identifier
        """
        if name not in self.queue_dict:
            self.queue_dict[name] = Queue()
        self.process_dict[name+"_simplify"] = Process(
            target=self._simplify_parallel, args=(name,), name=name+"_simplify")
        self.process_dict[name+"_simplify"].start()

    def _get_value(self, name):
        """Get value from queue in self.queue_dict and put it in again

        Args:
            name (str): Identifier

        Returns:
            any: Value
        """
        value = self.queue_dict[name].get()
        self.queue_dict[name].put(value)
        return value

    def _simplify_parallel(self, name):
        """Take value from self.queue_dict, simplify it and put it in again.

        Args:
            name (str): Identifier
        """
        value = simplify(self.queue_dict[name].get())
        self.queue_dict[name].put(value)

    def _flush_queue(self, q):
        """Flush all items in queue

        Args:
            q (Queue): Queue to flush
        """
        try:
            while True:
                q.get(block=False)
        except queue.Empty:
            pass

    def closed_form_kinematics_body_fixed_parallel(self, q, qd, q2d, simplify_expressions=True):
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
        print("Forward kinematics calculation")
        self.var_syms.update(q.free_symbols)
        self.var_syms.update(qd.free_symbols)
        self.var_syms.update(q2d.free_symbols)

        self.n = len(q)

        if self._FK_C is not None:
            FK_C = self._FK_C
        elif self.A is not None:
            print("Using absolute configuration (A) of the body frames")
            FK_f = [self.SE3Exp(self.Y[0], q[0])]
            FK_C = [FK_f[0]*self.A[0]]
            for i in range(1, self.n):
                FK_f.append(FK_f[i-1]*self.SE3Exp(self.Y[i], q[i]))
                FK_C.append(FK_f[i]*self.A[i])
            self._FK_C = FK_C
        elif self.B is not None:
            print('Using relative configuration (B) of the body frames')
            FK_C = [self.B[0]*self.SE3Exp(self.X[0], q[0])]
            for i in range(1, self.n):
                FK_C.append(FK_C[i-1]*self.B[i]*self.SE3Exp(self.X[i], q[i]))
            self._FK_C = FK_C
        else:
            'Absolute (A) or Relative (B) configuration of the bodies should be provided in class!'
            return

        self._set_value("fkin", FK_C[self.n-1]*self.ee)
        if simplify_expressions:
            self._start_simplificaton_process("fkin")

        # Block diagonal matrix A (6n x 6n) of the Adjoint of body frame
        if self._A is not None:
            A = self._A
        else:
            A = Matrix(Identity(6*self.n))
            for i in range(self.n):
                for j in range(i):
                    Crel = self.SE3Inv(FK_C[i])*FK_C[j]
                    AdCrel = self.SE3AdjMatrix(Crel)
                    r = 6*(i)
                    c = 6*(j)
                    A[r:r+6, c:c+6] = AdCrel
            self._A = A

        if self.J is not None:
            self._set_value("J", self.J)
            self._set_value("V", self._V)
        else:
            # Block diagonal matrix X (6n x n) of the screw coordinate vector associated to all joints in the body frame (Constant)
            X = zeros(6*self.n, self.n)
            for i in range(self.n):
                X[6*i:6*i+6, i] = self.X[i]

            # System level Jacobian
            # J = A*X
            self._set_value("J", A*X)
            if simplify_expressions:
                self._start_simplificaton_process("J")

            # System twist (6n x 1)
            # V = J*qd
            self._set_value_as_process("V", lambda: self._get_value("J")*qd)

        # Different Jacobians
        self._set_value_as_process("R_i", lambda: Matrix(self._get_value("fkin")[:3, :3]).row_join(
            zeros(3, 1)).col_join(Matrix([0, 0, 0, 1]).T))

        if simplify_expressions:  # fastens later simmplifications
            self._start_simplificaton_process("R_i")

        self._set_value("R_BFn", Matrix(FK_C[-1][:3, :3]).row_join(
            zeros(3, 1)).col_join(Matrix([0, 0, 0, 1]).T))

        # Body fixed Jacobian of last moving body (This may not correspond to end-effector frame)
        self._set_value_as_process("Jb", lambda: self._get_value("J")[-6:, :])
        # Jb = J[-6:, :]
        if simplify_expressions:
            self._start_simplificaton_process("Jb")

        self._set_value_as_process("Vb_BFn", lambda: self._get_value("Jb")*qd)
        # Vb_BFn = Jb*qd  # Body fixed twist of last moving body
        if simplify_expressions:
            self._start_simplificaton_process("Vb_BFn")
        # Vh_BFn = self.SE3AdjMatrix(R_BFn)*Vb_BFn
        self._set_value_as_process("Vh_BFn", lambda: self.SE3AdjMatrix(
            self._get_value("R_BFn"))*self._get_value("Vb_BFn"))
        if simplify_expressions:
            self._start_simplificaton_process("Vh_BFn")

        # Body fixed twist of end-effector frame
        # Vb_ee = self.SE3AdjMatrix(self.SE3Inv(self.ee))*Vb_BFn
        self._set_value_as_process("Vb_ee", lambda: self.SE3AdjMatrix(
            self.SE3Inv(self.ee))*self._get_value("Vb_BFn"))
        if simplify_expressions:
            self._start_simplificaton_process("Vb_ee")
        # Hybrid twist of end-effector frame
        # Vh_ee = self.SE3AdjMatrix(R_i)*Vb_ee
        self._set_value_as_process("Vh_ee", lambda: self.SE3AdjMatrix(
            self._get_value("R_i"))*self._get_value("Vb_ee"))
        if simplify_expressions:
            self._start_simplificaton_process("Vh_ee")

        # Body fixed Jacobian of end-effector frame
        # Jb_ee = self.SE3AdjMatrix(self.SE3Inv(self.ee))*Jb
        self._set_value_as_process("Jb_ee", lambda: self.SE3AdjMatrix(
            self.SE3Inv(self.ee))*self._get_value("Jb"))
        if simplify_expressions:
            self._start_simplificaton_process("Jb_ee")

        # Hybrid Jacobian of end-effector frame
        # Jh_ee = self.SE3AdjMatrix(R_i)*Jb_ee
        self._set_value_as_process("Jh_ee", lambda: self.SE3AdjMatrix(
            self._get_value("R_i"))*self._get_value("Jb_ee"))
        # Jh = self.SE3AdjMatrix(R_i)*Jb  # Hybrid Jacobian of last moving body
        self._set_value_as_process("Jh", lambda: self.SE3AdjMatrix(
            self._get_value("R_i"))*self._get_value("Jb"))

        if simplify_expressions:
            self._start_simplificaton_process("Jh_ee")
            self._start_simplificaton_process("Jh")

        # Acceleration computations
        if self._a is not None:
            self._set_value("a", self._a)
        else:
            # Block diagonal matrix a (6n x 6n)
            a = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                a[6*i:6*i+6, 6*i:6*i+6] = self.SE3adMatrix(self.X[i])*qd[i]
            self._set_value("a", a)
            if simplify_expressions:
                self._start_simplificaton_process("a")

        # System acceleration (6n x 1)
        # Jdot = -A*a*J  # Sys-level Jacobian time derivative
        self._set_value_as_process(
            "Jdot", lambda: -A*self._get_value("a")*self._get_value("J"))
        if simplify_expressions:
            self._start_simplificaton_process("Jdot")

        # self.Jdot = Jdot

        # Vbd = J*q2d - A*a*V
        self._set_value_as_process("Vbd", lambda: self._get_value(
            "J")*q2d - A*self._get_value("a")*self._get_value("V"))

        # Hybrid acceleration of the last body
        # Vbd_BFn = Vbd[-6:, :]
        self._set_value_as_process(
            "Vbd_BFn", lambda: self._get_value("Vbd")[-6:, :])

        if simplify_expressions:
            self._start_simplificaton_process("Vbd_BFn")

        # Vhd_BFn = self.SE3AdjMatrix(R_BFn)*Vbd_BFn + self.SE3adMatrix(Matrix(Vh_BFn[:3, :]).col_join(
            # Matrix([0, 0, 0])))*self.SE3AdjMatrix(R_BFn)*Vb_BFn  # Hybrid twist of end-effector frame
        self._set_value_as_process("Vhd_BFn", lambda: self.SE3AdjMatrix(self._get_value("R_BFn"))*self._get_value("Vbd_BFn") + self.SE3adMatrix(Matrix(self._get_value("Vh_BFn")[:3, :]).col_join(
            Matrix([0, 0, 0])))*self.SE3AdjMatrix(self._get_value("R_BFn"))*self._get_value("Vb_BFn"))

        if simplify_expressions:
            self._start_simplificaton_process("Vhd_BFn")

        # Body fixed twist of end-effector frame
        # Hybrid acceleration of the EE
        # Vbd_ee = self.SE3AdjMatrix(self.SE3Inv(self.ee))*Vbd_BFn
        self._set_value_as_process("Vbd_ee", lambda: self.SE3AdjMatrix(
            self.SE3Inv(self.ee))*self._get_value("Vbd_BFn"))
        if simplify_expressions:
            self._start_simplificaton_process("Vbd_ee")
        # Vhd_ee = self.SE3AdjMatrix(R_i)*Vbd_ee + self.SE3adMatrix(Matrix(
        #     Vh_ee[:3, :]).col_join(Matrix([0, 0, 0])))*self.SE3AdjMatrix(R_i)*Vb_ee  # Hybrid twist of end-effector frame
        self._set_value_as_process("Vhd_ee", lambda: self.SE3AdjMatrix(self._get_value("R_i")) * self._get_value("Vbd_ee") + self.SE3adMatrix(Matrix(
            self._get_value("Vh_ee")[:3, :]).col_join(Matrix([0, 0, 0])))*self.SE3AdjMatrix(self._get_value("R_i"))*self._get_value("Vb_ee"))  # Hybrid twist of end-effector frame

        if simplify_expressions:
            self._start_simplificaton_process("Vhd_ee")

        # Body Jacobian time derivative

        # For the last moving body
        # Jb_dot = Jdot[-6:, :]
        self._set_value_as_process(
            "Jb_dot", lambda: self._get_value("Jdot")[-6:, :])

        # For the EE
        # Jb_ee_dot = self.SE3AdjMatrix(self.SE3Inv(self.ee))*Jb_dot
        self._set_value_as_process("Jb_ee_dot", lambda: self.SE3AdjMatrix(
            self.SE3Inv(self.ee))*self._get_value("Jb_dot"))
        if simplify_expressions:
            self._start_simplificaton_process("Jb_ee_dot")

        # Hybrid Jacobian time derivative
        # For the last moving body
        # Jh_dot = self.SE3AdjMatrix(R_BFn)*Jb_dot + self.SE3adMatrix(
        #     Matrix(Vh_BFn[:3, :]).col_join(Matrix([0, 0, 0])))*self.SE3AdjMatrix(R_BFn)*Jb
        self._set_value_as_process("Jh_dot", lambda: self.SE3AdjMatrix(self._get_value("R_BFn"))*self._get_value("Jb_dot") + self.SE3adMatrix(
            Matrix(self._get_value("Vh_BFn")[:3, :]).col_join(Matrix([0, 0, 0])))*self.SE3AdjMatrix(self._get_value("R_BFn"))*self._get_value("Jb"))
        if simplify_expressions:
            self._start_simplificaton_process("Jh_dot")
        # self.Jh_dot = Jh_dot

        # For the EE
        # Jh_ee_dot = self.SE3AdjMatrix(R_i)*Jb_ee_dot + self.SE3adMatrix(
        #     Matrix(Vh_ee[:3, :]).col_join(Matrix([0, 0, 0])))*self.SE3AdjMatrix(R_i)*Jb_ee
        self._set_value_as_process("Jh_ee_dot", lambda: self.SE3AdjMatrix(self._get_value("R_i"))*self._get_value("Jb_ee_dot") + self.SE3adMatrix(
            Matrix(self._get_value("Vh_ee")[:3, :]).col_join(Matrix([0, 0, 0])))*self.SE3AdjMatrix(self._get_value("R_i"))*self._get_value("Jb_ee"))
        if simplify_expressions:
            self._start_simplificaton_process("Jh_ee_dot")
        self._a = self._get_value("a")
        self._V = self._get_value("V")

        # variables for Code Generation:
        self.fkin = self._get_value("fkin")
        self.J = self._get_value("J")
        self.Jb = self._get_value("Jb")
        self.Jh = self._get_value("Jh")
        self.Jdot = self._get_value("Jdot")
        self.Vb_ee = self._get_value("Vb_ee")
        self.Vh_ee = self._get_value("Vh_ee")
        self.Jb_ee = self._get_value("Jb_ee")
        self.Jh_ee = self._get_value("Jh_ee")
        self.Vh_BFn = self._get_value("Vh_BFn")
        self.Vb_BFn = self._get_value("Vb_BFn")
        self.Vhd_BFn = self._get_value("Vhd_BFn")
        self.Vbd_BFn = self._get_value("Vbd_BFn")
        self.Vhd_ee = self._get_value("Vhd_ee")
        self.Vbd_ee = self._get_value("Vhd_ee")
        self.Jh_dot = self._get_value("Jh_dot")
        self.Jb_dot = self._get_value("Jb_dot")
        self.Jh_ee_dot = self._get_value("Jh_ee_dot")
        self.Jb_ee_dot = self._get_value("Jb_ee_dot")

        # empty Queues
        for i in self.queue_dict:
            self._flush_queue(self.queue_dict[i])
        self.queue_dict = {}

        # join Processes
        for i in self.process_dict:
            self.process_dict[i].join()
        self.process_dict = {}

        print("Done")
        return self.fkin

    def closed_form_inv_dyn_body_fixed_parallel(self, q, qd, q2d, WEE=zeros(6, 1), simplify_expressions=True):
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
        print("Inverse dynamics calculation")

        self.var_syms.update(q.free_symbols)
        self.var_syms.update(qd.free_symbols)
        self.var_syms.update(q2d.free_symbols)
        self.var_syms.update(WEE.free_symbols)

        self.n = len(q)

        if self._FK_C is not None:
            FK_C = self._FK_C
        elif self.A is not None:
            print("Using absolute configuration (A) of the body frames")
            FK_f = [self.SE3Exp(self.Y[0], q[0])]
            FK_C = [FK_f[0]*self.A[0]]
            for i in range(1, self.n):
                FK_f.append(FK_f[i-1]*self.SE3Exp(self.Y[i], q[i]))
                FK_C.append(FK_f[i]*self.A[i])
            self._FK_C = FK_C
        elif self.B is not None:
            print('Using relative configuration (B) of the body frames')
            FK_C = [self.B[0]*self.SE3Exp(self.X[0], q[0])]
            for i in range(1, self.n):
                FK_C.append(FK_C[i-1]*self.B[i]*self.SE3Exp(self.X[i], q[i]))
            self._FK_C = FK_C
        else:
            'Absolute (A) or Relative (B) configuration of the bodies should be provided in class!'
            return

        # Block diagonal matrix A (6n x 6n) of the Adjoint of body frame
        if self._A is not None:
            A = self._A
        else:
            A = Matrix(Identity(6*self.n))
            for i in range(self.n):
                for j in range(i):
                    Crel = self.SE3Inv(FK_C[i])*FK_C[j]
                    AdCrel = self.SE3AdjMatrix(Crel)
                    r = 6*(i)
                    c = 6*(j)
                    A[r:r+6, c:c+6] = AdCrel
            self._A = A

        if self.J is not None:
            self._set_value("J", self.J)
            self._set_value("V", self._V)
        else:
            # Block diagonal matrix X (6n x n) of the screw coordinate vector associated to all joints in the body frame (Constant)
            X = zeros(6*self.n, self.n)
            for i in range(self.n):
                X[6*i:6*i+6, i] = self.X[i]

            # System level Jacobian
            # J = A*X
            self._set_value("J", A*X)
            if simplify_expressions:
                self._start_simplificaton_process("J")

            # System twist (6n x 1)
            # V = J*qd
            self._set_value_as_process("V", lambda: self._get_value("J")*qd)

        # Acceleration computations

        if self._a is not None:
            a = self._a
        else:
            # Block diagonal matrix a (6n x 6n)
            a = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                a[6*i:6*i+6, 6*i:6*i+6] = self.SE3adMatrix(self.X[i])*qd[i]
            self._a = a

        # System acceleration (6n x 1)
        # Vd = J*q2d - A*a*V
        self._set_value_as_process("Vd", lambda: self._get_value(
            "J")*q2d - A*a*self._get_value("V"))

        # Block Diagonal Mb (6n x 6n) Mass inertia matrix in body frame (Constant)
        Mb = zeros(6*self.n, 6*self.n)
        for i in range(self.n):
            Mb[i*6:i*6+6, i*6:i*6+6] = self.Mb[i]

        # Block diagonal matrix b (6n x 6n) used in Coriolis matrix
        def _b():
            nonlocal self
            b = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                b[i*6:i*6+6, i*6:i*6 +
                    6] = self.SE3adMatrix(Matrix(self._get_value("V")[6*i:6*i+6]))
            return b
        self._set_value_as_process("b", _b)

        # Block diagonal matrix Cb (6n x 6n)
        # Cb = -Mb*A*a - b.T * Mb
        self._set_value_as_process(
            "Cb", lambda: -Mb*A*a - self._get_value("b").T * Mb)

        # Lets setup the Equations of Motion

        # Mass inertia matrix in joint space (n x n)
        # M = J.T*Mb*J
        self._set_value_as_process(
            "M", lambda: self._get_value("J").T*Mb*self._get_value("J"))
        if simplify_expressions:
            self._start_simplificaton_process("M")

        # Coriolis-Centrifugal matrix in joint space (n x n)
        # C = J.T * Cb * J
        self._set_value_as_process("C", lambda: self._get_value(
            "J").T*self._get_value("Cb")*self._get_value("J"))
        if simplify_expressions:
            self._start_simplificaton_process("C")

        # Gravity Term
        U = self.SE3AdjInvMatrix(FK_C[0])
        for k in range(1, self.n):
            U = U.col_join(self.SE3AdjInvMatrix(FK_C[k]))

        Vd_0 = zeros(6, 1)
        Vd_0[3:6, 0] = self.gravity_vector
        # Qgrav = J.T*Mb*U*Vd_0
        self._set_value_as_process(
            "Qgrav", lambda: self._get_value("J").T*Mb*U*Vd_0)
        if simplify_expressions:
            # Qgrav = simplify(Qgrav)
            self._start_simplificaton_process("Qgrav")

        # External Wrench
        Wext = zeros(6*self.n, 1)
        # WEE (t) is the time varying wrench on the EE link.
        Wext[-6:, 0] = WEE
        # Qext = J.T * Wext
        self._set_value_as_process("Qext", lambda: self._get_value("J").T * Wext)

        # Generalized forces Q
        # Q = M*q2d + C*qd   # without gravity
        # Q = M*q2d + C*qd + Qgrav + Qext
        self._set_value_as_process("Q", lambda: self._get_value(
            "M")*q2d + self._get_value("C")*qd + self._get_value("Qgrav") + self._get_value("Qext"))

        if simplify_expressions:
            self._start_simplificaton_process("Q")

        self._V = self._get_value("V")
        self.J = self._get_value("J")
        self.M = self._get_value("M")
        self.C = self._get_value("C")
        self.Qgrav = self._get_value("Qgrav")
        self.Q = self._get_value("Q")

        # empty Queues
        for i in self.queue_dict:
            self._flush_queue(self.queue_dict[i])
        self.queue_dict = {}

        # join Processes
        for i in self.process_dict:
            self.process_dict[i].join()
        self.process_dict = {}

        print("Done")
        return self.Q

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
        xhat = Matrix([[0, -x[2, 0], x[1, 0]],
                       [x[2, 0], 0, -x[0, 0]],
                       [-x[1, 0], x[0, 0], 0]])
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
            frame = DH_param_table[i, :]
            gamma = frame[0]
            alpha = frame[1]
            d = frame[2]
            theta = frame[3]
            r = frame[4]
            self.B.append(self.SO3Exp(Matrix([1, 0, 0]), alpha).row_join(Matrix([d, 0, 0])).col_join(Matrix([0, 0, 0, 1]).T)
                          * self.SO3Exp(Matrix([0, 0, 1]), theta).row_join(Matrix([0, 0, r])).col_join(Matrix([0, 0, 0, 1]).T))

            #  Joint screw coordinates in body-fixed representation
            if gamma == 0:
                self.X.append(Matrix([0, 0, 1, 0, 0, 0]))
            else:
                self.X.append(Matrix([0, 0, 0, 0, 0, 1]))

    # def load_from_urdf(self, path= "/home/hannah/DFKI/hopping_leg/model/with_rails/urdf/v7.urdf"):
    #     robot = URDF.load("/home/hannah/DFKI/hopping_leg/model/with_rails/urdf/v7.urdf")
    #     self.B = []
    #     self.X = []
    #     for joint in robot.joints:
    #         if joint.joint_type == "revolute":
    #             origin = joint.origin
    #             for i in range(4):
    #                 for j in range(4):
    #                     origin[i,j] = nsimplify(origin[i,j], [pi], tolerance=0.00001)
    #             self.B.append(Matrix(origin)) # Probably B
    #             axis = joint.axis
    #             for i in range(3):
    #                 axis[i] = nsimplify(axis[i], [pi], tolerance=0.00001)
    #             self.X.append(Matrix(axis).col_join(Matrix([0,0,0])))
    #     # self.Mb = []
    #     # for link in robot.links:
    #     #     self.Mb.append(self.MassMatrixMixedData())


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
    e1 = Matrix([0, 0, 1])  # joint axis of revolute joint
    y1 = Matrix([0, 0, 0])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spacial representation
    s.Y.append(Matrix([e1, y1.cross(e1)]))

    e2 = Matrix([0, 0, 1])  # joint axis of revolute joint
    y2 = Matrix([L1, 0, 0])  # Vector to joint axis from inertial Frame
    # Joint screw coordinates in spacial representation
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

    # Kinematics
    # F = s.closed_form_kinematics_body_fixed(q, qd, q2d)
    # Q = s.closed_form_inv_dyn_body_fixed(q, qd, q2d)
    F = s.closed_form_kinematics_body_fixed_parallel(q, qd, q2d)
    Q = s.closed_form_inv_dyn_body_fixed_parallel(q, qd, q2d)

    s.generateCode(python=True, C=True, Matlab=True,
                   use_global_vars=True, name="plant", project="Project")
