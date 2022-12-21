import multiprocessing
import os
import queue
import random
import regex
from multiprocessing import Process, Queue
from typing import Union, List, Any, Tuple, Generator, Dict, Callable

import numpy
import sympy
from sympy import (Identity, Matrix, cancel, cos, cse, factor,
                   lambdify, powsimp, sin, symbols, zeros, pi, nsimplify, 
                   octave_code, ccode, MutableDenseMatrix)
from sympy.printing.numpy import NumPyPrinter
from sympy.simplify.cse_main import numbered_symbols
from sympy.simplify.fu import fu
from sympy.utilities.codegen import codegen
from urdfpy import URDF, matrix_to_xyz_rpy

from KinematicsGenerator.matrices import (
    SE3AdjInvMatrix, SE3AdjMatrix, SE3adMatrix, SE3Exp, SE3Inv, SO3Exp, 
    InertiaMatrix, TransformationMatrix, MassMatrixMixedData, rpy_to_matrix, 
    xyz_rpy_to_matrix, generalized_vectors)

class _AbstractCodeGeneration():
    def __init__(self) -> None:
        # variables for Code Generation:
        self.fkin = None  # forward_kinematics
        self.J = None  # system_jacobian_matrix
        self.Jb = None  # body_jacobian_matrix
        self.Jh = None  # hybrid_jacobian_matrix
        self.Jdot = None  # system_jacobian_dot
        self.Vb_ee = None  # body_twist_ee
        self.Vh_ee = None  # hybrid_twist_ee
        self.Jb_ee = None  # body_jacobian_matrix_ee
        self.Jh_ee = None  # hybrid_jacobian_matrix_ee
        self.M = None  # generalized_mass_inertia_matrix
        self.C = None  # coriolis_centrifugal_matrix
        self.Qgrav = None  # gravity_vector
        self.Q = None  # inverse_dynamics
        self.Vh_BFn = None
        self.Vb_BFn = None
        self.Vhd_BFn = None  # hybrid_acceleration
        self.Vbd_BFn = None  # body_acceleration
        self.Vhd_ee = None  # hybrid_acceleration_ee
        self.Vbd_ee = None  # body_acceleration_ee
        self.Jh_dot = None  # hybrid_jacobian_matrix_dot
        self.Jb_dot = None  # body_jacobian_matrix_dot
        self.Jh_ee_dot = None  # hybrid_jacobian_matrix_ee_dot
        self.Jb_ee_dot = None  # body_jacobian_matrix_ee_dot

        # set of variable symbols to use in generated functions as arguments
        self.var_syms = set()
        
        # Value assignment
        # dict with assigned variables for code generation
        self.assignment_dict = {}  
        # dict for subexpressions fro common subexpression elimination
        self.subex_dict = {}  

        self.all_symbols = set()  # set with all used symbols


    def get_expressions_dict(self, filterNone: bool=True) -> Dict[str,Union[sympy.Expr, None]]:
        """Get dictionary with expression names (key) and generated 
        expressions (value).

        Args:
            filterNone (bool, optional): 
                Exclude expressions which haven't been generate yet. 
                Defaults to True.

        Returns:
            dict: dictionary with expression names (key) and generated 
                expressions (value).
        """
        # all expressions in this dictionary can be code generated 
        # using the generate_code function.
        all_expressions = {"forward_kinematics": self.fkin,
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
                           "body_jacobian_matrix_ee_dot": self.Jb_ee_dot}  
                            # not included: self.Vh_BFn, self.Vb_BFn
        # exclude expressions which are None
        if filterNone:
            filtered = {k: v for k, v in all_expressions.items()
                        if v is not None}
            return filtered
        return all_expressions

    def generateCode(self, python: bool=False, C: bool=False, Matlab: bool=False, cython: bool=False,
                     folder: str="./generated_code", use_global_vars: bool=True, 
                     name: str="plant", project: str="Project") -> None:
        """Generate code of generated Expressions. 
        It can generate Python, C (C99) and Matlab/Octave code.  
        Needs 'closed_form_inv_dyn_body_fixed' and/or 
        'closed_form_kinematics_body_fixed' to run first.


        Args:
            python (bool, optional): 
                Generate Python code. Defaults to False.
            C (bool, optional): 
                Generate C99 code. Defaults to False.
            Matlab (bool, optional): 
                Generate Matlab/Octave code. Defaults to False.
            cython (bool, optional):
                Generate cython code. Defaults to False.
            folder (str, optional): 
                Folder where to save code. 
                Defaults to "./generated_code".
            use_global_vars (bool, optional): 
                Constant vars like mass etc are no arguments of the 
                generated expressions. Defaults to True.
            name (str, optional): 
                Name of class and file. 
                Defaults to "plant".
            project (str, optional): 
                Project name in C header. Defaults to "Project".

        """
        # create Folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        # dict of expression names and expressions
        all_expressions = self.get_expressions_dict()

        # get individual tuples for expression names and expressions
        (names, expressions) = zip(*all_expressions.items())

        # generate set with all used symbols
        all_syms = set()
        for e in expressions:
            all_syms.update(e.free_symbols)

        if use_global_vars:
            # generate list of constant symbols
            constant_syms = self._sort_variables(
                all_syms.difference(self.var_syms).union(self.subex_dict))
            # generate list with preassigned symbols like subexpressions
            # from common subexpression elimination
            not_assigned_syms = self._sort_variables(
                all_syms
                .difference(self.var_syms)
                .difference(self.assignment_dict)
                .difference(self.subex_dict)
                )
        else:
            constant_syms = []
            not_assigned_syms = []

        if python:
            print("Generate Python code")
            # create folder
            if not os.path.exists(os.path.join(folder, "python")):
                os.mkdir(os.path.join(folder, "python"))

            p = NumPyPrinter()

            # start python file with import
            s = ["import numpy\n\n"]
            # class name
            s.append("class "+name.capitalize()+"():")
            # define __init__ function
            s.append("    def __init__(self, %s):" % (
                ", ".join(
                    [str(not_assigned_syms[i]) 
                     for i in range(len(not_assigned_syms))] 
                    + [str(i)+" = " + str(self.assignment_dict[i]) 
                       for i in self.assignment_dict])))
            if len(not_assigned_syms) > 0:
                s.append("        "
                         + ", ".join(["self."+str(not_assigned_syms[i]) 
                                      for i in range(len(not_assigned_syms))])
                         + " = " 
                         + ", ".join([str(not_assigned_syms[i]) 
                                      for i in range(len(not_assigned_syms))])
                         )

            # append preassigned values to __init__ function
            if len(self.assignment_dict) > 0:
                s.append("        "
                         + ", ".join(sorted(["self."+str(i) 
                                             for i in self.assignment_dict]))
                         + " = " 
                         + ", ".join(sorted([str(i) 
                                             for i in self.assignment_dict]))
                         )

            # append cse expressions to __init__ function
            if len(self.subex_dict) > 0:
                for i in sorted([str(j) for j in self.subex_dict], key=lambda x: int(regex.findall("(?<=sub)\d*",x)[0])):
                    modstring = p.doprint(self.subex_dict[symbols(i)])
                    for j in sorted([str(h) 
                                     for h in self.subex_dict[symbols(i)].free_symbols],
                                    reverse=1):
                        modstring = regex.sub(
                            str(j), "self."+str(j), modstring)
                        # remove double self
                        modstring = regex.sub("self.self.", "self.", modstring)
                    s.append("        self."+str(i)+" = " + modstring)

            # define functions
            for i in range(len(expressions)):
                var_syms = self._sort_variables(self.var_syms.intersection(
                    expressions[i].free_symbols))
                const_syms = self._sort_variables(
                    set(constant_syms).intersection(
                        expressions[i].free_symbols))
                if len(var_syms) > 0:
                    s.append("\n    def "+names[i]+"(self, %s):" % (
                        ", ".join([str(var_syms[i]) 
                                   for i in range(len(var_syms))])))

                else:
                    s.append("\n    def "+names[i]+"(self):")
                # if len(const_syms) > 0:
                #     s.append("        "
                #              + ", ".join([str(const_syms[i]) 
                #                           for i in range(len(const_syms))])
                #              + " = " 
                #              + ", ".join(["self."+str(const_syms[i]) 
                #                           for i in range(len(const_syms))])
                #              )

                s.append("        "
                         + names[i] 
                         + " = " 
                         + regex.sub(f"(?<=\W|^)(?={'|'.join([str(i) for i in const_syms])}(\W|\Z))","self.",p.doprint(expressions[i])))
                        #  + p.doprint(expressions[i]))
                s.append("        return " + names[i])

            # replace numpy with np for better readability
            s = list(map(lambda x: x.replace("numpy.", "np."), s))
            s[0] = "import numpy as np\n\n"

            # join list to string
            s = "\n".join(s)

            # write python file
            with open(os.path.join(folder, "python", name + ".py"), "w+") as f:
                f.write(s)
            print("Done")

        if cython:
            print("Generate Cython code")
            # create folder
            if not os.path.exists(os.path.join(folder, "cython")):
                os.mkdir(os.path.join(folder, "cython"))

            p = NumPyPrinter()

            # start python file with import
            s = ["import numpy"]
            s.append("cimport numpy\n\n")
            
            s.append("numpy.import_array()")
            s.append("DTYPE = numpy.float64")
            s.append("ctypedef numpy.float64_t DTYPE_t")
            
            s.append("\n")
            # class name
            s.append("cdef class "+name.capitalize()+"():")
            s.append("    cdef public double %s\n"%(
                ", ".join(
                    [str(not_assigned_syms[i]) 
                        for i in range(len(not_assigned_syms))] 
                    + [str(i) 
                      for i in self.assignment_dict])))
            if self.subex_dict:
                s.append("    cdef double %s\n"%(
                    ", ".join(
                        [str(i) 
                        for i in self.subex_dict])))
                    
            
            # define __init__ function
            s.append("    def __cinit__(self, %s):" % (
                ", ".join(
                    ["double "+str(not_assigned_syms[i]) 
                     for i in range(len(not_assigned_syms))] 
                    + ["double "+str(i)+" = " + str(self.assignment_dict[i]) 
                       for i in self.assignment_dict])))
            if len(not_assigned_syms) > 0:
                s.append("        "
                         + ", ".join(["self."+str(not_assigned_syms[i]) 
                                      for i in range(len(not_assigned_syms))])
                         + " = " 
                         + ", ".join([str(not_assigned_syms[i]) 
                                      for i in range(len(not_assigned_syms))])
                         )

            # append preassigned values to __init__ function
            if len(self.assignment_dict) > 0:
                s.append("        "
                         + ", ".join(sorted(["self."+str(i) 
                                             for i in self.assignment_dict]))
                         + " = " 
                         + ", ".join(sorted([str(i) 
                                             for i in self.assignment_dict]))
                         )

            # append cse expressions to __init__ function
            if len(self.subex_dict) > 0:
                for i in sorted([str(j) for j in self.subex_dict], key=lambda x: int(regex.findall("(?<=sub)\d*",x)[0])):
                    modstring = p.doprint(self.subex_dict[symbols(i)])
                    for j in sorted([str(h) 
                                     for h in self.subex_dict[symbols(i)].free_symbols],
                                    reverse=1):
                        modstring = regex.sub(
                            str(j), "self."+str(j), modstring)
                        # remove double self
                        modstring = regex.sub("self.self.", "self.", modstring)
                    s.append("        self."+str(i)+" = " + modstring)

            # define functions
            for i in range(len(expressions)):
                var_syms = self._sort_variables(self.var_syms.intersection(
                    expressions[i].free_symbols))
                const_syms = self._sort_variables(
                    set(constant_syms).intersection(
                        expressions[i].free_symbols))
                if len(var_syms) > 0:
                    s.append(f"\n    cpdef "+names[i]+"(self, %s):" % (
                    # s.append(f"\n    cpdef np.ndarray[DTYPE_t,ndim={len(expressions[i].shape)}] "+names[i]+"(self, %s):" % (
                        ", ".join(["double "+str(var_syms[i]) 
                                   for i in range(len(var_syms))])))

                else:
                    s.append("\n    cpdef "+names[i]+"(self):")
                # if len(const_syms) > 0:
                #     s.append("        "
                #              + ", ".join(["double "+str(const_syms[i]) 
                #                           for i in range(len(const_syms))])
                #              + " = " 
                #              + ", ".join(["self."+str(const_syms[i]) 
                #                           for i in range(len(const_syms))])
                #              )
                s.append(f"        cdef numpy.ndarray[DTYPE_t,ndim={len(expressions[i].shape)}] "
                         + names[i])
                s.append("        "
                         + names[i] 
                         + " = " 
                         + regex.sub(f"(?<=\W|^)(?={'|'.join([str(i) for i in const_syms])}(\W|\Z))","self.",p.doprint(expressions[i])))
                        #  + regex.sub("(?<=((?<=[^\.])\W)\d+)(?=\W)(?!\.)",".0",regex.sub(f"(?<=\W|^)(?={'|'.join([str(i) for i in const_syms])}(\W|\Z))","self.",p.doprint(expressions[i]))))
                s.append("        return " + names[i])

            # replace numpy with np for better readability
            s = list(map(lambda x: x.replace("numpy.", "np."), s))
            s[0] = "import numpy as np"
            s[1] = "cimport numpy as np\n\n"

            # join list to string
            s = "\n".join(s)

            # write python file
            with open(os.path.join(folder, "cython", name + ".pyx"), "w+") as f:
                f.write(s)
            print("Done")

        if C:
            print("Generate C code")
            if not os.path.exists(os.path.join(folder, "C")):
                os.mkdir(os.path.join(folder, "C"))

            # generate c files
            if use_global_vars:
                [(c_name, c_code), (h_name, c_header)] = codegen(
                    [tuple((names[i], expressions[i])) 
                     for i in range(len(expressions))],
                    "C99", name, project, header=False, 
                    empty=True, global_vars=constant_syms)
            else:
                [(c_name, c_code), (h_name, c_header)] = codegen(
                    [tuple((names[i], expressions[i])) 
                     for i in range(len(expressions))],
                    "C99", name, project, header=False, empty=True)
            # change strange variable names
            c_code = regex.sub(r"out_\d{10}[\d]+", "out", c_code)
            c_header = regex.sub(r"out_\d{10}[\d]+", "out", c_header)

            c_lines = c_code.splitlines(True)
            i = 0
            # correct dimension of output array pointers
            while i < len(c_lines):
                # find function definition
                if any(n+"(" in c_lines[i] for n in names):
                    # which expression is defined
                    [fname] = [n for n in names if n+"(" in c_lines[i]]
                    # find shape of expression
                    cols = all_expressions[fname].shape[1]
                    i += 1
                    # replace all 1D arrays with 2D arrays for matrices
                    while "}" not in c_lines[i]:
                        out = regex.findall("out\[[\d]+\]", c_lines[i])
                        if out and cols > 1:
                            [num] = regex.findall("[\d]+", out[0])
                            num = int(num)
                            c_lines[i] = c_lines[i].replace(
                                out[0], f"out[{num//cols}][{num%cols}]")
                        i += 1
                i += 1
            c_code = "".join(c_lines)
            
            # save assinged parameters 
            c_def_name = f"{c_name[:-2]}_parameters.c"
            header_insert = []
            c_definitions = [f'#include "{h_name}"\n']
            c_definitions.append("#include <math.h>\n")
            c_definitions.append("\n")
            
            if not_assigned_syms:
                header_insert.append(f"/* Please uncomment and assign values in '{c_def_name}'\n")
                c_definitions.append(f"/* Please assign values and uncomment in '{h_name}'\n")
                for var in sorted([str(i) for i in not_assigned_syms]):
                    header_insert.append(f"extern const float {var};\n")
                    if var == "g":
                        c_definitions.append(f"const float {var} = 9.81;\n")
                    else:
                        c_definitions.append(f"const float {var} = 0;\n")
                header_insert.append("*/ \n")
                c_definitions.append("*/ \n")
            
            for var in sorted([str(i) for i in self.assignment_dict]):
                val = ccode(self.assignment_dict[symbols(var)])
                header_insert.append(f"extern const float {var};\n")
                c_definitions.append(f"const float {var} = {val};\n")


            # append cse expressions
            for var in sorted([str(j) for j in self.subex_dict], key=lambda x: int(regex.findall("(?<=sub)\d*",x)[0])):
                val = ccode(self.subex_dict[symbols(var)])
                header_insert.append(f"extern const float {var};\n")
                c_definitions.append(f"const float {var} = {val};\n")

            
            
            if header_insert:
                header_insert.append("\n")
                h_lines = c_header.splitlines(True)
                for i in range(len(h_lines)):
                    if "#endif" in h_lines[i]:
                        h_lines[i:i] = header_insert 
                        break
                c_header = "".join(h_lines)

            # write code files
            with open(os.path.join(folder, "C", c_name), "w+") as f:
                f.write(c_code)
            with open(os.path.join(folder, "C", h_name), "w+") as f:
                f.write(c_header)
            if header_insert:
                with open(os.path.join(folder, "C", c_def_name), "w+") as f:
                    f.writelines(c_definitions)
                
            print("Done")

        if Matlab:
            m_class = []
            m_class.append(f"classdef {name}\n")
            m_properties = not_assigned_syms[:]
            m_properties.extend([i for i in self.assignment_dict])
            # properties
            m_class.append(f"\tproperties\n")
            for var in m_properties:
                m_class.append(f"\t\t{str(var)}\n")
            # add cse subexpressions as private properties
            if self.subex_dict:
                m_class.append("\tend\n\n")
                m_class.append(f"\tproperties (Access = private)\n")
                for var in [i for i in sorted([str(j) for j in self.subex_dict], key=lambda x: int(regex.findall("(?<=sub)\d*",x)[0]))]:
                    m_class.append(f"\t\t{str(var)}\n")
                # add subex to m_properties list
                m_properties.extend([i for i in sorted([str(j) for j in self.subex_dict], key=lambda x: int(regex.findall("(?<=sub)\d*",x)[0]))])
            m_class.append("\tend\n\n")
            
            # methods
            m_class.append(f"\tmethods\n")
            
            # init function
            # function argumens
            var_substr = ", ".join(
                    [str(not_assigned_syms[i]) 
                     for i in range(len(not_assigned_syms))] 
                    + [str(j) for j in self.assignment_dict])
            m_class.append(f"\t\tfunction obj = {name}({var_substr})\n")
            
            # default values for not assigned variables
            if not_assigned_syms:
                m_class.append(f"\t\t\t% TODO: Assign missing variables here:\n")
                for var in not_assigned_syms:
                    if str(var) == "g":
                        m_class.append(f"\t\t\t% if ~exist('{str(var)}','var'); {str(var)} = 9.81; end\n")
                    else:
                        m_class.append(f"\t\t\t% if ~exist('{str(var)}','var'); {str(var)} = 0; end\n")
            # default values for assigned variables
            for var in self.assignment_dict:
                val = octave_code(self.assignment_dict[var])
                m_class.append(f"\t\t\tif ~exist('{str(var)}','var'); {str(var)} = {val}; end\n")
            m_class.append("")
            # save variables to parameters
            for var in not_assigned_syms:
                m_class.append(f"\t\t\tobj.{str(var)} = {str(var)};\n")
            for var in self.assignment_dict:
                m_class.append(f"\t\t\tobj.{str(var)} = {str(var)};\n")
            # calculate subexpressions
            for var in sorted([str(j) for j in self.subex_dict], key=lambda x: int(regex.findall("(?<=sub)\d*",x)[0])):
                val = regex.sub("(?<=\W|^)sub","obj.sub",octave_code(self.subex_dict[symbols(var)]))
                m_class.append(f"\t\t\tobj.{str(var)} = {val};\n")       
            m_class.append("\t\tend\n\n")
            
            # Add generated functions
            for i in range(len(expressions)):
                # generate function
                [(m_name, m_code)] = codegen(
                    (names[i], expressions[i]), "Octave", project=project, 
                    header=False, empty=True, 
                    argument_sequence=self._sort_variables(self.all_symbols)
                    )
                # remove already set variables
                m_func = m_code.splitlines(True)
                m_func = [f"\t\t{line}" for line in m_func]
                m_func[0] = m_func[0].replace("(","(obj, ")
                m_func[0] = regex.sub(
                    "(" + '|'.join([f', {str(v)}(?=\W)' for v in m_properties])+")",
                    "", m_func[0])
                # remove unused variable symbols
                m_func[0] = regex.sub(
                    "(" + '|'.join([f', {str(v)}(?=\W)' for v in self.var_syms.difference(expressions[i].free_symbols)])+")",
                    "", m_func[0])
                # use obj.variables defined in class parameters
                for i in range(1, len(m_func)):
                    m_func[i] = regex.sub(f"(?<=\W|^)(?=({'|'.join([str(s) for s in m_properties])})\W)","obj.",m_func[i])        
                m_func.append("\n")
                m_class.extend(m_func)
            
            m_class.append("\tend\n")
            m_class.append("end\n")
            with open(os.path.join(folder, "matlab", f"{name}.m"), "w+") as f:
                f.writelines(m_class)

class SymbolicKinDyn(_AbstractCodeGeneration):
    BODY_FIXED = "body_fixed"
    SPACIAL = "spacial"
    
    # compatibility with older version, where these functions have been 
    # member functions of class.
    SE3AdjInvMatrix = staticmethod(SE3AdjInvMatrix)
    SE3AdjMatrix = staticmethod(SE3AdjMatrix)
    SE3adMatrix = staticmethod(SE3adMatrix)
    SE3Exp = staticmethod(SE3Exp)
    SE3Inv = staticmethod(SE3Inv)
    SO3Exp = staticmethod(SO3Exp)
    InertiaMatrix = staticmethod(InertiaMatrix)
    TransformationMatrix = staticmethod(TransformationMatrix)
    MassMatrixMixedData = staticmethod(MassMatrixMixedData)
    rpy_to_matrix = staticmethod(rpy_to_matrix)
    xyz_rpy_to_matrix = staticmethod(xyz_rpy_to_matrix)
    
    
    
    def __init__(self, gravity_vector: MutableDenseMatrix=None, 
                 ee: MutableDenseMatrix=None, 
                 body_ref_config: List[MutableDenseMatrix]=[], 
                 joint_screw_coord: List[MutableDenseMatrix]=[], 
                 config_representation: str="spacial", 
                 Mb: List[MutableDenseMatrix]=[], 
                 parent: List[int]=[], support: List[List[int]]=[], 
                 child: List[List[int]]=[], **kwargs) -> None:
        """SymbolicKinDyn
        Symbolic tool to compute equations of motion of serial chain 
        robots and autogenerate code from the calculated equations. 
        This tool supports generation of python, C and Matlab code.

        Args:
            gravity_vector (sympy.Matrix, optional): 
                Vector of gravity. Defaults to None.
            ee (sympy.Matrix, optional): 
                End-effector configuration with reference to last link 
                body fixed frame in the chain. Defaults to None.
            body_ref_config (list of sympy.Matrix, optional): 
                List of reference configurations of bodies in body-fixed
                or spacial representation, dependent on selected 
                config_representation. 
                Leave empty for dH Parameter usage (dhToScrewCoord(...)). 
                Defaults to [].
            joint_screw_coord (list of sympy.Matrix, optional): 
                List of joint screw coordinates in body-fixed 
                or spacial representation, dependent on selected 
                config_representation. 
                Leave empty for dH Parameter usage (dhToScrewCoord(...)). 
                Defaults to [].
            config_representation (str):
                Use body fixed or spacial representation for reference 
                configuration of bodies and joint screw coordinates.
                Has to be "body_fixed" or "spacial". 
                Defaults to "spacial".
            Mb (list of sympy.Matrix, optional): 
                List of Mass Inertia matrices for all links. Only 
                necessary for inverse dynamics. Defaults to [].
            parent (list, optional):
                TODO
                Defaults to [].
            support (list, optional):
                TODO
                Defaults to [].
            child (list, optional):
                TODO
                Defaults to [].
                
        Usage:
            Example of an 2R Serial chain robot:
            Imports:
                >>> import sympy 
            
            Declaration of symbolic variables:
                joint positions:
                >>> q1, q2 = sympy.symbols("q1 q2")
                
                joint velocities:
                >>> dq1, dq2 = sympy.symbols("dq1 dq2")
                
                joint accelerations:
                >>> ddq1, ddq2 = sympy.symbols("ddq1 ddq2")
                
                mass:
                >>> m1, m2 = sympy.symbols("m1 m2", real=1, constant=1)
                
                center of gravity and gravity
                >>> cg1, cg2, g = sympy.symbols("cg1 cg2 g", constant=1)
                
                link lengths:
                >>> L1, L2 = sympy.symbols("L1 L2", real=1, constant=1)
                
            Definition of arguments:
                Gravity vector:
                >>> gravity_vector = sympy.Matrix([0, g, 0])
                
                Joint screw coordinate in body fixed representation:
                >>> joint_screw_coord = []
                >>> joint_screw_coord.append(sympy.Matrix([0, 0, 1, 0, 0, 0]).T)
                >>> joint_screw_coord.append(sympy.Matrix([0, 0, 1, 0, -L1, 0]).T)
                
                Reference configurations of bodies 
                (i.e. of body-fixed reference frames):
                >>> body_ref_config = []
                >>> body_ref_config.append(
                ...     TransformationMatrix(
                ...     t=sympy.Matrix([0, 0, 0]))
                >>> body_ref_config.append(
                ...     TransformationMatrix(
                ...     t=sympy.Matrix([L1, 0, 0]))
            
                End-effector configuration wrt last link body fixed 
                frame in the chain:
                >>> ee = TransformationMatrix(
                ...     t=sympy.Matrix([L2, 0, 0]))

                Mass-Intertia parameters:
                >>> Mb = []
                >>> Mb.append(MassMatrixMixedData(
                ...     m1, (m1*L1**2) * sympy.Identity(3), cg1))
                >>> Mb.append(MassMatrixMixedData(
                ...     m2, (m2*L2**2) * sympy.Identity(3), cg2))

                Declaring generalized vectors:
                >>> q = Matrix([q1, q2])
                >>> qd = Matrix([dq1, dq2])
                >>> q2d = Matrix([ddq1, ddq2])
            
            Initialization and usage of SymbolicKinDyn:
                Init:
                >>> skd = SymbolicKinDyn(gravity_vector=gravity_vector, 
                ...                      ee=ee, 
                ...                      body_ref_config = body_ref_config, 
                ...                      joint_screw_coord = joint_screw_coord, 
                ...                      config_representation = "body_fixed", 
                ...                      Mb=Mb)
                
                Generate Kinematics:
                >>> skd.closed_form_kinematics_body_fixed(q, qd, q2d)
                
                Generate Dynamics:
                >>> skd.closed_form_inv_dyn_body_fixed(q, qd, q2d)
                
                Generate Code:
                >>> skd.generateCode(python=True, C=True, Matlab=True,  
                ...                  name="R2_plant_example")
        """
        super().__init__()
        self.n = None  # degrees of freedom
        self.gravity_vector = gravity_vector
        self.ee = ee

        self.B = [] # List of reference configurations of bodies in body-fixed representation.
        self.X = [] # List of joint screw coordinates in body-fixed representation.

        self.A = [] # List of reference configurations of bodies in spacial representation.
        self.Y = [] # List of joint screw coordinates in spacial representation.
        
        self.config_representation = config_representation # @property: checks for valid value 
        if body_ref_config != []:
            self.body_ref_config = body_ref_config # @property: sets A or B
        if joint_screw_coord != []:
            self.joint_screw_coord = joint_screw_coord # @property: sets X or Y
        # support of old syntax
        if "A" in kwargs:
            self.A = kwargs["A"]
            n = len(self.A)
            if n:
                self.n = n
        if "B" in kwargs:
            self.B = kwargs["B"]
            n = len(self.B)
            if n:
                self.n = n
        if "X" in kwargs:
            self.X = kwargs["X"]
        if "Y" in kwargs:
            self.Y = kwargs["Y"]

            
        self.Mb = Mb
        self.parent = parent
        self.child = child
        self.support = support

        # temporary vars
        self._FK_C = None
        self._A = None
        self._a = None
        self._V = None  # system twist

        
        # Multiprocessing
        # dict of queues, which saves values and results
        self.queue_dict = {}  
        # dict of running processes
        self.process_dict = {}  

        
    @property
    def config_representation(self) -> str:
        return self._config_representation
    
    @config_representation.setter
    def config_representation(self, value: str) -> None:
        if value not in {self.BODY_FIXED, self.SPACIAL}:
            raise ValueError("config_representation has to be 'body_fixed' or 'spacial'")
        self._config_representation = value
    
    @property
    def body_ref_config(self) -> list:
        if self.config_representation == self.BODY_FIXED:
            return self.B
        elif self.config_representation == self.SPACIAL:
            return self.A
    
    @body_ref_config.setter
    def body_ref_config(self, value: List[MutableDenseMatrix]) -> None:
        n = len(value)
        if n:
            self.n = n
        if self.config_representation == self.BODY_FIXED:
            self.B = value
        elif self.config_representation == self.SPACIAL:
            self.A = value
    
    @property
    def joint_screw_coord(self) -> list:
        if self.config_representation == self.BODY_FIXED:
            return self.X
        elif self.config_representation == self.SPACIAL:
            return self.Y
    
    @joint_screw_coord.setter
    def joint_screw_coord(self, value: List[MutableDenseMatrix]) -> None:
        if self.config_representation == self.BODY_FIXED:
            self.X = value
        elif self.config_representation == self.SPACIAL:
            self.Y = value
    
    def closed_form_kinematics_body_fixed(
        self, q:sympy.MutableDenseMatrix=None, qd: MutableDenseMatrix=None, 
        q2d: MutableDenseMatrix=None, simplify_expressions: bool=True, 
        cse_ex: bool=False, parallel: bool=True) -> MutableDenseMatrix:
        """Position, Velocity and Acceleration Kinematics using Body 
        fixed representation of the twists in closed form.

        The following expressions are saved in the class and can be 
        code generated afterwards:
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

            Needs class parameters body_ref_config, joint_screw_coord and ee to be defined.

        Args:
            q (sympy.Matrix): 
                (n,1) Generalized position vector. Defaults to None.
            qd (sympy.Matrix): 
                (n,1) Generalized velocity vector. Defaults to None.
            q2d (sympy.Matrix): 
                (n,1) Generalized acceleration vector. Defaults to None.
            simplify_expressions (bool, optional): 
                Use simplify command on saved expressions. 
                Defaults to True.
            cse_ex (bool, optional): 
                Use common subexpression elimination. Defaults to False.
            parallel (bool, optional): 
                Use parallel computation via multiprocessing. 
                Defaults to True.
        Raises:
            ValueError:
                Joint screw coordinates and/or reference configuration 
                of bodies not set.
                
        Returns:
            sympy.Matrix: Forward kinematics.
        """
        if not q or not qd or not q2d:
            if not self.n:
                self.n = len(self.body_ref_config)
            q, qd, q2d = generalized_vectors(self.n,self._find_start_index())
            
        if parallel:
            self._closed_form_kinematics_body_fixed_parallel(
                q, qd, q2d, simplify_expressions, cse_ex)
        else:
            self._closed_form_kinematics_body_fixed(
                q, qd, q2d, simplify_expressions, cse_ex)
        return self.fkin

    def closed_form_inv_dyn_body_fixed(
        self, q:sympy.MutableDenseMatrix=None, qd: MutableDenseMatrix=None, 
        q2d: MutableDenseMatrix=None, WEE: MutableDenseMatrix=zeros(6, 1), 
        simplify_expressions: bool=True, cse_ex: bool=False, 
        parallel: bool=True) -> MutableDenseMatrix:
        """Inverse dynamics using body fixed representation of the 
        twists in closed form. 

        The following expressions are saved in the class and can be 
        code generated afterwards:
            coriolis_centrifugal_matrix
            generalized_mass_inertia_matrix
            gravity_vector
            inverse_dynamics

        Args:
            q (sympy.Matrix): 
                (n,1) Generalized position vector. Defaults to None.
            qd (sympy.Matrix): 
                (n,1) Generalized velocity vector. Defaults to None.
            q2d (sympy.Matrix): 
                (n,1) Generalized acceleration vector. Defaults to None.
            WEE (sympy.Matrix, optional): 
                (6,1) WEE (t) is the time varying wrench on the EE link. 
                Defaults to zeros(6, 1).
            simplify_expressions (bool, optional): 
                Use simplify command on saved expressions. 
                Defaults to True.
            cse_ex (bool, optional): 
                Use common subexpression elimination. Defaults to False.
            parallel (bool, optional): 
                Use parallel computation via multiprocessing. 
                Defaults to True.

        Raises:
            ValueError:
                Joint screw coordinates and/or reference configuration 
                of bodies not set.
        
        Returns:
            sympy.Matrix: Generalized forces
        """
        if not q or not qd or not q2d:
            if not self.n:
                self.n = len(self.body_ref_config)
            q, qd, q2d = generalized_vectors(self.n,self._find_start_index())
        
        if parallel:
            self._closed_form_inv_dyn_body_fixed_parallel(
                q, qd, q2d, WEE, simplify_expressions, cse_ex)
        else:
            self._closed_form_inv_dyn_body_fixed(
                q, qd, q2d, WEE, simplify_expressions, cse_ex)
        return self.Q

    def _closed_form_kinematics_body_fixed(
        self, q: MutableDenseMatrix, qd: MutableDenseMatrix, 
        q2d: MutableDenseMatrix, simplify_expressions: bool=True, 
        cse_ex: bool=False) -> MutableDenseMatrix:
        """Position, velocity and acceleration kinematics using 
        body fixed representation of the twists in closed form.

        The following expressions are saved in the class and can be 
        code generated afterwards:
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

            Needs class parameters body_ref_config, joint_screw_coord 
            and ee to be defined.

        Args:
            q (sympy.Matrix): 
                (n,1) Generalized position vector.
            qd (sympy.Matrix): 
                (n,1) Generalized velocity vector.
            q2d (sympy.Matrix): 
                (n,1) Generalized acceleration vector.
            simplify_expressions (bool, optional): 
                Use simplify command on saved expressions. 
                Defaults to True.
            cse_ex (bool, optional): 
                Use common subexpression elimination. Defaults to False.


        Returns:
            sympy.Matrix: Forward kinematics.
        """
        print("Forward kinematics calculation")
        self.var_syms.update(q.free_symbols)
        self.var_syms.update(qd.free_symbols)
        self.var_syms.update(q2d.free_symbols)

        self.n = len(q)  # DOF

        # calc Forward kinematics
        if self.parent and self.support:
            FK_C, A = self._calc_A_matrix_tree(q)
        else:    
            FK_C, A = self._calc_A_matrix(q)
            
        fkin = FK_C[self.n-1]*self.ee
        if simplify_expressions:
            fkin = self.simplify(fkin, cse_ex)
        elif cse_ex:
            fkin = self._cse_expression(fkin)    
        self.fkin = fkin


        if self.J is not None:
            J = self.J
            V = self._V
        else:
            # Block diagonal matrix X (6n x n) of the screw coordinate 
            # vector associated to all joints in the body frame (Constant)
            X = zeros(6*self.n, self.n)
            for i in range(self.n):
                X[6*i:6*i+6, i] = self.X[i]

            # System level Jacobian
            J = A*X
            if simplify_expressions:
                J = self.simplify(J, cse_ex)
            elif cse_ex:
                J = self._cse_expression(J)
            self.J = J

            # System twist (6n x 1)
            V = J*qd
            self._V = V

        # Different Jacobians
        R_i = Matrix(fkin[:3, :3]).row_join(
            zeros(3, 1)).col_join(Matrix([0, 0, 0, 1]).T)
        if simplify_expressions:  # fastens later simplifications
            R_i = self.simplify(R_i, cse_ex)
        elif cse_ex:
            R_i = self._cse_expression(R_i)

        R_BFn = Matrix(FK_C[-1][:3, :3]).row_join(
            zeros(3, 1)).col_join(Matrix([0, 0, 0, 1]).T)

        # Body fixed Jacobian of last moving body 
        # (This may not correspond to end-effector frame)
        Jb = J[-6:, :]
        if simplify_expressions:
            Jb = self.simplify(Jb, cse_ex)
        elif cse_ex:
            Jb = self._cse_expression(Jb)

        Vb_BFn = Jb*qd  # Body fixed twist of last moving body
        if simplify_expressions:
            Vb_BFn = self.simplify(Vb_BFn, cse_ex)
        elif cse_ex:
            Vb_BFn = self._cse_expression(Vb_BFn)
        Vh_BFn = SE3AdjMatrix(R_BFn)*Vb_BFn
        if simplify_expressions:
            Vh_BFn = self.simplify(Vh_BFn, cse_ex)
        elif cse_ex:
            Vh_BFn = self._cse_expression(Vh_BFn)
        self.Vb_BFn = Vb_BFn
        self.Vh_BFn = Vh_BFn

        # Body fixed twist of end-effector frame
        Vb_ee = SE3AdjMatrix(SE3Inv(self.ee))*Vb_BFn
        if simplify_expressions:
            Vb_ee = self.simplify(Vb_ee, cse_ex)
        elif cse_ex:
            Vb_ee = self._cse_expression(Vb_ee)
        # Hybrid twist of end-effector frame
        Vh_ee = SE3AdjMatrix(R_i)*Vb_ee
        if simplify_expressions:
            Vh_ee = self.simplify(Vh_ee, cse_ex)
        elif cse_ex:
            Vh_ee = self._cse_expression(Vh_ee)

        self.Vb_ee = Vb_ee
        self.Vh_ee = Vh_ee

        # Body fixed Jacobian of end-effector frame
        Jb_ee = SE3AdjMatrix(SE3Inv(self.ee))*Jb
        if simplify_expressions:
            Jb_ee = self.simplify(Jb_ee, cse_ex)
        elif cse_ex:
            Jb_ee = self._cse_expression(Jb_ee)

        # Hybrid Jacobian of end-effector frame
        Jh_ee = SE3AdjMatrix(R_i)*Jb_ee
        # Hybrid Jacobian of last moving body
        Jh = SE3AdjMatrix(R_i)*Jb  

        if simplify_expressions:
            Jh_ee = self.simplify(Jh_ee, cse_ex)
            Jh = self.simplify(Jh, cse_ex)
        elif cse_ex:
            Jh_ee = self._cse_expression(Jh_ee)
            Jh = self._cse_expression(Jh)

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
                a[6*i:6*i+6, 6*i:6*i+6] = SE3adMatrix(self.X[i])*qd[i]
            if simplify_expressions:
                a = self.simplify(a, cse_ex)
            elif cse_ex:
                a = self._cse_expression(a)
            self._a = a

        # System acceleration (6n x 1)
        Jdot = -A*a*J  # Sys-level Jacobian time derivative
        if simplify_expressions:
            Jdot = self.simplify(Jdot, cse_ex)
        elif cse_ex:
            Jdot = self._cse_expression(Jdot)

        self.Jdot = Jdot

        Vbd = J*q2d - A*a*V

        # Hybrid acceleration of the last body
        Vbd_BFn = Vbd[-6:, :]
        if simplify_expressions:
            Vbd_BFn = self.simplify(Vbd_BFn, cse_ex)
        elif cse_ex:
            Vbd_BFn = self._cse_expression(Vbd_BFn)
        # Hybrid twist of end-effector frame 
        # TODO: check comments
        Vhd_BFn = (SE3AdjMatrix(R_BFn)*Vbd_BFn 
                   + SE3adMatrix(Matrix(Vh_BFn[:3, :])
                                      .col_join(Matrix([0, 0, 0])))
                   * SE3AdjMatrix(R_BFn)*Vb_BFn)  

        if simplify_expressions:
            Vhd_BFn = self.simplify(Vhd_BFn, cse_ex)
        elif cse_ex:
            Vhd_BFn = self._cse_expression(Vhd_BFn)

        self.Vbd_BFn = Vbd_BFn
        self.Vhd_BFn = Vhd_BFn

        # Body fixed twist of end-effector frame
        # Hybrid acceleration of the EE
        Vbd_ee = SE3AdjMatrix(SE3Inv(self.ee))*Vbd_BFn
        if simplify_expressions:
            Vbd_ee = self.simplify(Vbd_ee, cse_ex)
        elif cse_ex:
            Vbd_ee = self._cse_expression(Vbd_ee)
        # Hybrid twist of end-effector frame
        Vhd_ee = SE3AdjMatrix(R_i)*Vbd_ee + SE3adMatrix(Matrix(
            Vh_ee[:3, :]).col_join(Matrix([0, 0, 0])))*\
                SE3AdjMatrix(R_i)*Vb_ee  
        if simplify_expressions:
            Vhd_ee = self.simplify(Vhd_ee, cse_ex)
        elif cse_ex:
            Vhd_ee = self._cse_expression(Vhd_ee)

        self.Vbd_ee = Vbd_ee
        self.Vhd_ee = Vhd_ee

        # Body Jacobian time derivative

        # For the last moving body
        Jb_dot = Jdot[-6:, :]
        self.Jb_dot = Jb_dot

        # For the EE
        Jb_ee_dot = SE3AdjMatrix(SE3Inv(self.ee))*Jb_dot
        if simplify_expressions:
            Jb_ee_dot = self.simplify(Jb_ee_dot, cse_ex)
        elif cse_ex:
            Jb_ee_dot = self._cse_expression(Jb_ee_dot)
        self.Jb_ee_dot = Jb_ee_dot

        # Hybrid Jacobian time derivative
        # For the last moving body
        Jh_dot = SE3AdjMatrix(R_BFn)*Jb_dot + SE3adMatrix(
            Matrix(Vh_BFn[:3, :]).col_join(Matrix([0, 0, 0])))*\
                SE3AdjMatrix(R_BFn)*Jb
        if simplify_expressions:
            Jh_dot = self.simplify(Jh_dot, cse_ex)
        elif cse_ex:
            Jh_dot = self._cse_expression(Jh_dot)
        self.Jh_dot = Jh_dot

        # For the EE
        Jh_ee_dot = SE3AdjMatrix(R_i)*Jb_ee_dot + SE3adMatrix(
            Matrix(Vh_ee[:3, :]).col_join(Matrix([0, 0, 0])))*\
                SE3AdjMatrix(R_i)*Jb_ee
        if simplify_expressions:
            Jh_ee_dot = self.simplify(Jh_ee_dot, cse_ex)
        elif cse_ex:
            Jh_ee_dot = self._cse_expression(Jh_ee_dot)
        self.Jh_ee_dot = Jh_ee_dot

        for e in self._get_expressions():
            self.all_symbols.update(e.free_symbols)

        print("Done")
        return fkin

    def _closed_form_inv_dyn_body_fixed(self, q: MutableDenseMatrix, 
                                        qd: MutableDenseMatrix, 
                                        q2d: MutableDenseMatrix, 
                                        WEE: MutableDenseMatrix=zeros(6, 1), 
                                        simplify_expressions: bool=True, 
                                        cse_ex: bool=False) -> MutableDenseMatrix:
        """Inverse dynamics using body fixed representation of the 
        twists in closed form. 

        The following expressions are saved in the class and can be code 
        generated afterwards:
            coriolis_centrifugal_matrix
            generalized_mass_inertia_matrix
            gravity_vector
            inverse_dynamics

        Args:
            q (sympy.Matrix): (n,1) Generalized position vector.
            qd (sympy.Matrix): (n,1 )Generalized velocity vector.
            q2d (sympy.Matrix): (n,1) Generalized acceleration vector.
            WEE (sympy.Matrix, optional): (6,1) WEE (t) is the time 
                varying wrench on the EE link. Defaults to zeros(6, 1).
            simplify_expressions (bool, optional): Use simplify command 
                on saved expressions. Defaults to True.
            cse_ex (bool, optional): Use common subexpression 
                elimination. Defaults to False.

        Returns:
            sympy.Matrix: Generalized forces
        """
        print("Inverse dynamics calculation")

        self.var_syms.update(q.free_symbols)
        self.var_syms.update(qd.free_symbols)
        self.var_syms.update(q2d.free_symbols)
        self.var_syms.update(WEE.free_symbols)

        self.n = len(q)

        # calc Forward kinematics
        if self.parent and self.support:
            FK_C, A = self._calc_A_matrix_tree(q)
        else:    
            FK_C, A = self._calc_A_matrix(q)
        
        if self.J is not None:
            J = self.J  # system level Jacobian
            V = self._V  # system twist
        else:
            # Block diagonal matrix X (6n x n) of the screw coordinate 
            # vector associated to all joints in the body frame (constant)
            X = zeros(6*self.n, self.n)
            for i in range(self.n):
                X[6*i:6*i+6, i] = self.X[i]

            # System level Jacobian
            J = A*X
            if simplify_expressions:
                J = self.simplify(J, cse_ex)
            elif cse_ex:
                J = self._cse_expression(J)
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
                a[6*i:6*i+6, 6*i:6*i+6] = SE3adMatrix(self.X[i])*qd[i]
            self._a = a

        # System acceleration (6n x 1)
        Vd = J*q2d - A*a*V

        # Block Diagonal Mb (6n x 6n) Mass inertia matrix in body frame 
        # (constant)
        Mb = zeros(6*self.n, 6*self.n)
        for i in range(self.n):
            Mb[i*6:i*6+6, i*6:i*6+6] = self.Mb[i]

        # Block diagonal matrix b (6n x 6n) used in Coriolis matrix
        b = zeros(6*self.n, 6*self.n)
        for i in range(self.n):
            b[i*6:i*6+6, i*6:i*6+6] = SE3adMatrix(Matrix(V[6*i:6*i+6]))

        # Block diagonal matrix Cb (6n x 6n)
        Cb = -Mb*A*a - b.T * Mb

        # Lets setup the Equations of Motion

        # Mass inertia matrix in joint space (n x n)
        M = J.T*Mb*J
        if simplify_expressions:
            M = self.simplify(M, cse_ex)
        elif cse_ex:
            M = self._cse_expression(M)

        # Coriolis-Centrifugal matrix in joint space (n x n)
        C = J.T * Cb * J
        if simplify_expressions:
            C = self.simplify(C, cse_ex)
        elif cse_ex:
            C = self._cse_expression(C)

        # Gravity Term
        U = SE3AdjInvMatrix(FK_C[0])
        for k in range(1, self.n):
            U = U.col_join(SE3AdjInvMatrix(FK_C[k]))

        Vd_0 = zeros(6, 1)
        Vd_0[3:6, 0] = self.gravity_vector
        Qgrav = J.T*Mb*U*Vd_0
        if simplify_expressions:
            Qgrav = self.simplify(Qgrav, cse_ex)
        elif cse_ex:
            Qgrav = self._cse_expression(Qgrav)

        # External Wrench
        Wext = zeros(6*self.n, 1)
        # WEE (t) is the time varying wrench on the EE link.
        Wext[-6:, 0] = WEE
        Qext = J.T * Wext

        # Generalized forces Q
        # Q = M*q2d + C*qd   # without gravity
        Q = M*q2d + C*qd + Qgrav + Qext

        if simplify_expressions:
            Q = self.simplify(Q, cse_ex)
        elif cse_ex:
            Q = self._cse_expression(Q)

        self.M = M
        self.C = C
        self.Q = Q
        self.Qgrav = Qgrav

        # save used symbols
        for e in self._get_expressions():
            self.all_symbols.update(e.free_symbols)

        print("Done")
        return Q

    def _closed_form_kinematics_body_fixed_parallel(
        self, q: MutableDenseMatrix, qd: MutableDenseMatrix, 
        q2d: MutableDenseMatrix, simplify_expressions: bool=True, 
        cse_ex: bool=False) -> MutableDenseMatrix:
        """Position, velocity and acceleration kinematics using 
        body fixed representation of the twists in closed form.

        The following expressions are saved in the class and can be 
        code generated afterwards:
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
            q (sympy.Matrix): 
                (n,1) Generalized position vector.
            qd (sympy.Matrix): 
                (n,1) Generalized velocity vector.
            q2d (sympy.Matrix): 
                (n,1) Generalized acceleration vector.
            simplify_expressions (bool, optional): 
                Use simplify command on saved expressions. 
                Defaults to True.
            cse_ex (bool, optional): 
                Use common subexpression elimination. Defaults to False.

        Returns:
            sympy.Matrix: Forward kinematics.
        """
        # This method does the same as _closed_form_kinematics_body_fixed.
        # Parallel computation is implemented by writing most values 
        # in queues, organized in a dict.
        # This ensures the correct order for the execution.
        # To understand the calculations it is recommended to read the 
        # code in _closed_form_kinematics_body_fixed since it is more 
        # readable and has the same structure.

        print("Forward kinematics calculation")
        self.var_syms.update(q.free_symbols)
        self.var_syms.update(qd.free_symbols)
        self.var_syms.update(q2d.free_symbols)

        self.n = len(q)
        self.queue_dict["subex_dict"] = Queue()

        # calc Forward kinematics
        if self.parent and self.support:
            FK_C, A = self._calc_A_matrix_tree(q)
        else:    
            FK_C, A = self._calc_A_matrix(q)
        
        self._set_value("fkin", FK_C[self.n-1]*self.ee)
        if simplify_expressions:
            self._start_simplification_process("fkin", cse_ex)
        elif cse_ex:
            self._start_cse_process("fkin")


        if self.J is not None:
            self._set_value("J", self.J)
            self._set_value("V", self._V)
        else:
            # Block diagonal matrix X (6n x n) of the screw coordinate 
            # vector associated to all joints in the body frame (Constant)
            X = zeros(6*self.n, self.n)
            for i in range(self.n):
                X[6*i:6*i+6, i] = self.X[i]

            # System level Jacobian
            self._set_value("J", A*X)
            if simplify_expressions:
                self._start_simplification_process("J", cse_ex)
            elif cse_ex:
                self._start_cse_process("J")
            # System twist (6n x 1)
            self._set_value_as_process("V", lambda: self._get_value("J")*qd)

        # Different Jacobians
        self._set_value_as_process(
            "R_i", 
            lambda: 
                Matrix(self._get_value("fkin")[:3, :3])
                .row_join(zeros(3, 1))
                .col_join(Matrix([0, 0, 0, 1]).T)
            )

        if simplify_expressions:  # fastens later simplifications
            self._start_simplification_process("R_i", cse_ex)
        elif cse_ex:
            self._start_cse_process("R_i")

        self._set_value("R_BFn", Matrix(FK_C[-1][:3, :3]).row_join(
            zeros(3, 1)).col_join(Matrix([0, 0, 0, 1]).T))

        # Body fixed Jacobian of last moving body 
        # (This may not correspond to end-effector frame)
        self._set_value_as_process("Jb", lambda: self._get_value("J")[-6:, :])
        if simplify_expressions:
            self._start_simplification_process("Jb", cse_ex)
        elif cse_ex:
            self._start_cse_process("Jb")

        self._set_value_as_process("Vb_BFn", lambda: self._get_value("Jb")*qd)
        # Body fixed twist of last moving body
        if simplify_expressions:
            self._start_simplification_process("Vb_BFn", cse_ex)
        elif cse_ex:
            self._start_cse_process("Vb_BFn")

        self._set_value_as_process("Vh_BFn", lambda: SE3AdjMatrix(
            self._get_value("R_BFn"))*self._get_value("Vb_BFn"))
        if simplify_expressions:
            self._start_simplification_process("Vh_BFn", cse_ex)
        elif cse_ex:
            self._start_cse_process("Vh_BFn")

        # Body fixed twist of end-effector frame
        self._set_value_as_process("Vb_ee", lambda: SE3AdjMatrix(
            SE3Inv(self.ee))*self._get_value("Vb_BFn"))
        if simplify_expressions:
            self._start_simplification_process("Vb_ee", cse_ex)
        elif cse_ex:
            self._start_cse_process("Vb_ee")
        # Hybrid twist of end-effector frame
        self._set_value_as_process("Vh_ee", lambda: SE3AdjMatrix(
            self._get_value("R_i"))*self._get_value("Vb_ee"))
        if simplify_expressions:
            self._start_simplification_process("Vh_ee", cse_ex)
        elif cse_ex:
            self._start_cse_process("Vh_ee")

        # Body fixed Jacobian of end-effector frame
        self._set_value_as_process("Jb_ee", lambda: SE3AdjMatrix(
            SE3Inv(self.ee))*self._get_value("Jb"))
        if simplify_expressions:
            self._start_simplification_process("Jb_ee", cse_ex)
        elif cse_ex:
            self._start_cse_process("Jb_ee")

        # Hybrid Jacobian of end-effector frame
        self._set_value_as_process("Jh_ee", lambda: SE3AdjMatrix(
            self._get_value("R_i"))*self._get_value("Jb_ee"))
        # Hybrid Jacobian of last moving body
        self._set_value_as_process("Jh", lambda: SE3AdjMatrix(
            self._get_value("R_i"))*self._get_value("Jb"))

        if simplify_expressions:
            self._start_simplification_process("Jh_ee", cse_ex)
            self._start_simplification_process("Jh", cse_ex)
        elif cse_ex:
            self._start_cse_process("Jh_ee")
            self._start_cse_process("Jh")

        # Acceleration computations
        if self._a is not None:
            self._set_value("a", self._a)
        else:
            # Block diagonal matrix a (6n x 6n)
            a = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                a[6*i:6*i+6, 6*i:6*i+6] = SE3adMatrix(self.X[i])*qd[i]
            self._set_value("a", a)
            if simplify_expressions:
                self._start_simplification_process("a", cse_ex)
            elif cse_ex:
                self._start_cse_process("a")

        # System acceleration (6n x 1)
        # System-level Jacobian time derivative
        self._set_value_as_process(
            "Jdot", lambda: -A*self._get_value("a")*self._get_value("J"))
        if simplify_expressions:
            self._start_simplification_process("Jdot", cse_ex)
        elif cse_ex:
            self._start_cse_process("Jdot")

        self._set_value_as_process("Vbd", lambda: self._get_value(
            "J")*q2d - A*self._get_value("a")*self._get_value("V"))

        # Hybrid acceleration of the last body
        self._set_value_as_process(
            "Vbd_BFn", lambda: self._get_value("Vbd")[-6:, :])

        if simplify_expressions:
            self._start_simplification_process("Vbd_BFn", cse_ex)
        elif cse_ex:
            self._start_cse_process("Vbd_BFn")

        # Hybrid twist of end-effector frame
        self._set_value_as_process(
            "Vhd_BFn", 
            lambda: 
                SE3AdjMatrix(self._get_value("R_BFn"))
                * self._get_value("Vbd_BFn") 
                + SE3adMatrix(Matrix(self._get_value("Vh_BFn")[:3, :])
                                   .col_join(Matrix([0, 0, 0])))
                * SE3AdjMatrix(self._get_value("R_BFn"))
                * self._get_value("Vb_BFn")
            )

        if simplify_expressions:
            self._start_simplification_process("Vhd_BFn", cse_ex)
        elif cse_ex:
            self._start_cse_process("Vhd_BFn")

        # Body fixed twist of end-effector frame
        # Hybrid acceleration of the EE
        self._set_value_as_process("Vbd_ee", lambda: SE3AdjMatrix(
            SE3Inv(self.ee))*self._get_value("Vbd_BFn"))
        if simplify_expressions:
            self._start_simplification_process("Vbd_ee", cse_ex)
        elif cse_ex:
            self._start_cse_process("Vbd_ee")
        # Hybrid twist of end-effector frame
        self._set_value_as_process(
            "Vhd_ee", 
            lambda: 
                SE3AdjMatrix(self._get_value("R_i")) 
                * self._get_value("Vbd_ee") 
                + SE3adMatrix(Matrix(self._get_value("Vh_ee")[:3, :])
                                   .col_join(Matrix([0, 0, 0])))
                * SE3AdjMatrix(self._get_value("R_i"))
                * self._get_value("Vb_ee")
            )  # Hybrid twist of end-effector frame

        if simplify_expressions:
            self._start_simplification_process("Vhd_ee", cse_ex)
        elif cse_ex:
            self._start_cse_process("Vhd_ee")

        # Body Jacobian time derivative

        # For the last moving body
        self._set_value_as_process(
            "Jb_dot", lambda: self._get_value("Jdot")[-6:, :])

        # For the EE
        self._set_value_as_process("Jb_ee_dot", lambda: SE3AdjMatrix(
            SE3Inv(self.ee))*self._get_value("Jb_dot"))
        if simplify_expressions:
            self._start_simplification_process("Jb_ee_dot", cse_ex)
        elif cse_ex:
            self._start_cse_process("Jb_ee_dot")

        # Hybrid Jacobian time derivative
        # For the last moving body
        self._set_value_as_process(
            "Jh_dot", 
            lambda: 
                SE3AdjMatrix(self._get_value("R_BFn"))
                * self._get_value("Jb_dot") 
                + SE3adMatrix(Matrix(self._get_value("Vh_BFn")[:3, :])
                                   .col_join(Matrix([0, 0, 0])))
                * SE3AdjMatrix(self._get_value("R_BFn"))
                * self._get_value("Jb")
            )
        if simplify_expressions:
            self._start_simplification_process("Jh_dot", cse_ex)
        elif cse_ex:
            self._start_cse_process("Jh_dot")

        # For the EE
        self._set_value_as_process(
            "Jh_ee_dot", 
            lambda: 
                SE3AdjMatrix(self._get_value("R_i"))
                * self._get_value("Jb_ee_dot") 
                + SE3adMatrix(Matrix(self._get_value("Vh_ee")[:3, :])
                                   .col_join(Matrix([0, 0, 0])))
                * SE3AdjMatrix(self._get_value("R_i"))
                * self._get_value("Jb_ee")
            )
        if simplify_expressions:
            self._start_simplification_process("Jh_ee_dot", cse_ex)
        elif cse_ex:
            self._start_cse_process("Jh_ee_dot")
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
        self.Vbd_ee = self._get_value("Vbd_ee")
        self.Jh_dot = self._get_value("Jh_dot")
        self.Jb_dot = self._get_value("Jb_dot")
        self.Jh_ee_dot = self._get_value("Jh_ee_dot")
        self.Jb_ee_dot = self._get_value("Jb_ee_dot")

        try:
            while True:
                self.subex_dict.update(
                    self.queue_dict["subex_dict"].get(timeout=0.05))
        except queue.Empty:
            pass

        # empty Queues
        for i in self.queue_dict:
            self._flush_queue(self.queue_dict[i])
        self.queue_dict = {}

        # join Processes
        for i in self.process_dict:
            self.process_dict[i].join()
        self.process_dict = {}

        # save used symbols
        for e in self._get_expressions():
            self.all_symbols.update(e.free_symbols)

        print("Done")
        return self.fkin

    def _closed_form_inv_dyn_body_fixed_parallel(
        self, q: MutableDenseMatrix, qd: MutableDenseMatrix, 
        q2d: MutableDenseMatrix, WEE: MutableDenseMatrix=zeros(6, 1), 
        simplify_expressions: bool=True, cse_ex: bool=False) -> MutableDenseMatrix:
        """Inverse dynamics using body fixed representation of the 
        twists in closed form. 

        The following expressions are saved in the class and can be 
        code generated afterwards:
            coriolis_centrifugal_matrix
            generalized_mass_inertia_matrix
            gravity_vector
            inverse_dynamics

        Args:
            q (sympy.Matrix): 
                (n,1) Generalized position vector.
            qd (sympy.Matrix): 
                (n,1 )Generalized velocity vector.
            q2d (sympy.Matrix): 
                (n,1) Generalized acceleration vector.
            WEE (sympy.Matrix, optional): 
                (6,1) WEE (t) is the time varying wrench on the EE link. 
                Defaults to zeros(6, 1).
            simplify_expressions (bool, optional): 
                Use simplify command on saved expressions. 
                Defaults to True.
            cse_ex (bool, optional): 
                Use common subexpression elimination. Defaults to False.

        Returns:
            sympy.Matrix: Generalized Forces
        """
        # This method does the same as _closed_form_inv_dyn_body_fixed.
        # Parallel computation is implemented by writing most values 
        # in queues, organized in a dict.
        # This ensures the correct order for the execution.
        # To understand the calculations it is recommended to read the 
        # code in _closed_form_inv_dyn_body_fixed since it is more 
        # readable and has the same structure.

        print("Inverse dynamics calculation")

        self.var_syms.update(q.free_symbols)
        self.var_syms.update(qd.free_symbols)
        self.var_syms.update(q2d.free_symbols)
        self.var_syms.update(WEE.free_symbols)

        self.n = len(q)
        self.queue_dict["subex_dict"] = Queue()

        # calc Forward kinematics
        if self.parent and self.support:
            FK_C, A = self._calc_A_matrix_tree(q)
        else:    
            FK_C, A = self._calc_A_matrix(q)
        
        if self.J is not None:
            self._set_value("J", self.J)
            self._set_value("V", self._V)
        else:
            # Block diagonal matrix X (6n x n) of the screw coordinate 
            # vector associated to all joints in the body frame (Constant)
            X = zeros(6*self.n, self.n)
            for i in range(self.n):
                X[6*i:6*i+6, i] = self.X[i]

            # System level Jacobian
            self._set_value("J", A*X)
            if simplify_expressions:
                self._start_simplification_process("J", cse_ex)
            elif cse_ex:
                self._start_cse_process("J")

            # System twist (6n x 1)
            self._set_value_as_process("V", lambda: self._get_value("J")*qd)

        # Acceleration computations

        if self._a is not None:
            a = self._a
        else:
            # Block diagonal matrix a (6n x 6n)
            a = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                a[6*i:6*i+6, 6*i:6*i+6] = SE3adMatrix(self.X[i])*qd[i]
            self._a = a

        # System acceleration (6n x 1)
        # Vd = J*q2d - A*a*V
        self._set_value_as_process("Vd", lambda: self._get_value(
            "J")*q2d - A*a*self._get_value("V"))

        # Block Diagonal Mb (6n x 6n) Mass inertia matrix in body frame 
        # (Constant)
        Mb = zeros(6*self.n, 6*self.n)
        for i in range(self.n):
            Mb[i*6:i*6+6, i*6:i*6+6] = self.Mb[i]

        # Block diagonal matrix b (6n x 6n) used in Coriolis matrix
        def _b():
            nonlocal self
            b = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                b[i*6:i*6+6, i*6:i*6 + 6] = SE3adMatrix(
                    Matrix(self._get_value("V")[6*i:6*i+6]))
            return b
        self._set_value_as_process("b", _b)

        # Block diagonal matrix Cb (6n x 6n)
        self._set_value_as_process(
            "Cb", lambda: -Mb*A*a - self._get_value("b").T * Mb)

        # Lets setup the Equations of Motion

        # Mass inertia matrix in joint space (n x n)
        self._set_value_as_process(
            "M", lambda: self._get_value("J").T*Mb*self._get_value("J"))
        if simplify_expressions:
            self._start_simplification_process("M", cse_ex)
        elif cse_ex:
            self._start_cse_process("M")

        # Coriolis-Centrifugal matrix in joint space (n x n)
        self._set_value_as_process("C", lambda: self._get_value(
            "J").T*self._get_value("Cb")*self._get_value("J"))
        if simplify_expressions:
            self._start_simplification_process("C", cse_ex)
        elif cse_ex:
            self._start_cse_process("C")

        # Gravity Term
        U = SE3AdjInvMatrix(FK_C[0])
        for k in range(1, self.n):
            U = U.col_join(SE3AdjInvMatrix(FK_C[k]))

        Vd_0 = zeros(6, 1)
        Vd_0[3:6, 0] = self.gravity_vector
        self._set_value_as_process(
            "Qgrav", lambda: self._get_value("J").T*Mb*U*Vd_0)
        if simplify_expressions:
            self._start_simplification_process("Qgrav", cse_ex)
        elif cse_ex:
            self._start_cse_process("Qgrav")

        # External Wrench
        Wext = zeros(6*self.n, 1)
        # WEE (t) is the time varying wrench on the EE link.
        Wext[-6:, 0] = WEE
        self._set_value_as_process(
            "Qext", lambda: self._get_value("J").T * Wext)

        # Generalized forces Q
        self._set_value_as_process(
            "Q", 
            lambda: 
                self._get_value("M") * q2d 
                + self._get_value("C") * qd 
                + self._get_value("Qgrav") 
                + self._get_value("Qext")
            )

        if simplify_expressions:
            self._start_simplification_process("Q", cse_ex)
        elif cse_ex:
            self._start_cse_process("Q")

        self._V = self._get_value("V")
        self.J = self._get_value("J")
        self.M = self._get_value("M")
        self.C = self._get_value("C")
        self.Qgrav = self._get_value("Qgrav")
        self.Q = self._get_value("Q")

        try:
            while True:
                self.subex_dict.update(
                    self.queue_dict["subex_dict"].get(timeout=0.05))
        except queue.Empty:
            pass

        # empty Queues
        for i in self.queue_dict:
            self._flush_queue(self.queue_dict[i])
        self.queue_dict = {}

        # join Processes
        for i in self.process_dict:
            self.process_dict[i].join()
        self.process_dict = {}

        # save used symbols
        for e in self._get_expressions():
            self.all_symbols.update(e.free_symbols)

        print("Done")
        return self.Q

    def simplify(self, exp: sympy.Expr, cse_ex: bool=False) -> sympy.Expr:
        """Faster simplify implementation for sympy expressions.
        Expressions can be different simplified as with sympy.simplify.

        Args:
            exp (sympy expression): 
                Expression to simplify.
            cse_ex (bool, optional): 
                Use common subexpression elimination. Defaults to False.

        Returns:
            sympy expression: Simplified expression.
        """
        if cse_ex:
            exp = self._cse_expression(exp)
        if (type(exp) == sympy.matrices.immutable.ImmutableDenseMatrix
                or type(exp) == sympy.matrices.dense.MutableDenseMatrix):
            # fasten simplification of symmetric matrices
            if exp.is_square:
                # test if matrix is symmetric
                # numeric test is faster than is_symmetric method  for 
                # long expressions
                
                # create matrix with randon values
                num = lambdify(list(exp.free_symbols), exp, "numpy")(
                    *(random.random() for i in exp.free_symbols))
                # if (random) matrix is symmetric, we have to simplify 
                # less values
                if numpy.allclose(num, num.T):
                    shape = exp.shape
                    m_exp = exp.as_mutable()
                    # simplify values only once in symmetric matrices
                    for i in range(shape[0]):
                        for j in range(i):
                            m_exp[i, j] = self.simplify(exp[i, j])
                            if i != j:
                                m_exp[j, i] = exp[j, i]
                    return Matrix(m_exp)
        if type(exp) == sympy.matrices.dense.MutableDenseMatrix:
            exp = exp.as_immutable()
        exp = fu(exp)  # fast function to simplify sin and cos expressions
        exp = cancel(exp)
        exp = factor(exp)
        exp = powsimp(exp)
        exp = exp.doit()
        return exp

    def _create_topology_lists(self,robot: URDF) -> None:
        # names of all links in urdf
        link_names = [link.name for link in robot.links] 
        parent_names = [] # names of parent links corresponding link_names
        connection_type = [] # 0 for fixed, None for base, 1 else
        body_index = [] # index of link; -1 for fixed links
        parent = [] # parent array
        child = [] # child array
        support = [] # support array
        # find parent link names and search for fixed joints
        for name in link_names:
            for joint in robot.joints:
                if joint.child == name:
                    parent_names.append(joint.parent)
                    body_index.append(None) # specify later
                    if joint.joint_type == "fixed":
                        connection_type.append(0)
                    else:
                        connection_type.append(1)
                    break
            else: # base link
                parent_names.append(None)
                connection_type.append(None)
                body_index.append(0)

        # generate body indices concatenating fixed bodies
        while None in body_index:
            i1 = body_index.index(None) # i of current link
            # update until parent is already specified
            while body_index[link_names.index(parent_names[i1])] is None:
                i1 = link_names.index(parent_names[i1])
            # fixed links get index -1
            if connection_type[i1] == 0:
                body_index[i1] = -1
                continue
            i2 = link_names.index(parent_names[i1]) # i of parent link
            while body_index[i2] == -1: # find forst non fixed parent
                i2 = link_names.index(parent_names[i2])
            index = body_index[i2]+1 # body index
            while index in body_index: # find first unused index
                index+=1
            body_index[i1] = index
            
        parent = [None for _ in range(max(body_index))] 
        child = [[] for _ in range(max(body_index))]
        support = [[] for _ in range(max(body_index))]

        # fill parent, child and support array
        for i in range(len(body_index)):
            idx = body_index[i] # get index of current body
            if idx <= 0: # ignore base and fixed bodys
                continue
            i1 = link_names.index(parent_names[i]) # parent index
            while body_index[i1] == -1: # find first non fixed parent
                i1 = link_names.index(parent_names[i1])
            parent[idx-1] = body_index[i1] # save parent index
            if body_index[i1] > 0: # ignore base
                child[body_index[i1]-1].append(idx) # save child to parent
            i2 = i
            while body_index[i2] != 0: # save all indices in support path
                if  body_index[i2] > 0: # ignore fixed links
                    support[idx-1].append(body_index[i2])
                i2 = link_names.index(parent_names[i2])
            support[idx-1].reverse()
        self.support = support
        self.child = child
        self.parent = parent
            
    def _nsimplify(
        self,num: float, *args, max_denominator: int=0, **kwargs
        ) -> Union[sympy.Expr, float]:
        ex = nsimplify(num,*args,**kwargs)
        if not max_denominator:
            return ex
        if ex.is_rational:
            try:
                d = ex.denominator()
                if d > max_denominator:
                    return num
            except ValueError:
                return ex
        elif type(ex) in {sympy.core.add.Add, sympy.core.power.Pow}:
            return num
        for i in regex.findall("(?<=Rational\(\d*, )\d*", sympy.srepr(ex)):
            if abs(int(i)) > max_denominator:
                return num
        return ex
        
    def load_from_urdf(self, path: str, symbolic: bool=True, 
                       simplify_numbers: bool=True, cse_ex: bool=False, 
                       tolerance: float=0.0001, max_denominator: int=9) -> None:
        robot = URDF.load(path)
        self.B = []
        self.X = []
        self._create_topology_lists(robot)
        fixed_origin = None
        fixed_links = []
        DOF = 0
        xyz_rpy_syms = []
        for joint in robot.joints:
            if joint.joint_type in ["revolute", "continuous", "prismatic"]:
                DOF += 1
            elif joint.joint_type in ["fixed"]:
                pass
            else:
                raise NotImplementedError(
                    "Joint type '" + joint.joint_type+"' not implemented yet!")

        ji = 0  # joint index of used joints
        jia = 0  # joint index of all joints (fixed included)
        joint_origins = []
        for joint in robot.joints:
            name = joint.name
            origin = Matrix(joint.origin)
            if symbolic:
                xyz_rpy = matrix_to_xyz_rpy(joint.origin)
                xyz_rpy_syms.append(symbols(
                    " ".join([name+"_%s" % s for s in ["x", "y", "z", "roll", "pitch", "yar"]])))
                xyzrpylist = []
                if simplify_numbers:
                    for i in range(6):
                        if (self._nsimplify(xyz_rpy[i], 
                                           tolerance=tolerance, 
                                           max_denominator=max_denominator) 
                            in [0, -1, 1, pi, -pi, pi/2, -pi/2, 3*pi/2, -3*pi/2]
                            ):
                            xyzrpylist.append(
                                self._nsimplify(xyz_rpy[i], tolerance=tolerance,
                                                max_denominator=max_denominator))
                        # elif nsimplify(xyz_rpy[i],tolerance=tolerance) == 1:
                            # xyzrpylist.append(1)
                        # elif nsimplify(xyz_rpy[i],tolerance=tolerance) == -1:
                            # xyzrpylist.append(-1)
                        else:
                            xyzrpylist.append(xyz_rpy_syms[jia][i])
                            self.assignment_dict[xyz_rpy_syms[jia]
                                                 [i]] = xyz_rpy[i]
                else:
                    for i in range(6):
                        if xyz_rpy[i] == 0:
                            xyzrpylist.append(0)
                        elif xyz_rpy[i] == 1:
                            xyzrpylist.append(1)
                        elif xyz_rpy[i] == -1:
                            xyzrpylist.append(-1)
                        else:
                            xyzrpylist.append(xyz_rpy_syms[jia][i])
                            self.assignment_dict[xyz_rpy_syms[jia]
                                                 [i]] = xyz_rpy[i]
                origin = xyz_rpy_to_matrix(xyzrpylist)
                if cse_ex:
                    origin = self._cse_expression(origin)
            elif simplify_numbers:
                for i in range(4):
                    for j in range(4):
                        origin[i, j] = self._nsimplify(
                            origin[i, j], [pi], tolerance=tolerance,
                            max_denominator=max_denominator)
            joint_origins.append(origin)
            if joint.joint_type in ["revolute", "continuous", "prismatic"]:
                # origin = Matrix(joint.origin)
                axis = Matrix(joint.axis)
                if simplify_numbers:
                    for i in range(3):
                        axis[i] = self._nsimplify(axis[i], [pi], tolerance=tolerance,
                                            max_denominator=max_denominator)
                if fixed_origin:
                    origin *= fixed_origin
                    fixed_origin = None
                self.B.append(Matrix(origin))

                if joint.joint_type in ["revolute", "continuous"]:
                    self.X.append(Matrix(axis).col_join(Matrix([0, 0, 0])))
                else:
                    self.X.append(Matrix(Matrix([0, 0, 0])).col_join(axis))
                ji += 1
            elif joint.joint_type == "fixed":
                if fixed_origin:
                    fixed_origin *= origin
                else:
                    fixed_origin = origin
                fixed_links.append((joint.parent, joint.child))
            jia += 1

        self.Mb = []
        # I_syms = []
        # m_syms = []
        # I_syms = [symbols("I%dxx I%dxy I%dxz I%dyy I%dyz I%dzz"%(i,i,i,i,i,i)) for i in range(DOF)]
        # m_syms = [symbols("m%d cx%d cy%d cz%d"%(i,i,i,i)) for i in range(DOF)]
        i = 0
        first_non_fixed = 1
        for link in robot.links:
            name = link.name
            # ignore base link
            if i < first_non_fixed:
                if name in [x[1] for x in fixed_links]:
                    first_non_fixed += 1
                i += 1
                continue
            inertia = Matrix(link.inertial.inertia)
            mass = link.inertial.mass
            inertiaorigin = Matrix(link.inertial.origin)
            if symbolic:
                I_syms = symbols("Ixx_%s Ixy_%s Ixz_%s Iyy_%s Iyz_%s Izz_%s" % (
                    name, name, name, name, name, name))
                c_syms = symbols("cx_%s cy_%s cz_%s" % (name, name, name))
                I = InertiaMatrix(*I_syms)
                m = symbols("m_%s" % name)
                cg = Matrix([*c_syms])
            else:
                if simplify_numbers:
                    for i in range(4):
                        for j in range(4):
                            inertiaorigin[i, j] = self._nsimplify(
                                inertiaorigin[i, j], [pi], tolerance=tolerance,
                                max_denominator=max_denominator)
                    for i in range(3):
                        for j in range(3):
                            inertia[i, j] = self._nsimplify(
                                inertia[i, j], [pi], tolerance=tolerance,
                                max_denominator=max_denominator)
                    # mass = nsimplify(mass, [pi], tolerance=tolerance)
                I = Matrix(inertia)
                m = mass
                cg = Matrix(inertiaorigin[0:3, 3])
            # if is fixed child: # TODO: find out what to do
            # cg =
            # I =
            # m =
            M = MassMatrixMixedData(m, I, cg)
            if name in [x[1] for x in fixed_links]:
                j = i
                # transform Mass matrix
                while robot.links[j].name in [x[1] for x in fixed_links]:
                    M = SE3AdjInvMatrix(
                        joint_origins[j-1]).T * M * SE3AdjInvMatrix(joint_origins[j-1])
                    j -= 1
                self.Mb[-1] += M
                i += 1
                continue
            self.Mb.append(M)
            i += 1

        # for link in robot.links:
        #     self.Mb.append(MassMatrixMixedData())
        return

    def dhToScrewCoord(self, DH_param_table: MutableDenseMatrix) -> None:
        """Build screw coordinate paramters (joint axis frames and 
        body reference frames) from a given modified Denavit-Hartenberg 
        (DH) parameter table.
        Joint screw coordinates and reference configurations of bodies 
        are directly applied to class.

        Args:
            DH_param_table (array_like): 
                Table with modified DH parameters (n,5) 
                -> (gamma,alpha,d,theta,r)
        """
        number_of_frames = DH_param_table.shape[0]
        self.B = []
        self.X = []
        for i in range(number_of_frames):
            # Reference configurations of bodies (i.e. of body-fixed 
            # reference frames) w.r.t their previous bodies
            # gamma, alpha, d, theta,r
            frame = DH_param_table[i, :]
            gamma = frame[0]
            alpha = frame[1]
            d = frame[2]
            theta = frame[3]
            r = frame[4]
            self.B.append(SO3Exp(Matrix([1, 0, 0]), alpha)
                          .row_join(Matrix([d, 0, 0]))
                          .col_join(Matrix([0, 0, 0, 1]).T)
                          * SO3Exp(Matrix([0, 0, 1]), theta)
                          .row_join(Matrix([0, 0, r]))
                          .col_join(Matrix([0, 0, 0, 1]).T)
                          )

            #  Joint screw coordinates in body-fixed representation
            if gamma == 0:
                self.X.append(Matrix([0, 0, 1, 0, 0, 0]))
            else:
                self.X.append(Matrix([0, 0, 0, 0, 0, 1]))

    def _set_value_as_process(self, name: str, target: Callable) -> None:
        """Set return value of target as value to queue in 
        self.queue_dict with identifier name.

        Args:
            name (str): Identifier.
            target (function): function, which returns value. 
        """
        if name not in self.queue_dict:
            self.queue_dict[name] = Queue()
        if name in self.process_dict:
            print("already there")
        self.process_dict[name] = Process(
            target=lambda: self._set_value(name, target()), args=(), name=name)
        self.process_dict[name].start()

    def _set_value(self, name: str, var: Any) -> None:
        """Set value to queue in self.queue_dict.

        Args:
            name (str): Identifier.
            var (any): Value to save.
        """
        if name not in self.queue_dict:
            self.queue_dict[name] = Queue()
        self.queue_dict[name].put(var)

    def _start_simplification_process(
        self, name: str, cse_ex: bool=False) -> None:
        """Start Process, which simplifies and overwrites value in 
        queue from self.queue_dict.

        Args:
            name (str): Identifier
            cse_ex (bool, optional): 
                Use common subexpression elimination. Defaults to False.
        """
        if name not in self.queue_dict:
            self.queue_dict[name] = Queue()
        self.process_dict[name+"_simplify"] = Process(
            target=self._simplify_parallel, 
            args=(name, cse_ex,), 
            name=name+"_simplify")
        self.process_dict[name+"_simplify"].start()

    def _start_cse_process(self, name: str) -> None:
        """Start Process, which generates cse expression and overwrites value in 
        queue from self.queue_dict.

        Args:
            name (str): Identifier
        """
        if name not in self.queue_dict:
            self.queue_dict[name] = Queue()
        self.process_dict[name+"_cse"] = Process(
            target=self._cse_parallel, 
            args=(name,), 
            name=name+"_cse")
        self.process_dict[name+"_cse"].start()

    def _get_value(self, name: str) -> Any:
        """Get value from queue in self.queue_dict and put it in again.

        Args:
            name (str): Identifier.

        Returns:
            any: Value
        """
        value = self.queue_dict[name].get()
        self.queue_dict[name].put(value)
        return value

    def _simplify_parallel(self, name: str, cse_ex: bool=False) -> None:
        """Take value from self.queue_dict, simplify it and put it in 
        again.

        Args:
            name (str): Identifier
            cse_ex (bool, optional): 
                Use common subexpression elimination. Defaults to False.
        """
        value = self.simplify(self.queue_dict[name].get(), cse_ex)
        self.queue_dict[name].put(value)
        
    def _cse_parallel(self, name: str) -> None:
        """Take value from self.queue_dict, generate cse_expressions and 
        put it in again.

        Args:
            name (str): Identifier
        """
        value = self._cse_expression(self.queue_dict[name].get())
        self.queue_dict[name].put(value)

    def _flush_queue(self, q: Queue) -> None:
        """Flush all items in queue

        Args:
            q (Queue): Queue to flush
        """
        try:
            while True:
                q.get(block=False)
        except queue.Empty:
            pass

    def _individual_numbered_symbols(
        self, exclude: list=[], i: List[int]=[0]) -> Generator[sympy.Symbol, None, None]:
        """create individual symbol names for subexpressions using 
        multiprocessing.

        Args:
            exclude (list, optional): 
                List of names, which should not be used. Defaults to [].
            i (list, optional): 
                List with starting value -1 as first value. Is used as 
                counter and should not be set. Defaults to [0].

        Returns:
            sympy.numbered_symbols:  Symbols
        """
        i[0] += 1
        prefix="sub%s_%s_" % (
            "_".join([str(j) for j in multiprocessing.current_process()._identity]), 
            i[0]
            )
        prefix = prefix.replace("sub_","sub0_")
        return numbered_symbols(
            prefix=prefix, 
            exclude=exclude)

    def _sort_variables(self, vars:List[sympy.Symbol]) -> List[sympy.Symbol]:
        """Sort variables for code generation starting with q, qd, qdd, 
        continuing with variable symbols and ending with constant 
        symbols.

        Args:
            vars (list of sympy.symbols): Variables to sort.

        Returns:
            list: Sorted list of variables.
        """
        # vars as set
        vars = set(vars)
        # divide into variable and constant symbols
        var_syms = self.var_syms.intersection(vars)
        rest = list(vars.difference(var_syms))
        # divide variable symbols into q, dq, ddq and other variable symbols
        q = []
        dq = []
        ddq = []
        var_rest = []
        for i in var_syms:
            if str(i).startswith("ddq"):
                ddq.append(i)
            elif str(i).startswith("dq"):
                dq.append(i)
            elif str(i).startswith("q"):
                q.append(i)
            else:
                var_rest.append(i)

        def symsort(data: List[sympy.Symbol]) -> List[sympy.Symbol]:
            """Sort symbols

            Args:
                data (list): symbols

            Returns:
                list: sorted symbols
            """
            return [x for _, x in sorted(zip(list(map(str, data)), data))]
        # return sorted list
        return symsort(q) + symsort(dq) + symsort(ddq) + symsort(var_rest) \
            + symsort(rest)

    def _cse_expression(self, exp: sympy.Expr) -> sympy.Expr:
        """Use common subexpression elimination to shorten expression.
        The used subexpressions are saved to the class internal 
        subex_dict.

        Args:
            exp (Sympy expression): Expression to shorten using cse.

        Returns:
            Sympy expression: Shortened expression.
        """
        # cse expression
        r, e = cse([exp, exp], self._individual_numbered_symbols(
            exclude=self.all_symbols), order="canonical", ignore=self.var_syms)
        # add subexpressions to dict
        for (sym, val) in r:
            self.subex_dict[sym] = val
            # for multiprocessing save in queue
            try:
                self.queue_dict["subex_dict"].put(self.subex_dict)
            except:
                pass
            # update used symbols
            self.all_symbols.update({sym})
        return e[0]

    def _get_expressions(self) -> List[sympy.Expr]:
        """Get list of all generated expressions.

        Returns:
            list: generated expressions.
        """
        expression_dict = self.get_expressions_dict()
        expressions = [expression_dict[i] for i in expression_dict]
        return expressions

    def _calc_A_matrix(
        self, q: MutableDenseMatrix
        ) -> Tuple[List[MutableDenseMatrix], MutableDenseMatrix]:
        # calc Forward kinematics
        if self._FK_C is not None:
            FK_C = self._FK_C
        elif self.A:
            # print("Using absolute configuration (A) of the body frames")
            FK_f = [SE3Exp(self.Y[0], q[0])]
            FK_C = [FK_f[0]*self.A[0]]
            for i in range(1, self.n):
                FK_f.append(FK_f[i-1]*SE3Exp(self.Y[i], q[i]))
                FK_C.append(FK_f[i]*self.A[i])
            self._FK_C = FK_C
            if not self.X:
                # Joint screw coordinates in body-fixed representation 
                # computed from screw coordinates in IFR
                self.X = [SE3AdjInvMatrix(
                    self.A[i])*self.Y[i] for i in range(self.n)]

        elif self.B:
            # print('Using relative configuration (B) of the body frames')
            FK_C = [self.B[0]*SE3Exp(self.X[0], q[0])]
            for i in range(1, self.n):
                FK_C.append(FK_C[i-1]*self.B[i]*SE3Exp(self.X[i], q[i]))
            self._FK_C = FK_C
        else:
            # 'Absolute (A) or Relative (B) configuration of the bodies should be provided in class!'
            raise ValueError("Joint screw coordinates and/or reference configuration of bodies not set.")

        # Block diagonal matrix A (6n x 6n) of the Adjoint of body frame
        if self._A is not None:
            A = self._A
        else:
            A = Matrix(Identity(6*self.n))
            for i in range(self.n):
                for j in range(i):
                    Crel = SE3Inv(FK_C[i])*FK_C[j]
                    AdCrel = SE3AdjMatrix(Crel)
                    r = 6*(i)
                    c = 6*(j)
                    A[r:r+6, c:c+6] = AdCrel
            self._A = A
        return FK_C, A

    def _calc_A_matrix_tree(self, q: MutableDenseMatrix):
        if self._FK_C is not None:
            FK_C = self._FK_C
        elif self.A:
            # print("Using absolute configuration (A) of the body frames")
            FK_f = []
            FK_C = []
            for i in range(self.n):
                if self.parent[i] == 0: # bodies with no predecessor
                    # Initialization for the first body
                    FK_f.append(SE3Exp(self.Y[i], q[i]))
                    FK_C.append(FK_f[i]*self.A[i])
                else:
                    FK_f.append(FK_f[self.parent[i]-1]*SE3Exp(self.Y[i], q[i]))
                    FK_C.append(FK_f[i]*self.A[i])      
            self._FK_C = FK_C
            if not self.X:
                # Joint screw coordinates in body-fixed representation 
                # computed from screw coordinates in IFR
                self.X = [SE3AdjInvMatrix(
                    self.A[i])*self.Y[i] for i in range(self.n)]

        elif self.B:
            # print('Using relative configuration (B) of the body frames')
            FK_C = []
            for i in range(self.n):
                if self.parent[i] == 0: # bodies with no predecessor
                    # Initialization for the first body
                    FK_C.append(self.B[i]*SE3Exp(self.X[i], q[i]))
                else:
                    FK_C.append(FK_C[self.parent[i]-1]*self.B[i]*SE3Exp(self.X[i], q[i]))
        else:
            # 'Absolute (A) or Relative (B) configuration of the bodies should be provided in class!'
            raise ValueError("Joint screw coordinates and/or reference configuration of bodies not set.")

        # Block diagonal matrix A (6n x 6n) of the Adjoint of body frame
        if self._A is not None:
            A = self._A
        else:
            A = Matrix(Identity(6*self.n))
            for i in range(self.n):        
                if True:
                # if self.parent[i] != 0:
                    for k in self.support[i]:
                        j = k-1
                        Crel = SE3Inv(FK_C[i])*FK_C[j]
                        AdCrel = SE3AdjMatrix(Crel)
                        r = 6*(i)
                        c = 6*(j)
                        A[r:r+6, c:c+6] = AdCrel
            self._A = A
        return FK_C, A
    
    def _find_start_index(self) -> int:
        # search all symbols
        syms = set()
        for i in self.Mb:
            syms.update(i.free_symbols)
        for i in self.body_ref_config:
            syms.update(i.free_symbols)
        for i in self.joint_screw_coord:
            syms.update(i.free_symbols)
        # search all indices in symbols
        indices = []
        for i in syms:
            indices.extend(regex.findall("\d+",str(i)))
        #find smalles used index
        if indices:
            if min([int(i) for i in indices]) == 0:
                return 0
            if max([int(i) for i in indices]) == self.n -1:
                return 0
        return 1
