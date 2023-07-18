from __future__ import annotations
import copy
import multiprocessing
import os
import queue
import random
from collections import defaultdict
from itertools import combinations
from multiprocessing import Process, Queue
from typing import Any, Callable, Generator, Optional

import numpy
import pydot
import regex
import sympy
import sympy.physics.mechanics
from pylatex import Command, Document, NoEscape, Section
from sympy import (Identity, Matrix, MutableDenseMatrix, cancel, ccode, factor,
                   julia_code, lambdify, nsimplify, octave_code, pi, powsimp,
                   symbols, zeros)
from sympy.printing.latex import LatexPrinter
from sympy.printing.numpy import NumPyPrinter
from sympy.simplify.cse_main import numbered_symbols
from sympy.simplify.fu import fu
from sympy.utilities.codegen import JuliaCodeGen, codegen
from urdf_parser_py.urdf import URDF

import skidy
from skidy.matrices import (SE3AdjInvMatrix, SE3AdjMatrix, SE3adMatrix, SE3Exp,
                            SE3Inv, SO3Exp, generalized_vectors,
                            inertia_matrix, mass_matrix_mixed_data,
                            matrix_to_xyz_rpy, xyz_rpy_to_matrix,
                            transformation_matrix, mass_matrix_to_parameter_vector)

class CodeGenerator_():
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
        self.Md = None # generalized_mass_inertia_matrix_dot 
        self.Cd = None # coriolis_centrifugal_matrix_dot
        self.Qdgrav = None # gravity_vector_dot
        self.Qd = None # inverse_dynamics_dot
        self.M2d = None # generalized_mass_inertia_matrix_ddot
        self.C2d = None # coriolis_centrifugal_matrix_ddot
        self.Q2dgrav = None # gravity_vector_ddot
        self.Q2d = None # inverse_dynamics_ddot
        self.Vh_BFn = None # hybrid_twist
        self.Vb_BFn = None # body_twist
        self.Vhd_BFn = None  # hybrid_acceleration
        self.Vbd_BFn = None  # body_acceleration
        self.Vhd_ee = None  # hybrid_acceleration_ee
        self.Vbd_ee = None  # body_acceleration_ee
        self.Jh_dot = None  # hybrid_jacobian_matrix_dot
        self.Jb_dot = None  # body_jacobian_matrix_dot
        self.Jh_ee_dot = None  # hybrid_jacobian_matrix_ee_dot
        self.Jb_ee_dot = None  # body_jacobian_matrix_ee_dot
        self.Yr = None # regressor_matrix
        self.Xr = None # parameter_vector

        # set of variable symbols to use in generated functions as arguments
        self.var_syms = set()
        self.optional_var_syms = set() # for wrenches
        
        # Value assignment
        # dict with assigned variables for code generation
        self.assignment_dict = {}  
        # dict for subexpressions fro common subexpression elimination
        self.subex_dict = {}  

        self.all_symbols = set()  # set with all used symbols


    def get_expressions_dict(self, filterNone: bool=True) -> dict[str,sympy.Expr | None]:
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
        expressions = {"forward_kinematics": self.fkin,
                       "system_jacobian_matrix": self.J,
                       "body_jacobian_matrix": self.Jb,
                       "hybrid_jacobian_matrix": self.Jh,
                       "system_jacobian_dot": self.Jdot,
                       "body_twist_ee": self.Vb_ee,
                       "hybrid_twist_ee": self.Vh_ee,
                       "body_twist": self.Vb_BFn,
                       "hybrid_twist": self.Vh_BFn,
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
                       "body_jacobian_matrix_ee_dot": self.Jb_ee_dot,
                       "generalized_mass_inertia_matrix_dot": self.Md, 
                       "coriolis_centrifugal_matrix_dot": self.Cd,
                       "gravity_vector_dot": self.Qdgrav,
                       "inverse_dynamics_dot": self.Qd,
                       "generalized_mass_inertia_matrix_ddot": self.M2d,
                       "coriolis_centrifugal_matrix_ddot": self.C2d,
                       "gravity_vector_ddot": self.Q2dgrav,
                       "inverse_dynamics_ddot": self.Q2d,
                       "regressor_matrix": self.Yr,
                       "parameter_vector": self.Xr,
                       }  
        
        # deal with multiple ee
        all_expressions = {}
        for k, v in expressions.items():
            if type(v) is list:
                if len(v) > 1:
                    for i in range(len(v)):
                        all_expressions[f"{k}_{i+1}"] = v[i]
                else:
                    all_expressions[k] = v[0]
            else:
                all_expressions[k] = v
        
        # exclude expressions which are None
        if filterNone:
            filtered = {k: v for k, v in all_expressions.items()
                        if v is not None}
            return filtered
        return all_expressions

    def generate_python_code(self, name: str="plant", folder: str="./python", 
                             cache: bool=False, use_global_vars: bool = True):
        """Generate python code from generated expressions. 
        Needs 'closed_form_inv_dyn_body_fixed' and/or 
        'closed_form_kinematics_body_fixed' to run first.


        Args:
            name (str, optional): 
                Name of class and file. 
                Defaults to "plant".
            folder (str, optional): 
                Folder where to save code. 
                Defaults to "./python".
            cache (bool, optional):
                Cache results of sin and cos function in generated python 
                code. Defaults to False.
            use_global_vars (bool, optional): 
                Constant vars are class variables. Defaults to True.
        """
        
        names, expressions, _, constant_syms, not_assigned_syms = self._prepare_code_generation(folder, use_global_vars)
        
        
        print("Generate Python code")
        # create folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        p = NumPyPrinter()

        # start python file with import
        s = ["import numpy"]
        if cache:
            s.append("from functools import lru_cache")
        s.append("\n")
        # class name
        s.append("class "+regex.sub("^\w",lambda x: x.group().upper(),name)+"():")
        # define __init__ function
        s.append("    def __init__(self, %s) -> None:" % (
            ", ".join(
                [str(not_assigned_syms[i])+": float" 
                    for i in range(len(not_assigned_syms))] 
                + [str(i)+": float=" + str(self.assignment_dict[i]) 
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

        # avoid empty init function
        if (not len(self.subex_dict) > 0 
            and not len(self.assignment_dict) > 0
            and not len(not_assigned_syms) > 0
            ):
            s.append("        pass")
        
        # define functions
        for i in range(len(expressions)):
            var_syms = self._sort_variables(self.var_syms.intersection(
                expressions[i].free_symbols))
            optional_var_syms = self._sort_variables(self.optional_var_syms.intersection(
                expressions[i].free_symbols))
            const_syms = self._sort_variables(
                set(constant_syms).intersection(
                    expressions[i].free_symbols))
            if len(var_syms) > 0 or len(optional_var_syms) > 0:
                s.append("\n    def "+names[i]+"(self, %s) -> numpy.ndarray:" % (
                    ", ".join([str(var_syms[i])+": float" 
                                for i in range(len(var_syms))]
                                + [str(optional_var_syms[i])+": float=0" 
                                for i in range(len(optional_var_syms))])))

            else:
                s.append("\n    def "+names[i]+"(self) -> numpy.ndarray:")
            
            if const_syms: # insert self. in front of const_syms
                s.append("        "
                        + names[i] 
                        + " = " 
                        + regex.sub(f"(?<=\W|^)(?={'|'.join([str(i) for i in const_syms])}(\W|\Z))",
                                    "self.",
                                    p.doprint(expressions[i])))
            else: 
                s.append("        "
                        + names[i] 
                        + " = " 
                        + p.doprint(expressions[i]))
            s.append("        return " + names[i])

        if cache:
            s = list(map(lambda x: x.replace("numpy.sin", "cached_sin"), s))
            s = list(map(lambda x: x.replace("numpy.cos", "cached_cos"), s))
            
            s.append("")
            s.append("@lru_cache(maxsize=128)")
            s.append("def cached_sin(x: float) -> float:")
            s.append("    return numpy.sin(x)")
            s.append("")
            s.append("@lru_cache(maxsize=128)")
            s.append("def cached_cos(x: float) -> float:")
            s.append("    return numpy.cos(x)")
            
        # replace numpy with np for better readability
        s = list(map(lambda x: x.replace("numpy.", "np."), s))
        s[0] = "import numpy as np\n\n"

        # join list to string
        s = "\n".join(s)

        # write python file
        with open(os.path.join(folder, name + ".py"), "w+") as f:
            f.write(s)
        print("Done")
    
    def generate_cython_code(self, name: str="plant", folder: str="./cython", 
                             cache: bool=False, use_global_vars: bool = True):
        """Generate cython code from generated expressions. 
        Needs 'closed_form_inv_dyn_body_fixed' and/or 
        'closed_form_kinematics_body_fixed' to run first.


        Args:
            name (str, optional): 
                Name of class and file. 
                Defaults to "plant".
            folder (str, optional): 
                Folder where to save code. 
                Defaults to "./cython".
            cache (bool, optional):
                Cache results of sin and cos function in generated cython 
                code. Defaults to False.
            use_global_vars (bool, optional): 
                Constant vars are class variables. Defaults to True.
        """
        
        names, expressions, _, constant_syms, not_assigned_syms = self._prepare_code_generation(folder, use_global_vars)
        
        print("Generate Cython code")
        # create folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        p = NumPyPrinter()

        # start python file with import
        s = ["import numpy"]
        s.append("cimport numpy\n")
        s.append("cimport cython\n")
        if cache:
            s.append("from functools import lru_cache\n")
        s.append("")
        s.append("numpy.import_array()")
        # s.append("DTYPE = numpy.float64")
        # s.append("ctypedef numpy.float64_t DTYPE_t")
        
        s.append("\n")
        # class name
        s.append("@cython.boundscheck(False)")
        s.append("@cython.wraparound(False)")
        s.append("cdef class "+regex.sub("^\w",lambda x: x.group().upper(),name)+"():")
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
                
        
        # define __cinit__ function
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

        # append preassigned values to __cinit__ function
        if len(self.assignment_dict) > 0:
            s.append("        "
                        + ", ".join(sorted(["self."+str(i) 
                                            for i in self.assignment_dict]))
                        + " = " 
                        + ", ".join(sorted([str(i) 
                                            for i in self.assignment_dict]))
                        )

        # append cse expressions to __cinit__ function
        if len(self.subex_dict) > 0:
            for i in sorted([str(j) for j in self.subex_dict], key=lambda x: int(regex.findall("(?<=sub)\d*",x)[0])):
                modstring = p.doprint(self.subex_dict[symbols(i)])
                for j in sorted([str(h) 
                                    for h in self.subex_dict[symbols(i)].free_symbols],
                                reverse=1):
                    modstring = regex.sub(
                        str(j), "self."+str(j), modstring)
                    modstring = regex.sub("(?<=(\W|^)(?<!\.)\d+)(?!\.)(?=\W|\Z)",".0", modstring) # replace integer with floats
                    # remove double self
                    modstring = regex.sub("self.self.", "self.", modstring)
                s.append("        self."+str(i)+" = " + modstring)

        # avoid empty init function
        if (not len(self.subex_dict) > 0 
            and not len(self.assignment_dict) > 0
            and not len(not_assigned_syms) > 0
            ):
            s.append("        pass")
        
        # define functions
        for i in range(len(expressions)):
            var_syms = self._sort_variables(self.var_syms.intersection(
                expressions[i].free_symbols))
            optional_var_syms = self._sort_variables(self.optional_var_syms.intersection(
                expressions[i].free_symbols))
            const_syms = self._sort_variables(
                set(constant_syms).intersection(
                    expressions[i].free_symbols))
            s.append("") # empty line
            s.append("    @cython.boundscheck(False)")
            s.append("    @cython.wraparound(False)")
            if len(var_syms) > 0 or len(optional_var_syms) > 0:
                # s.append(f"    cpdef double[:,::1] "+names[i]+"(self, %s):" % (
                s.append(f"    cpdef numpy.ndarray[double, ndim={len(expressions[i].shape)}] "+names[i]+"(self, %s):" % (
                # s.append(f"\n    cpdef np.ndarray[DTYPE_t,ndim={len(expressions[i].shape)}] "+names[i]+"(self, %s):" % (
                    ", ".join(["double "+str(var_syms[i]) 
                                for i in range(len(var_syms))]
                                + ["double "+str(optional_var_syms[i])+"=0.0" 
                                    for i in range(len(optional_var_syms))])))

            else:
                s.append(f"    cpdef numpy.ndarray[double, ndim={len(expressions[i].shape)}] "+names[i]+"(self):")
            
            # s.append(f"        cdef double[:,::1] {names[i]} = numpy.empty({expressions[i].shape}, dtype=double)")
            s.append(f"        cdef numpy.ndarray[double, ndim={len(expressions[i].shape)}] {names[i]} = numpy.empty({expressions[i].shape}, dtype=np.double)")
            for j in range(expressions[i].shape[0]):
                for k in range(expressions[i].shape[1]):
                    if const_syms:
                        # add self before const_syms
                        ex_string = regex.sub(f"(?<=\W|^)(?={'|'.join([str(i) for i in const_syms])}(\W|\Z))","self.",p.doprint(expressions[i][j,k]))
                        # replace integer with floats
                        ex_string = regex.sub("(?<=(\W|^)(?<!\.)\d+)(?!\.)(?=\W|\Z)",".0", ex_string) 
                        s.append("        "
                                + names[i]
                                + f"[{j}, {k}]" 
                                + " = " 
                                + ex_string)
                                #  + regex.sub("(?<=((?<=[^\.])\W)\d+)(?=\W)(?!\.)",".0",regex.sub(f"(?<=\W|^)(?={'|'.join([str(i) for i in const_syms])}(\W|\Z))","self.",p.doprint(expressions[i]))))
                    else:
                        ex_string = p.doprint(expressions[i])
                        # replace integer with floats
                        ex_string = regex.sub("(?<=(\W|^)(?<!\.)\d+)(?!\.)(?=\W|\Z)",".0", ex_string) 
                        s.append("        "
                                + names[i] 
                                + f"[{j}, {k}]" 
                                + " = " 
                                + ex_string)
                        
            s.append("        return " + names[i])

        if cache:
            s = list(map(lambda x: x.replace("numpy.sin", "cached_sin"), s))
            s = list(map(lambda x: x.replace("numpy.cos", "cached_cos"), s))
            
            s.append("")
            s.append("@lru_cache(maxsize=128)")
            s.append("def cached_sin(double x):")
            s.append("    return numpy.sin(x)")
            s.append("")
            s.append("@lru_cache(maxsize=128)")
            s.append("def cached_cos(double x):")
            s.append("    return numpy.cos(x)")
        
        # replace numpy with np for better readability
        s = list(map(lambda x: x.replace("numpy.", "np."), s))
        s[0] = "import numpy as np"
        s[1] = "cimport numpy as np\n"

        # join list to string
        s = "\n".join(s)

        # create setup file to compile cython code
        su = ("import os.path\n"
                + "from setuptools import setup\n"
                + "from Cython.Build import cythonize\n"
                + "\n"
                + "setup(\n"
                + f"    name='{name}',\n"
                + f"    ext_modules=cythonize(os.path.join(os.path.dirname(__file__), '{name + '.pyx'}'), \n"
                +  "                          compiler_directives={'language_level' : '3'}),\n"
                + "    zip_safe=False,\n"
                + ")\n"
        )


        # write cython file
        with open(os.path.join(folder, name + ".pyx"), "w+") as f:
            f.write(s)
        
        # write setup file
        with open(os.path.join(folder, "setup_" + name + ".py"), "w+") as f:
            f.write(su)
        
        print("Done")

        
    def generate_C_code(self, name: str="plant", folder: str="./C", 
                        project: str="project", use_global_vars: bool = True):
        """Generate C code from generated expressions. 
        Needs 'closed_form_inv_dyn_body_fixed' and/or 
        'closed_form_kinematics_body_fixed' to run first.


        Args:
            name (str, optional): 
                Name of file (without extension). 
                Defaults to "plant".
            folder (str, optional): 
                Folder where to save code. 
                Defaults to "./C".
            project (str, optional): 
                Project name in C header. Defaults to "project".
            use_global_vars (bool, optional): 
                Constant vars like mass etc are global variables. Defaults to True.
        """
        names, expressions, all_expressions, constant_syms, not_assigned_syms = self._prepare_code_generation(folder, use_global_vars)
        
        print("Generate C code")
        # generate folder
        if not os.path.exists(folder):
            os.mkdir(folder)

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
                if cols > 1:
                    c_lines[i] = c_lines[i].replace("double *out","double **out")
                    c_header = regex.sub(f"(?<= {fname}\(.*)\*out(?=\))","**out",c_header)
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
        
        # save assinged parameters to another c file
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
        with open(os.path.join(folder, c_name), "w+") as f:
            f.write(c_code)
        with open(os.path.join(folder, h_name), "w+") as f:
            f.write(c_header)
        if header_insert:
            with open(os.path.join(folder, c_def_name), "w+") as f:
                f.writelines(c_definitions)
            
        print("Done")

    def generate_julia_code(self, name: str="plant", folder: str="./julia", 
                            use_global_vars: bool = True):
        """Generate julia code from generated expressions. 
        Needs 'closed_form_inv_dyn_body_fixed' and/or 
        'closed_form_kinematics_body_fixed' to run first.


        Args:
            name (str, optional): 
                Name of file (without extension). 
                Defaults to "plant".
            folder (str, optional): 
                Folder where to save code. 
                Defaults to "./julia".
            use_global_vars (bool, optional): 
                Constant vars like mass etc are global variables. Defaults to True.
        """
        names, expressions, _, constant_syms, not_assigned_syms = self._prepare_code_generation(folder, use_global_vars)
        print("Generate julia code")
        # generate folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        jcg = JuliaCodeGen()
        if use_global_vars:
            routines = [jcg.routine(names[i],
                                    expressions[i],
                                    self._sort_variables(
                                        (self.var_syms 
                                            | self.optional_var_syms)
                                        & expressions[i].free_symbols),
                                    global_vars=constant_syms) 
                        for i in range(len(expressions))]
            
            # save assigned parameters to beginning of julia file
            j_definitions = []
            
            if not_assigned_syms:
                j_definitions.append("# Please assign values\n")
                for var in sorted([str(i) for i in not_assigned_syms]):
                    if var == "g":
                        j_definitions.append(f"{var} = 9.81\n")
                    else:
                        j_definitions.append(f"{var} = 0\n")
                j_definitions.append("\n")

            if self.assignment_dict:                
                for var in sorted([str(i) for i in self.assignment_dict]):
                    val = julia_code(self.assignment_dict[symbols(var)])
                    j_definitions.append(f"{var} = {val}\n")
                j_definitions.append("\n")

            # append cse expressions
            if self.subex_dict:
                j_definitions.append("# subexpressions due to cse\n")
                for var in sorted([str(j) for j in self.subex_dict], key=lambda x: int(regex.findall("(?<=sub)\d*",x)[0])):
                    val = julia_code(self.subex_dict[symbols(var)])
                    j_definitions.append(f"{var} = {val}\n")
                j_definitions.append("\n")

            j_constants = "".join(j_definitions)
        else:
            routines = [jcg.routine(names[i],
                                    expressions[i],
                                    self._sort_variables(
                                        expressions[i].free_symbols),
                                    global_vars=None) 
                        for i in range(len(expressions))]
            j_constants = ""
            
        j_name, j_code = jcg.write(routines, name, header=False)[0]
        j_code = j_constants + j_code
        
        # ensure operator ^ instead of **
        j_code = j_code.replace("**","^")
        
        # write code files
        with open(os.path.join(folder, j_name), "w+") as f:
            f.write(j_code)
        
        print("Done")

    def generate_matlab_code(self, name: str="plant", 
                             folder: str="./matlab"):
        """Generate matlab / octave code from generated expressions. 
        Needs 'closed_form_inv_dyn_body_fixed' and/or 
        'closed_form_kinematics_body_fixed' to run first.


        Args:
            name (str, optional): 
                Name of class and file. 
                Defaults to "plant".
            folder (str, optional): 
                Folder where to save code. 
                Defaults to "./matlab".
        """
        names, expressions, _, _, not_assigned_syms = self._prepare_code_generation(folder, True)
        
        # create folder
        if not os.path.exists(folder):
            os.mkdir(folder)
        
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
        # function arguments
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
                (names[i], expressions[i]), "Octave", 
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
                "(" + '|'.join([f', {str(v)}(?=\W)' for v in self.var_syms.union(self.optional_var_syms).difference(expressions[i].free_symbols)])+")",
                "", m_func[0])
            # use obj.variables defined in class parameters
            for i in range(1, len(m_func)):
                m_func[i] = regex.sub(f"(?<=\W|^)(?=({'|'.join([str(s) for s in m_properties])})\W)","obj.",m_func[i])        
            m_func.append("\n")
            m_class.extend(m_func)
        
        m_class.append("\tend\n")
        m_class.append("end\n")
        with open(os.path.join(folder, f"{name}.m"), "w+") as f:
            f.writelines(m_class)
    
    def generate_latex_document(self, name: str="plant", folder: str="./latex", 
                                landscape: bool=False):
        """Generate LaTeX document from generated expressions. 
        Needs 'closed_form_inv_dyn_body_fixed' and/or 
        'closed_form_kinematics_body_fixed' to run first.

        Args:
            name (str, optional): 
                Name of file. 
                Defaults to "plant".
            folder (str, optional): 
                Folder where to save code. 
                Defaults to "./latex".
            landscape (bool, optional):
                Generate LaTeX document in landscape mode to fit longer equations.
                Defaults to False.
        """
        names, expressions, _, _, _ = self._prepare_code_generation(folder, False)
        # create folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Document with `\maketitle` command activated
        doc = Document(documentclass="article", inputenc="utf8")
        if landscape:
            doc.packages.append(NoEscape(r"\usepackage[landscape,a4paper,top=2cm,bottom=2cm,left=2.5cm,right=2.5cm,marginparwidth=1.75cm]{geometry}"))
        else: 
            doc.packages.append(NoEscape(r"\usepackage[a4paper,top=2cm,bottom=2cm,left=2.5cm,right=2.5cm,marginparwidth=1.75cm]{geometry}"))
        doc.packages.append(NoEscape(r"\usepackage{amsmath}"))
        doc.packages.append(NoEscape(r"\usepackage{graphicx}"))
        doc.packages.append(NoEscape(r"\usepackage{breqn}[breakdepth={100}]"))
        
        doc.preamble.append(Command("title", "Equations of Motion"))
        doc.preamble.append(Command("author", "Author: SymbolicKinDyn"))
        doc.preamble.append(Command("date", NoEscape(r"\today")))
        doc.append(NoEscape(r"\maketitle"))
        doc.append(NoEscape(r"\tableofcontents"))
        doc.append(NoEscape(r"\newpage"))
        
        # create symbols and indices for equations
        for i in range(len(expressions)):
            letter = ""
            if "jacobian" in names[i]: letter = "J"
            elif "twist" in names[i]: letter = r"V"
            elif "kinematics" in names[i]: letter = r"^0T_E"
            elif "inertia" in names[i]: letter = "M"
            elif "coriolis" in names[i]: letter = "C"
            elif "gravity" in names[i]: letter = "g"
            elif "regressor" in names[i]: letter = "Y"
            elif "parameter_vector" in names[i]: letter = "X"
            elif "dynamics" in names[i]: letter = r"\tau"
            elif "acceleration" in names[i]: letter = r"\dot V"
            if "ddot" in names[i]: letter = r"\ddot "+letter
            elif "dot" in names[i]: letter = r"\dot "+letter
            if "_ee" in names[i]: letter = r"^E" + letter
            elif ("twist" in names[i] 
                    or "jacobian" in names[i] 
                    or "acceleration" in names[i]): letter = r"^0" + letter
            if "hybrid" in names[i]: letter += r"_h"
            elif "body" in names[i]: letter += r"_b"
            
            replacements = [("ddddq", r"\\ddddot q"),
                            ("ddddtheta", r"\\ddddot \\theta"),
                            ("dddq", r"\\dddot q"),
                            ("dddtheta", r"\\dddot \\theta"),
                            ("ddq", r"\\ddot q"), 
                            ("ddtheta", r"\\ddot \\theta"), 
                            ("dq", r"\\dot q"),
                            ("dtheta", r"\\dot \\theta")
                            ]
            with doc.create(Section(regex.sub("_"," ",names[i]))):
                maxlen = 0
                for row in range(expressions[i].shape[0]):
                    length = 0
                    for column in range(expressions[i].shape[1]):
                            length += len(regex.sub(r"(\\left|\\right|\{|\}|\\|_|\^|dot|ddot| |begin|matrix)","",str(expressions[i][row,column])))
                    maxlen = max(maxlen, length)
                if maxlen < 120+int(landscape)*100:
                    eq = LatexPrinter().doprint(expressions[i])
                    for pat, repl in replacements:
                        eq = regex.sub(pat, repl, eq)
                    # doc.append(NoEscape(r"\begin{footnotesize}"))
                    doc.append(NoEscape(r"\[ \resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{$%"))
                    doc.append(NoEscape(r"\boldsymbol{"f"{letter}""} = "f"{eq}"))
                    doc.append(NoEscape(r"$} \]"))
                else:
                    doc.append(NoEscape(r"\begin{dgroup*}"))
                    for row in range(expressions[i].shape[0]):
                        for column in range(expressions[i].shape[1]):
                            eq = LatexPrinter().doprint(expressions[i][row,column])
                            for pat, repl in replacements:
                                eq = regex.sub(pat, repl, eq)
                            doc.append(NoEscape(r"\begin{dmath*}"))
                            doc.append(NoEscape(f"{letter}_"+"{"+f"{row+1}{','+str(column+1) if expressions[i].shape[1]>1 else ''}" + "}" + " = " + f"{eq}"))
                            doc.append(NoEscape(r"\end{dmath*}"))
                    doc.append(NoEscape(r"\end{dgroup*}"))
                    
                # doc.append(NoEscape(r"\end{footnotesize}"))
                # doc.append("\n")
        
        # save tex file and compile pdf
        doc.generate_pdf(os.path.join(folder, name), clean_tex=False)

    
    def generate_code(self, python: bool=False, C: bool=False, Matlab: bool=False, 
                      cython: bool=False, julia: bool=False, latex: bool=False, 
                      cache: bool=False, landscape: bool=False,
                      folder: str="./generated_code", use_global_vars: bool=True, 
                      name: str="plant", project: str="Project") -> None:
        """Generate code from generated expressions. 
        Generates Python, C (C99), Matlab/Octave, Cython and/or LaTeX code.  
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
            julia (bool, optional):
                Generate julia code. Defaults to False.
            latex (bool, optional):
                Generate latex code with all equations and generate pdf from it. 
                Defaults to False.
            cache (bool, optional):
                Cache results sin and cos function in generated python 
                and cython code. Defaults to False.
            landscape (bool, optional):
                Generate LaTeX document in landscape mode to fit longer equations.
                Defaults to False.
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
        # create folder
        if not os.path.exists(folder):
            os.mkdir(folder)
            
        if python:
            self.generate_python_code(name, os.path.join(folder,"python"),cache, use_global_vars)
        if cython:
            self.generate_cython_code(name, os.path.join(folder,"cython"),cache, use_global_vars)
        if C:
            self.generate_C_code(name, os.path.join(folder,"C"), project, use_global_vars)
        if julia:
            self.generate_julia_code(name, os.path.join(folder,"julia"), use_global_vars)
        if Matlab:
            self.generate_matlab_code(name, os.path.join(folder,"matlab"))
        if latex:
            self.generate_latex_document(name, os.path.join(folder,"latex"), landscape)

    def generate_graph(self, path: str="output.pdf", include_mb: bool=False) -> None:
        """Write pdf graph from robot structure.

        Args:
            path (str, optional): Path where to save graph. 
                Defaults to "output.pdf".
            include_mb (bool, optional): Include mass inertia matrix. 
                Defaults to False.
        """
        graph = pydot.Dot("Robot structure", graph_type="digraph")
        graph.add_node(pydot.Node("0_l", label="0 (base)", shape="box"))
        for i in range(len(self.parent) if self.parent else self.n):
            lab = "[[{}, {}, {}, {}], \l [{}, {}, {}, {}], \l [{}, {}, {}, {}], \l [{}, {}, {}, {}]]\l".format(
                *self.body_ref_config[i]
            )
            # joints
            vars = [str(x[i]) for x in [self.q,self.qd,self.q2d,self.q3d, self.q4d] if x]
            graph.add_node(pydot.Node(f"{i+1}",label="[{}, {}, {}, {}, {}, {}]{}"
                                      .format(*self.joint_screw_coord[i],"\n{"+", ".join(vars)+"}" if vars else ""), color="blue"))
            graph.add_edge(pydot.Edge(f"{self.parent[i] if self.parent else i}_l",f"{i+1}",
                                      label=lab))
            # links
            if include_mb:
                lab = (("[[{}, {}, {}, {}, {}, {}], \l [{}, {}, {}, {}, {}, {}], \l"
                        " [{}, {}, {}, {}, {}, {}], \l [{}, {}, {}, {}, {}, {}], \l"
                        " [{}, {}, {}, {}, {}, {}], \l [{}, {}, {}, {}, {}, {}]]\l")
                    .format(*self.Mb[i]))
            graph.add_node(pydot.Node(f"{i+1}_l",shape="box", label=f"{i+1}"+(":\n"+lab if include_mb else "")))
            graph.add_edge(pydot.Edge(f"{i+1}",f"{i+1}_l"))
        
        # end effector
        # set ee_parent if not already set
        self.ee_parent = self.ee_parent or (len(self.parent) if self.parent else self.n)
        if type(self.ee_parent) is list and len(self.parent)>1:
            for i in range(len(self.ee_parent)):
                # ee node with WEE, WDEE, and W2DEE label in case they are non zero
                lab = (f"ee_{i+1}"
                       # WEE
                       +"{}".format(
                           "\n[{}, {}, {}, {}, {}, {}]"
                           .format(*self.WEE[i] 
                                   if type(self.WEE) is list 
                                   else self.WEE)
                           if not self.WEE == zeros(6,1) 
                           and not (type(self.WEE) is list 
                                    and self.WEE[i] == zeros(6,1))
                           else "")
                       # WDEE
                       +"{}".format(
                           "\n[{}, {}, {}, {}, {}, {}]"
                           .format(*self.WDEE[i] 
                                   if type(self.WDEE) is list 
                                   else self.WDEE)
                           if not self.WDEE == zeros(6,1) 
                           and not (type(self.WDEE) is list 
                                    and self.WDEE[i] == zeros(6,1))
                           else "")
                       # W2DEE
                       +"{}".format(
                           "\n[{}, {}, {}, {}, {}, {}]"
                           .format(*self.W2DEE[i] 
                                   if type(self.W2DEE) is list 
                                   else self.W2DEE)
                           if not self.W2DEE == zeros(6,1) 
                           and not (type(self.W2DEE) is list 
                                    and self.W2DEE[i] == zeros(6,1))
                           else ""))
                graph.add_node(
                    pydot.Node(f"ee_{i+1}",  shape="plaintext", 
                              label=lab
                    )
                )
                lab = "[[{}, {}, {}, {}], \l [{}, {}, {}, {}], \l [{}, {}, {}, {}], \l [{}, {}, {}, {}]]\l".format(
                    *self.ee[i]
                )
                graph.add_edge(pydot.Edge(f"{self.ee_parent[i]}_l",f"ee_{i+1}", 
                                            label=lab))
        else:
            # ee node with WEE, WDEE, and W2DEE label in case they are non zero
            lab = (f"ee"
                   # WEE
                   +"{}".format(
                       "\n[{}, {}, {}, {}, {}, {}]"
                       .format(*self.WEE[0] 
                               if type(self.WEE) is list # decide wether [Matrix] or just Matrix
                               else self.WEE)
                       if not self.WEE == zeros(6,1) # only if non zero
                       and not (type(self.WEE) is list and self.WEE[0] == zeros(6,1)) # can be a list too...
                       else "")
                   # WDEE
                   +"{}".format(
                       "\n[{}, {}, {}, {}, {}, {}]"
                       .format(*self.WDEE[0] 
                               if type(self.WDEE) is list 
                               else self.WDEE)
                       if not self.WDEE == zeros(6,1) 
                       and not (type(self.WDEE) is list and self.WDEE[0] == zeros(6,1))
                       else "")
                   # W2DEE
                   +"{}".format(
                       "\n[{}, {}, {}, {}, {}, {}]"
                       .format(*self.W2DEE[0] 
                               if type(self.W2DEE) is list 
                               else self.W2DEE)
                       if not self.W2DEE == zeros(6,1) 
                       and not (type(self.W2DEE) is list and self.W2DEE[0] == zeros(6,1))
                       else ""))
            graph.add_node(
                pydot.Node(f"ee", shape="plaintext",
                           label=lab
                )
            )
            lab = "[[{}, {}, {}, {}], \l [{}, {}, {}, {}], \l [{}, {}, {}, {}], \l [{}, {}, {}, {}]]\l".format(
                    *self.ee[0] if type(self.ee) is list else self.ee
                )
            graph.add_edge(pydot.Edge(
                f"{self.ee_parent[0] if type(self.ee_parent) is list else self.ee_parent}_l",f"ee",
                label=lab
                ))
        graph.write_pdf(path)
        
    def _prepare_code_generation(self, folder, use_global_vars):
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
                all_syms
                .difference(self.var_syms)
                .difference(self.optional_var_syms)
                .union(self.subex_dict))
            # generate list with preassigned symbols like subexpressions
            # from common subexpression elimination
            not_assigned_syms = self._sort_variables(
                all_syms
                .difference(self.var_syms)
                .difference(self.optional_var_syms)
                .difference(self.assignment_dict)
                .difference(self.subex_dict)
                )
        else:
            constant_syms = []
            not_assigned_syms = []
            
        return names, expressions, all_expressions, constant_syms, not_assigned_syms

        
    
    def _sort_variables(self, vars:list[sympy.Symbol]) -> list[sympy.Symbol]:
        """Sort variables for code generation starting with q, qd, qdd, 
        continuing with variable symbols (like fx in WEE) and ending 
        with constant symbols.

        Args:
            vars (list of sympy.symbols): Variables to sort.

        Returns:
            list: Sorted list of variables.
        """
        # vars as set
        vars = set(vars)
        # divide into variable and constant symbols
        var_syms = self.var_syms.intersection(vars)
        optional_var_syms = self.optional_var_syms.intersection(vars)
        rest = list(vars.difference(var_syms).difference(optional_var_syms))
        
        # function to sort symbols by name; returns list
        symsort = lambda data: [x for _, x in sorted(zip(list(map(str, data)), data))]
        
        sorted_syms = []
        # go through generalized vectors and sort variables according to their order
        for i in self.q:
            if i in var_syms:
                sorted_syms.append(i)
                var_syms.remove(i)
        for i in self.qd:
            if i in var_syms:
                sorted_syms.append(i)
                var_syms.remove(i)
        for i in self.q2d:
            if i in var_syms:
                sorted_syms.append(i)
                var_syms.remove(i)
        if self.q3d:
            for i in self.q3d:
                if i in var_syms:
                    sorted_syms.append(i)
                    var_syms.remove(i)
        if self.q4d:
            for i in self.q4d:
                if i in var_syms:
                    sorted_syms.append(i)
                    var_syms.remove(i)
        if var_syms:
            sorted_syms += symsort(var_syms)
        # now append symbols in external wrenches
        for i in self.WEE:
            if i in optional_var_syms:
                sorted_syms.append(i)
                optional_var_syms.remove(i)
        for i in self.WDEE:
            if i in optional_var_syms:
                sorted_syms.append(i)
                optional_var_syms.remove(i)
        for i in self.W2DEE:
            if i in optional_var_syms:
                sorted_syms.append(i)
                optional_var_syms.remove(i)
        if optional_var_syms:
            sorted_syms += symsort(optional_var_syms)
        # finally sort additional symbols by their name
        sorted_syms += symsort(rest)
        return sorted_syms

class SymbolicKinDyn(CodeGenerator_):
    BODY_FIXED = "body_fixed"
    SPATIAL = "spatial"
    
    def __init__(self, 
                 gravity_vector: Optional[MutableDenseMatrix | list]=None, 
                 ee: Optional[MutableDenseMatrix | list[MutableDenseMatrix]]=None, 
                 body_ref_config: list[MutableDenseMatrix]=[], 
                 joint_screw_coord: list[MutableDenseMatrix]=[], 
                 config_representation: str="spatial", 
                 Mb: list[MutableDenseMatrix]=[], 
                 parent: list[int]=[], 
                 support: list[list[int]]=[], 
                 child: list[list[int]]=[], 
                 ee_parent: Optional[int | list[int]]=None,
                 q: Optional[MutableDenseMatrix]=None, 
                 qd: Optional[MutableDenseMatrix]=None, 
                 q2d: Optional[MutableDenseMatrix]=None, 
                 q3d: Optional[MutableDenseMatrix]=None, 
                 q4d: Optional[MutableDenseMatrix]=None, 
                 WEE: MutableDenseMatrix | list[MutableDenseMatrix]=zeros(6, 1),
                 WDEE: MutableDenseMatrix | list[MutableDenseMatrix]=zeros(6, 1),
                 W2DEE: MutableDenseMatrix | list[MutableDenseMatrix]=zeros(6, 1),
                 **kwargs) -> None:
        """SymbolicKinDyn
        Symbolic tool to compute equations of motion of serial chain 
        robots and autogenerate code from the calculated equations. 
        This tool supports generation of python, C and Matlab code.

        Args:
            gravity_vector (sympy.Matrix, optional): 
                Vector of gravity. Defaults to None.
            ee (sympy.Matrix | list of sympy.Matrix, optional): 
                End-effector configuration with reference to last link 
                body fixed frame in the chain. This link can be selected 
                manually using the parameter ee_parent. 
                If there is more than one end-effector use a list of 
                transforms instead.
                Defaults to None.
            body_ref_config (list of sympy.Matrix, optional): 
                List of reference configurations of bodies in body-fixed
                or spatial representation, dependent on selected 
                config_representation. 
                Leave empty for dH Parameter usage (dhToScrewCoord(...)). 
                Defaults to [].
            joint_screw_coord (list of sympy.Matrix, optional): 
                List of joint screw coordinates in body-fixed 
                or spatial representation, dependent on selected 
                config_representation. 
                Leave empty for dH Parameter usage (dhToScrewCoord(...)). 
                Defaults to [].
            config_representation (str, optional): 
                Use body fixed or spatial representation for reference 
                configuration of bodies and joint screw coordinates.
                Has to be "body_fixed" or "spatial". 
                Defaults to "spatial".
            Mb (list of sympy.Matrix, optional): 
                List of Mass Inertia matrices for all links. Only 
                necessary for inverse dynamics. Defaults to [].
            parent (list, optional): 
                list of parent link indices for any joint. Use 0 for world.
                Only necessary for tree-like robot structures. 
                Defaults to [].
            support (list, optional): 
                list of lists with one list per link which includes all 
                support links beginning with the first link in the chain 
                and including the current link.
                Only necessary for tree-like robot structures. 
                Defaults to [].
            child (list, optional): 
                list of lists with one list per link which includes all
                child links. Use empty list if no child link is present.
                Only necessary for tree-like robot structures. 
                Defaults to [].
            ee_parent (int | list, optional): 
                parent link of the end effector frame. If there is more 
                than one end-effector, use a list of indices instead.
                Defaults to None (= last link).
            q (sympy.Matrix, optional): 
                (n,1) Generalized position vector. Defaults to None.
            qd (sympy.Matrix, optional): 
                (n,1) Generalized velocity vector. Defaults to None.
            q2d (sympy.Matrix, optional): 
                (n,1) Generalized acceleration vector. Defaults to None.
            q3d (sympy.Matrix, optional): 
                (n,1) Generalized jerk vector. Defaults to None.
            q4d (sympy.Matrix, optional): 
                (n,1) Generalized jounce vector. Defaults to None.
            WEE (sympy.Matrix | list, optional): 
                (6,1) WEE (t) = [Mx,My,Mz,Fx,Fy,Fz] is the time varying 
                wrench on the EE link. 
                If there is more than one end-effector, you can use a 
                list containing all wrenches instead.
                Defaults to zeros(6, 1).
            WDEE (sympy.Matrix | list, optional): 
                (6,1) WDEE (t) = [dMx,dMy,dMz,dFx,dFy,dFz] is the derivative 
                of the time varying wrench on the EE link. 
                If there is more than one end-effector, you can use a 
                list containing all wrenches instead.
                Defaults to zeros(6, 1).
            W2DEE (sympy.Matrix | list, optional): 
                (6,1) W2DEE (t) = [ddMx,ddMy,ddMz,ddFx,ddFy,ddFz] is the 
                2nd derivative of the time varying wrench on the EE link. 
                If there is more than one end-effector, you can use a 
                list containing all wrenches instead.
                Defaults to zeros(6, 1).
            
        """
        super().__init__()
        self.n = None  # degrees of freedom
        self.gravity_vector = gravity_vector
        self.ee = ee

        self.B = [] # List of reference configurations of bodies in body-fixed representation.
        self.X = [] # List of joint screw coordinates in body-fixed representation.

        self.A = [] # List of reference configurations of bodies in spatial representation.
        self.Y = [] # List of joint screw coordinates in spatial representation.
        
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
        
        # costom ee link
        self.ee_parent = ee_parent
        
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
        
        # generalized vectors end external forces
        self.q = q
        self.qd = qd
        self.q2d = q2d
        self.q3d = q3d
        self.q4d = q4d
        self.WEE = WEE
        self.WDEE = WDEE
        self.W2DEE = W2DEE

        
    @property
    def config_representation(self) -> str:
        return self._config_representation
    
    @config_representation.setter
    def config_representation(self, value: str) -> None:
        if value not in {self.BODY_FIXED, self.SPATIAL}:
            raise ValueError("config_representation has to be 'body_fixed' or 'spatial'")
        self._config_representation = value
    
    @property
    def body_ref_config(self) -> list:
        if self.config_representation == self.BODY_FIXED:
            return self.B
        elif self.config_representation == self.SPATIAL:
            return self.A
    
    @body_ref_config.setter
    def body_ref_config(self, value: list[MutableDenseMatrix]) -> None:
        n = len(value)
        if n:
            self.n = n
        if self.config_representation == self.BODY_FIXED:
            self.B = value
        elif self.config_representation == self.SPATIAL:
            self.A = value
    
    @property
    def joint_screw_coord(self) -> list:
        if self.config_representation == self.BODY_FIXED:
            return self.X
        elif self.config_representation == self.SPATIAL:
            return self.Y
    
    @joint_screw_coord.setter
    def joint_screw_coord(self, value: list[MutableDenseMatrix]) -> None:
        if self.config_representation == self.BODY_FIXED:
            self.X = value
        elif self.config_representation == self.SPATIAL:
            self.Y = value
    
    def closed_form_kinematics_body_fixed(
        self, 
        q: Optional[MutableDenseMatrix]=None, 
        qd: Optional[MutableDenseMatrix]=None, 
        q2d: Optional[MutableDenseMatrix]=None, 
        simplify: bool=True, 
        cse: bool=False, 
        parallel: bool=True) -> MutableDenseMatrix:
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

        Needs class parameters body_ref_config, joint_screw_coord and ee 
        to be defined.

        Args:
            q (sympy.Matrix, optional): 
                (n,1) Generalized position vector. Defaults to None.
            qd (sympy.Matrix, optional): 
                (n,1) Generalized velocity vector. Defaults to None.
            q2d (sympy.Matrix, optional): 
                (n,1) Generalized acceleration vector. Defaults to None.
            simplify (bool, optional): 
                Use simplify command on saved expressions. 
                Defaults to True.
            cse (bool, optional): 
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
            if not self.q or not self.qd or not self.q2d:
                print(self.q, self.qd, self.q2d)
                q, qd, q2d = generalized_vectors(self.n,self._find_start_index())
            else:
                q, qd, q2d = self.q, self.qd, self.q2d
                
        self._save_vectors(q, qd, q2d, None, None, ..., ..., ...)
        self.n = len(q)
        
        # prepare ee link selection
        if self.ee_parent is None:
            self.ee_parent = [self.n]
        if type(self.ee) is not list: self.ee = [self.ee]
        if type(self.ee_parent) is not list: self.ee_parent = [self.ee_parent]
        assert(len(self.ee) == len(self.ee_parent))    
        self.n_ee = len(self.ee_parent)
        
        if parallel:
            self._closed_form_kinematics_body_fixed_parallel(
                q, qd, q2d, simplify, cse)
        else:
            self._closed_form_kinematics_body_fixed(
                q, qd, q2d, simplify, cse)
        return self.fkin

    def closed_form_inv_dyn_body_fixed(
        self, 
        q: Optional[MutableDenseMatrix]=None, 
        qd: Optional[MutableDenseMatrix]=None, 
        q2d: Optional[MutableDenseMatrix]=None, 
        q3d: Optional[MutableDenseMatrix]=None, 
        q4d: Optional[MutableDenseMatrix]=None, 
        WEE: MutableDenseMatrix | list[MutableDenseMatrix]=..., 
        WDEE: MutableDenseMatrix | list[MutableDenseMatrix]=..., 
        W2DEE: MutableDenseMatrix | list[MutableDenseMatrix]=..., 
        simplify: bool=True, cse: bool=False, 
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
            q (sympy.Matrix, optional): 
                (n,1) Generalized position vector. Defaults to None.
            qd (sympy.Matrix, optional): 
                (n,1) Generalized velocity vector. Defaults to None.
            q2d (sympy.Matrix, optional): 
                (n,1) Generalized acceleration vector. Defaults to None.
            q3d (sympy.Matrix, optional): 
                (n,1) Generalized jerk vector. Defaults to None.
            q4d (sympy.Matrix, optional): 
                (n,1) Generalized jounce vector. Defaults to None.
            WEE (list | sympy.Matrix, optional): 
                (6,1) WEE (t) = [Mx,My,Mz,Fx,Fy,Fz] is the time varying 
                wrench on the EE link. 
                If there is more than one end-effector, you can use a 
                list containing all wrenches instead.
                Defaults to zeros(6, 1).
            WDEE (list | sympy.Matrix, optional): 
                (6,1) WDEE (t) = [dMx,dMy,dMz,dFx,dFy,dFz] is the derivative 
                of the time varying wrench on the EE link. 
                If there is more than one end-effector, you can use a 
                list containing all wrenches instead.
                Defaults to zeros(6, 1).
            W2DEE (list | sympy.Matrix, optional): 
                (6,1) W2DEE (t) = [ddMx,ddMy,ddMz,ddFx,ddFy,ddFz] is the 
                2nd derivative of the time varying wrench on the EE link. 
                If there is more than one end-effector, you can use a 
                list containing all wrenches instead.
                Defaults to zeros(6, 1).
            simplify (bool, optional): 
                Use simplify command on saved expressions. 
                Defaults to True.
            cse (bool, optional): 
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
            if not self.q or not self.qd or not self.q2d:
                q, qd, q2d = generalized_vectors(self.n,self._find_start_index())
            else:
                q, qd, q2d, q3d, q4d = self.q, self.qd, self.q2d, self.q3d, self.q4d
        
        if WEE is Ellipsis:
            WEE = self.WEE
        if WDEE is Ellipsis:
            WDEE = self.WDEE
        if W2DEE is Ellipsis:
            W2DEE = self.W2DEE
        
        self._save_vectors(q, qd, q2d, q3d, q4d, WEE, WDEE, W2DEE)
        self.n = len(q)
        
        # prepare ee link selection
        if self.ee_parent is None:
            self.ee_parent = [self.n]
        if type(self.ee_parent) is not list: self.ee_parent = [self.ee_parent]
        self.n_ee = len(self.ee_parent)
        if self.n_ee > 1:
            if type(WEE) is list and len(WEE) > 1:
                assert(len(WEE) == len(self.ee_parent))
            if type(WDEE) is list and len(WDEE) > 1:
                assert(len(WDEE) == len(self.ee_parent))
            if type(W2DEE) is list and len(W2DEE) > 1:
                assert(len(W2DEE) == len(self.ee_parent)) 
        if type(WEE) is not list: WEE = [WEE]
        if type(WDEE) is not list: WDEE = [WDEE]
        if type(W2DEE) is not list: W2DEE = [W2DEE]
        
        if parallel:
            self._closed_form_inv_dyn_body_fixed_parallel(
                q, qd, q2d, q3d, q4d, WEE, WDEE, W2DEE, simplify, cse)
        else:
            self._closed_form_inv_dyn_body_fixed(
                q, qd, q2d, q3d, q4d, WEE, WDEE, W2DEE, simplify, cse)
        return self.Q

    def _closed_form_kinematics_body_fixed(
        self, q: MutableDenseMatrix, qd: MutableDenseMatrix, 
        q2d: MutableDenseMatrix, simplify: bool=True, 
        cse: bool=False) -> MutableDenseMatrix:
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
            simplify (bool, optional): 
                Use simplify command on saved expressions. 
                Defaults to True.
            cse (bool, optional): 
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
        
        fkin = [None]*self.n_ee
        for i in range(self.n_ee):
            fkin[i] = FK_C[self.ee_parent[i]-1]*self.ee[i]
            fkin[i] = self.simplify(fkin[i], cse, simplify)    
        self.fkin = fkin if len(fkin)>1 else fkin[0]
        
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
            J = self.simplify(J, cse, simplify)
            self.J = J

            # System twist (6n x 1)
            V = J*qd
            self._V = V

        R_i = [None]*self.n_ee
        R_BFn = [None]*self.n_ee
        Vb_BFn = [None]*self.n_ee
        Vh_BFn = [None]*self.n_ee
        Vb_ee = [None]*self.n_ee
        Vh_ee = [None]*self.n_ee
        Jh_ee = [None]*self.n_ee
        Jb_ee = [None]*self.n_ee
        Jh = [None]*self.n_ee
        Jb = [None]*self.n_ee
        for i in range(self.n_ee):
            # Different Jacobians
            R_i[i] = Matrix(fkin[i][:3, :3]).row_join(
                zeros(3, 1)).col_join(Matrix([0, 0, 0, 1]).T)
            R_i[i] = self.simplify(R_i[i], cse, simplify)
            
            R_BFn[i] = Matrix(FK_C[self.ee_parent[i]-1][:3, :3]).row_join(
                zeros(3, 1)).col_join(Matrix([0, 0, 0, 1]).T)

            # Body fixed Jacobian of last moving body 
            # (This may not correspond to end-effector frame)
            Jb[i] = J[6*(self.ee_parent[i]-1):6*self.ee_parent[i], :] # used to be [-6:,:]
            Jb[i] = self.simplify(Jb[i], cse, simplify)
            
            Vb_BFn[i] = Jb[i]*qd  # Body fixed twist of last moving body
            Vb_BFn[i] = self.simplify(Vb_BFn[i], cse, simplify)
            
            Vh_BFn[i] = SE3AdjMatrix(R_BFn[i])*Vb_BFn[i]
            Vh_BFn[i] = self.simplify(Vh_BFn[i], cse, simplify)
            
            # Body fixed twist of end-effector frame
            Vb_ee[i] = SE3AdjMatrix(SE3Inv(self.ee[i]))*Vb_BFn[i]
            Vb_ee[i] = self.simplify(Vb_ee[i], cse, simplify)
            
            # Hybrid twist of end-effector frame
            Vh_ee[i] = SE3AdjMatrix(R_i[i])*Vb_ee[i]
            Vh_ee[i] = self.simplify(Vh_ee[i], cse, simplify)
            
            # Body fixed Jacobian of end-effector frame
            Jb_ee[i] = SE3AdjMatrix(SE3Inv(self.ee[i]))*Jb[i]
            Jb_ee[i] = self.simplify(Jb_ee[i], cse, simplify)
            
            # Hybrid Jacobian of end-effector frame
            Jh_ee[i] = SE3AdjMatrix(R_i[i])*Jb_ee[i]
            # Hybrid Jacobian of last moving body
            Jh[i] = SE3AdjMatrix(R_i[i])*Jb[i]  

            Jh_ee[i] = self.simplify(Jh_ee[i], cse, simplify)
            Jh[i] = self.simplify(Jh[i], cse, simplify)
            
        self.Vb_BFn = Vb_BFn if len(Vb_BFn)>1 else Vb_BFn[0]
        self.Vh_BFn = Vh_BFn if len(Vh_BFn)>1 else Vh_BFn[0]
        self.Vb_ee = Vb_ee if len(Vb_ee)>1 else Vb_ee[0]
        self.Vh_ee = Vh_ee if len(Vh_ee)>1 else Vh_ee[0]
        self.Jh_ee = Jh_ee if len(Jh_ee)>1 else Jh_ee[0]
        self.Jb_ee = Jb_ee if len(Jb_ee)>1 else Jb_ee[0]
        self.Jh = Jh if len(Jh)>1 else Jh[0]
        self.Jb = Jb if len(Jb)>1 else Jb[0]
        
        # Acceleration computations
        if self._a is not None:
            a = self._a
        else:
            # Block diagonal matrix a (6n x 6n)
            a = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                a[6*i:6*i+6, 6*i:6*i+6] = SE3adMatrix(self.X[i])*qd[i]
            a = self.simplify(a, cse, simplify)
            self._a = a

        # System acceleration (6n x 1)
        Jdot = -A*a*J  # Sys-level Jacobian time derivative
        Jdot = self.simplify(Jdot, cse, simplify)
        
        self.Jdot = Jdot

        Vbd = J*q2d - A*a*V

        Vbd_BFn = [None]*self.n_ee
        Vhd_BFn = [None]*self.n_ee
        Vbd_ee = [None]*self.n_ee
        Vhd_ee = [None]*self.n_ee
        Jb_dot = [None]*self.n_ee
        Jb_ee_dot = [None]*self.n_ee
        Jh_dot = [None]*self.n_ee
        Jh_ee_dot = [None]*self.n_ee
        for i in range(self.n_ee):
            # Hybrid acceleration of the last body
            Vbd_BFn[i] = Vbd[6*(self.ee_parent[i]-1):6*self.ee_parent[i], :] # used to be [-6:,:]
            Vbd_BFn[i] = self.simplify(Vbd_BFn[i], cse, simplify)
            
            # Hybrid twist of end-effector frame 
            # TODO: check comments
            Vhd_BFn[i] = (SE3AdjMatrix(R_BFn[i])*Vbd_BFn[i] 
                    + SE3adMatrix(Matrix(Vh_BFn[i][:3, :])
                                        .col_join(Matrix([0, 0, 0])))
                    * SE3AdjMatrix(R_BFn[i])*Vb_BFn[i])  

            Vhd_BFn[i] = self.simplify(Vhd_BFn[i], cse, simplify)
            
            # Body fixed twist of end-effector frame
            # Hybrid acceleration of the EE
            Vbd_ee[i] = SE3AdjMatrix(SE3Inv(self.ee[i]))*Vbd_BFn[i]
            Vbd_ee[i] = self.simplify(Vbd_ee[i], cse, simplify)
            
            # Hybrid twist of end-effector frame
            Vhd_ee[i] = SE3AdjMatrix(R_i[i])*Vbd_ee[i] + SE3adMatrix(Matrix(
                Vh_ee[i][:3, :]).col_join(Matrix([0, 0, 0])))*\
                    SE3AdjMatrix(R_i[i])*Vb_ee[i]  
            Vhd_ee[i] = self.simplify(Vhd_ee[i], cse, simplify)
            
            # Body Jacobian time derivative

            # For the last moving body
            Jb_dot[i] = Jdot[6*(self.ee_parent[i]-1):6*self.ee_parent[i], :] # used to be [-6:,:]

            # For the EE
            Jb_ee_dot[i] = SE3AdjMatrix(SE3Inv(self.ee[i]))*Jb_dot[i]
            Jb_ee_dot[i] = self.simplify(Jb_ee_dot[i], cse, simplify)
            
            # Hybrid Jacobian time derivative
            # For the last moving body
            Jh_dot[i] = (SE3AdjMatrix(R_BFn[i])*Jb_dot[i] 
                    + SE3adMatrix(Matrix(Vh_BFn[i][:3, :]).col_join(Matrix([0, 0, 0])))
                    *SE3AdjMatrix(R_BFn[i])*Jb[i])
            Jh_dot[i] = self.simplify(Jh_dot[i], cse, simplify)
            
            # For the EE
            Jh_ee_dot[i] = (SE3AdjMatrix(R_i[i])*Jb_ee_dot[i] 
                        + SE3adMatrix(Matrix(Vh_ee[i][:3, :]).col_join(Matrix([0, 0, 0])))
                        * SE3AdjMatrix(R_i[i])*Jb_ee[i])
            Jh_ee_dot[i] = self.simplify(Jh_ee_dot[i], cse, simplify)
            
        self.Vbd_BFn = Vbd_BFn if len(Vbd_BFn)>1 else Vbd_BFn[0]
        self.Vhd_BFn = Vhd_BFn if len(Vhd_BFn)>1 else Vhd_BFn[0]
        self.Vbd_ee = Vbd_ee if len(Vbd_ee)>1 else Vbd_ee[0]
        self.Vhd_ee = Vhd_ee if len(Vhd_ee)>1 else Vhd_ee[0]
        self.Jb_dot = Jb_dot if len(Jb_dot)>1 else Jb_dot[0]
        self.Jb_ee_dot = Jb_ee_dot if len(Jb_ee_dot)>1 else Jb_ee_dot[0]
        self.Jh_dot = Jh_dot if len(Jh_dot)>1 else Jh_dot[0]
        self.Jh_ee_dot = Jh_ee_dot if len(Jh_ee_dot)>1 else Jh_ee_dot[0]

        for e in self._get_expressions():
            self.all_symbols.update(e.free_symbols)

        print("Done")
        return fkin

    def _closed_form_inv_dyn_body_fixed(self, q: MutableDenseMatrix, 
                                        qd: MutableDenseMatrix, 
                                        q2d: MutableDenseMatrix,
                                        q3d: Optional[MutableDenseMatrix]=None, 
                                        q4d: Optional[MutableDenseMatrix]=None, 
                                        WEE: list[MutableDenseMatrix]=[zeros(6, 1)], 
                                        WDEE: list[MutableDenseMatrix]=[zeros(6, 1)], 
                                        W2DEE: list[MutableDenseMatrix]=[zeros(6, 1)], 
                                        simplify: bool=True, 
                                        cse: bool=False) -> MutableDenseMatrix:
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
            q3d (sympy.Matrix, optional): 
                (n,1) Generalized jerk vector. Defaults to None.
            q4d (sympy.Matrix, optional): 
                (n,1) Generalized jounce vector. Defaults to None.
            WEE (list of sympy.Matrix, optional): 
                (6,1) WEE (t) = [Mx,My,Mz,Fx,Fy,Fz] is the time varying 
                wrench on the EE link. 
                Defaults to [zeros(6, 1)].
            WDEE (list of sympy.Matrix, optional): 
                (6,1) WDEE (t) = [dMx,dMy,dMz,dFx,dFy,dFz] is the derivative 
                of the time varying wrench on the EE link. 
                Defaults to [zeros(6, 1)].
            W2DEE (list of sympy.Matrix, optional): 
                (6,1) W2DEE (t) = [ddMx,ddMy,ddMz,ddFx,ddFy,ddFz] is the 
                2nd derivative of the time varying wrench on the EE link. 
                Defaults to [zeros(6, 1)].
            simplify (bool, optional): Use simplify command 
                on saved expressions. Defaults to True.
            cse (bool, optional): Use common subexpression 
                elimination. Defaults to False.

        Returns:
            sympy.Matrix: Generalized forces
        """
        print("Inverse dynamics calculation")

        self.var_syms.update(q.free_symbols)
        self.var_syms.update(qd.free_symbols)
        self.var_syms.update(q2d.free_symbols)
        if q3d: self.var_syms.update(q3d.free_symbols)
        if q4d: self.var_syms.update(q4d.free_symbols)
        for W in WEE:
            self.optional_var_syms.update(W.free_symbols)
        for WD in WDEE:
            self.optional_var_syms.update(WD.free_symbols)
        for W2D in W2DEE:
            self.optional_var_syms.update(W2D.free_symbols)

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
            J = self.simplify(J, cse, simplify)
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
        M = self.simplify(M, cse, simplify)
        
        # Coriolis-Centrifugal matrix in joint space (n x n)
        C = J.T * Cb * J
        C = self.simplify(C, cse, simplify)
        
        # Gravity Term
        U = SE3AdjInvMatrix(FK_C[0])
        for k in range(1, self.n):
            U = U.col_join(SE3AdjInvMatrix(FK_C[k]))

        Vd_0 = zeros(6, 1)
        Vd_0[3:6, 0] = -Matrix(self.gravity_vector)
        Qgrav = J.T*Mb*U*Vd_0
        Qgrav = self.simplify(Qgrav, cse, simplify)
        
        # External Wrench
        Wext = zeros(6*self.n, 1)
        # WEE (t) is the time varying wrench on the EE link.
        for i in range(self.n_ee):
            Wext[6*(self.ee_parent[i]-1):6*self.ee_parent[i], 0] = WEE[i if len(WEE) == self.n_ee else 0]
        Qext = J.T * Wext

        # Generalized forces Q
        # Q = M*q2d + C*qd   # without gravity
        Q = M*q2d + C*qd + Qgrav + Qext
        Q = self.simplify(Q, cse, simplify)
        
        self.M = M
        self.C = C
        self.Q = Q
        self.Qgrav = Qgrav
        
        # Regressor matrix for standard inertial parameter identification, Q = Y*X in SE(3)
        # mass-inertia parameter vector X         
        # m, m*c, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz        
        Xr = Matrix([value for i in self.Mb for value in mass_matrix_to_parameter_vector(i)])
        # dont calculate with numeric values
        if not any([i.is_Float for i in Xr]):
            S = Matrix(Identity(len(Xr)))
            # Regressor matrix Y   
            Yr = zeros(self.n, len(Xr))
            for i in range(self.n):
                for j in range(len(Xr)):                        
                    Yr[i,j] = self.collect_and_subs(Q[i], {k: v for k, v in zip(Xr, S[j,:]) if k not in {0,1}})
            Yr = self.simplify(Yr, cse, simplify)
            
            self.Xr = Xr
            self.Yr = Yr

        ##### First Order Derivatives of EOM #####
        if q3d is not None:        
            # First time derivative of Block diagonal matrix a (6n x 6n)
            ad = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                ad[i*6:i*6+6, i*6:i*6+6] = q2d[i] * SE3adMatrix(self.X[i])

            # Third order Forward Kinematics
            V2d = J*q3d - A*ad*V - 2*A*a*Vd - A*a*a*V
                        
            #  First time derivative of Block diagonal matrix b (6n x 6n) 
            # used in Coriolis matrix
            bd = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                bd[i*6:i*6+6, i*6:i*6+6] = SE3adMatrix(Matrix(Vd[i*6:i*6+6]))
            
            # First time derivative of Mass inertia matrix in joint space (n x n)
            Mbd = -Mb*A*a - (Mb*A*a).T
            Md = J.T * Mbd * J
            Md = self.simplify(Md, cse, simplify)
            
            # First time derivative of Coriolis-Centrifugal matrix in joint space (n x n)
            Cbd = Mb*A*a*A*a - Mb*A*a*a - Mb*A*ad - bd.T * Mb - Cb*A*a - a.T*A.T*Cb
            Cd = J.T*Cbd*J
            Cd = self.simplify(Cd, cse, simplify)
            
            # First time derivative of gravity force
            Qdgrav = J.T*Mbd*U*Vd_0
            Qdgrav = self.simplify(Qdgrav, cse, simplify)
            
            # First time derivative of External Wrench
            Wdext = zeros(6*self.n,1)
            for i in range(self.n_ee):
                Wdext[6*(self.ee_parent[i]-1):6*self.ee_parent[i], 0] = WDEE[i if len(WDEE) == self.n_ee else 0]
            Qdext = J.T*(Wdext - (A*a).T * Wext)
            
            # Qd = M*q3d + (Md + C)*q2d + Cd*qd  # without gravity 
            Qd = M*q3d + (Md + C)*q2d + Cd*qd + Qdgrav + Qdext # with gravity
            Qd = self.simplify(Qd, cse, simplify)
            
            self.Md = Md
            self.Cd = Cd
            self.Qdgrav = Qdgrav
            self.Qd = Qd

        ##### Second Order Derivatives of EOM #####
        if q3d is not None and q4d is not None:        
            # Second time derivative of Block diagonal matrix a (6n x 6n)
            a2d = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                a2d[i*6:i*6+6, i*6:i*6+6] = SE3adMatrix(self.X[i]) * q3d[i]
                
            # Second time derivative of Block diagonal matrix b (6n x 6n) 
            # used in Coriolis matrix
            b2d = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                b2d[i*6:i*6+6, i*6:i*6+6] = SE3adMatrix(Matrix(V2d[i*6:i*6+6]))
                
            # Second time derivative of Mass inertia matrix in joint space (n x n)
            Mb2d = (- Mb*A*ad - (Mb*A*ad).T + 2*Mb*A*a*A*a + 2*(Mb*A*a*A*a).T 
                    + 2*a.T*A.T*Mb*A*a - Mb*A*a*a - (Mb*A*a*a).T)
            M2d = J.T*Mb2d*J
            M2d = self.simplify(M2d, cse, simplify)
            
            # Second time derivative of Coriolis-Centrifugal matrix in joint space (n x n)
            Cddot = (- Mb*A*a2d - 3*Mb*A*a*ad - Mb*A*a*a*a - b2d.T*Mb 
                     + Mb*A*ad*A*a + Mb*A*a*a*A*a + 2*Mb*A*a*A*ad 
                     + 2*Mb*A*a*A*a*a - 2*Mb*A*a*A*a*A*a)
            Cb2d = (Cddot - (Cbd + a.T*A.T*Cb)*A*a - a.T*A.T*(Cbd + Cb*A*a) 
                    - Cb*A*ad - ad.T*A.T*Cb - Cb*A*a*a - a.T*a.T*A.T*Cb 
                    - Cbd*A*a - a.T*A.T*Cbd)
            C2d = J.T*Cb2d*J
            C2d = self.simplify(C2d, cse, simplify)
            
            # Second time derivative of gravity force
            Q2dgrav = J.T*Mb2d*U*Vd_0
            Q2dgrav = self.simplify(Q2dgrav, cse, simplify)
            
            # Second time derivative of External Wrench
            W2dext = zeros(6*self.n,1)
            for i in range(self.n_ee):
                W2dext[6*(self.ee_parent[i]-1):6*self.ee_parent[i], 0] = W2DEE[i if len(W2DEE) == self.n_ee else 0]
            Q2dext = J.T*(W2dext - 2*(A*a).T*Wdext + (2*(A*a*A*a).T - (A*ad).T - (A*a*a).T)*Wext)
            
            # Second time derivative of generalized forces
            # without gravity:
            # Q2d = M*q4d + (2*Md + C)*q3d + (M2d + 2*Cd)*q2d + C2d*qd   
            # with gravity and external forces:
            Q2d = M*q4d + (2*Md + C)*q3d + (M2d + 2*Cd)*q2d + C2d*qd + Q2dgrav + Q2dext
            Q2d = self.simplify(Q2d, cse, simplify)
            
            self.M2d = M2d
            self.C2d = C2d
            self.Q2dgrav = Q2dgrav
            self.Q2d = Q2d

        # save used symbols
        for e in self._get_expressions():
            self.all_symbols.update(e.free_symbols)

        print("Done")
        return Q

    def _closed_form_kinematics_body_fixed_parallel(
        self, q: MutableDenseMatrix, qd: MutableDenseMatrix, 
        q2d: MutableDenseMatrix, simplify: bool=True, 
        cse: bool=False) -> MutableDenseMatrix:
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
            simplify (bool, optional): 
                Use simplify command on saved expressions. 
                Defaults to True.
            cse (bool, optional): 
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
        
        for i in range(self.n_ee):
            self._set_value(f"fkin{i}", FK_C[self.ee_parent[i]-1]*self.ee[i])
            self._start_simplification_process(f"fkin{i}", cse, simplify)
            
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
            self._start_simplification_process("J", cse, simplify)
            # System twist (6n x 1)
            self._set_value_as_process("V", lambda: self._get_value("J")*qd)

        for i in range(self.n_ee):
            # Different Jacobians
            self._set_value_as_process(
                f"R_i{i}", 
                lambda: 
                    Matrix(self._get_value(f"fkin{i}")[:3, :3])
                    .row_join(zeros(3, 1))
                    .col_join(Matrix([0, 0, 0, 1]).T)
                )

            self._start_simplification_process(f"R_i{i}", cse, simplify)
            
            self._set_value(f"R_BFn{i}", Matrix(FK_C[self.ee_parent[i]-1][:3, :3]).row_join(
                zeros(3, 1)).col_join(Matrix([0, 0, 0, 1]).T)) 

            # Body fixed Jacobian of last moving body 
            # (This may not correspond to end-effector frame)
            self._set_value_as_process(
                f"Jb{i}", 
                lambda: self._get_value("J")[6*(self.ee_parent[i]-1):6*self.ee_parent[i], :])
            self._start_simplification_process(f"Jb{i}", cse, simplify)
            
            self._set_value_as_process(f"Vb_BFn{i}", lambda: self._get_value(f"Jb{i}")*qd)
            # Body fixed twist of last moving body
            self._start_simplification_process(f"Vb_BFn{i}", cse, simplify)
            
            self._set_value_as_process(f"Vh_BFn{i}", lambda: SE3AdjMatrix(
                self._get_value(f"R_BFn{i}"))*self._get_value(f"Vb_BFn{i}"))
            self._start_simplification_process(f"Vh_BFn{i}", cse, simplify)
            
            # Body fixed twist of end-effector frame
            self._set_value_as_process(f"Vb_ee{i}", lambda: SE3AdjMatrix(
                SE3Inv(self.ee[i]))*self._get_value(f"Vb_BFn{i}"))
            self._start_simplification_process(f"Vb_ee{i}", cse, simplify)
            
            # Hybrid twist of end-effector frame
            self._set_value_as_process(f"Vh_ee{i}", lambda: SE3AdjMatrix(
                self._get_value(f"R_i{i}"))*self._get_value(f"Vb_ee{i}"))
            self._start_simplification_process(f"Vh_ee{i}", cse, simplify)
            
            # Body fixed Jacobian of end-effector frame
            self._set_value_as_process(f"Jb_ee{i}", lambda: SE3AdjMatrix(
                SE3Inv(self.ee[i]))*self._get_value(f"Jb{i}"))
            self._start_simplification_process(f"Jb_ee{i}", cse, simplify)
            
            # Hybrid Jacobian of end-effector frame
            self._set_value_as_process(f"Jh_ee{i}", lambda: SE3AdjMatrix(
                self._get_value(f"R_i{i}"))*self._get_value(f"Jb_ee{i}"))
            # Hybrid Jacobian of last moving body
            self._set_value_as_process(f"Jh{i}", lambda: SE3AdjMatrix(
                self._get_value(f"R_i{i}"))*self._get_value(f"Jb{i}"))

            self._start_simplification_process(f"Jh_ee{i}", cse, simplify)
            self._start_simplification_process(f"Jh{i}", cse, simplify)
            
        # Acceleration computations
        if self._a is not None:
            self._set_value("a", self._a)
        else:
            # Block diagonal matrix a (6n x 6n)
            a = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                a[6*i:6*i+6, 6*i:6*i+6] = SE3adMatrix(self.X[i])*qd[i]
            self._set_value("a", a)
            self._start_simplification_process("a", cse, simplify)
            
        # System acceleration (6n x 1)
        # System-level Jacobian time derivative
        self._set_value_as_process(
            "Jdot", lambda: -A*self._get_value("a")*self._get_value("J"))
        self._start_simplification_process("Jdot", cse, simplify)
        
        self._set_value_as_process("Vbd", lambda: self._get_value(
            "J")*q2d - A*self._get_value("a")*self._get_value("V"))

        for i in range(self.n_ee):
            # Hybrid acceleration of the last body
            self._set_value_as_process(
                f"Vbd_BFn{i}", lambda: self._get_value("Vbd")[6*(self.ee_parent[i]-1):6*self.ee_parent[i], :])

            self._start_simplification_process(f"Vbd_BFn{i}", cse, simplify)
            
            # Hybrid twist of end-effector frame
            self._set_value_as_process(
                f"Vhd_BFn{i}", 
                lambda: 
                    SE3AdjMatrix(self._get_value(f"R_BFn{i}"))
                    * self._get_value(f"Vbd_BFn{i}") 
                    + SE3adMatrix(Matrix(self._get_value(f"Vh_BFn{i}")[:3, :])
                                    .col_join(Matrix([0, 0, 0])))
                    * SE3AdjMatrix(self._get_value(f"R_BFn{i}"))
                    * self._get_value(f"Vb_BFn{i}")
                )

            self._start_simplification_process(f"Vhd_BFn{i}", cse, simplify)
            
            # Body fixed twist of end-effector frame
            # Hybrid acceleration of the EE
            self._set_value_as_process(f"Vbd_ee{i}", lambda: SE3AdjMatrix(
                SE3Inv(self.ee[i]))*self._get_value(f"Vbd_BFn{i}"))
            self._start_simplification_process(f"Vbd_ee{i}", cse, simplify)
            
            # Hybrid twist of end-effector frame
            self._set_value_as_process(
                f"Vhd_ee{i}", 
                lambda: 
                    SE3AdjMatrix(self._get_value(f"R_i{i}")) 
                    * self._get_value(f"Vbd_ee{i}") 
                    + SE3adMatrix(Matrix(self._get_value(f"Vh_ee{i}")[:3, :])
                                    .col_join(Matrix([0, 0, 0])))
                    * SE3AdjMatrix(self._get_value(f"R_i{i}"))
                    * self._get_value(f"Vb_ee{i}")
                )  # Hybrid twist of end-effector frame

            self._start_simplification_process(f"Vhd_ee{i}", cse, simplify)
            
            # Body Jacobian time derivative

            # For the last moving body
            self._set_value_as_process(
                f"Jb_dot{i}", lambda: self._get_value("Jdot")[6*(self.ee_parent[i]-1):6*self.ee_parent[i], :])

            # For the EE
            self._set_value_as_process(f"Jb_ee_dot{i}", lambda: SE3AdjMatrix(
                SE3Inv(self.ee[i]))*self._get_value(f"Jb_dot{i}"))
            self._start_simplification_process(f"Jb_ee_dot{i}", cse, simplify)
            
            # Hybrid Jacobian time derivative
            # For the last moving body
            self._set_value_as_process(
                f"Jh_dot{i}", 
                lambda: 
                    SE3AdjMatrix(self._get_value(f"R_BFn{i}"))
                    * self._get_value(f"Jb_dot{i}") 
                    + SE3adMatrix(Matrix(self._get_value(f"Vh_BFn{i}")[:3, :])
                                    .col_join(Matrix([0, 0, 0])))
                    * SE3AdjMatrix(self._get_value(f"R_BFn{i}"))
                    * self._get_value(f"Jb{i}")
                )
            self._start_simplification_process(f"Jh_dot{i}", cse, simplify)
            
            # For the EE
            self._set_value_as_process(
                f"Jh_ee_dot{i}", 
                lambda: 
                    SE3AdjMatrix(self._get_value(f"R_i{i}"))
                    * self._get_value(f"Jb_ee_dot{i}") 
                    + SE3adMatrix(Matrix(self._get_value(f"Vh_ee{i}")[:3, :])
                                    .col_join(Matrix([0, 0, 0])))
                    * SE3AdjMatrix(self._get_value(f"R_i{i}"))
                    * self._get_value(f"Jb_ee{i}")
                )
            self._start_simplification_process(f"Jh_ee_dot{i}", cse, simplify)
            
        self._a = self._get_value("a")
        self._V = self._get_value("V")

        # variables for Code Generation:
        self.J = self._get_value("J")
        self.Jdot = self._get_value("Jdot")
        self.fkin = [self._get_value(f"fkin{i}") for i in range(self.n_ee)]
        self.Jb = [self._get_value(f"Jb{i}") for i in range(self.n_ee)]
        self.Jh = [self._get_value(f"Jh{i}") for i in range(self.n_ee)]
        self.Vb_ee = [self._get_value(f"Vb_ee{i}") for i in range(self.n_ee)]
        self.Vh_ee = [self._get_value(f"Vh_ee{i}") for i in range(self.n_ee)]
        self.Jb_ee = [self._get_value(f"Jb_ee{i}") for i in range(self.n_ee)]
        self.Jh_ee = [self._get_value(f"Jh_ee{i}") for i in range(self.n_ee)]
        self.Vh_BFn = [self._get_value(f"Vh_BFn{i}") for i in range(self.n_ee)]
        self.Vb_BFn = [self._get_value(f"Vb_BFn{i}") for i in range(self.n_ee)]
        self.Vhd_BFn = [self._get_value(f"Vhd_BFn{i}") for i in range(self.n_ee)]
        self.Vbd_BFn = [self._get_value(f"Vbd_BFn{i}") for i in range(self.n_ee)]
        self.Vhd_ee = [self._get_value(f"Vhd_ee{i}") for i in range(self.n_ee)]
        self.Vbd_ee = [self._get_value(f"Vbd_ee{i}") for i in range(self.n_ee)]
        self.Jh_dot = [self._get_value(f"Jh_dot{i}") for i in range(self.n_ee)]
        self.Jb_dot = [self._get_value(f"Jb_dot{i}") for i in range(self.n_ee)]
        self.Jh_ee_dot = [self._get_value(f"Jh_ee_dot{i}") for i in range(self.n_ee)]
        self.Jb_ee_dot = [self._get_value(f"Jb_ee_dot{i}") for i in range(self.n_ee)]

        if self.n_ee == 1:
            self.fkin = self.fkin[0] 
            self.Jb = self.Jb[0] 
            self.Jh = self.Jh[0] 
            self.Vb_ee = self.Vb_ee[0] 
            self.Vh_ee = self.Vh_ee[0] 
            self.Jb_ee = self.Jb_ee[0] 
            self.Jh_ee = self.Jh_ee[0] 
            self.Vh_BFn = self.Vh_BFn[0] 
            self.Vb_BFn = self.Vb_BFn[0] 
            self.Vhd_BFn = self.Vhd_BFn[0] 
            self.Vbd_BFn = self.Vbd_BFn[0] 
            self.Vhd_ee = self.Vhd_ee[0] 
            self.Vbd_ee = self.Vbd_ee[0] 
            self.Jh_dot = self.Jh_dot[0] 
            self.Jb_dot = self.Jb_dot[0] 
            self.Jh_ee_dot = self.Jh_ee_dot[0] 
            self.Jb_ee_dot = self.Jb_ee_dot[0] 
        
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
        self, 
        q: MutableDenseMatrix, 
        qd: MutableDenseMatrix, 
        q2d: MutableDenseMatrix, 
        q3d: Optional[MutableDenseMatrix]=None, 
        q4d: Optional[MutableDenseMatrix]=None, 
        WEE: list[MutableDenseMatrix]=[zeros(6, 1)], 
        WDEE: list[MutableDenseMatrix]=[zeros(6, 1)], 
        W2DEE: list[MutableDenseMatrix]=[zeros(6, 1)], 
        simplify: bool=True, cse: bool=False) -> MutableDenseMatrix:
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
            q3d (sympy.Matrix, optional): 
                (n,1) Generalized jerk vector. Defaults to None.
            q4d (sympy.Matrix, optional): 
                (n,1) Generalized jounce vector. Defaults to None.
            WEE (list of sympy.Matrix, optional): 
                (6,1) WEE (t) = [Mx,My,Mz,Fx,Fy,Fz] is the time varying 
                wrench on the EE link. 
                Defaults to [zeros(6, 1)].
            WDEE (list of sympy.Matrix, optional): 
                (6,1) WDEE (t) = [dMx,dMy,dMz,dFx,dFy,dFz] is the derivative 
                of the time varying wrench on the EE link. 
                Defaults to [zeros(6, 1)].
            W2DEE (list of sympy.Matrix, optional): 
                (6,1) W2DEE (t) = [ddMx,ddMy,ddMz,ddFx,ddFy,ddFz] is the 
                2nd derivative of the time varying wrench on the EE link. 
                Defaults to [zeros(6, 1)].
            simplify (bool, optional): 
                Use simplify command on saved expressions. 
                Defaults to True.
            cse (bool, optional): 
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
        if q3d: self.var_syms.update(q3d.free_symbols)
        if q4d: self.var_syms.update(q4d.free_symbols)
        for W in WEE:
            self.optional_var_syms.update(W.free_symbols)
        for WD in WDEE:
            self.optional_var_syms.update(WD.free_symbols)
        for W2D in W2DEE:
            self.optional_var_syms.update(W2D.free_symbols)
        
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
            self._start_simplification_process("J", cse, simplify)
            
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
        def _b() -> MutableDenseMatrix:
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
        self._start_simplification_process("M", cse, simplify)
        
        # Coriolis-Centrifugal matrix in joint space (n x n)
        self._set_value_as_process("C", lambda: self._get_value(
            "J").T*self._get_value("Cb")*self._get_value("J"))
        self._start_simplification_process("C", cse, simplify)
        
        # Gravity Term
        U = SE3AdjInvMatrix(FK_C[0])
        for k in range(1, self.n):
            U = U.col_join(SE3AdjInvMatrix(FK_C[k]))

        Vd_0 = zeros(6, 1)
        Vd_0[3:6, 0] = -Matrix(self.gravity_vector)
        self._set_value_as_process(
            "Qgrav", lambda: self._get_value("J").T*Mb*U*Vd_0)
        self._start_simplification_process("Qgrav", cse, simplify)
        
        # External Wrench
        Wext = zeros(6*self.n, 1)
        # WEE (t) is the time varying wrench on the EE link.
        for i in range(self.n_ee):
                Wext[6*(self.ee_parent[i]-1):6*self.ee_parent[i], 0] = WEE[i if len(WEE) == self.n_ee else 0]
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

        self._start_simplification_process("Q", cse, simplify)
        
        # Regressor matrix for standard inertial parameter identification, Q = Y*X in SE(3)
        # mass-inertia parameter vector X         
        # m, m*c, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz        
        Xr = Matrix([value for i in self.Mb for value in mass_matrix_to_parameter_vector(i)])
        
        if not any([i.is_Float for i in Xr]):
        
        
            def _Yr() -> MutableDenseMatrix:
                nonlocal self
                S = Matrix(Identity(len(Xr)))
                
                # Regressor matrix Y   
                Yr = zeros(self.n, len(Xr))
                for i in range(self.n):
                    for j in range(len(Xr)):
                        Yr[i,j] = self.collect_and_subs(self._get_value("Q")[i], {k: v for k, v in zip(Xr, S[j,:]) if k not in {0,1}})
                return Yr
            
            self._set_value_as_process("Yr", _Yr)
            self._start_simplification_process("Yr", cse, simplify)
        
            self.Xr = Xr

        
        ##### First Order Derivatives of EOM #####
        if q3d is not None:
            # First time derivative of Block diagonal matrix a (6n x 6n)
            ad = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                ad[i*6:i*6+6, i*6:i*6+6] = q2d[i] * SE3adMatrix(self.X[i])

            # Third order Forward Kinematics
            self._set_value_as_process(
                "V2d", 
                lambda: self._get_value("J")*q3d - A*ad*self._get_value("V")
                - 2*A*a*self._get_value("Vd") - A*a*a*self._get_value("V"))
                        
            #  First time derivative of Block diagonal matrix b (6n x 6n) 
            # used in Coriolis matrix
            def _bd() -> MutableDenseMatrix:
                nonlocal self
                bd = zeros(6*self.n, 6*self.n)
                for i in range(self.n):
                    bd[i*6:i*6+6, i*6:i*6+6] = SE3adMatrix(Matrix(self._get_value("Vd")[i*6:i*6+6]))
                return bd
            self._set_value_as_process("bd", _bd)
            
            # First time derivative of Mass inertia matrix in joint space (n x n)
            Mbd = -Mb*A*a - (Mb*A*a).T
            self._set_value_as_process(
                "Md",
                lambda: self._get_value("J").T * Mbd * self._get_value("J"))
            self._start_simplification_process("Md", cse, simplify)
            
            # First time derivative of Coriolis-Centrifugal matrix in joint space (n x n)
            self._set_value_as_process(
                "Cbd",
                lambda: Mb*A*a*A*a - Mb*A*a*a - Mb*A*ad - self._get_value("bd").T * Mb 
                        - self._get_value("Cb")*A*a - a.T*A.T*self._get_value("Cb"))
            self._set_value_as_process(
                "Cd", 
                lambda: self._get_value("J").T*self._get_value("Cbd")*self._get_value("J"))
            self._start_simplification_process("Cd", cse, simplify)
            
            # First time derivative of gravity force
            self._set_value_as_process(
                "Qdgrav",
                lambda: self._get_value("J").T*Mbd*U*Vd_0)
            self._start_simplification_process("Qdgrav", cse, simplify)
            
            # First time derivative of External Wrench
            Wdext = zeros(6*self.n,1)
            for i in range(self.n_ee):
                Wdext[6*(self.ee_parent[i]-1):6*self.ee_parent[i], 0] = WDEE[i if len(WDEE) == self.n_ee else 0]
            self._set_value_as_process(
                "Qdext",
                lambda: self._get_value("J").T*(Wdext - (A*a).T * Wext))
            
            self._set_value_as_process(
                "Qd",
                lambda: self._get_value("M")*q3d 
                        + (self._get_value("Md") + self._get_value("C"))*q2d 
                        + self._get_value("Cd")*qd + self._get_value("Qdgrav") 
                        + self._get_value("Qdext")
            )
            self._start_simplification_process("Qd", cse, simplify)
            
        ##### Second Order Derivatives of EOM #####
        if q3d is not None and q4d is not None:        
            # Second time derivative of Block diagonal matrix a (6n x 6n)
            a2d = zeros(6*self.n, 6*self.n)
            for i in range(self.n):
                a2d[i*6:i*6+6, i*6:i*6+6] = SE3adMatrix(self.X[i]) * q3d[i]
                
            # Second time derivative of Block diagonal matrix b (6n x 6n) 
            # used in Coriolis matrix
            def _b2d() -> MutableDenseMatrix:
                nonlocal self
                b2d = zeros(6*self.n, 6*self.n)
                for i in range(self.n):
                    b2d[i*6:i*6+6, i*6:i*6+6] = SE3adMatrix(
                        Matrix(self._get_value("V2d")[i*6:i*6+6]))
                return b2d
            self._set_value_as_process("b2d", _b2d)
                
            # Second time derivative of Mass inertia matrix in joint space (n x n)
            Mb2d = (- Mb*A*ad - (Mb*A*ad).T + 2*Mb*A*a*A*a + 2*(Mb*A*a*A*a).T 
                    + 2*a.T*A.T*Mb*A*a - Mb*A*a*a - (Mb*A*a*a).T)
            self._set_value_as_process(
                "M2d",
                lambda: self._get_value("J").T*Mb2d*self._get_value("J"))
            self._start_simplification_process("M2d", cse, simplify)
            
            # Second time derivative of Coriolis-Centrifugal matrix in joint space (n x n)
            self._set_value_as_process(
                "Cddot",
                lambda: - Mb*A*a2d - 3*Mb*A*a*ad - Mb*A*a*a*a 
                        - self._get_value("b2d").T*Mb 
                        + Mb*A*ad*A*a + Mb*A*a*a*A*a + 2*Mb*A*a*A*ad 
                        + 2*Mb*A*a*A*a*a - 2*Mb*A*a*A*a*A*a
            )
            self._set_value_as_process(
                "Cb2d",
                lambda: self._get_value("Cddot") 
                        - (self._get_value("Cbd") + a.T*A.T*self._get_value("Cb"))*A*a 
                        - a.T*A.T*(self._get_value("Cbd") + self._get_value("Cb")*A*a) 
                        - self._get_value("Cb")*A*ad - ad.T*A.T*self._get_value("Cb") 
                        - self._get_value("Cb")*A*a*a - a.T*a.T*A.T*self._get_value("Cb") 
                        - self._get_value("Cbd")*A*a - a.T*A.T*self._get_value("Cbd")
            )
            self._set_value_as_process(
                "C2d",
                lambda: self._get_value("J").T*self._get_value("Cb2d")*self._get_value("J")
            )
            self._start_simplification_process("C2d", cse, simplify)
            
            # Second time derivative of gravity force
            self._set_value_as_process(
                "Q2dgrav",
                lambda: self._get_value("J").T*Mb2d*U*Vd_0
            )
            self._start_simplification_process("Q2dgrav", cse, simplify)
            
            # Second time derivative of External Wrench
            W2dext = zeros(6*self.n,1)
            for i in range(self.n_ee):
                W2dext[6*(self.ee_parent[i]-1):6*self.ee_parent[i], 0] = W2DEE[i if len(W2DEE) == self.n_ee else 0]
            self._set_value_as_process(
                "Q2dext",
                lambda: self._get_value("J").T
                        * (W2dext - 2*(A*a).T*Wdext + (2*(A*a*A*a).T - (A*ad).T - (A*a*a).T)*Wext) 
            )
            
            # Second time derivative of generalized forces
            # with gravity and external forces:
            self._set_value_as_process(
                "Q2d",
                lambda: self._get_value("M")*q4d 
                        + (2*self._get_value("Md") + self._get_value("C"))*q3d 
                        + (self._get_value("M2d") + 2*self._get_value("Cd"))*q2d 
                        + self._get_value("C2d")*qd 
                        + self._get_value("Q2dgrav") 
                        + self._get_value("Q2dext") 
            )
            self._start_simplification_process("Q2d", cse, simplify)
            

        self._V = self._get_value("V")
        self.J = self._get_value("J")
        self.M = self._get_value("M")
        self.C = self._get_value("C")
        self.Qgrav = self._get_value("Qgrav")
        self.Q = self._get_value("Q")
        if self.Xr is not None:
            self.Yr = self._get_value("Yr")    
        if q3d is not None:
            self.Md = self._get_value("Md")
            self.Cd = self._get_value("Cd")
            self.Qdgrav = self._get_value("Qdgrav")
            self.Qd = self._get_value("Qd")
        if q3d is not None and q4d is not None:
            self.M2d = self._get_value("M2d")
            self.C2d = self._get_value("C2d")
            self.Q2dgrav = self._get_value("Q2dgrav")
            self.Q2d = self._get_value("Q2d")

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

    def partial_factor(self, exp: sympy.Expr) -> sympy.Expr:
        """Partial factor simplification for sympy expression.

        Args:
            exp (sympy.Expr): sympy expression.

        Returns:
            sympy.Expr: modified sympy expression.
        """
        # split up matrices
        if (type(exp) == sympy.matrices.immutable.ImmutableDenseMatrix
            or type(exp) == sympy.matrices.dense.MutableDenseMatrix):
            new_expr = zeros(*exp.shape)
            for i in range(exp.shape[0]):
                for j in range(exp.shape[1]):
                    new_expr[i,j] = self.partial_factor(exp[i,j])
            return new_expr
            
        # seach for factors
        factor_map = defaultdict(set)
        const, additive_terms = exp.as_coeff_add()
        for term1, term2 in combinations(additive_terms, 2):
            common_terms = (
                set(term1.as_coeff_mul()[-1])
                & set(term2.as_coeff_mul()[-1])
            )
            if common_terms:
                common_factor = sympy.Mul(*common_terms)
                factor_map[common_factor] |= {term1, term2}
        
        # sort by number of operations represented by the terms
        factor_list = sorted(
            factor_map.items(),
            key = lambda i: (i[0].count_ops() + 1) * len(i[1])
        )[::-1]

        # rebuild expression
        used = set()
        new_expr = nsimplify(0)
        for item in factor_list:
            factor = item[0]
            appearances = item[-1]
            terms = 0
            for instance in appearances:
                if instance not in used:
                    terms += instance.as_coefficient(factor)
                    used.add(instance)
            new_expr += factor * terms
        for term in set(additive_terms) - used:
            new_expr += term
        return new_expr + const

    def collect_and_subs(self, ex: sympy.Expr, terms: dict) -> sympy.Expr:
        for key in terms:
            if key.is_Atom:
                ex = ex.subs(key,terms[key])
            else:
                ex = ex.expand().collect(key).subs(key,terms[key])
        return ex
    
    def simplify(self, exp: sympy.Expr, cse: bool=False, simplify: bool=True) -> sympy.Expr:
        """Faster simplify implementation for sympy expressions.
        Expressions can be different simplified as with sympy.simplify.

        Args:
            exp (sympy expression): 
                Expression to simplify.
            cse (bool, optional): 
                Use common subexpression elimination. Defaults to False.
            simplify (bool, optional): 
                deactivate simplification by setting simplify to False. 
                Defaults to True.

        Returns:
            sympy expression: Simplified expression.
        """
        if cse:
            exp = self._cse_expression(exp)
        if simplify:
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
            exp = self.partial_factor(exp)
            exp = exp.doit()
        return exp

    def _create_topology_lists(self,robot: URDF) -> None:
        """Generate parent, support and child list from URDF.

        Args:
            robot (URDF): URDF of robot.
        """
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
                if joint.child == name and joint.parent in link_names:
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
        print(link_names)
        print(parent_names)
        print(connection_type)
        
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
            while body_index[i2] == -1: # find first non fixed parent
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
        self, num: float, *args, max_denominator: int=0, tolerance: Optional[float]=None, **kwargs
        ) -> sympy.Expr | float:
        """Find a simple sympy representation for a number like 1/2 
        instead of 0.5. This function extends sympy.nsimplify with a 
        parameter to specify a maximum denominator to avoid simplifications
        like 13/157.  

        Args:
            num (float): number to simplify.
            max_denominator (int, optional): 
                maximum denominator to use. Use 0 to deactivate. 
                Defaults to 0.
            tolerance (float, optional): 
                tolerance for simplify_numbers. Defaults to 0.0001.
            
        Returns:
            sympy.Expr | float: simplified number.
        """
        if tolerance:
            if numpy.abs(num) < tolerance:
                return 0
            elif numpy.abs(1-num) < tolerance:
                return 1
            elif numpy.abs(-1-num) < tolerance:
                return -1 
        ex = nsimplify(num,*args, rational=False, tolerance=tolerance, **kwargs)
        if not max_denominator:
            return ex
        if ex.is_rational:
            try:
                try:
                    d = ex.denominator()
                except TypeError:
                    d = ex.denominator 
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
        
    def load_from_urdf(self, path: str, symbolic: bool=False, 
                       cse: bool=False, simplify_numbers: bool=True,  
                       tolerance: float=0.0001, max_denominator: int=9) -> None:
        """Load robot from urdf.

        Args:
            path (str): path to URDF.
            symbolic (bool, optional): 
                generate symbols for numeric values. 
                Defaults to False.
            cse (bool, optional): 
                use common subexpression elimination. Defaults to False.
            simplify_numbers (bool, optional): 
                Use eg. pi/2 instead of 1.5708. Defaults to True.
            tolerance (float, optional): 
                tolerance for simplify_numbers. Defaults to 0.0001.
            max_denominator (int, optional): 
                Maximum denominator to use for simplify numbers to avoid
                values like 13/153. Use 0 to deactivate. Defaults to 9.

        Raises:
            NotImplementedError: supports only the joint types 
                "revolute", "continuous" and "prismatic".
        """        
        # load URDF
        robot = URDF.from_xml_file(path)
        
        self.config_representation = self.BODY_FIXED
        self.B = []
        self.X = []
        
        # get parent, child and support array
        self._create_topology_lists(robot)
        # init some variables for later
        fixed_origin = None
        fixed_links = []
        DOF = 0
        xyz_rpy_syms = []
        
        # count DOF
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
        joint_origins_dict = {}
        for joint in robot.joints:
            name = joint.name
            
            # get SE3 pose from xyzrpy
            origin = xyz_rpy_to_matrix(joint.origin.xyz+joint.origin.rpy)
            if symbolic:
                xyz_rpy = Matrix(joint.origin.xyz+joint.origin.rpy)
                # substitute variables
                xyz_rpy_syms.append(
                    symbols(" ".join([name+"_%s" % s 
                                      for s in ["x", "y", "z", "roll", "pitch", "yar"]])))
                xyzrpylist = []
                if simplify_numbers:
                    # replace values like 3.142 with pi
                    for i in range(6):
                        if (self._nsimplify(xyz_rpy[i], 
                                           tolerance=tolerance, 
                                           max_denominator=max_denominator) 
                            in [0, -1, 1, pi, -pi, pi/2, -pi/2, 3*pi/2, -3*pi/2]
                            ):
                            xyzrpylist.append(
                                self._nsimplify(xyz_rpy[i], tolerance=tolerance,
                                                max_denominator=max_denominator))
                        else:
                            xyzrpylist.append(xyz_rpy_syms[jia][i])
                            self.assignment_dict[xyz_rpy_syms[jia]
                                                 [i]] = xyz_rpy[i]
                else:
                    # replace all values unequal 0, abs(1) with symbolic values
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
                # get new SE3 Pose from xyzrpy
                origin = xyz_rpy_to_matrix(xyzrpylist)
                if cse:
                    origin = self._cse_expression(origin)
            elif simplify_numbers:
                for i in range(4):
                    for j in range(4):
                        origin[i, j] = self._nsimplify(
                            origin[i, j], [pi], tolerance=tolerance,
                            max_denominator=max_denominator)
            # save SE3 pose in list
            joint_origins.append(origin)
            joint_origins_dict[name] = origin
            
            # add fixed parents to origin
            if joint.joint_type in ["revolute", "continuous", "prismatic"]:
                axis = Matrix(joint.axis)
                
                # calc fixed origin from parent origins:
                fixed_origin = Matrix(Identity(4))
                parent_joint = copy.copy(joint)
                while True :
                    try:
                        parent_joint = robot.joint_map[robot.parent_map[parent_joint.parent][0]]
                    except KeyError: # base_link has no parent
                        break
                    if parent_joint.type == "fixed":
                        fixed_origin = joint_origins_dict[parent_joint.name] * fixed_origin
                    else: # stop when last non fixed joint is reached
                        break

                # transform
                origin = fixed_origin * origin 
                
                if simplify_numbers:
                    for i in range(3):
                        axis[i] = self._nsimplify(axis[i], [pi], tolerance=tolerance,
                                            max_denominator=max_denominator)
                
                self.B.append(Matrix(origin))

                if joint.joint_type in ["revolute", "continuous"]:
                    self.X.append(Matrix(axis).col_join(Matrix([0, 0, 0])))
                else:
                    self.X.append(Matrix(Matrix([0, 0, 0])).col_join(axis))
                ji += 1
            elif joint.joint_type == "fixed":
                fixed_links.append((joint.parent, joint.child))
            jia += 1

        self.Mb = []
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
            inertia = Matrix(link.inertial.inertia.to_matrix())
            mass = link.inertial.mass
            inertiaorigin = xyz_rpy_to_matrix(link.inertial.origin.xyz+link.inertial.origin.rpy)
            if symbolic:
                I_syms = symbols("Ixx_%s Ixy_%s Ixz_%s Iyy_%s Iyz_%s Izz_%s" % (
                    name, name, name, name, name, name))
                c_syms = symbols("cx_%s cy_%s cz_%s" % (name, name, name))
                I = inertia_matrix(*I_syms)
                m = symbols("m_%s" % name)
                cg = Matrix([*c_syms])
            else:
                I = Matrix(inertia)
                m = mass
                cg = Matrix(inertiaorigin[0:3, 3])
            M = mass_matrix_mixed_data(m, I, cg)
            # if link is child of fixed joint
            if name in [x[1] for x in fixed_links]:
                parent_joint = robot.joint_map[robot.parent_map[name][0]]
                while True:
                    if parent_joint.type == 'fixed':
                        M = (SE3AdjInvMatrix(joint_origins_dict[parent_joint.name]).T 
                             * M * SE3AdjInvMatrix(joint_origins_dict[parent_joint.name]))
                    else:
                        break
                    try:
                        parent_joint = robot.joint_map[robot.parent_map[parent_joint.parent][0]]
                    except KeyError:
                        break      
                # j = i
                # # transform Mass matrix
                # while robot.links[j].name in [x[1] for x in fixed_links]:
                #     M = (SE3AdjInvMatrix(joint_origins[j-1]).T 
                #          * M * SE3AdjInvMatrix(joint_origins[j-1]))
                #     j -= 1
                try:
                    self.Mb[-1] += M
                except IndexError: # Base dynamics not important
                    pass
                i += 1
                continue
            self.Mb.append(M)
            i += 1
        if simplify_numbers and not symbolic:
            for M in self.Mb:
                for i in range(6):
                    for j in range(6):
                        M[i, j] = self._nsimplify(
                            M[i, j], [pi], tolerance=tolerance,
                            max_denominator=max_denominator)
        if self.ee is None:
            self.ee = [transformation_matrix()]
        if self.gravity_vector is None:
            self.gravity_vector = Matrix([0,0,-9.81])
        if self.ee_parent is None:
            self.ee_parent = DOF
        return

    def to_yaml(self, path: str="robot.yaml") -> None:
        """Save robot as YAML file.

        Args:
            path (str, optional): Path where to save .yaml file. 
                Defaults to "robot.yaml".
        """
        return skidy.parser.skd_to_yaml(self,path)
    
    def to_json(self, path: str="robot.json") -> None:
        """Save robot as JSON file.

        Args:
            path (str, optional): Path where to save .json file. 
                Defaults to "robot.json".
        """
        return skidy.parser.skd_to_json(self,path)
    
    def dh_to_screw_coord(self, DH_param_table: MutableDenseMatrix) -> None:
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
        self, name: str, cse: bool=False, simplify: bool=True) -> None:
        """Start Process, which simplifies and overwrites value in 
        queue from self.queue_dict.

        Args:
            name (str): Identifier
            cse (bool, optional): 
                Use common subexpression elimination. Defaults to False.
            simplify (bool, optional): 
                deactivate simplification by setting simplify to False. 
                Defaults to True.
        """
        if not simplify: 
            if not cse:
                return
            else:
                self._start_cse_process(name)
                return
        if name not in self.queue_dict:
            self.queue_dict[name] = Queue()
        self.process_dict[name+"_simplify"] = Process(
            target=self._simplify_parallel, 
            args=(name, cse,), 
            name=name+"_simplify")
        self.process_dict[name+"_simplify"].start()
        return

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

    def _simplify_parallel(self, name: str, cse: bool=False) -> None:
        """Take value from self.queue_dict, simplify it and put it in 
        again.

        Args:
            name (str): Identifier
            cse (bool, optional): 
                Use common subexpression elimination. Defaults to False.
        """
        value = self.simplify(self.queue_dict[name].get(), cse)
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
        self, exclude: list=[], i: list[int]=[0]) -> Generator[sympy.Symbol, None, None]:
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
        r, e = sympy.cse([exp, exp], self._individual_numbered_symbols(
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

    def _get_expressions(self) -> list[sympy.Expr]:
        """Get list of all generated expressions.

        Returns:
            list: generated expressions.
        """
        expression_dict = self.get_expressions_dict()
        expressions = [expression_dict[i] for i in expression_dict]
        return expressions

    def _calc_A_matrix(
        self, q: MutableDenseMatrix
        ) -> tuple[list[MutableDenseMatrix], MutableDenseMatrix]:
        """Calculate forward kinematics and the block diagonal matrix 
        A (6n x 6n) of the Adjoint of body frame for serial robots.

        Args:
            q (sympy.MutableDenseMatrix): 
                Generalized position vector.

        Raises:
            ValueError: 
                Joint screw coordinates or body reference configuration 
                not found.

        Returns:
            tuple[list[MutableDenseMatrix], MutableDenseMatrix]: (FK, A)
        """
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

    def _calc_A_matrix_tree(
        self, q: MutableDenseMatrix
        ) -> tuple[list[MutableDenseMatrix], MutableDenseMatrix]:
        """Calculate forward kinematics and the block diagonal matrix 
        A (6n x 6n) of the Adjoint of body frame for tree like robot 
        structures.

        Args:
            q (sympy.MutableDenseMatrix): 
                Generalized position vector.

        Raises:
            ValueError: 
                Joint screw coordinates or body reference configuration 
                not found.

        Returns:
            tuple[list[MutableDenseMatrix], MutableDenseMatrix]: (FK, A)
        """
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
        """Guess if 0 or 1 is the first index in the robot by analysing 
        used variable names.

        Returns:
            int: index.
        """
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

    def _save_vectors(self, q: MutableDenseMatrix, qd: MutableDenseMatrix, 
                      q2d: MutableDenseMatrix, q3d: MutableDenseMatrix, 
                      q4d: MutableDenseMatrix, WEE: MutableDenseMatrix, 
                      WDEE: MutableDenseMatrix, W2DEE: MutableDenseMatrix) -> None:
        """Save generalized vectors als class parameter.

        Args:
            q (sympy.Matrix): 
                (n,1) Generalized position vector.
            qd (sympy.Matrix): 
                (n,1) Generalized velocity vector.
            q2d (sympy.Matrix): 
                (n,1) Generalized acceleration vector.
            q3d (sympy.Matrix): 
                (n,1) Generalized jerk vector.
            q4d (sympy.Matrix): 
                (n,1) Generalized jounce vector.
            WEE (sympy.Matrix): 
                (6,1) WEE (t) = [Mx,My,Mz,Fx,Fy,Fz] is the time varying 
                wrench on the EE link. 
            WDEE (sympy.Matrix): 
                (6,1) WDEE (t) = [dMx,dMy,dMz,dFx,dFy,dFz] is the derivative 
                of the time varying wrench on the EE link. 
            W2DEE (sympy.Matrix): 
                (6,1) W2DEE (t) = [ddMx,ddMy,ddMz,ddFx,ddFy,ddFz] is the 
                2nd derivative of the time varying wrench on the EE link. 
        """
        if not self.q: self.q = q 
        if not self.qd: self.qd = qd 
        if not self.q2d: self.q2d = q2d 
        if not self.q3d: self.q3d = q3d 
        if not self.q4d: self.q4d = q4d
        if self.WEE == zeros(6,1) and WEE is not Ellipsis: self.WEE = WEE
        if self.WDEE == zeros(6,1) and WDEE is not Ellipsis: self.WDEE = WDEE
        if self.W2DEE == zeros(6,1) and W2DEE is not Ellipsis: self.W2DEE = W2DEE
        
    def _time_derivative(self, expression: sympy.Expr, level: int=1) -> sympy.Expr:
        """Generate time derivative of expression by substituting 
        generalized vectors with time dependent functions,
        let sympy calculate the time derivative, and resubstitute the 
        generalized vectors.

        Used for unit testing.
        
        Warning: This does not support external wrenches yet.

        Args:
            expression (sympy.Expr): 
                expression dependent on generalized vectors.
            level (int, optional): 
                Order of derivative to evaluate. Defaults to 1.

        Returns:
            sympy.Expr: Derived expression.
        """
        # generate dynamic symbols as substitutes
        startindex = 1
        if self.n > 1:
            subq = Matrix(sympy.physics.mechanics.dynamicsymbols(
                " ".join(f"subq{i}" for i in range(startindex,startindex+self.n))))
            subqd =  subq.diff()
            subq2d = subqd.diff() 
            subq3d = subq2d.diff() 
            subq4d = subq3d.diff() 
        else:
            subq = Matrix([sympy.physics.mechanics.dynamicsymbols(
                " ".join(f"subq{i}" for i in range(startindex,startindex+self.n)))])
            subqd =  subq.diff()
            subq2d = subqd.diff() 
            subq3d = subq2d.diff() 
            subq4d = subq3d.diff()
        # substitute
        expression = (expression.subs(zip(self.q,subq))
                                .subs(zip(self.qd,subqd))
                                .subs(zip(self.q2d,subq2d))
                                .subs(zip(self.q3d,subq3d))
                                .subs(zip(self.q4d,subq4d)))
        # derivative
        for _ in range(level):
            expression = expression.diff("t")
        # resubstitute
        expression = (expression.subs(zip(subq4d,self.q4d))
                                .subs(zip(subq3d,self.q3d))
                                .subs(zip(subq2d,self.q2d))
                                .subs(zip(subqd,self.qd))
                                .subs(zip(subq,self.q))
                                )
        return expression
    
    def __repr__(self) -> str:
        return (f"SymbolicKinDyn(ee={self.ee}, "
               f"body_ref_config={self.body_ref_config}, "
               f"joint_screw_coord={self.joint_screw_coord}, "
               f"config_representation=\'{self.config_representation}\', "
               f"Mb={self.Mb}, "
               f"parent={self.parent}, "
               f"support={self.support}, "
               f"child={self.child}, "
               f"ee_parent={self.ee_parent}, "
               f"q={self.q}, "
               f"qd={self.qd}, "
               f"q2d={self.q2d}, "
               f"q3d={self.q3d}, "
               f"q4d={self.q4d}, "
               f"WEE={self.WEE}, "
               f"WDEE={self.WDEE}, "
               f"W2DEE={self.W2DEE})")