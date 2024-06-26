#!/usr/bin/env python3

import os
import sys
import argparse
from skidy.parser import (
    robot_from_yaml, generate_template_yaml,
    robot_from_json, generate_template_json,
    robot_from_urdf, generate_template_python,
    urdf_to_yaml, urdf_to_json 
)
from skidy import __version__

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m skidy",
        description="Symbolic kinematics and dynamics model generation using Equations of Motion in closed form.",
    )
    # TODO: add urdf to description as soon as supported.
    parser.add_argument("filename",type=str, help=".yaml or .json file with definition of robot to analyse or location where to save generated template with '--template' option.")
    parser.add_argument('--version', action='version',
                    version='skidy {version}'.format(version=__version__))
    parser.add_argument("-s", "--simplify", action="store_true", help="simplify expressions")
    parser.add_argument("--cse", action="store_true", help="use common subexpression elimination to shorten expressions")
    parser.add_argument("-p", "--python", action="store_true", help="generate python code")
    parser.add_argument("-c","--cpp", action="store_true", help="generate C++ code")
    parser.add_argument("-C","--C", action="store_true", help="generate C code")
    parser.add_argument("-m","--matlab", action="store_true", help="generate Matlab/Octave code")
    parser.add_argument("-j","--julia", action="store_true", help="generate julia code")
    parser.add_argument("-l","--latex", action="store_true", help="generate LaTeX code and pdf")
    parser.add_argument("-g","--graph", action="store_true", help="generate structure graph from input and save it as pdf")
    parser.add_argument("--cython", action="store_true", help="generate Cython code")
    parser.add_argument("--no-kinematics", action="store_true", help="skip generation of kinematics equations.")
    parser.add_argument("--no-dynamics", action="store_true", help="skip generation of dynamics equations.")
    parser.add_argument("-f", "--folder", type=str, default="./generated_code", help="folder where to save generated code. Defaults to './generated_code'")
    parser.add_argument("--serial", "--not-parallel", action="store_false", default=True, help="don't use parallel computation")
    parser.add_argument("-n","--name", type=str, default="", help="name of class and file. Defaults to filename")
    parser.add_argument("--project", type=str, default="Project", help="project name in C header. Defaults to 'Project'.")
    parser.add_argument("--cache", action="store_true", help="Use LRU cache for sin and cos function in generated python and cython code to speed up calculations.")
    # parser.add_argument("--yaml",action="store_true", help="enforce yaml file")
    # parser.add_argument("--json",action="store_true", help="enforce json file")
    # parser.add_argument("--urdf",action="store_true", help="enforce urdf file")

    # options for template generation
    generate_template = parser.add_argument_group("Options for template Generation (available formats: yaml, json, python)")
    generate_template.add_argument("-T","--template", "--please", action="store_true", help="store template yaml or json file to edit instead of analyzing existing robot. Format is chosen by extension of filename")
    generate_template.add_argument("-S", "--structure", type=str, help="structure of robot template: String containing only 'r' and 'p' of joint order. E.g.: Use 'prr' for a robot which has 1 prismatic joint followed by 2 revolute joints.")
    generate_template.add_argument("-d", "--dof", type=int, default=0, help="degrees of freedom. Is usually calculated by length of 'structure'.")
    generate_template.add_argument("--urdf",action="store_true", help="use urdf in generated python file")
    generate_template.add_argument("--yaml",action="store_true", help="enforce yaml file generation")
    generate_template.add_argument("--json",action="store_true", help="enforce json file generation")
    generate_template.add_argument("--convert", metavar="URDF_PATH", type=str, default=None, help="Convert URDF to YAML or JSON. Use -s option to round numbers if close to 0 or 1 (recommended)")
    generate_template.add_argument("--symbolify", action="store_true", help="Symbolify numbers during conversion. Defaults to False.")
    # generate_template.add_argument("--python", action="store_true", help="enforce python file generation")


    args = parser.parse_args()
    
    path = args.filename
    name, ext = os.path.splitext(path)
    
    # generate empty yaml or json file
    if args.convert or args.template:
        structure = args.structure
        dof = args.dof
        # yaml
        if args.yaml or ext in {".yaml",".YAML","yml"}:
            if not ext in {".yaml",".YAML","yml"}:
                path = name+".yaml"
            if (os.path.exists(path) 
                and not input(f"{path} already exists.\nOverwrite existing file? [Y/n] ") in {"","y","Y"}):
                print("Abort execution.")
                exit()
            if args.convert:
                urdf_to_yaml(args.convert,path,symbolic=args.symbolify, cse=args.cse,
                                  simplify_numbers=args.simplify,tolerance=0.0001, 
                                  max_denominator=9)
            else:
                generate_template_yaml(path,structure,dof)    
        # json
        if args.json or ext in {".json",".JSON"}:
            if ext not in {".json",".JSON"}:
                path = name+".json"
            if (os.path.exists(path) 
                and not input(f"{path} already exists.\nOverwrite existing file? [Y/n] ") in {"","y","Y"}):
                print("Abort execution.")
                exit()
            if args.convert:
                urdf_to_json(args.convert,path,symbolic=args.symbolify, cse=args.cse,
                                  simplify_numbers=args.simplify,tolerance=0.0001, 
                                  max_denominator=9)
            else:
                generate_template_json(path,structure,dof)
        # python
        if args.python or ext in {".py"}:
            if ext not in {".py"}:
                path = name+".py"
            if (os.path.exists(path) 
                and not input(f"{path} already exists.\nOverwrite existing file? [Y/n] ") in {"","y","Y"}):
                print("Abort execution.")
                exit()
            generate_template_python(path,structure,dof,urdf=args.urdf)
    else:
        if not os.path.exists(path) or not os.path.isfile(path):
            raise ValueError(f"{path} is no existing file.")
        
        if args.name == "": args.name = os.path.basename(name)
        
        if (not args.python 
            and not args.C 
            and not args.cpp
            and not args.matlab 
            and not args.julia 
            and not args.latex 
            and not args.cython
            and not args.graph):
            raise ValueError("Please provide at least one programming language in which I should generate code.")
        
        if ext in {".yaml",".YAML","yml"}:
            skd = robot_from_yaml(path)
            
        elif ext in {".json",".JSON"}:
            skd = robot_from_json(path)
            
        elif ext in {".urdf"}:
            skd = robot_from_urdf(path,symbolic=False, cse=args.cse,
                                  simplify_numbers=True,tolerance=0.0001, 
                                  max_denominator=8)
        
        # elif ext in {".urdf"}:
        #     skd = robot_from_urdf(path,cse=args.cse) # TODO: add options
        #     # TODO: deal with ee

        elif ext == ".py":
            if input(f"{path} is a python file. Should I run this file? [y/N] ") in {"y","Y"}:
                os.system(f"{sys.executable} {path}")
            else:
                print("Abort execution.")
            exit()
        else: 
            raise ValueError("File extension not recognized.")
        
        if (args.python 
            or args.C 
            or args.cpp
            or args.matlab 
            or args.julia 
            or args.latex 
            or args.cython):
            
            if not args.no_kinematics:
                skd.closed_form_kinematics_body_fixed(
                    simplify=args.simplify, cse=args.cse, parallel=args.serial)
            if not args.no_dynamics:
                skd.closed_form_inv_dyn_body_fixed(
                    simplify=args.simplify, cse=args.cse, parallel=args.serial)
    
            skd.generate_code(python=args.python, 
                              C=args.C, 
                              cpp=args.cpp,
                              Matlab=args.matlab, 
                              cython=args.cython, 
                              julia=args.julia, 
                              latex=args.latex, 
                              cse=args.cse,
                              cache=args.cache,
                              folder=args.folder, 
                              use_global_vars=True, 
                              name=args.name, 
                              project=args.project)
        if args.graph:
            if not os.path.exists(os.path.join(args.folder,"graphs")):
                os.mkdir(os.path.join(args.folder,"graphs"))
            skd.generate_graph(os.path.join(args.folder,"graphs",args.name+".pdf"),1)

if __name__ == "__main__":
    main()